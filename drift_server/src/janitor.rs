use crate::persistence::PersistenceManager;
use drift_core::index::{MaintenanceStatus, VectorIndex};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::{collections::HashSet, sync::atomic::Ordering};
use tokio::time;
use tracing::{error, info, instrument};

pub struct Janitor {
    index: Arc<VectorIndex>,
    persistence: PersistenceManager,
    flush_threshold: usize,
    check_interval: Duration,
    ignore_set: std::sync::Mutex<HashSet<u32>>,
}

impl Janitor {
    pub fn new(
        index: Arc<VectorIndex>,
        persistence: PersistenceManager,
        flush_threshold: usize,
        check_interval: Duration,
    ) -> Self {
        Self {
            index,
            persistence,
            flush_threshold,
            check_interval,
            ignore_set: std::sync::Mutex::new(HashSet::new()),
        }
    }

    pub async fn run(&self) {
        let mut interval = time::interval(self.check_interval);
        loop {
            interval.tick().await;

            // 1. Flush Logic
            // Check if we need to flush (Threshold met OR pending frozen work from a failed retry)
            let mem_size = self.index.memtable_len();
            let has_frozen = self.index.frozen_memtable.read().is_some();

            if mem_size >= self.flush_threshold || has_frozen {
                match self.perform_flush().await {
                    Ok(Some(p)) => info!("Janitor: Flushed to {:?}", p),
                    Ok(None) => {} // No work needed
                    Err(e) => error!("Janitor Error: Failed to flush: {}", e),
                }
            }

            // 2. Scavenge (Disk -> Disk Compaction)
            if let Err(e) = self.scavenge().await {
                error!("Janitor: Scavenge failed: {}", e);
            }

            // 3. Maintenance
            self.perform_maintenance().await;
        }
    }

    async fn perform_maintenance(&self) {
        let buckets = self.index.get_all_bucket_headers();
        let target_cap = self.index.config.max_bucket_capacity as u32;
        let mut ops_budget = 1;

        for header in buckets {
            if ops_budget == 0 {
                break;
            }
            {
                let guard = self.ignore_set.lock().unwrap();
                if guard.contains(&header.id) {
                    continue;
                }
            }

            let _ =
                header
                    .stats
                    .temperature
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| Some(t * 0.98));

            let urgency = header.calculate_urgency(target_cap as usize);

            // Make that value configurable
            if header.count > (target_cap as f32 * 1.5) as u32 {
                info!(
                    "Janitor: ✂️ Splitting Bucket {} (Count {})",
                    header.id, header.count
                );
                match self.index.split_and_steal(header.id).await {
                    Ok(MaintenanceStatus::Completed) => ops_budget -= 1,
                    Ok(MaintenanceStatus::SkippedSingularity) => {
                        info!("Janitor: Bucket {} is a Singularity. Ignoring.", header.id);
                        self.ignore_set.lock().unwrap().insert(header.id);
                    }
                    Ok(_) => {}
                    Err(e) => error!("Split failed: {}", e),
                }
            // } else if header.count == 0 || (header.count < 50 && capacity_ratio < 0.1) {
            } else if urgency > 1.5 {
                info!(
                    "Janitor: 🚑 Scatter Merging Bucket {} (Count {})",
                    header.id, header.count
                );
                self.ignore_set.lock().unwrap().remove(&header.id);
                match self.index.scatter_merge(header.id).await {
                    Ok(MaintenanceStatus::Completed) => ops_budget -= 1,
                    Ok(_) => {}
                    Err(e) => error!("Merge failed: {}", e),
                }
            }
        }
    }

    #[instrument(skip(self), level = "info")]
    async fn perform_flush(&self) -> std::io::Result<Option<std::path::PathBuf>> {
        let start = std::time::Instant::now();

        // 1. Extract Data from MemTable
        let memtable_arc = {
            let frozen = self.index.frozen_memtable.read();
            if let Some(m) = frozen.as_ref() {
                m.clone()
            } else {
                drop(frozen);
                match self.index.rotate_and_freeze()? {
                    Some(m) => m,
                    None => return Ok(None),
                }
            }
        };

        let data = memtable_arc.extract_all();
        if data.is_empty() {
            self.index.confirm_flush()?;
            return Ok(None);
        }

        let vectors: Vec<Vec<f32>> = data.iter().map(|(_, v)| v.clone()).collect();
        let ids: Vec<u64> = data.iter().map(|(id, _)| *id).collect();

        // 2. Train Index (if needed)
        {
            let q_guard = self.index.get_quantizer();
            if q_guard.is_none() {
                info!(count = data.len(), "Starting Index Training");
                self.index.train(&vectors).await?;
            }
        }

        // 3. ⚡ PARTITION (Calculate, don't write) ⚡
        let partitions = self.index.calculate_partitions(&ids, &vectors).await?;

        if partitions.is_empty() {
            self.index.confirm_flush()?;
            return Ok(None);
        }

        // 4. ⚡ PERSIST (Write Segment to S3 with Compression) ⚡
        // This uses SegmentWriter internally, preserving ALP/SQ8 structure.
        let run_id = self
            .persistence
            .write_partitioned_segment(&partitions, &self.index)
            .await?;

        // 5. ⚡ REGISTER (Update Memory) ⚡
        self.index.register_partitions(&partitions, &run_id).await?;

        // 6. Flush Tombstones
        let deleted_snapshot: Vec<u64> = self.index.deleted_ids.read().iter().cloned().collect();
        if !deleted_snapshot.is_empty() {
            self.persistence
                .flush_tombstones(&deleted_snapshot, &run_id)
                .await?;
        }

        // 7. Cleanup
        self.index.confirm_flush()?;

        info!(
            vectors = data.len(),
            buckets = partitions.len(),
            duration_ms = start.elapsed().as_millis(),
            "Flush Complete"
        );

        // Return dummy path for logging, or actual object key
        Ok(Some(PathBuf::from(format!("segment_{}.drift", run_id))))
    }

    // async fn perform_flush(&self) -> std::io::Result<Option<std::path::PathBuf>> {
    //     let start = std::time::Instant::now();

    //     // 1. Get Data (Retry logic)
    //     // If a frozen memtable exists (retry scenario), use it.
    //     // Otherwise, try to rotate.
    //     let memtable_arc = {
    //         let frozen = self.index.frozen_memtable.read();
    //         if let Some(m) = frozen.as_ref() {
    //             m.clone()
    //         } else {
    //             drop(frozen); // Release lock
    //             match self.index.rotate_and_freeze()? {
    //                 Some(m) => m,
    //                 None => return Ok(None),
    //             }
    //         }
    //     };

    //     let data = memtable_arc.extract_all();
    //     if data.is_empty() {
    //         self.index.confirm_flush()?; // Clear empty frozen slot
    //         return Ok(None);
    //     }

    //     let vectors: Vec<Vec<f32>> = data.iter().map(|(_, v)| v.clone()).collect();
    //     let ids: Vec<u64> = data.iter().map(|(id, _)| *id).collect();

    //     // 2. Train
    //     {
    //         let q_guard = self.index.get_quantizer();
    //         if q_guard.is_none() {
    //             info!(count = data.len(), "Starting Index Training");
    //             self.index.train(&vectors).await?;
    //             info!(
    //                 duration_ms = start.elapsed().as_millis(),
    //                 "Training Complete"
    //             );
    //         }
    //     }

    //     // 3. Persist
    //     // let new_id = self.index.allocate_next_bucket_id();
    //     let _new_bucket_ids = self.index.partition_and_flush(&ids, &vectors).await?;

    //     let run_id = uuid::Uuid::new_v4().to_string();
    //     let path = self
    //         .persistence
    //         .flush_memtable_to_segment(&data, &self.index, &run_id)
    //         .await?;

    //     // 4. Flush Tombstones
    //     // We snapshot the entire set. This is safe and robust for now.
    //     // Compaction (future) will delete these files when they are no longer needed.
    //     let deleted_snapshot: Vec<u64> = self.index.deleted_ids.read().iter().cloned().collect();

    //     // Cleanup
    //     // Only verify flush if persistence succeeded.
    //     if !deleted_snapshot.is_empty() {
    //         self.persistence
    //             .flush_tombstones(&deleted_snapshot, &run_id)
    //             .await?;
    //     }

    //     self.index.confirm_flush()?;

    //     info!(
    //         vectors = data.len(),
    //         duration_ms = start.elapsed().as_millis(),
    //         "Flush Complete"
    //     );
    //     Ok(Some(PathBuf::from(path)))
    // }

    // In drift_server/src/janitor.rs

    pub async fn scavenge(&self) -> std::io::Result<()> {
        // Optimization: If global deleted set is small, don't bother scanning.
        if self.index.deleted_ids.read().len() < 100 {
            return Ok(());
        }

        // 1. Identify Dirty Buckets
        // We scan headers and check the AtomicU32 tombstone_count.
        let headers = self.index.get_all_bucket_headers();
        let mut dirty_candidates = Vec::new();

        for header in headers {
            let total = header.count as f32;
            if total == 0.0 {
                continue;
            }

            // SAFETY: This load is atomic and safe.
            // It reflects the current count updated by delete() calls.
            let dead = header
                .stats
                .tombstone_count
                .load(std::sync::atomic::Ordering::Relaxed) as f32;

            // Threshold: > 20% dead items
            if (dead / total) > 0.20 {
                dirty_candidates.push(header.id);
            }
        }

        // 2. Compact One Candidate
        // We process one per cycle to be gentle on system resources.
        if let Some(bucket_id) = dirty_candidates.first() {
            info!("Janitor: Scavenging bucket {} (Dirty)", bucket_id);

            if let Ok(Some(purged_ids)) = self.index.compact_bucket(*bucket_id).await
                && !purged_ids.is_empty()
            {
                info!("Janitor: Purged {} IDs from memory.", purged_ids.len());

                // 3. FREE MEMORY
                // Now we can safely remove these IDs from the global filter
                let mut write_guard = self.index.deleted_ids.write();
                for id in purged_ids {
                    write_guard.remove(&id);
                }
            }
        }

        Ok(())
    }
}
