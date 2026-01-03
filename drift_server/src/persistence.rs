use bit_set::BitSet;
use byteorder::{LittleEndian, WriteBytesExt};
use drift_core::aligned::AlignedBytes;
use drift_core::bucket::BucketData;
use drift_core::index::{IndexOptions, PartitionResult, VectorIndex};
use drift_core::quantizer::Quantizer;
use drift_core::tombstone::TombstoneFile;
use drift_storage::disk_manager::{DiskManager, DriftPageManager};
use drift_storage::segment_reader::SegmentReader;
use drift_storage::segment_writer::{BucketLocation, SegmentWriter};
use opendal::Entry;
use opendal::Operator;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn};

// const MAGIC: u64 = 0x44524946545631; // "DRIFTV1" in hex
pub const MAGIC_BYTES: &[u8; 8] = b"DRIFT_V1";

#[derive(Clone)]
pub struct PersistenceManager {
    op: Operator,
    local_base_path: PathBuf,
}

impl PersistenceManager {
    pub fn new(op: Operator, local_path: impl Into<PathBuf>) -> Self {
        Self {
            op,
            local_base_path: local_path.into(),
        }
    }

    // pub async fn flush_to_segment(
    //     &self,
    //     index: &VectorIndex,
    //     run_id: &str,
    // ) -> std::io::Result<String> {
    //     // Returns object key (filename)
    //     let file_name = format!("segment_{}.drift", run_id);

    //     let quantizer_arc = index.get_quantizer().ok_or(std::io::Error::new(
    //         std::io::ErrorKind::InvalidInput,
    //         "Untrained",
    //     ))?;
    //     let q_bytes = bincode::encode_to_vec(&*quantizer_arc, bincode::config::standard())
    //         .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    //     // ⚡ Use DI: DiskManager::new(op, path)
    //     let manager = DiskManager::new(self.op.clone(), file_name.clone());
    //     let mut writer = SegmentWriter::new(manager, q_bytes).await?;

    //     let headers = index.get_all_bucket_headers();
    //     for header in headers {
    //         if let Ok(data) = index.cache.get(&header.page_id).await
    //             && !data.vids.is_empty()
    //         {
    //             writer
    //                 .write_bucket_sq8(
    //                     header.id,
    //                     &data.vids,
    //                     data.codes.as_slice(),
    //                     index.config.dim,
    //                 )
    //                 .await?;
    //         }
    //     }
    //     writer.finalize().await?;
    //     Ok(file_name)
    // }

    // pub async fn flush_memtable_to_segment(
    //     &self,
    //     data: &[(u64, Vec<f32>)],
    //     index: &VectorIndex,
    //     run_id: &str,
    // ) -> std::io::Result<String> {
    //     let file_name = format!("segment_l0_{}.drift", run_id);

    //     let quantizer_arc = index.get_quantizer().ok_or(std::io::Error::new(
    //         std::io::ErrorKind::InvalidInput,
    //         "Untrained",
    //     ))?;
    //     let q_bytes = bincode::encode_to_vec(&*quantizer_arc, bincode::config::standard())
    //         .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    //     // ⚡ Use DI
    //     let manager = DiskManager::new(self.op.clone(), file_name.clone());
    //     let mut writer = SegmentWriter::new(manager, q_bytes).await?;

    //     let ids: Vec<u64> = data.iter().map(|(id, _)| *id).collect();
    //     let vecs: Vec<Vec<f32>> = data.iter().map(|(_, v)| v.clone()).collect();

    //     if !vecs.is_empty() {
    //         let dim = index.config.dim;
    //         let mut flat_codes = Vec::with_capacity(vecs.len() * dim);
    //         for v in &vecs {
    //             flat_codes.extend_from_slice(&quantizer_arc.encode(v));
    //         }

    //         writer
    //             .write_bucket_dual(0, &ids, &vecs, &flat_codes, dim)
    //             .await?;
    //     }

    //     writer.finalize().await?;
    //     Ok(file_name)
    // }

    // Note: 'path' here is a relative object key string, not a PathBuf
    // pub async fn load_from_segment(&self, object_key: &str) -> std::io::Result<VectorIndex> {
    //     // ⚡ Use DI: SegmentReader must act on the injected operator
    //     // We will add a helper to SegmentReader to accept (Operator, path)
    //     let reader = SegmentReader::open_with_op(self.op.clone(), object_key).await?;

    //     let q_bytes = reader.read_metadata();
    //     let (quantizer, _): (Quantizer, usize) =
    //         bincode::decode_from_slice(q_bytes, bincode::config::standard()).map_err(|_| {
    //             std::io::Error::new(std::io::ErrorKind::InvalidData, "Corrupt Quantizer")
    //         })?;
    //     let dim = quantizer.min.len();

    //     // Use default options for recovery
    //     let options = IndexOptions {
    //         dim,
    //         num_centroids: 0,
    //         training_sample_size: 0,
    //         max_bucket_capacity: 1000,
    //         ..Default::default()
    //     };

    //     // Rebuild storage layout
    //     let storage_path = self.local_base_path.join("storage");
    //     std::fs::create_dir_all(&storage_path)?;

    //     // Use DriftPageManager with the same Operator!
    //     let storage = Arc::new(DriftPageManager::new(self.op.clone()));
    //     let wal_path = self.local_base_path.join("current.wal");

    //     let index = VectorIndex::new(options, &wal_path, storage)?;
    //     index.set_quantizer(quantizer.clone());

    //     let bucket_ids: Vec<u32> = reader.index.buckets.keys().cloned().collect();
    //     for id in bucket_ids {
    //         let vectors = match reader.read_bucket_high_fidelity(id).await {
    //             Ok(v) => v,
    //             Err(_) => {
    //                 info!("Recovering bucket {} from SQ8", id);
    //                 let (_, codes) = reader.read_bucket(id).await?;
    //                 let count = codes.len() / dim;
    //                 let mut rec_vecs = Vec::with_capacity(count);
    //                 for i in 0..count {
    //                     let start = i * dim;
    //                     rec_vecs.push(quantizer.reconstruct(&codes[start..start + dim]));
    //                 }
    //                 rec_vecs
    //             }
    //         };
    //         let (ids, _) = reader.read_bucket(id).await?;
    //         index
    //             .force_register_bucket_with_ids(id, &ids, &vectors)
    //             .await?;
    //     }
    //     Ok(index)
    // }

    pub async fn hydrate_index(&self, index: &VectorIndex) -> std::io::Result<()> {
        // List files from the Operator
        let _lister = self.op.lister("").await.map_err(std::io::Error::other)?;
        // We need to collect and filter
        // OpenDAL Lister is async stream
        // Basic list loop (simplified for brevity, assume < 1000 files for now)
        // In production use streaming
        let entries: Vec<Entry> = self.op.list("").await.map_err(std::io::Error::other)?;

        let mut paths = Vec::new();
        for entry in entries {
            let path = entry.path();
            if path.ends_with(".drift") {
                paths.push(path.to_string());
            }
        }
        paths.sort();

        for path in paths {
            // ⚡ Use DI
            let reader = SegmentReader::open_with_op(self.op.clone(), &path).await?;

            if index.get_quantizer().is_none() {
                let q_bytes = reader.read_metadata();
                let (quantizer, _): (Quantizer, usize) = bincode::decode_from_slice(
                    q_bytes,
                    bincode::config::standard(),
                )
                .map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "Corrupt Quantizer")
                })?;
                index.set_quantizer(quantizer);
            }

            let q_ref = index.get_quantizer();
            let bucket_ids: Vec<u32> = reader.index.buckets.keys().cloned().collect();
            for _old_id in bucket_ids {
                let vectors = match reader.read_bucket_high_fidelity(_old_id).await {
                    Ok(v) => v,
                    Err(_) => {
                        // Fallback logic (same as before)
                        let (_, codes) = reader.read_bucket(_old_id).await?;
                        if let Some(q) = &q_ref {
                            let dim = index.config.dim;
                            let count = codes.len() / dim;
                            let mut rec_vecs = Vec::with_capacity(count);
                            for i in 0..count {
                                rec_vecs.push(q.reconstruct(&codes[i * dim..(i + 1) * dim]));
                            }
                            rec_vecs
                        } else {
                            return Err(std::io::Error::other("Missing Quantizer"));
                        }
                    }
                };
                let (ids, _) = reader.read_bucket(_old_id).await?;
                let new_id = index.allocate_next_bucket_id();
                index
                    .force_register_bucket_with_ids(new_id, &ids, &vectors)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn write_segment_file(
        &self,
        buckets: &[(u32, BucketData)],
        dim: usize,
    ) -> std::io::Result<String> {
        let run_id = uuid::Uuid::new_v4().to_string();
        let file_name = format!("segment_{}.drift", run_id);

        // We buffer the segment in memory before uploading.
        // In a high-throughput scenario, you would stream this via SegmentWriter,
        // but for correctness and simplicity now, we pack the buffer manually.
        let mut buffer = Vec::new();

        // 1. Write Data Blocks (The Buckets)
        let mut bucket_index = Vec::new(); // (ID, Offset, Length)

        for (id, bucket) in buckets {
            let start_offset = buffer.len() as u64;
            // Serialize bucket using existing logic
            let bytes = bucket.to_bytes(dim)?;
            buffer.write_all(&bytes)?;
            let length = buffer.len() as u64 - start_offset;

            bucket_index.push((*id, start_offset, length));
        }

        // 2. Write Bucket Index
        let index_start = buffer.len() as u64;
        buffer.write_u32::<LittleEndian>(bucket_index.len() as u32)?;

        for (id, offset, length) in bucket_index {
            buffer.write_u32::<LittleEndian>(id)?;
            buffer.write_u64::<LittleEndian>(offset)?;
            buffer.write_u64::<LittleEndian>(length)?;
        }

        // 3. Write Footer (Fixed 64 bytes)
        // Layout: [Magic(8) | IndexOffset(8) | BloomFilterOffset(8) | Padding...]
        let footer_start = buffer.len();
        buffer.write_all(MAGIC_BYTES)?;
        buffer.write_u64::<LittleEndian>(index_start)?;
        buffer.write_u64::<LittleEndian>(0)?; // Bloom Filter Offset (0 = None)

        // Pad to 64 bytes
        while buffer.len() - footer_start < 64 {
            buffer.write_u8(0)?;
        }

        // 4. Upload to S3/MinIO
        self.op
            .write(&file_name, buffer)
            .await
            .map_err(std::io::Error::other)?;

        info!(
            "Persistence: Wrote segment {} with {} buckets",
            file_name,
            buckets.len()
        );

        // Return the RUN_ID (raw UUID), not the filename.
        // The Janitor uses this ID to name related files (like tombstones).
        Ok(run_id)
    }

    // /// Writes multiple partitions into a single Segment File.
    // /// Preserves ALP compression and Dual-Tier layout.
    // pub async fn write_partitioned_segment(
    //     &self,
    //     partitions: &[PartitionResult],
    //     index: &VectorIndex,
    // ) -> std::io::Result<(String, HashMap<u32, BucketLocation>)> {
    //     let run_id = uuid::Uuid::new_v4().to_string();
    //     let file_name = format!("segment_{}.drift", run_id);

    //     let quantizer_arc = index
    //         .get_quantizer()
    //         .ok_or(std::io::Error::other("Untrained"))?;
    //     let q_bytes = bincode::encode_to_vec(&*quantizer_arc, bincode::config::standard())
    //         .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    //     let manager = DiskManager::new(self.op.clone(), file_name.clone());
    //     let mut writer = SegmentWriter::new(manager, q_bytes).await?;

    //     for p in partitions {
    //         // Reconstruct the BucketData object for the writer
    //         // (We created one for the cache in index.rs, here we create it for disk)
    //         let bucket_data = BucketData {
    //             codes: AlignedBytes::from_slice(&p.codes),
    //             vids: p.ids.clone(),
    //             tombstones: BitSet::with_capacity(p.ids.len()),
    //         };

    //         writer
    //             .write_partition(
    //                 p.bucket_id,
    //                 &bucket_data, // ⚡ Pass the struct
    //                 &p.vectors,   // Raw floats for ALP
    //                 index.config.dim,
    //             )
    //             .await?;
    //     }

    //     let location = writer.finalize().await?;
    //     info!(
    //         "Persistence: Wrote segment {} with {} buckets",
    //         file_name,
    //         partitions.len()
    //     );
    //     Ok((run_id, location))
    // }

    /// This is the primary path used by the Janitor.
    pub async fn write_partitioned_segment(
        &self,
        partitions: &[PartitionResult],
        index: &VectorIndex,
    ) -> std::io::Result<(String, HashMap<u32, BucketLocation>)> {
        let run_id = uuid::Uuid::new_v4().to_string();
        let file_name = format!("segment_{}.drift", run_id);

        let quantizer_arc = index
            .get_quantizer()
            .ok_or(std::io::Error::other("Untrained"))?;
        let q_bytes = bincode::encode_to_vec(&*quantizer_arc, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let manager = DiskManager::new(self.op.clone(), file_name.clone());
        let mut writer = SegmentWriter::new(manager, q_bytes).await?;

        for p in partitions {
            // Reconstruct the BucketData object for the writer
            // This ensures we have the correct structure for to_bytes()
            let bucket_data = BucketData {
                codes: AlignedBytes::from_slice(&p.codes),
                vids: p.ids.clone(),
                tombstones: BitSet::with_capacity(p.ids.len()),
            };

            writer
                .write_partition(
                    p.bucket_id,
                    &bucket_data, // ⚡ Pass full struct (Format Source of Truth)
                    &p.vectors,   // Raw floats for ALP
                    index.config.dim,
                )
                .await?;
        }

        let location = writer.finalize().await?;
        info!(
            "Persistence: Wrote segment {} with {} buckets",
            file_name,
            partitions.len()
        );
        Ok((run_id, location))
    }

    /// Updated to use `write_partition` to ensure format consistency.
    pub async fn flush_memtable_to_segment(
        &self,
        data: &[(u64, Vec<f32>)],
        index: &VectorIndex,
        run_id: &str,
    ) -> std::io::Result<String> {
        let file_name = format!("segment_l0_{}.drift", run_id);

        let quantizer_arc = index.get_quantizer().ok_or(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Untrained",
        ))?;
        let q_bytes = bincode::encode_to_vec(&*quantizer_arc, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let manager = DiskManager::new(self.op.clone(), file_name.clone());
        let mut writer = SegmentWriter::new(manager, q_bytes).await?;

        let ids: Vec<u64> = data.iter().map(|(id, _)| *id).collect();
        let vecs: Vec<Vec<f32>> = data.iter().map(|(_, v)| v.clone()).collect();

        if !vecs.is_empty() {
            let dim = index.config.dim;

            // Encode Codes
            let mut flat_codes = Vec::with_capacity(vecs.len() * dim);
            for v in &vecs {
                flat_codes.extend_from_slice(&quantizer_arc.encode(v));
            }

            // Wrap in BucketData
            let bucket = BucketData {
                codes: AlignedBytes::from_slice(&flat_codes),
                vids: ids,
                tombstones: BitSet::new(),
            };

            // Write as Bucket 0
            writer.write_partition(0, &bucket, &vecs, dim).await?;
        }

        writer.finalize().await?;
        Ok(file_name)
    }

    // Note: 'path' here is a relative object key string, not a PathBuf
    pub async fn load_from_segment(&self, object_key: &str) -> std::io::Result<VectorIndex> {
        // ⚡ Use DI: SegmentReader must act on the injected operator
        let reader = SegmentReader::open_with_op(self.op.clone(), object_key).await?;

        let q_bytes = reader.read_metadata();
        let (quantizer, _): (Quantizer, usize) =
            bincode::decode_from_slice(q_bytes, bincode::config::standard()).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Corrupt Quantizer")
            })?;
        let dim = quantizer.min.len();

        // Use default options for recovery
        let options = IndexOptions {
            dim,
            num_centroids: 0,
            training_sample_size: 0,
            max_bucket_capacity: 1000,
            ..Default::default()
        };

        // Rebuild storage layout
        let storage_path = self.local_base_path.join("storage");
        std::fs::create_dir_all(&storage_path)?;

        // Use DriftPageManager with the same Operator!
        let storage = Arc::new(DriftPageManager::new(self.op.clone()));
        let wal_path = self.local_base_path.join("current.wal");

        let index = VectorIndex::new(options, &wal_path, storage)?;
        index.set_quantizer(quantizer.clone());

        let bucket_ids: Vec<u32> = reader.index.buckets.keys().cloned().collect();

        // Register the file mapping for all buckets (needed for lazy loading)
        // We do this eagerly so force_register can work if it needs to read?
        // Actually force_register writes *new* pages to cache.
        // But here we want to restore from S3.

        for id in bucket_ids {
            // 1. Try High Fidelity (ALP)
            let vectors = match reader.read_bucket_high_fidelity(id).await {
                Ok(v) => v,
                Err(_) => {
                    info!("Recovering bucket {} from SQ8", id);
                    let (_, codes) = reader.read_bucket(id).await?;
                    let count = codes.len() / dim;
                    let mut rec_vecs = Vec::with_capacity(count);
                    for i in 0..count {
                        let start = i * dim;
                        rec_vecs.push(quantizer.reconstruct(&codes[start..start + dim]));
                    }
                    rec_vecs
                }
            };

            let (ids, _) = reader.read_bucket(id).await?;

            // This re-writes the cache page.
            // In a pure lazy system we wouldn't do this, but for crash recovery
            // it ensures the index is hot.
            index
                .force_register_bucket_with_ids(id, &ids, &vectors)
                .await?;
        }
        Ok(index)
    }

    pub async fn flush_tombstones(&self, ids: &[u64], run_id: &str) -> std::io::Result<String> {
        if ids.is_empty() {
            return Ok("".to_string());
        }

        let file_name = format!("tombstones_{}.drift", run_id);
        let file = TombstoneFile::new(ids.to_vec());
        let bytes = file.to_bytes()?;

        let manager = DiskManager::new(self.op.clone(), file_name.clone());
        manager.upload(bytes).await?;

        info!("Flushed {} tombstones to {}", ids.len(), file_name);
        Ok(file_name)
    }

    pub async fn load_all_tombstones(&self) -> std::io::Result<Vec<u64>> {
        let mut all_deleted = Vec::new();
        // Listing
        let entries: Vec<opendal::Entry> = self.op.list("").await.map_err(std::io::Error::other)?;

        for entry in entries {
            let path = entry.path();
            if path.starts_with("tombstones_") && path.ends_with(".drift") {
                let manager = DiskManager::new(self.op.clone(), path.to_string());
                let len = manager.len().await?;
                if len > 0 {
                    let bytes = manager.read_at(0, len as usize).await?;
                    match TombstoneFile::from_bytes(&bytes) {
                        Ok(file) => all_deleted.extend(file.deleted_ids),
                        Err(e) => warn!("Failed to load tombstone file {:?}: {}", path, e),
                    }
                }
            }
        }
        Ok(all_deleted)
    }
}
