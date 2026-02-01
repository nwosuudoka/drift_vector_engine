# **Global Master Plan: Drift-Aware Vector Engine (V2 LBR)**

#### **Phase 1: The Unified Storage Format (`drift_storage`)**

**Status:** ✅ **Complete**

- [x] **Physical Layout:** `DriftHeader`, `RowGroupHeader`, `DriftFooter`.
- [x] **RowGroupWriter:** Transpose -> Compress (ALP) -> Quantize (SQ8) -> Serialize.
- [x] **BucketFileWriter:** Supports `Append Mode` (Local) and `Stream Mode` (S3).
- [x] **BucketFileReader:** Stream-first scan with footer validation.

#### **Phase 2: The Control Plane (Durability & Metadata)**

**Status:** ✅ **Complete**

- [x] **Manifest V2:** Atomic `apply_atomic` updates for Buckets and Centroids.
- [x] **WAL V2:** Transactional WAL (Begin/Commit/Rollback) with CRC checksums.
- [x] **Recovery Manager:**
  - [x] Rebuild Router from Manifest.
  - [x] Re-register Local Staging files.
  - [x] Replay WAL for MemTable restoration.

#### **Phase 3: Core Logic Pivot (Incrementalism)**

**Status:** ✅ **Complete**

- [x] **Static Router:** Router is now stable; points don't move until explicit Split/Merge.
- [x] **BucketManager:** The new "Source of Truth" for where data lives (Local vs Remote vs Tiered).
- [x] **VectorIndex V2:**
  - [x] Decoupled Storage (`StorageEngine` trait).
  - [x] Atomic ID Allocation (`AtomicU32`).
  - [x] `insert_batch` with internal Rotation.

#### **Phase 4: The Local-Buffered Write Path (LBR)**

**Status:** ✅ **Complete**

- [x] **Local Staging Manager:** Manages `bucket_{id}.drift` files in `data/staging/`.
- [x] **Append Logic:** `append_batch` detects existing file, reads footer, appends RG, rewrites footer.
- [x] **Janitor Flush:**
  - [x] `rotate_and_freeze` MemTable.
  - [x] Partition data by Router.
  - [x] Append to Local Staging files.
  - [x] Update Drift Stats (`vector_sum`).

#### **Phase 5: Maintenance & Self-Healing**

**Status:** ✅ **Complete**

- [x] **Drift Tracking:** Real-time `calculate_drift()` based on `vector_sum`.
- [x] **Smart Split:**
  - [x] `calculate_split`: K-Means (K=2) + Defector Loopback.
  - [x] **Singularity Guard:** Abort if variance is too low.
  - [x] **Atomic Execution:** Write 2 new buckets -> Swap Manifest -> Update Router.
- [x] **Promotion (Tiering):**
  - [x] `promote_to_s3`: Merge Local + Remote -> New S3 Segment.
  - [x] **Reaper:** Background deletion of obsolete files.
- [x] **Scatter-Merge (Zombie Healing):**
  - [x] **Detection:** Identify buckets with `Urgency > 1.5`.
  - [x] **Calculation:** `calculate_merge` routes orphans to nearest neighbors.
  - [x] **Execution:** Delta-CoW to Neighbor Staging Files.
  - [x] **Cleanup:** Atomic Manifest Update + Physical Deletion.

#### **Phase 6: Unified Search & Operations**

**Status:** 🚧 **In Progress**

- [x] **Unified Searcher:**
  - [x] `search_async` scans MemTable (L0).
  - [x] `BucketManager` scans Local + Remote files (L1).
  - [x] `Refine`: Loads High-Fidelity data (ALP) for top candidates.
- [ ] **Tombstone Handling V2 (Hardening):**
  - [x] `mark_delete` updates in-memory bitsets.
  - [ ] **Persistent Deletes:** Ensure deletes are propagated correctly during Promotion/Compaction (currently we might be losing some delete signals if a crash happens between marking and flushing).
  - [ ] **Global Filter:** Optimize global tombstone filter for large-scale deletes.

#### **Phase 7: Cleanup & Hardening**

**Status:** 📝 **Next Steps**

- [ ] **Distributed Consensus:** Prepare for stateless workers.
- [ ] **Final Chaos Test:** Random kill -9 loop while running heavy ingest.
