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

#### Phase 6: Unified Search & Operations

**Status:** ✅ **Complete**

- [x] **Unified Searcher:**
  - [x] `search_async` scans MemTable (L0).
  - [x] `BucketManager` scans Local + Remote files (L1).
  - [x] `Refine`: Loads High-Fidelity data (ALP) for top candidates.
- [x] **Tombstone Handling V2 (Hardening):**
  - [x] `mark_delete` updates in-memory bitsets.
  - [x] **Persistent Deletes:** Verified tombstone propagation during Promotion/Compaction cycles.
  - [x] **Global Filter:** Optimize global tombstone filter for large-scale deletes.

#### Phase 7: Cleanup & Hardening

**Status:** ✅ **Complete**

- [x] **Reaper Verification:** Integration test to ensure physical file deletion (Local + S3) after compaction/promotion.
- [x] **Chaos Testing:** Validated durability via `chaos_test` (kill -9 loops).
- [x] **Split Safety:** Implemented parent-child consistency checks to prevent data loss during splits.
- [x] **Simulation Harness:** Updated `billion_scale`, `churn_sim`, and `drift_sim` for V2.

---

# **Global Master Plan: Drift Cluster (Phase 8)**

#### Phase 8: Distributed Consensus (The "Stateless Worker" Model)

**Status:** 🚧 **Planning**

- [ ] **Consensus Layer:** Integrate `openraft` or Etcd to manage the "Shard Map" (Which node owns which bucket?).
- [ ] **Remote WAL:** Abstract `WalManager` to support Kafka/Redpanda or S3-Append, allowing any node to replay another's log.
- [ ] **Stateless Worker:** Refactor `DriftService` to mount any collection by pulling state from S3, rather than relying on local disk affinity.
- [ ] **Gateway Node:** Create a gRPC proxy that hashes vector IDs and routes requests to the correct Worker node.
