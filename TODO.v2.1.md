Here is the comprehensive, unified **Todo List for Drift v2.0**.

This plan executes the transition from the current "Segment-Based" architecture to the **"Unified Row-Group + Local-Buffered Rewrite (LBR)"** architecture. It prioritizes **Durability** (WAL/Manifest) and **Latency** (Stream-First Layout).

---

### **Phase 1: The Unified Storage Format (`drift_storage`)**

_Goal: Create a single file format (`.drift`) that works for both efficient local appending and fast S3 streaming._

- **1.1. Define Physical Layout Specs**
- [x] Define `DriftHeader` struct (128 bytes fixed): Magic, Version, Global Vector Count, Index Length.
- [x] Define `RowGroupHeader` struct: Count, Checksum, SQ8 Offset, ALP Offset.
- [x] Define `DriftFooter` struct: Offset to first RG, Magic, Global Checksum.

- **1.2. Implement `RowGroupWriter**`
- [x] Create `RowGroupBuffer`: A structure to hold raw vectors in RAM until `chunk_size` (e.g., 2000 vectors).
- [x] Implement `flush()`:
- Transpose vectors (Row -> Col).
- Compress Cold Cols (ALP).
- Quantize Hot Cols (SQ8).
- Serialize to bytes: `[Header][Hot Index][Cold Data]`.

- **1.3. Implement `BucketFileWriter` (The Streamer)**
- [x] Support **Append Mode** (for Local): Open file, seek to end, append new RG, update/rewrite Footer.
- [x] Support **Stream Mode** (for S3 Promotion): Write Header once, stream RGs continuously, write Footer.

- **1.4. Implement `BucketFileReader` (The Scanner)**
- [x] Implement **Stream-First Scan**:
- Read Header.
- Read `Hot Index` of RG 0.
- **Action:** Callback to `Scanner`. If match, read `Cold Data`. If no match, skip bytes (seek).

- [ ] Implement **Early Abort**: If Bloom Filter (in Footer/Header) fails, drop the reader immediately.

- **1.5. Verification**
- [ ] Test: Append 10 small RGs to a local file. Read it back as one stream. Verify data integrity.

---

### **Phase 2: The Control Plane (Durability & Metadata)**

_Goal: Ensure no data loss on crash and prevent "Split Brain" during splits._

- **2.1. The Manifest (`drift_core/src/manifest.rs`)**
- [ ] Define `manifest.proto` (Protobuf):
- `version`: monotonic u64.
- `centroids`: List of `[id, vector]`.
- `buckets`: Map `id -> { run_id, vector_count, tombstone_count }`.

- [ ] Implement `ManifestManager`:
- `load_latest()`: Fetch `manifest_vN.pb` from S3.
- `cas_update(old_v, new_manifest)`: Optimistic locking write (e.g., `If-None-Match` or DynamoDB lock).

- **2.2. The Write-Ahead Log (`drift_core/src/wal.rs`)**
- [ ] _Audit existing WAL:_ Ensure strictly append-only and `fsync` on every batch.
- [ ] Implement `WalReplayer`:
- Read log from byte 0.
- Filter entries that are already in the Manifest (using Log Sequence Numbers or IDs).
- Return list of "Lost Vectors" to be re-inserted.

- **2.3. The Recovery Orchestrator**
- [ ] Create `RecoveryManager::startup()`:

1. Load Manifest (Restore Routing Table).
2. Scan Local Disk (Discover un-promoted `.drift` files).
3. Replay WAL (Restore RAM MemTable).

---

### **Phase 3: Core Logic Pivot (Incrementalism)**

_Goal: Stop "Centroid Proliferation". Assign vectors to fixed buckets._

- **3.1. Implement The Router**
- [ ] Create `StaticRouter` struct in `VectorIndex`.
- [ ] Initialize with centroids from `Manifest`.
- [ ] Implement `assign(vector) -> BucketID` (Nearest Neighbor search on centroids).

- **3.2. Refactor `VectorIndex` Insert Path**
- [ ] **Delete** dynamic K-Means partitioning on flush.
- [ ] **Add** `IncrementalPartitioner`:
- Take `MemTable` snapshot.
- Group vectors by `assign(vec)`.
- Return `HashMap<BucketID, Vec<Vector>>`.

- **3.3. Verify**
- [ ] Test: Insert 1M vectors. Ensure Bucket Count == Initial Centroids (no new buckets created).

---

### **Phase 4: The Local-Buffered Write Path (LBR)**

_Goal: Buffer writes locally. Write to S3 only when data is "big enough"._

- **4.1. Local Staging Manager**
- [ ] Manage `data/buckets/bucket_{id}.drift` files.
- [ ] Implement `append_batch(bucket_id, vectors)`:
- Uses `BucketFileWriter` (Append Mode).
- Updates in-memory "Dirty Set" (track which buckets were modified).

- **4.2. Update `Janitor` Flush Loop**
- [ ] **Old:** `MemTable -> S3`.
- [ ] **New:** `MemTable -> Partition -> Local Staging`.
- [ ] **Constraint:** Flush MUST verify local file sync before truncating WAL.

---

### **Phase 5: Promotion & Compaction (S3 Path)**

_Goal: Move data from Local to S3 and merge fragmented Row Groups._

- **5.1. Implement `Promoter` Task**
- [ ] Scan `LocalStaging` for files > 16MB (Threshold).
- [ ] **Download:** Fetch `s3://bucket_{id}_{old_run}.drift`.
- [ ] **Merge:** Stream Local + Remote iterators into a `BucketFileWriter` (Stream Mode).
- [ ] **Upload:** Write `s3://bucket_{id}_{new_run}.drift`.
- [ ] **Commit:** Update `Manifest` (CAS): `Bucket {id} now points to {new_run}`.
- [ ] **Cleanup:** Delete local file and old S3 file.

- **5.2. Implement `Split` Operation**
- [ ] Trigger: `Bucket.count > 1M`.
- [ ] Logic:
- Download Bucket.
- Train K-Means (K=2).
- Split into `bucket_A.drift`, `bucket_B.drift`.
- Update Manifest: Remove `BucketID`, Add `ID_A`, `ID_B`. Add 2 new Centroids.

---

### **Phase 6: Unified Search & Operations**

_Goal: Query the hybrid state (S3 + Local + RAM) seamlessly._

- **6.1. Unified Searcher**
- [ ] `search(query)` logic:

1. **Route:** Find target `BucketID` + Neighbors.
2. **S3 Scan:** Stream `s3://bucket_{id}.drift` (Hot Index).
3. **Local Scan:** Read `data/bucket_{id}.drift` (Hot Index).
4. **Merge:** Combine candidates.
5. **Fetch:** Load Cold Data (ALP) for Top K.

- **6.2. Tombstone Handling**
- [ ] **Global:** Maintain `DeletedBitSet` in RAM (restored from `tombstones.log`).
- [ ] **Query Time:** Filter results against `DeletedBitSet`.
- [ ] **Compaction Time:** When Promoter runs, permanently remove vectors present in `DeletedBitSet`.

---

### **Phase 7: Cleanup & Cutover**

- [ ] Remove `drift_storage::segment_writer` (Legacy).
- [ ] Remove `drift_core::bucket::BucketData` (Legacy Memory Blob).
- [ ] Remove `drift_kv` (No longer needed, routing is geometric).
- [ ] Final Integration Test: "The Chaos Monkey" (Random kills during Flush/Split/Merge).

This list is complete. It covers the **Physics** (Storage), the **Logic** (Incremental), and the **Safety** (Durability).
