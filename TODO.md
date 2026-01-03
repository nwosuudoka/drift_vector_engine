# **Global Master Plan: Drift-Aware Vector Engine**

#### **Section 1: Storage Layer (Level 1)**

**Status:** ✅ **Complete**

- ✅ **Custom `.drift` File Format:** `SegmentWriter`/`SegmentReader` with Versioning & Magic Bytes.
- ✅ **Disk Manager:** Abstracted via `opendal` for local/cloud transparency.
- ✅ **Block Alignment:** 4KB aligned pages for O_DIRECT compatibility.
- ✅ **Compression:** ALP/ALP_RD quantization for high-ratio float compression.
- ✅ **Dual-Tier Storage Strategy:** Implemented architecture for Fast (SQ8) vs Cold (ALP) paths.
- ✅ **"The Blob" Alignment:**
  - [x] `SegmentWriter` writes contiguous `BucketData` blobs (Header+Codes+IDs) for the index.
  - [x] `SegmentReader` supports fetching Raw SQ8 blobs and lazy-loading High-Fidelity data.
  - [x] `PersistenceManager` implements fallback logic to hydrate from SQ8 if ALP is missing (Robustness).

#### **Section 2: Core Indexing Logic (Level 1)**

**Status:** ✅ **Complete**

- ✅ **Bucket Structure:** RAM Header + Disk Data (Hybrid Layout).
- ✅ **Maintenance:** Drift-Aware Split & Strict Hysteresis Merge.
- ✅ **Safety:** Singularity Guard prevents infinite loops on duplicate data.

#### **Section 3: Memory Structure (Level 0)**

**Status:** ✅ **Complete**

- ✅ **MemTable:** Thread-safe HNSW Graph for low-latency ingest.
- [cite_start]✅ **Lazy Indexing:** Removed synchronous HNSW build from the hot write path. [cite: 1067]
- ✅ **Durability:** Write-Ahead Log (WAL) with crash recovery.
- ✅ **Janitor:** Background process for operation budgeting and auto-flushing.
- ✅ **Tiered Cache:** Implemented chunk-aligned read-ahead for S3/NVMe.
- ✅ **Tombstone Persistence:**
  - [x] Implemented `TombstoneFile` format (Magic + CRC + IDs).
  - [x] Added `flush_tombstones` to PersistenceManager.
  - [x] Startup hydration merges deleted IDs into global filter.
- ✅ **Compaction (The Scavenger):**
  - [x] Implemented Copy-on-Write `compact_bucket` to rewrite dirty buckets.
  - [x] Added `scavenge()` loop to Janitor to identify and clean buckets > 20% dirty.
  - [x] Verified memory reclamation and KV update integrity.

#### **Section 4: Execution Engine**

**Status:** ✅ **Complete**

- ✅ **Async Architecture:** Fully non-blocking core using `tokio`.
- [cite_start]✅ **Hybrid Search:** Merges results from Parallel Scan (RAM) and HNSW (Disk). [cite: 701]
- ✅ **Routing:** Saturating Density model (Lambda/Tau) for query routing.
- ✅ **Parallelism:** `rayon` integration for high-speed brute-force scanning of unindexed data.
- ✅ **Correctness Hardening:**
  - [x] **Phase Separation:** Split Search into Async I/O -> Sync CPU phases to fix `!Send` panic.
  - [x] **RAM-First Search:** Fixed "Hole in the Timeline" race condition by searching MemTable before yielding.
  - [x] **Recall Guardrail:** Added geometric fallback to prevent density starvation for new/small buckets.
- ✅ **ADC Optimization (Critique A):**
  - [x] Implement `Quantizer::precompute_lut`.
  - [x] Update `Bucket::scan` to use LUT instead of float math.
- ✅ **Storage Format Alignment (Critique B):**
  - [x] Store raw SQ8 bytes in `index_blob` for fast mapping.
  - [x] Store ALP bytes in `data_blob` for high-fidelity retrieval.

#### **Section 5: Server & API**

**Status:** ✅ **Complete**

- ✅ **gRPC Interface:** `Train`, `Insert`, `Search` via `tonic`.
- ✅ **CLI Tool:** `drift-cli` for human interaction and management.
- ✅ **Dynamic Config:** Auto-dimension sizing and URI-based storage handling.

#### **Section 6: Scaling & Optimization**

**Status:** ✅ **Complete**

- ✅ **Global ID Index:** O(1) `VectorID -> BucketID` mapping.
- ✅ **Bloom Filters:** Integrated per-segment probabilistic filters for fast negative lookups.
- ✅ **Drift Correction:** Geometric center tracking for data distribution shifts.

#### **Section 7: Cloud-Native Infrastructure**

**Status:** 🚧 **In Progress**

- ✅ **Storage Abstraction:** Replaced `std::fs` with `apache-opendal` to support S3, GCS, Azure, and Local FS uniformly.
- ✅ **Immutable Write Pattern:** Implemented "Scratch File" strategy to build segments locally and upload atomically.
- ✅ **Error Propagation:** Hardened `DiskManager` against silent IO failures.
- ✅ **Metric Unification:** Standardize on Squared Euclidean distance to fix L0/L1 ranking mismatches.
- ⬜ **Distributed Consensus:** Design the "Stateless Worker" clustering model for horizontal scaling.

#### **Section 8: Benchmarking & Correctness**

**Status:** ✅ **Complete** (Core Verification)

- ✅ **Drift Simulation:** Implemented `drift_sim` harness to inject concept drift (moving clusters).
- ✅ **Recall Verification:** Verified >90% recall under heavy drift (Speed 5.0).
- ✅ **Adaptive Indexing Test:** Validated `Split` logic expands capacity and heals recall drops automatically.

#### **Section 9: Garbage Collection (Disk Maintenance)**

**Status:** 🚧 **In Progress**

- [ ] **Segment Compaction (The Vacuum):**
  - [ ] Implement `get_physical_path` trait in `PageManager` to map logical IDs to physical files.
  - [ ] Create `SegmentCompactor` struct to perform Mark-and-Sweep GC on S3/Disk.
  - [ ] Implement `vacuum_segments()`: Identify and delete `.drift` files no longer referenced by the Index.
  - [ ] Implement `compact_tombstones()`: Consolidate scattered tombstone logs into a single snapshot.
  - [ ] Integrate `SegmentCompactor` into the `Janitor` background loop.
