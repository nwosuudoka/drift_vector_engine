# **Global Master Plan: Drift-Aware Vector Engine**

#### **Section 1: Storage Layer (Level 1)**

**Status:** ✅ **Complete**

- ✅ **Custom `.drift` File Format:** `SegmentWriter`/`SegmentReader` with Versioning & Magic Bytes.
- ✅ **Disk Manager:** Abstracted via `opendal` for local/cloud transparency.
- ✅ **Block Alignment:** 4KB aligned pages for O_DIRECT compatibility.
- ✅ **Compression:** ALP/ALP_RD quantization for high-ratio float compression.

#### **Section 2: Core Indexing Logic (Level 1)**

**Status:** ✅ **Complete**

- ✅ **Bucket Structure:** RAM Header + Disk Data (Hybrid Layout).
- ✅ **Maintenance:** Drift-Aware Split & Strict Hysteresis Merge.
- ✅ **Safety:** Singularity Guard prevents infinite loops on duplicate data.

#### **Section 3: Memory Structure (Level 0)**

**Status:** ✅ **Complete**

- ✅ **MemTable:** Thread-safe HNSW Graph for low-latency ingest.
- ✅ **Durability:** Write-Ahead Log (WAL) with crash recovery.
- ✅ **Janitor:** Background process for operation budgeting and auto-flushing.

#### **Section 4: Execution Engine**

**Status:** ✅ **Complete**

- ✅ **Async Architecture:** Fully non-blocking core using `tokio`.
- ✅ **Routing:** Saturating Density model (Lambda/Tau) for query routing.

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
- ⬜ **Metric Unification:** Standardize on Squared Euclidean distance to fix L0/L1 ranking mismatches.
- ⬜ **Distributed Consensus:** Design the "Stateless Worker" clustering model for horizontal scaling.
