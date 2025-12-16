Here is the Global Master Plan for the Drift-Aware Vector Engine, organized by section with the requested checkmark emojis.

### **Section 1: Storage Layer (Level 1)**

**Status:** ✅ Mostly Complete. The on-disk format and compression engines are solid.

- ✅ **Custom `.drift` File Format:** Implemented `SegmentWriter`/`SegmentReader` with Footer and Index.

- ✅ **Disk Manager:** Implemented basic I/O and seeking.

- ✅ **Block Alignment:** `PageBlock` implemented for 4KB alignment (NVMe prep).

- ✅ **Compression (SQ8):** `Quantizer` implemented with training and clamping.

- ✅ **Compression (ALP/ALP_RD):** Float compression implemented.

- ✅ **Compression (FastLanes):** Integer compression implemented.

- ✅ **Bloom Filters:** The SDD requires a Bloom Filter offset in the footer for quick "negative lookups" (checking if an ID exists without decompressing).

### **Section 2: Core Indexing Logic (Level 1)**

**Status:** ✅ Functional. The maintenance primitives are verified.

- ✅ **Bucket Structure:** SoA layout with `AlignedBytes` and `BitSet` for tombstones.

- ✅ **ADC Scanning:** SIMD-optimized `scan_adc` (AVX2/Neon) is implemented.

- ✅ **K-Means Clustering:** Robust initialization and training implemented.

- ✅ **Split Operation:** "Neighbor Stealing" logic implemented.

- ✅ **Merge Operation:** "Scatter Merge" logic implemented.

- ✅ **Urgency Calculation:** Implemented the "Hot Zombie" formula correctly ().

- ✅ Maintenance: Split/Steal/Merge primitives are verified.

### **Section 3: Memory Structure (Level 0)**

**Status:** ⬜ **Missing.** The system currently writes directly to L1 (Buckets), skipping the "MemTable" layer.

- ✅ **HNSW Graph:** Implement the Level 0 in-memory graph for recent data.
- ✅ Hybrid Search: VectorIndex queries both L0 and L1 and merges results.

- ⬜ **Flushing Logic:** Implement the async job that freezes L0, runs K-Means, and writes a `.drift` file.

- ✅ **Write-Ahead Log (WAL):** Implement append-only persistence to survive crashes before flushing.

### **Section 4: Execution Engine**

**Status:** ✅ **Complete.**

- ✅ **Epoch-Based Reclamation:** Integrate `crossbeam-epoch` to manage memory safety without locks.

- ✅ **Probabilistic Stopping Condition:** Implement the logic to stop searching early.

- ✅ **Concurrency:** Lock-free reads via crossbeam-epoch.

- ✅ **Scoring:** Saturating Density math implemented.

### **Section 5: Server & API**

**Status:** 🚧 Barebones.

- ✅ **Persistence Manager:** Basic save/load lifecycle tests exist.

- ⬜ **gRPC Interface:** Define the Protobuf service and handlers.

- ⬜ **Request Router:** Implement consistent hashing to route queries to correct nodes.

- ⬜ **Background Workers:** The "Janitor" thread that periodically calls `calculate_urgency` and triggers maintenance.

---

### **Section 6: Required Refactoring**

These items are implemented but deviate from the specification or performance requirements.

- ✅ **Refactor Concurrency Model**
- **Current:** Uses `parking_lot::RwLock` on the critical path (`search_drift_aware` locks buckets).

- **Target:** Must be **Lock-Free**. Replace `RwLock<HashMap>` with atomic pointer swapping (using `crossbeam-epoch`) so searches _never_ block.

- ✅ **Refactor Search Scoring**
- **Current:** Uses a linear penalty: `dist *= 1.0 + frag`.

- **Target:** Must use the **Saturating Density** probability model: .

---

### **Immediate Next Step**

We should prioritize **Section 6 (Refactoring Concurrency)**. If we build L0 (HNSW) on top of the current blocking `RwLock` architecture, we will have to rewrite significantly more code later.
