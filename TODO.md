# **Global Master Plan: Drift-Aware Vector Engine**

#### **Section 1: Storage Layer (Level 1)**

**Status:** ✅ **Complete.** The bedrock is solid.

- ✅ **Custom `.drift` File Format:** `SegmentWriter`/`SegmentReader` with Footer/Index.

- ✅ **Disk Manager:** Async I/O with seek support.

- ✅ **Block Alignment:** `PageBlock` for 4KB alignment.

- ✅ **Compression:**

  - SQ8 Quantizer (1-99% clipping).
  - ALP/ALP_RD for Floats.
  - FastLanes for Integers.

- ✅ **Bloom Filters:** Integrated into footer for O(1) negative lookups.

#### **Section 2: Core Indexing Logic (Level 1)**

**Status:** ✅ **Complete.** The engine logic works.

- ✅ **Bucket Structure:** SoA layout with `AlignedBytes`.

- ✅ **ADC Scanning:** SIMD-optimized `scan_adc`.

- ✅ **Maintenance Primitives:**
  - Split (Neighbor Stealing).
  - Merge (Scatter Merge).
  - Urgency Calculation ("Hot Zombie" formula).

#### **Section 3: Memory Structure (Level 0)**

**Status:** ✅ **Complete.** The ingest path is fully operational.

- ✅ **HNSW Graph:** Thread-safe MemTable for hot data.

- ✅ **Hybrid Search:** Merges L0 (Graph) and L1 (Disk) results seamlessly.

- ✅ **Flushing Logic:** `Janitor` rotates MemTable, trains Quantizer (cold start), and flushes to Disk.

- ✅ **Write-Ahead Log (WAL):** Durability guaranteed. Crashes recover data from WAL before hydration.

#### **Section 4: Execution Engine**

**Status:** ✅ **Complete.**

- ✅ **Epoch-Based Reclamation:** `crossbeam-epoch` manages lock-free memory safety.

- ✅ **Probabilistic Stopping:** Saturating Density scoring implemented.

- ✅ **Concurrency:** Lock-free reads on the hot path.

#### **Section 5: Server & API**

**Status:** ✅ **Complete.** We have a working server.

- ✅ **Persistence Manager:** Handles Hydration (Disk -> RAM) and Flushing (RAM -> Disk).

- ✅ **gRPC Interface:** `DriftService` implements `Insert` and `Search` via Protobuf.

- ✅ **Multi-Tenancy:** `CollectionManager` creates isolated indices on-the-fly (`/data/users`, `/data/products`).

- ✅ **Background Workers:** `Janitor` runs per-collection to manage lifecycle.

---

### **Where We Are: The "Golden Path" is Live**

We have built a system that:

1. **Accepts Data** via gRPC.
2. **Writes safely** to a WAL.
3. **Serves immediately** from RAM (HNSW).
4. **Autonomously flushes** to compressed Disk Segments.
5. **Recovers perfectly** from crashes (Hydration + WAL Replay).
6. **Isolates Tenants** via Collections.

### **The Final Frontier: Distributed Routing**

The only item remaining from the original plan is the **Request Router**. Currently, `Drift` is a "Scale-Up" database (single node, many cores). To become "Scale-Out" (infinite storage), we need to shard data across multiple nodes.

**Updated Todo List:**

Section 6: Scaling & Optimization (New)

Status: 🚧 In Progress.

- ⬜ **Global ID Index**: Integrate BitStore (Disk Hash Table) to map VectorID -> BucketID for O(1) deletes/updates.
- ⬜ **Distributed Consensus**: Use a lightweight consensus (or consistent hashing) to map Collection -> Node.
- ⬜ **Request Router**: Forward gRPC requests to the correct node/shard.
- ⬜ **CLI Tooling**: A proper command-line interface (drift-cli) to admin the cluster.
