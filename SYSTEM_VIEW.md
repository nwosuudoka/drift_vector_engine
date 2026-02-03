# System View (v2)

This document describes the end-to-end flow of a vector through the v2 system and how the
major subsystems interact. The design centers on a **Logical Bucket** that can span multiple
physical files and states, while search and maintenance operate on a consistent view.

## Logical Bucket (v2)
A bucket is no longer a single file. It is a logical entity with storage state managed by the
BucketManager:

- **Local**: one mutable local `.drift` file in `data/<collection>/staging/` (multiple row groups).
- **Remote**: one immutable remote `.drift` file in the configured storage root (single run id).
- **Tiered**: a remote base file plus a local delta file.
- **Promoting**: a transient state while a local file is being merged into remote.

All of these are expressed by `StorageClass` and hidden behind `BucketManager` + `StorageEngine`.

---

## 1) Ingest: MemTable + WAL

- **Arrival**: Vector `V` (ID=100) arrives via Insert/Train.
- **Durability**: the insert is written to the per-collection WAL first.
- **Visibility**: the vector is appended to the active MemTable (L0).
- **Tombstones**: deletes mark an L0 tombstone set (copy-on-write for lock-free reads).

When the MemTable reaches capacity it is rotated into a frozen table and queued for flush.

---

## 2) Flush: Local Staging + Router/KV updates

The Janitor periodically flushes frozen MemTables:

1. **Partition**: vectors are assigned to buckets using the Router (centroid routing).
2. **Append**: each bucket's vectors are appended as a new row group in
   `data/<collection>/staging/bucket_<id>.drift`.
3. **KV Mapping**: `VectorID -> BucketID` is updated in BitStore for each flushed ID.
4. **Bucket Stats**: BucketManager updates per-bucket drift stats (count + vector sum).
5. **Router Update**: global centroid and count are recalculated and applied to the Router.
6. **Manifest Update**: bucket metadata and current tombstone file pointer are updated
   atomically.

Result: L0 drains into L1 (local staging), routing metadata stays consistent, and the ID map is
kept correct for delete propagation.

---

## 3) Promotion: Local -> Remote (Tiered state)

When a local staging file exceeds a size threshold, the Janitor promotes it:

1. **Lock**: the bucket is write-locked via `BucketCoordinator`.
2. **Rotate**: the active local file becomes a frozen staging file, and a new empty local
   file is created.
3. **Merge**: the remote base (if any) and the frozen local file are read, tombstones are
   applied, and a new remote file is written.
4. **Finalize**: the bucket becomes **Tiered** (remote base + local delta), the manifest
   run id is updated, and processed tombstones are pruned.

Result: remote is compacted, local continues as a mutable delta, and search reads both.

---

## 4) Split + Defector Loopback (Split/Steal update)

When a bucket is too large or drifted:

1. **Fetch**: the bucket's full data is loaded (merging tiers if needed).
2. **K-Means (K=2)**: produces two candidate child centroids.
3. **Partition**: vectors are assigned to left/right children.
4. **Defector Check**: if a vector is much closer to a *different* global centroid than its
   assigned child, it is **looped back** to L0 instead of being written to a child bucket.
5. **Write**: two new local bucket files are created; old bucket metadata is removed.
6. **Router/Manifest**: atomic updates replace the parent bucket with the two children.

This replaces the older explicit neighbor stealing with a safer “defector loopback” guard.

---

## 5) Scatter Merge (Zombie healing)

When a bucket becomes a zombie (low count / high urgency):

1. **Fetch**: load the bucket's full data.
2. **Scatter**: each vector is assigned to the nearest neighboring centroid.
3. **Rewrite Neighbors**: for each target bucket, a new local file is written with
   the merged vectors (copy-on-write).
4. **Update**: manifest, router, and bucket manager are updated; zombie file is deleted.

Result: sparse buckets are eliminated and their vectors are redistributed to healthy neighbors.

---

## 6) Search: L0 + L1 Unified Scan

Search is executed as:

1. **Snapshot**: capture L0 tombstones and the current MemTable/Frozen tables.
2. **L0 Scan**: scan MemTables (parallel) and filter by L0 tombstones.
3. **Bucket Selection**: Router selects bucket IDs using target_confidence, lambda, tau,
   plus a distance guardrail.
4. **L1 Scan**: BucketManager scans the selected bucket files (Local/Remote/Tiered),
   applying per-bucket tombstones.
5. **Merge**: L0 + L1 results are merged, L0 tombstones are rechecked, top-K returned.

---

## 7) Tombstones + Recovery

- **Tombstones**: L0 and L1 deletes are merged and persisted periodically; the manifest
  tracks the active tombstone file for crash-safe recovery.
- **Recovery**: on startup, the manifest is loaded, Router is rebuilt, buckets are
  registered, WAL is replayed, and tombstones are rehydrated.

---

## Key Trade-offs (v2)

- **Logical buckets** simplify reasoning but require a BucketManager abstraction.
- **Local staging + remote base** minimizes remote writes while keeping search correctness.
- **Defector loopback** avoids misrouting during splits without expensive neighbor stealing.
- **Copy-on-write merges** keep maintenance safe at the cost of some rewrite overhead.

This is the current production mental model for v2 and aligns with the v2 server, janitor,
router, and bucket manager implementations.
