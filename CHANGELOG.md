# Changelog

## [Unreleased] - Phase 6: Scaling & Optimization

### Added

- **`BucketData` Serialization:** Implemented `Cacheable` trait for `BucketData`.
- **Bitpacking:** Integrated scalar bit-packing for Tombstone blocks, reducing metadata overhead by ~30x for sparse deletes.
- **Safety:** Added `0xBD47001` Magic Bytes validation and enforced 64-byte alignment during deserialization.

### Verified

- **Round-Trip:** Validated that `BucketData` -> `Bytes` -> `BucketData` is lossless and memory-safe.

### Added

- **`drift_cache` Crate:** Created new library to handle memory management and disk I/O.
- **S3-FIFO Eviction:** Implemented a state-of-the-art, thread-safe, sharded eviction policy to manage the Block Cache.

### Added

- **`drift_kv` Crate:** A persistent, disk-based Hash Table using Linear Hashing for O(1) key-value lookups.
- **Global Identity Map:** `VectorIndex` now maintains a persistent mapping of `VectorID -> BucketID`.
- **Drift Calculation:** Added `running_sum` to `Bucket` to track geometric centroid shifts in real-time.

### Changed

- **Deletion Logic:** Upgraded `delete(id)` from an $O(N)$ parallel scan to an $O(1)$ KV lookup + Bucket operation.
- **Urgency Formula:** Removed the "Overflow" term from `calculate_urgency`. It now strictly follows Equation (5) (Zombie/Emptiness ratio).
- **Split Trigger:** Implemented `should_split()` based on the SDD criteria: `Count > 80% Capacity` OR `Drift > 0.15`.

### Fixed

- **Maintenance Loop:** `Janitor` now distinctly separates "Healing" (Merge) and "Growth" (Split) checks to prevent logic conflation.
