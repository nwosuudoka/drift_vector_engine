# Changelog

## [Unreleased] - Phase 6: Scaling & Optimization

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
