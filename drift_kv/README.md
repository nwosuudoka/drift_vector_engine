# drift_kv

Persistent key-value store used by the drift vector engine to map `VectorID -> BucketID` efficiently. The `BitStore` offers an append-friendly disk layout, tombstones for deletes, iteration, and compaction.

## Implemented features
- **Disk-backed KV**: fixed-size slots with CRC-protected metadata and data pages.
- **Tombstones**: delete markers to avoid immediate rewrites.
- **O(1) lookup**: hash-based bucket selection with linear probing.
- **Iteration + compaction**: iterate entries and rebuild storage to reclaim space.

## Key modules
- `bitstore.rs`: `BitStore` implementation, on-disk layout, hashing, and compaction logic.
- `tests.rs`: correctness and persistence tests.

## Usage sketch
```rust
use drift_kv::bitstore::BitStore;

let store = BitStore::new("data/kv")?;
store.put(b"user:42", &123u64.to_le_bytes())?;
let value = store.get(b"user:42")?;
store.remove(b"user:42")?;
store.sync()?;
```
