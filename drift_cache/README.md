# drift_cache

Disk-native cache and page manager utilities used by the drift vector engine. This crate provides S3-FIFO eviction, sharding for concurrency, and a `PageManager` abstraction for async disk IO.

## Implemented features
- **S3-FIFO cache**: `FastS3Fifo` with lightweight metrics and admission control.
- **Sharded cache**: `ShardedFastS3Fifo` spreads contention across shards for multi-threaded access.
- **Block cache**: `BlockCache` wraps a `PageManager` and caches decoded `Cacheable` objects by `PageId`.
- **Local disk manager**: `LocalDiskManager` provides async `read_page`/`write_page` on local `.drift` files.

## Key modules
- `s3fifo.rs`: S3-FIFO eviction policy and cache metrics.
- `sharded_fifo.rs`: sharded wrapper for concurrent workloads.
- `block_cache.rs`: cache-fronted page loader for `Cacheable` types.
- `local_store.rs`: filesystem-backed `PageManager`.
- `store.rs`: `PageId`, `PageManager`, and `Cacheable` traits.

## Usage sketch
```rust
use drift_cache::{block_cache::BlockCache, local_store::LocalDiskManager, store::PageId};
use std::sync::Arc;

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
let storage = Arc::new(LocalDiskManager::new("data/storage"));
let cache = BlockCache::<MyCacheable>::new(storage, 10_000, 4);

let page_id = PageId { file_id: 1, offset: 0, length: 4096 };
let page = cache.get(&page_id).await?;
# Ok(())
# }
# struct MyCacheable;
# impl drift_cache::store::Cacheable for MyCacheable {
#     fn from_bytes(_data: &[u8]) -> std::io::Result<Self> { Ok(Self) }
# }
```
