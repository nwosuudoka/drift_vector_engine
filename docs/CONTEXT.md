# Drift Vector Engine Context

Last updated: 2026-02-23

## What This Repository Is
- A Rust workspace implementing a drift-aware vector engine.
- Core lifecycle: ingest to WAL/memtable, flush to storage buckets, search over L0 + L1, background maintenance.
- Unified storage work currently lives in `drift_storage`.

## Current Active Track
- Branch: `sidecar_payload_addition`.
- Focus: unified `.driftu` payload/index format expansion and exact-match index support.
- Recently modified files:
  - `drift_storage/src/unified_format.rs`
  - `drift_storage/src/unified_writer.rs`
  - `drift_storage/src/unified_reader.rs`

## Unified Storage Components
- `unified_format.rs`
  - File-format enums/structs, block descriptors, codec IDs, flags.
- `unified_writer.rs`
  - Writes vector chunks, payload columns, payload schema, exact index blocks.
  - Handles append semantics and consistency checks.
- `unified_reader.rs`
  - Reads blocks, payload schema/columns/rows, and exact indexes.
  - Applies payload decoding by codec and integrity validation.

## Current Codec Matrix (Payload Columns)
- `Bool` -> `Bitset`
- `Int64` -> `ForBitpack`
- `TimestampMicros` -> `ForBitpack`
- `Float32` -> `AlpRd`
- `Float64` -> `AlpRd`
- `Keyword` -> `DictBitpack`
- `Text` -> `DictBitpack` or `VarLen` (size-based choice)
- `Bytes` -> `DictBitpack` or `VarLen` (size-based choice)
- `LobRef` -> `VarLen`

## Known Invariants
- Header/footer flags must match actual block inventory.
- Append path must enforce payload schema consistency.
- Appending to payload-enabled files requires payload rows for appended data.
- Exact index data for same field across blocks must be mergeable and type-consistent.

## Validation Snapshot
- `cargo test -p drift_storage` passed on 2026-02-23.
- New targeted tests for codec matrix and append behavior are passing.
