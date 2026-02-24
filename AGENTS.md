# Drift Vector Engine: Agent Working Guide

## Project Context
- Repository type: Rust workspace (`edition = 2024`).
- Workspace crates: `drift_core`, `drift_storage`, `drift_server`, `drift_kv`, `drift_cache`, `drift_traits`.
- Current major track: payload sidecar and secondary-index groundwork in unified storage.
- Main active files for that track:
  - `drift_storage/src/unified_format.rs`
  - `drift_storage/src/unified_writer.rs`
  - `drift_storage/src/unified_reader.rs`

## Persistent Context Files
- `docs/CONTEXT.md`: stable architecture and constraints.
- `docs/PLAN.md`: current implementation plan.
- `docs/NEXT.md`: prioritized actionable next steps.
- `docs/DECISIONS.md`: dated technical decisions and rationale.
- `docs/SESSION_LOG.md`: session-by-session progress log.

## Working Agreements
1. Before code changes, restate the goal and acceptance criteria.
2. Use `docs/NEXT.md` as the default queue unless the user overrides priority.
3. Keep changes scoped; avoid unrelated refactors.
4. Run the smallest relevant test target first, then broader suites as needed.
5. After finishing work, update `docs/NEXT.md` and `docs/SESSION_LOG.md`.
6. In summaries, always include:
   - What changed
   - Files touched
   - Tests run
   - Follow-up items

## Standard Commands
- Format: `cargo fmt --all`
- Crate tests: `cargo test -p drift_storage`
- Workspace tests: `cargo test --workspace`
- Targeted tests:
  - `cargo test -p drift_storage payload_codec_matrix_roundtrip`
  - `cargo test -p drift_storage payload_rows_merges_exact_indexes`
  - `cargo test -p drift_storage payload_rows_missing_for_payload_file`

## Session Start Prompt Pattern
- "Read `@docs/CONTEXT.md` and `@docs/NEXT.md`, summarize current state, then execute the top unchecked step."

## Session End Prompt Pattern
- "Update `@docs/NEXT.md` and `@docs/SESSION_LOG.md` with completed work, exact commands run, and the next 3-10 steps."
