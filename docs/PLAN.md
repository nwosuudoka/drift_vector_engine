# Implementation Plan

Last updated: 2026-02-23

## Objective
Deliver robust unified `.driftu` payload/index support with codec-aware read/write paths, append safety, and exact-match index correctness.

## Scope
- In scope:
  - Payload codec additions and decode/encode compatibility.
  - Append behavior for payload-bearing files.
  - Exact-index merge behavior across appended blocks.
  - Test coverage for happy path and key invariants.
- Out of scope (for now):
  - Full hybrid query planner.
  - Range index implementation.
  - Distributed orchestration concerns.

## Execution Phases
1. Phase A: Storage wire contract hardening (`drift_storage`)
- Header/footer metadata expansion (metric + schema hash).
- Row-group payload stats metadata.
- Exact postings compact wire encoding.
- Range index block contract + implementation.

2. Phase B: Payload-preserving write path wiring (`drift_core` + `drift_server`)
- Carry payload rows through partitioning/staging.
- Keep payload through promotion/split/merge rewrites.
- Validate full lifecycle roundtrip of payload fields.

3. Phase C: Manifest + recovery metadata (`manifest.proto` + server recovery)
- Record payload/index capabilities and schema consistency metadata in manifest.
- Add startup validation and deterministic degraded-state behavior.

4. Phase D: API surface (`drift.proto` + handlers)
- Payload schema APIs.
- Payload-carrying insert APIs.
- Filter clauses in search APIs.

5. Phase E: Query planner integration
- Exact/range candidate pruning.
- Filter-first vs vector-first heuristics.
- Telemetry and latency/selectivity gates.

## Acceptance Criteria
- All `drift_storage` tests pass.
- New codec payloads roundtrip without data corruption.
- Append on payload-enabled files is deterministic and validated.
- Exact-match filtering remains correct after append.
