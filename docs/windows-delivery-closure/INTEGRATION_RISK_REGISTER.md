# Integration Risk Register — PR #32 × Windows Delivery

| ID | Risk | Severity | Status | Evidence / mitigation |
| --- | --- | --- | --- | --- |
| R1 | PR #32's hardened publication detector flags Windows test literals → repo-root gate fails on any combined tree | P1 (found) | **MITIGATED** | Commit `c432861` (hash-pinned WHITELIST entry, Windows side only). Combined tree `5b26f0fe…`: 459/0/10 green. Entry self-invalidates if the file changes. |
| R2 | Windows Job ownership bypassed by #32 components spawning processes | P0 (potential) | **CLEARED** | `watchdog.py` is a thread-only monitor; L1 scripts run inside the supervised command; merged-tree supervisor-death test passes. |
| R3 | Either side's publication protections silently dropped in conflict resolution | P0 (potential) | **CLEARED** | Disjoint hunks; automated assertion that all markers from both sides exist on the merged tree (`_WINDOWS_PATH_PATTERN`, `relative_to(root).parts`, `_reject_duplicate_keys`, `_contains_annotation_rows`, updated p15 hash); full fused gate suite green. |
| R4 | #32 policy admission weakens or is weakened by Windows runner changes | P1 (potential) | **CLEARED** | Disjoint files; both policy suites pass merged; production default path untouched; unauthorized budget/missing policy/wrong digest still fail closed. |
| R5 | Resource watchdog not wired into common runner → Windows production runs unwatched | P2 (observation) | **OPEN — carried to Phase 5** | Phase 5 admission policy requires resource thresholds; wiring the watchdog into the Windows runner is a future reviewed change, not part of this PR. |
| R6 | #32 merges first and `main` moves; Windows PR base becomes stale | P2 | **OPEN — procedural** | Windows PR targets `main@52d593e`; after #32 merges, GitHub will report mergeable status; both orders verified conflict-free. Re-evaluate only if `main` changes materially beyond #32. |
| R7 | Simulation environment leakage (editable install pointing at another branch) produces false failures | P2 (process) | **MITIGATED** | `PYTHONPATH` pinned to simulated tree; artifact documented in INTEGRATION_SIMULATION.md; future combined-tree runs must do the same. |
| R8 | L1/L1.1-B evidence (`eval/results/Linux gfx1151 verified/**`) misread as Windows qualification | P1 (labeling) | **GUARDED** | Path names keep platform scope; Phase 4 deliverables restate Linux≠Windows evidence separation. |
| R9 | Whitelist seen as "weakening" in review | P3 | **ACCEPTED** | Same pattern as pre-existing sanitizer-test entries; hash-pinned; detector strength unchanged; rationale in commit message and audit. |

Overall: no open P0. One P1 found and mitigated (R1); one P1 labeling guard (R8);
procedural risks tracked. Final classification `COMPATIBLE_WITH_CONDITIONS`
(condition met by `c432861`).
