# Policy identity & security audit (P4) — validate_budget, P2-A policy, P2B wording

Date: 2026-10-30. Scope: `inference.py::validate_budget`, the frozen
`eval/experiments/p2a-token-budget/policy.json`, `scripts/run_p2a_budget.py`,
the P2-A test gates, and `docs/P2B_READINESS_DECISION.md` wording. Audit only;
no behavior change on this branch.

## 1. validate_budget against the frozen policy

| Mission requirement | Finding | Verdict |
|---|---|---|
| Exact policy identity | `validate_budget` checks `schema == 1` and that the budget appears in `budgets_new_tokens` — it does **not** pin the file's SHA-256. Identity is enforced one level up: `run_p2a_budget.load_policy` requires `experiment == "P2A_OUTPUT_TOKEN_BUDGET_RCA"` and the run config freezes `policy_sha256` + repo-relative path; `run_native` records `experimental_budget_policy_sha256` in every report/receipt. | **Partial**: driver-path is identity-bound; a hand-written schema-1 JSON with a bigger budget list would still pass a *direct* `run_native` call. Local-opt-in trade, acceptable for P2-A, must be hardened for P2 (see §3). |
| Explicit experimental opt-in | Budgets > 8192 fail closed without a presented policy file; production defaults untouched (`PRODUCTION_MAX_NEW_TOKENS = 8192`); `prepare` in `run_mpdoc_sample.py` refuses > 8192 outright. | **PASS** |
| Approved budget values | Only values listed in the presented policy; 32768-style values rejected even with a policy present. | **PASS** |
| Correct source/model binding | Policy pins `model_revision 9cf077c…`; loader verifies all 18 model-code/weight hashes before load; receipts carry model evidence and wheel identity. | **PASS** |
| Context and memory admission | Context-token admission (pages×300 + budget + 4096 ≤ 131072) is code-enforced in `run_p2a_budget.admission_report` with tests (including a refusal case). GPU memory is admitted procedurally (runtime OOM → ledger OOM state), not by preflight; host RSS recorded. | **PASS for P2-A scope**; P2 plan adds explicit pre-run GPU admission (audit §10.4). |
| Fail-closed rejection tests | `test_p2a_budget_policy.py`: no-policy rejection, unlisted-budget rejection, malformed schema/list/type rejection, missing-file rejection; `run_native` fresh-output and ≤5-page gates unchanged. | **PASS** |

## 2. P2B readiness wording review

`docs/P2B_READINESS_DECISION.md` keeps the four required distinctions cleanly
separated and makes no inadmissible claim:

- **Measured total GPU peak** (8.55/8.78 GiB allocated/reserved, 4–5 pages,
  vision-prefill dominated, budget-insensitive) is attributed to P2-A/P1.5
  measurements only.
- **Decoder KV growth** and **vision activation** are discussed as
  decomposition, not conflated with the peak.
- **Input-page memory projection** is labeled arithmetic ("Likely fits — but
  this is arithmetic, not measurement") and explicitly refuses resource-only
  15-page diagnostics as evidence.
- **Physical 15-page measurement** is stated as nonexistent ("no correct
  15-page OCR result exists"); 15-page feasibility is nowhere claimed from
  4–5-page data.

Verdict: **no wording change required.** Gate 1 of its admissible-ladder list
("port and review the official n-gram suppression as its own identity change")
is exactly what `docs/rca/NGRAM35_SOURCE_AUDIT.md` (this branch) delivers the
source basis for; the decision document stays untouched as a dated record.

## 3. Required hardening for the P2 n-gram policy (carried into the P2 plan)

1. The n-gram gate must not repeat the schema-only weakness: pin the expected
   policy SHA-256 in the gate (a pinned-digest check, not just `schema == 1`),
   so a locally forged policy cannot authorize suppression.
2. Keep the driver-level bindings: experiment-name match, policy sha + path in
   the ledger config, budgets/ngram values listed explicitly.
3. Add the pre-run GPU memory admission check to the P2 runner procedure
   (procedural checklist item already in the audit §10.4 — make it a recorded
   field, not just a checklist line).
4. Fail-closed tests mirroring `test_p2a_budget_policy.py` for the n-gram gate:
   no-policy rejection, wrong-hash rejection, unlisted ngram/window rejection,
   malformed-policy rejection.

## 4. Conclusion

The P2-A budget gate and P2B decision text meet the mission's requirements for
identity, opt-in, approved values, source binding, admission and fail-closed
behavior at their scope. The single material gap — schema-only acceptance
inside `validate_budget` for direct callers — is documented here and becomes a
hardening requirement for the P2 n-gram policy gate before any GPU execution.
