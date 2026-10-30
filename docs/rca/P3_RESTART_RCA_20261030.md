# New P3 restart controls — original wheel

Decision: **KEEP EXPERIMENTAL**. PR #21's historical strict FAIL and PARTIAL
campaign receipts are unchanged.

A new no-kill process makes two MATH requests. A separate fault process reaches
three completed decoder forwards, synchronizes with HIP error zero, then its
own Windows Job Object is externally terminated before a native result returns.
A new recovery process makes the same two MATH requests. Every process uses the
original published wheel, locked model, frozen two-page inputs, seed 0 and the
same generation/source/observation configuration. Normal JIT remains enabled.

| Acceptance category | New result | Evidence/limit |
| --- | --- | --- |
| External process termination | PASS | Synchronized decoder-forward checkpoint 3; nonzero exit 1; no native return before kill. |
| Descendant cleanup | PASS | All 9 owned processes accounted for; zero active after termination and final cleanup. |
| Functional OCR recovery | PASS | Both recovery requests have correct content/order/structure, native stop, 12 DASH calls and zero HIP error. |
| Matched-context cold-start determinism | Exact agreement observed | No-kill cold equals post-kill cold; no-kill warm equals post-kill warm. Limited observations, not universal determinism. |
| Cold/warm equality | FAIL | Both paths reproduce the same one-coordinate difference. |
| Exact ordered-golden parity | FAIL cold / PASS warm | The unchanged ordered golden is not replaced. |
| New strict ledger accounting | PASS accounting; PARTIAL campaign | Two documents / four expected pages; no-kill has one failed strict attempt; post-kill has termination FAIL then strict recovery FAIL. Warm requests receive no ledger credit. |

The actual new production Run ledger is created before execution, with attempts
reserved at job launch. Its final counts are PASS 0 / FAIL 2 / NOT_RUN 0, with
three sealed attempts. This accounting result is independent of qualification:
correctly preserving failed attempts does not make the strict campaign pass.

The no-kill and post-kill first/second raw hashes are respectively
`e6408c4a652f7a738abdb14299041a346136528098eca7337443e4a00210e49d`
and `c08e7f728ea1e46dc20c2958bf6d3ac857902542935e257a0b64525b7d7b2a8e`.
The bbox difference is reproduced without kill. These results provide no basis
to claim kill caused it. They do not qualify physical OOM, all timeout modes,
concurrent serving, other hardware, or overall P3 readiness.

See `cold-start-20261030/restart-categories.json`, `new-strict-ledger.json` and
the bound local artifact hashes. Historical PR #21 still has its original
FAIL 1 / NOT_RUN 1 PARTIAL campaign; the new ledger is a separate experiment.
