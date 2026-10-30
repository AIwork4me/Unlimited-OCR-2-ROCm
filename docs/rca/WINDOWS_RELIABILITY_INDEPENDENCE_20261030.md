# Independent Windows reliability branch proposal

PR #21 targets `codex/synthetic-sdpa-ab`, draft PR #15's head
`3b817314c72b6f0b93e9ae4e13cc39115ce8bb99`, and imports its protocol and
Windows Job Object helpers. Its production baseline is integration commit
`82951c73b28bf735f5655e6f5c3306cd170a21c2`. This dependency is experimental
tooling, not evidence that candidate-wheel AUTO is needed for reliability.

The new `codex/cold-start-determinism-rca` branch starts directly from that
integration baseline in the primary checkout. The proposed reliability change
is additive: a local Job Object supervisor, original-wheel identity gates,
fresh-process matched controls, separate acceptance categories and CPU tests.
It must not bring in the candidate-wheel benchmark scheduler or require PR #15
to pass. The copied project-owned Job helper retains its original contents;
existing baseline model loader, inference and observer helpers are reused.

For a separate minimal reliability-only PR, propose a new branch
`codex/windows-reliability-independent` directly on `82951c7`. Its reviewable
scope would be the PR #21 original-wheel physical diagnostic with its P0 gate
implemented locally, a local copy of the Job Object helper, focused classifier/
ledger tests, and the reliability report. Remove imports from `p2_ab.worker`
and `p2_ab.protocol`; use the baseline observer and explicit original-wheel
identity instead. Exclude the candidate scheduler, AUTO experiments and timing
publication. This is a proposal for a new PR, not a base change to PR #21.

Retain PR #21 and its immutable FAILED/PARTIAL receipts as historical evidence.
Create a separate reviewable PR for the independent branch after new validation;
do not change either existing PR base, rewrite its commits or force-push.
Any later extraction of the reliability tooling should be reviewed against
this baseline and link the old evidence without relabeling its acceptance.

Keep external process termination, descendant cleanup, functional OCR recovery,
cold-start determinism, exact golden parity and ledger accounting separate.
A post-kill recovery can pass functional OCR and cleanup while failing exact
golden parity. A new campaign ledger must never promote the historical PARTIAL
campaign or count unrelated followup requests as its missing document.

No production MATH default, model source/weights, driver, SDK, global PATH,
security setting, release policy or historical tolerance changes are included.
