# Token10 common-prefix result — append-only clarification

KEEP EXPERIMENTAL. This note derives an equality already present in the sealed
`token-selection.json`; it adds no model request or new arithmetic computation.

Under the same frozen MM prefix tokens0–9, **all four backend conditions select
the same token at step10**. Each has a unique maximum, top1−top2 margin0.125, and
agrees with its own fixed-hidden-state FP64 projection argmax. Full-vocabulary
logit hashes remain different across conditions.

The earlier natural-generation step10 selected-token difference therefore
disappears for this chosen common prefix. This supports context dependence after
the token7 divergence; it does not prove a universal prefix-only cause or equal
backend logits/cache values. At step7 the common-prefix diagnostic still shows
MM's unique BF16 maximum versus AA/AM/MA's two-way ties.

Exact captured token IDs remain private. The accompanying JSON compares hashed
identities and references the original sealed source hash. Original public
report/checksum files are unchanged; this addendum has separate checksums.
