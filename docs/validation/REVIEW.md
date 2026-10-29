# Independent review of the completed wheel validation

The read-only reviewer independently checked the real wheel, all ZIP CRCs and
12,100 unique members, 15 binary hashes, seven official image files, source/ABI/HIP
identities and successful official build exit. The actual candidate profile and
all 16 core loaded native identities match the installed artifact.

Review found and closed three candidate-gate issues: stale/unbound pip-check
evidence, abbreviated error-clearing flags and incomplete CPU/Python/C10 native
coverage. Failing regression cases preceded fixes. The current focused gate
suite is **35 candidate + 30 existing tests**; final CPU suite is **121 passed,
two opt-in GPU markers skipped**. No numerical/fixture AST semantics changed.
Signed-URL log redaction findings were also reproduced, fixed and reviewed.

For GPU acceptance, the reviewer recalculated all 27 raw reports and confirmed
21 FUSED_PASS, four MATH_PASS, one CONTROL_PASS and the named optional
UNSUPPORTED result. Fixtures/arguments match baseline, each required group has
three runs, all numerical/nonfinite and HIP checks pass, and both logs show
actual attn_fwd load/lookup/nonnull launch/PAL execution. Producer hashes match
commit `3e641d97af952d3fa70a3fbec223d6faed41b047`. Checkout-context versus producer-byte
binding is accurately disclosed in [build provenance](BUILD_PROVENANCE.md).

**No actionable acceptance findings remain. FIX_VERIFIED is supported for the
tested local Windows gfx1151 BF16 forward artifact.**

Publication review independently verified all **710 checksum paths**, the
checksum file and **709 export records** against Git index blob bytes. Eight
gzip files have mtime=0 and match their uncompressed ledger hashes. All 174
unchanged initial files and four archived metadata blobs match the frozen
publication; only the three superseded root metadata files changed. Producer
blobs, the published profile and all 33 supplemental loaded/provider identities
match. No privacy residue or broken local report links were found. Handoff.md
agrees with measured results and scope.

Remote head/evidence integrity and hosted CI are checked after push. Hosted CPU
CI is not GPU acceptance. Scope exclusions are recorded in the decision report.
