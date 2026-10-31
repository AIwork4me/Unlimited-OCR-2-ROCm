# Independent read-only review

Verdict: **PASS for the bounded numerical-reference evidence and report. KEEP
EXPERIMENTAL. SAM and CLIP arithmetic-defect classifications remain INSUFFICIENT
EVIDENCE.** This is independent recomputation on the same physical host, not an
independent hardware reproduction. The reviewer made no file edits, GPU inference,
network submissions or system changes.

The reviewer independently verified prior isolation261, prior RCA379 and all406
historical hashes, both wheel identities and the18 pinned model/config/tokenizer/
source/weight files. Earlier exact repeatability and independent SAM/CLIP dispatch
evidence remain qualified at their existing scope.

Pre-execution review identified a stale guard binding in the first diagnostic
clone design; a dynamic guard resolver and red/green regression test corrected it
before execution. Two further failed attempts were preserved: a premature
Transformers import hit the unchanged fresh-module guard; then decorated-function
source was cloned with wrapper globals. The revised import order and unwrapped
target namespace were reviewed, with a real no_grad regression verifying preserved
grad semantics. Neither failed attempt completed OCR inference. No production
guard, cache, kernel or numerical criterion was changed to bypass a failure.

The final four diagnostics pass all per-worker gates. All424 captured tensor file
and logical-byte hashes, shapes/dtypes verify; four Jobs exit0 with zero active
descendants. Exact input/metadata/stride/offset equality holds for SAM MM/AA and
CLIP MM/MA plus AM/AA. The first measured differing attention return is followed
through CLIP attention projection, residual, norm2, MLP and block output. The
source and hooks support these boundaries.

The reviewer independently coded unchunked NumPy FP64 attention and obtained
bitwise-exact complete reference arrays for SAM and both CLIP input sets. All
reported backend maximum/mean errors agree with independent recomputation. These
are mathematical references, not exact-real arithmetic, backend emulators or new
acceptance tolerances.

The reviewer checked all eight full-vocabulary FP64 lm_head vectors, each
containing129280 entries. Independent CPU torch matmul
matches the authored NumPy result bitwise;16 top-two scalar dots recomputed using
math.fsum also match exactly. All observed top-two GPU values match the BF16-rounded
FP64 projections. At step7 all fixed-hidden FP64 projections select the MM winner;
AA/AM/MA GPU outputs are ties and select another token under unchanged argmax.
This is bounded final-projection quantization evidence, not higher-precision
decoder execution or an SDPA defect proof.

Supported DynamicCache indexing and sequence lengths are verified against the
installed4.52.1 source. All192 K/V append-prefix checks pass, with positions
899/906/909 across12 layers in each request. A preliminary reviewer message used
72 as a check-count typo;192 is the independently verified count. Original missing
offset/cache captures remain explicitly unmeasured at their old identities.

The CPU-only generic-dispatch filter was corrected after all GPU work and its
declared source preserved separately. No GPU rerun was used for that analysis-only
change. Fresh repository CPU receipts each exit0, pytest340 passed/7 skipped;
nine local experimental CPU tests pass.

Public report, matrices, scalar summaries and producer snapshots were reviewed
for private OCR, captured token arrays/IDs, raw tensors and user-directory paths;
none were found. Actual token IDs and full arrays remain in private local receipts.
The report distinguishes shared capture-point checks from absent historical full
prefill/cache observations, and defect qualification from descriptive consistency
with floating-point behavior. No material scientific/privacy objection remains
to a draft PR with this bounded scope. Final seals receive a separate closure
verification; merging, release and upstream issue submission are not authorized.
