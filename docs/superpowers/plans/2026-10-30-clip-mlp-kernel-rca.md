# CLIP MLP minimal kernel RCA plan

Goal: prove the first divergent operation on identical frozen BF16 tensors,
without changing production or historical evidence. KEEP EXPERIMENTAL.

- [x] Independently audit 10067ee, all 48 candidate receipts, 298/7 CPU logs,
  and 406 historical hashes; publish unchanged as draft PR #22.
- [x] One bounded original-wheel two-request capture on frozen synthetic pages
  1/2: block 0 pre-LN2, MLP input, fc1/activation input, activation output/fc2
  input, fc2 output; actual shapes/strides/dtype/device and executed JIT graphs.
  Captured bytes remain private under `.b/mlp/`. Reject capture if ordinary
  cold/warm OCR hashes no longer match the original evidence.
- [x] Standalone replay of the frozen activation tensor with no model/weights:
  fresh processes optimize-on/off, first call and repeated call; normal BF16
  GPU execution, layout/hash checks, actual last-executed graphs, operator and
  available GPU kernel traces; independent CPU FP32 reference. Two repeats per
  condition, no performance measurements. No `graph_for` calls that can execute
  hidden inference. Graph queries occur only after actual measured calls.
- [x] Adaptively predeclare narrow graph-transform controls after observing
  actual graphs; preserve schemas and tensor layout/dtype. Record unsupported
  profiling/kernel visibility explicitly. Identify no mechanism by assumption.
- [x] Independent reviewer reruns the minimal reproducer on the original wheel;
  sanitized numerical/graph evidence, updated separate P2/P3 gates, CPU checks,
  and a small draft follow-up PR based on PR #22. No merge or base rewrites.

API audit: the earlier intervention is `torch.jit.optimized_execution(False)`
around the entire original worker `run(options)`, not an activation-only scope.
Installed `_fuser.py` sets/restores graph executor optimize. Inspect the exact
published-wheel source commit 098172edc887399a593f0c74ae5c40b1fe043801 for how
this selects profiling/optimized/fallback execution plans; do not conflate it
with a direct CUDA initialization switch or a fusion-only switch.

No weights edited, production optimization switch/default MATH change,
historical relaxation, real documents, new bbox acceptance threshold, or
120-request measured P2 campaign. Commands run from the primary root; all
downloads/caches/evidence stay in-repository. Original 10067ee remains intact.

Publication: unchanged RCA draft PR #22; minimal follow-up draft PR #25.
Final CPU validation: 310 passed / 7 skipped, all six checks PASS.
