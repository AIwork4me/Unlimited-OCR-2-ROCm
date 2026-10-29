"""CPU tests of native metadata boundaries; no torch or model imports."""

import ast
import hashlib
import json
from pathlib import Path

import pytest

from unlimited_ocr_2_rocm import completion


def native_result(*, finish="stop", tokens=None, pages=2):
    ids = [128827, 5] * pages if tokens is None else tokens
    return {
        "text": "<PAGE><BLOCK>text[0,0,1,1]<FORK>hello" * pages,
        "token_ids": ids,
        "prompt_token_ids": [0] + [128827, 128815] * pages + [9],
        "finish_reason": finish,
        "num_pages_out": ids.count(128827),
        "expected_pages": pages,
        "time": 1.25,
        "prefill_time": 0.25,
        "prompt": "<PAGE><image>" * pages + "Multi page parsing.",
    }


def test_multi_mapping_matches_reviewed_official_infer_multi():
    """Wrong crop policy, input ordering or return_dict loses native equivalence."""
    root = Path(__file__).resolve().parents[2]
    snapshot = root / ".b/h/models--baidu--Unlimited-OCR-2/snapshots"
    source = snapshot / "9cf077c4b206e0e2a31879df937b0b0bcba3661c/modeling_unlimitedocr.py"
    if not source.is_file():
        pytest.skip("Optional local pinned-source characterization; no model download")
    payload = source.read_bytes()
    lock = json.loads((root / "src/unlimited_ocr_2_rocm/upstream-lock.json").read_text())
    assert hashlib.sha256(payload).hexdigest() == lock["files"][source.name]
    tree = ast.parse(payload)
    model_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UnlimitedOCRForCausalLM"
    )
    method = next(
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "infer_multi"
    )
    module = ast.Module(body=[method], type_ignores=[])
    scope = {}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)

    class Boundary:
        def __init__(self):
            self.calls = []

        def infer(self, tokenizer, **kwargs):
            self.calls.append((tokenizer, kwargs))
            return native_result()

    model = Boundary()
    files = ["page-2.png", "page-1.png"]
    reference = scope["infer_multi"](
        model,
        "tokenizer",
        image_files=files,
        max_new_tokens=8192,
        save_results=False,
        eval_mode=True,
    )
    actual = completion.infer_document(
        model,
        "tokenizer",
        files,
        multi=True,
        max_new_tokens=8192,
    )
    assert reference == (actual["text"], len(actual["token_ids"]))
    # The adapter may omit default kwargs, but every effective input must agree.
    reference_kwargs = model.calls[0][1]
    adapter_kwargs = model.calls[1][1]
    for name, value in adapter_kwargs.items():
        assert reference_kwargs[name] == value
    assert adapter_kwargs["image_file"] == ["page-2.png", "page-1.png"]
    assert adapter_kwargs["crop_mode"] is False
    assert adapter_kwargs["return_dict"] is True


def test_length_below_output_budget_stays_length():
    """A max_length break must never be guessed to be EOS from a small count."""
    result = native_result(finish="length")
    metadata = completion.completion_metadata(result, expected_pages=2, max_new_tokens=8192)
    assert metadata["finish_reason"] == "length"
    assert metadata["output_tokens"] == 4
    assert metadata["native_generation"] == result


def test_eos_excluded_from_native_count_needs_remaining_decode_iteration():
    result = native_result(tokens=[128827] + [5] * 8190, pages=1)
    assert (
        completion.completion_metadata(
            result,
            expected_pages=1,
            max_new_tokens=8192,
        )["finish_reason"]
        == "stop"
    )
    result["token_ids"].append(5)
    with pytest.raises(ValueError, match="EOS"):
        completion.completion_metadata(result, expected_pages=1, max_new_tokens=8192)


@pytest.mark.parametrize(
    "change",
    [
        {"finish_reason": "unknown"},
        {"token_ids": [True]},
        {"token_ids": [1.2]},
        {"token_ids": [-1]},
        {"expected_pages": 1},
        {"num_pages_out": 1},
        {"time": float("nan")},
        {"prompt_token_ids": []},
        {"text": None},
    ],
)
def test_malformed_native_metadata_fails_closed(change):
    result = native_result()
    result.update(change)
    with pytest.raises(ValueError):
        completion.completion_metadata(result, expected_pages=2, max_new_tokens=8192)


def test_unknown_tuple_cannot_become_completion():
    with pytest.raises(ValueError):
        completion.completion_metadata(("text", 4), expected_pages=2, max_new_tokens=8192)


@pytest.mark.parametrize(
    "raw",
    [
        "<PAGE><BLOCK><FORK>broken",
        "<PAGE>garbage<BLOCK>text[0,0,1,1]<FORK>ok",
        "<PAGE><BLOCK>text[bad]<FORK>broken",
        "<PAGE><BLOCK>text[0,0,1,1]<FORK>ok<BLOCK><FORK>broken",
        "<PAGE><BLOCK>title<LEVEL1}[0,0,1,1]<FORK>hello",
    ],
)
def test_converter_fallback_cannot_establish_valid_page_structure(raw):
    with pytest.raises(ValueError, match="malformed"):
        completion.require_complete_pages(raw, 1)


@pytest.mark.parametrize(
    "kind, state",
    [
        ("missing_page", "FAIL"),
        ("malformed", "FAIL"),
        ("fallback_header", "FAIL"),
        ("repaired_level", "FAIL"),
        ("below_budget_length", "TRUNCATED"),
        ("valid", "PASS"),
    ],
)
def test_native_metadata_real_ledger_requires_complete_pages(tmp_path, kind, state):
    from test_benchmark_ledger import make_run

    run = make_run(tmp_path)
    result = native_result()
    if kind == "missing_page":
        result = native_result(tokens=[128827, 5], pages=1)
        result["expected_pages"] = 2
        result["prompt_token_ids"] = [0, 128827, 128815, 128827, 128815, 9]
    elif kind == "malformed":
        result["text"] = "<PAGE>broken<PAGE>broken"
    elif kind == "fallback_header":
        result["text"] = "<PAGE><BLOCK><FORK>broken" * 2
    elif kind == "repaired_level":
        result["text"] = "<PAGE><BLOCK>title<LEVEL1}[0,0,1,1]<FORK>hello" * 2
    elif kind == "below_budget_length":
        result["finish_reason"] = "length"
    metadata = completion.completion_metadata(result, expected_pages=2, max_new_tokens=8192)
    metadata["memory"] = {"rss": 1}
    attempt = run.begin("z")
    run.finish("z", attempt, result["text"], "PASS", metadata)
    assert run.attempts("z")[-1]["state"] == state
    assert (run.path / "documents/z/000001/raw.txt").read_bytes() == result["text"].encode()
    assert run.summary()["expected_pages"] == 12


def test_generic_ledger_does_not_inherit_native_one_million_character_bound(tmp_path):
    from test_benchmark_ledger import make_run

    run = make_run(tmp_path)
    raw = ("<PAGE><BLOCK>text[0,0,1,1]<FORK>" + "x" * 500_001) * 2
    attempt = run.begin("z")
    run.finish(
        "z",
        attempt,
        raw,
        "PASS",
        {
            "finish_reason": "stop",
            "output_tokens": 7,
            "max_tokens": 100,
            "memory": {},
        },
    )
    assert run.attempts("z")[-1]["state"] == "PASS"
    assert (run.path / "documents/z/000001/raw.txt").read_bytes() == raw.encode()


def test_completion_retry_preserves_interrupted_and_committed_bytes(tmp_path, monkeypatch):
    from test_benchmark_ledger import make_run

    from unlimited_ocr_2_rocm.benchmark.ledger import Run

    run = make_run(tmp_path)
    run.begin("z")
    result = native_result(finish="length")
    meta = completion.completion_metadata(result, expected_pages=2, max_new_tokens=8192)
    meta["memory"] = {}
    second = run.begin("z")
    run.finish("z", second, result["text"], "PASS", meta)
    committed = run.path / "documents/z" / second
    before = {p.name: p.read_bytes() for p in committed.iterdir()}
    resumed = Run(run.path)
    assert resumed.attempts("z")[0]["reason"] == "INTERRUPTED"
    assert "z" in resumed.todo()
    third = resumed.begin("z")
    original_rename = Path.rename

    def interrupted_rename(path, target):
        if path.name == third + ".pending":
            raise OSError("injected interruption before atomic commit")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", interrupted_rename)
    with pytest.raises(OSError, match="atomic commit"):
        resumed.finish("z", third, result["text"], "PASS", meta)
    assert {p.name: p.read_bytes() for p in committed.iterdir()} == before
    assert resumed.attempts("z")[-1]["reason"] == "INTERRUPTED"
    assert not (run.path / "documents/z" / third).exists()


@pytest.mark.parametrize(
    "finish, kind, status",
    [
        ("stop", "valid", "INFERENCE_ONLY"),
        ("length", "valid", "FAIL"),
        ("unknown", "valid", "FAIL"),
        ("stop", "missing", "FAIL"),
        ("stop", "malformed", "FAIL"),
        ("stop", "terminal", "INFERENCE_ONLY"),
    ],
)
def test_run_native_retains_authoritative_result_and_raw(
    tmp_path, monkeypatch, finish, kind, status
):
    import contextlib
    import sys
    from types import SimpleNamespace

    import unlimited_ocr_2_rocm.inference as inference

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            inference_mode=contextlib.nullcontext,
            cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None, synchronize=lambda: None),
        ),
    )
    monkeypatch.setattr(inference, "memory_snapshot", lambda: {})
    result = native_result(finish=finish)
    if kind == "missing":
        result["text"] = "<PAGE><BLOCK>text[0,0,1,1]<FORK>hello"
        result["token_ids"] = [128827, 5]
        result["num_pages_out"] = 1
    elif kind == "malformed":
        result["text"] = "<PAGE>broken<PAGE>broken"
    elif kind == "terminal":
        result["text"] += "<PAGE>"
        result["token_ids"].append(128827)
        result["num_pages_out"] = 3
    native = SimpleNamespace(tokenizer=None, kernels=[], evidence={})

    def infer(tokenizer, **kwargs):
        assert kwargs["crop_mode"] is False
        assert kwargs["return_dict"] is True
        native.kernels.extend({"name": "unit-boundary-double"} for _ in range(12))
        return result

    native.model = SimpleNamespace(infer=infer)
    image = Path(__file__).resolve().parents[1] / "fixtures/english.png"
    output = tmp_path / "output"
    report = inference.run_native(native, [image, image], output, multi=True, max_new_tokens=8192)
    assert report["status"] == status
    assert (output / "raw.txt").read_bytes() == result["text"].encode()
    saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert saved["native_entrypoint"] == "infer"
    assert saved["native_generation"] == result
    assert saved["finish_reason"] == finish
    assert saved["native_kernel_count"] == 12


def test_runner_interrupt_writes_explicit_failure_and_keeps_prior_evidence(tmp_path, monkeypatch):
    import contextlib
    import sys
    from types import SimpleNamespace

    import unlimited_ocr_2_rocm.inference as inference

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            inference_mode=contextlib.nullcontext,
            cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None, synchronize=lambda: None),
        ),
    )
    monkeypatch.setattr(inference, "memory_snapshot", lambda: {})

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("injected user interruption")

    native = SimpleNamespace(
        model=SimpleNamespace(infer=interrupted),
        tokenizer=None,
        kernels=[],
        evidence={},
    )
    prior = tmp_path / "prior.txt"
    prior.write_bytes(b"earlier immutable evidence")
    image = Path(__file__).resolve().parents[1] / "fixtures/english.png"
    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        inference.run_native(native, [image], output)
    saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert saved["status"] == "FAIL"
    assert saved["error"] == "INTERRUPTED"
    assert "injected user interruption" in saved["traceback"]
    assert not (output / "raw.txt").exists()  # Upstream never returned partial IDs/text.
    assert prior.read_bytes() == b"earlier immutable evidence"


@pytest.mark.parametrize(
    "field, value, diagnostic",
    [
        ("time", object(), {"invalid_type": "builtins.object"}),
        ("token_ids", {128827, 5}, {"invalid_type": "builtins.set"}),
        ("unexpected", object(), {"invalid_type": "builtins.object"}),
        ("time", float("nan"), {"nonfinite_float": "nan"}),
        ("unexpected", float("inf"), {"nonfinite_float": "inf"}),
    ],
)
def test_malformed_metadata_cannot_prevent_strict_json_failure_manifest(
    tmp_path,
    monkeypatch,
    field,
    value,
    diagnostic,
):
    import contextlib
    import sys
    from types import SimpleNamespace

    import unlimited_ocr_2_rocm.inference as inference

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            inference_mode=contextlib.nullcontext,
            cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None, synchronize=lambda: None),
        ),
    )
    monkeypatch.setattr(inference, "memory_snapshot", lambda: {})
    result = native_result(pages=1)
    result[field] = value
    native = SimpleNamespace(tokenizer=None, kernels=[], evidence={})

    def infer(*args, **kwargs):
        native.kernels.extend({} for _ in range(12))
        return result

    native.model = SimpleNamespace(infer=infer)
    output = tmp_path / "invalid-native-metadata"
    image = Path(__file__).resolve().parents[1] / "fixtures/english.png"
    report = inference.run_native(native, [image], output)
    assert report["status"] == "FAIL"
    saved = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8"),
        parse_constant=lambda value: pytest.fail("nonstandard JSON constant: " + value),
    )
    assert saved["status"] == "FAIL"
    assert saved["finish_reason"] == "stop"
    assert saved["native_generation"][field] == diagnostic
    assert (output / "raw.txt").read_bytes() == result["text"].encode()


def test_diagnostic_evidence_preserves_valid_result_and_tags_cycles_without_repr():
    result = native_result()
    copied, invalid = completion.diagnostic_evidence(result)
    assert copied == result
    assert invalid is False
    copied["token_ids"].append(99)
    assert result["token_ids"] == [128827, 5, 128827, 5]
    circular = {}
    circular["self"] = circular
    safe, invalid = completion.diagnostic_evidence(circular)
    assert invalid is True
    assert safe == {"self": {"invalid_container": "cycle"}}
    assert json.loads(json.dumps(safe, allow_nan=False)) == safe


def test_diagnostic_evidence_handles_utf8_surrogate_metadata():
    safe, invalid = completion.diagnostic_evidence({"unexpected": "\ud800"})
    assert invalid is True
    json.dumps(safe, ensure_ascii=False, allow_nan=False).encode("utf-8")


def test_diagnostic_evidence_handles_interpreter_integer_encoding_limit():
    import sys

    limit = getattr(sys, "get_int_max_str_digits", lambda: 0)()
    if not limit:
        pytest.skip("Interpreter has no integer serialization digit limit")
    safe, invalid = completion.diagnostic_evidence({"unexpected": 10 ** (limit + 1)})
    assert invalid is True
    json.dumps(safe, ensure_ascii=False, allow_nan=False).encode("utf-8")


@pytest.mark.parametrize("nested", [False, True])
def test_invalid_mapping_keys_keep_identity_and_cannot_break_failure_manifest(
    tmp_path,
    monkeypatch,
    nested,
):
    import contextlib
    import sys
    from types import SimpleNamespace

    import unlimited_ocr_2_rocm.inference as inference

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            inference_mode=contextlib.nullcontext,
            cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None, synchronize=lambda: None),
        ),
    )
    monkeypatch.setattr(inference, "memory_snapshot", lambda: {})
    result = native_result(pages=1)
    mapping = {"\ud800": 1, "\\ud800": 2}
    if nested:
        result["unexpected"] = mapping
    else:
        result.update(mapping)
    native = SimpleNamespace(tokenizer=None, kernels=[], evidence={})

    def infer(*args, **kwargs):
        native.kernels.extend({} for _ in range(12))
        return result

    native.model = SimpleNamespace(infer=infer)
    output = tmp_path / "invalid-keys"
    image = Path(__file__).resolve().parents[1] / "fixtures/english.png"
    report = inference.run_native(native, [image], output)
    assert report["status"] == "FAIL"
    payload = (output / "manifest.json").read_bytes()
    assert payload
    saved = json.loads(payload)
    assert saved["status"] == "FAIL"
    assert saved["finish_reason"] == "stop"
    evidence = saved["native_generation"]
    if nested:
        evidence = evidence["unexpected"]
    entries = evidence["invalid_mapping"]
    assert {"key": {"invalid_string": "'\\ud800'"}, "value": 1} in entries
    assert {"key": "\\ud800", "value": 2} in entries
    assert (output / "raw.txt").read_bytes() == result["text"].encode()
