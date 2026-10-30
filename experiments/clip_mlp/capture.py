"""Private tensor capture using the immutable 10067ee original-wheel worker."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/cold_start"))
from worker import sha, write  # noqa: E402


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--execute-gpu":
        raise ValueError("Use --execute-gpu NEW_LOCAL_OUTPUT")
    output = Path(sys.argv[2]).resolve()
    if Path.cwd().resolve() != ROOT or not output.is_relative_to(ROOT / ".b/mlp"):
        raise ValueError("Primary root and local fresh output required")
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT / "src"))
    from unlimited_ocr_2_rocm import model

    original_load = model.load_model
    records = []
    counter = [0]
    pending = [False]

    def captured_load(*a, **kw):
        import torch

        native = original_load(*a, **kw)
        deep = importlib.import_module(
            native.model.__class__.__module__.rsplit(".", 1)[0] + ".deepencoder"
        )
        original_gelu = deep.quick_gelu
        write(
            output / "source.json",
            dict(
                producer_sha256=sha(__file__),
                deepencoder_sha256=sha(deep.__file__),
                static_graph=str(original_gelu.graph),
                schema=str(original_gelu.schema),
                model=native.evidence,
            ),
        )

        def capture(stage, tensor):
            slot = output / f"call-{counter[0]:02d}"
            slot.mkdir(exist_ok=True)
            cpu = torch.empty_strided(
                tensor.shape, tensor.stride(), dtype=tensor.dtype, device="cpu"
            )
            cpu.copy_(tensor.detach())
            path = slot / (stage + ".pt")
            torch.save(cpu, path)
            import hashlib

            records.append(
                dict(
                    call=counter[0],
                    stage=stage,
                    shape=list(tensor.shape),
                    stride=list(tensor.stride()),
                    storage_offset=tensor.storage_offset(),
                    dtype=str(tensor.dtype),
                    device=str(tensor.device),
                    cpu_stride=list(cpu.stride()),
                    sha256=hashlib.sha256(
                        cpu.contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                    file_sha256=sha(path),
                )
            )

        block = native.model.model.vision_model.transformer.layers[0]

        def before_norm(mod, inputs):
            counter[0] += 1
            capture("pre_norm", inputs[0])

        def before_fc1(mod, inputs):
            capture("mlp_input", inputs[0])

        def after_fc1(mod, inputs, result):
            capture("activation_input", result)
            pending[0] = True

        def before_fc2(mod, inputs):
            capture("fc2_input", inputs[0])

        def after_fc2(mod, inputs, result):
            capture("mlp_output", result)

        block.layer_norm2.register_forward_pre_hook(before_norm)
        block.mlp.fc1.register_forward_pre_hook(before_fc1)
        block.mlp.fc1.register_forward_hook(after_fc1)
        block.mlp.fc2.register_forward_pre_hook(before_fc2)
        block.mlp.fc2.register_forward_hook(after_fc2)

        def gelu(tensor):
            selected = pending[0]
            pending[0] = False
            result = original_gelu(tensor)
            if selected:
                graph = str(torch.jit.last_executed_optimized_graph())
                capture("activation_output", result)
                write(
                    output / f"call-{counter[0]:02d}/graph.json",
                    dict(graph=graph, optimize=torch._C._get_graph_executor_optimize()),
                )
            return result

        deep.quick_gelu = gelu
        return native

    model.load_model = captured_load
    import worker as frozen

    args = SimpleNamespace(
        output=output / "worker",
        wheel="original",
        modes="MATH,MATH",
        case="pages_2",
        init=False,
        trace=False,
        deep_trace=False,
        jit_unoptimized=False,
    )
    args.output.mkdir()
    try:
        frozen.run(args)
    finally:
        write(output / "captures.json", records)


if __name__ == "__main__":
    main()
