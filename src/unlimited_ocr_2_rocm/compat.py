"""Narrow, disclosed compatibility changes for the verified native stacks."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any


def wrap_vision_forward(
    forward: Callable[..., Any], name: str, events: list[dict[str, Any]]
) -> Callable[..., Any]:
    """Select correct GPU math SDPA only during one vision forward call."""

    @wraps(forward)
    def scoped(*args: Any, **kwargs: Any) -> Any:
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel

        tensors = [value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor)]
        if not tensors or any(tensor.device.type != "cuda" for tensor in tensors):
            raise RuntimeError("Vision compute tensors must remain on the physical GPU")
        event = {
            "module": name,
            "backend": "GPU MATH SDPA",
            "inputs": [
                {
                    "shape": list(tensor.shape),
                    "device": str(tensor.device),
                    "dtype": str(tensor.dtype),
                }
                for tensor in tensors
            ],
            "status": "ATTEMPTED",
        }
        events.append(event)
        with sdpa_kernel(SDPBackend.MATH):
            result = forward(*args, **kwargs)
            torch.cuda.synchronize()
        event["status"] = "PASS"
        return result

    return scoped
