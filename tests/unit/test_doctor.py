from copy import deepcopy

import pytest

from unlimited_ocr_2_rocm.doctor import require_stack


def valid():
    return {
        "platform": "Windows",
        "python": "3.10.20",
        "machine": "AMD64",
        "packages": {
            "torch": "2.12.0+rocm7.14.1",
            "torchvision": "0.27.0+rocm7.14.1",
            "torchaudio": "2.11.0+rocm7.14.1",
        },
        "hip": "7.14.60850",
        "rocm": "7.14.1",
        "cuda_build": None,
        "gpu_available": True,
        "gpu": {"name": "AMD Radeon(TM) 8060S Graphics", "arch": "gfx1151"},
    }


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("platform", "Linux"),
        ("python", "3.12.8"),
        ("machine", "ARM64"),
        ("hip", None),
        ("hip", "7.14.0"),
        ("rocm", "7.14.0"),
        ("cuda_build", "13.0"),
        ("gpu_available", False),
    ],
)
def test_refuses_wrong_environment(field, bad):
    report = valid()
    report[field] = bad
    with pytest.raises(RuntimeError):
        require_stack(report)


@pytest.mark.parametrize("package", ["torch", "torchvision", "torchaudio"])
def test_refuses_package_drift(package):
    report = deepcopy(valid())
    report["packages"][package] = "0.0.0"
    with pytest.raises(RuntimeError):
        require_stack(report)


@pytest.mark.parametrize(
    "gpu", [{"name": "NVIDIA", "arch": "gfx1151"}, {"name": "8060S", "arch": "gfx1100"}]
)
def test_refuses_wrong_device(gpu):
    report = valid()
    report["gpu"] = gpu
    with pytest.raises(RuntimeError):
        require_stack(report)


def test_accepts_target_identity():
    require_stack(valid())
    report = valid()
    report["gpu"]["arch"] = "gfx1151:sramecc-:xnack-"
    require_stack(report)
