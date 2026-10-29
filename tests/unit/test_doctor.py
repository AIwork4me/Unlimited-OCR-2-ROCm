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
        "windows_hardware": {
            "os": {"Caption": "Microsoft Windows 11 Home", "BuildNumber": "26200"},
            "cpu": {"Name": "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S"},
        },
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


@pytest.mark.parametrize(
    "hardware",
    [
        {},
        {
            "os": {"Caption": "Microsoft Windows 10 Home", "BuildNumber": "19045"},
            "cpu": {"Name": "AMD RYZEN AI MAX+ 395"},
        },
        {
            "os": {"Caption": "Microsoft Windows Server 2025", "BuildNumber": "26100"},
            "cpu": {"Name": "AMD RYZEN AI MAX+ 395"},
        },
        {
            "os": {"Caption": "Microsoft Windows 11 Home", "BuildNumber": "26200"},
            "cpu": {"Name": "AMD Ryzen 9 7950X"},
        },
    ],
)
def test_refuses_wrong_or_missing_os_and_cpu_evidence(hardware):
    report = valid()
    report["windows_hardware"] = hardware
    with pytest.raises(RuntimeError):
        require_stack(report)


def linux_valid():
    return {
        "platform": "Linux",
        "python": "3.10.20",
        "machine": "x86_64",
        "packages": {"torch": "2.14.1+rocm7.14"},
        "hip": "7.14.60850",
        "rocm": "7.14",
        "cuda_build": None,
        "gpu_available": True,
        "gpu": {"name": "AMD Radeon Graphics", "arch": "gfx1151"},
        "linux_hardware": {
            "cpu_model_name": "AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S",
            "product_name": "HP ZBook Ultra G1a 14 inch Mobile Workstation PC",
            "kernel_release": "6.17.0-1033-oem",
        },
    }


def test_accepts_linux_target_identity_and_full_arch_suffix():
    require_stack(linux_valid())
    report = linux_valid()
    report["gpu"]["arch"] = "gfx1151:sramecc-:xnack-"
    require_stack(report)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("python", "3.12.3"),
        ("machine", "aarch64"),
        ("hip", None),
        ("hip", "7.2.53211"),
        ("rocm", "7.2.1"),
        ("cuda_build", "13.0"),
        ("gpu_available", False),
    ],
)
def test_linux_refuses_wrong_environment(field, bad):
    report = linux_valid()
    report[field] = bad
    with pytest.raises(RuntimeError):
        require_stack(report)


def test_linux_refuses_package_drift():
    report = linux_valid()
    report["packages"]["torch"] = "2.12.0+rocm7.14.1"
    with pytest.raises(RuntimeError):
        require_stack(report)


@pytest.mark.parametrize(
    "gpu",
    [
        {"name": "AMD Radeon Graphics", "arch": "gfx1100"},
        {"name": "AMD Radeon Graphics", "arch": ""},
        {"name": "AMD Radeon Graphics", "arch": "gfx1150"},
    ],
)
def test_linux_refuses_wrong_device(gpu):
    report = linux_valid()
    report["gpu"] = gpu
    with pytest.raises(RuntimeError):
        require_stack(report)


@pytest.mark.parametrize(
    "hardware",
    [
        {},
        {"cpu_model_name": "AMD Ryzen 9 7950X", "product_name": "x"},
        {"cpu_model_name": "AMD RYZEN AI MAX+ PRO 395", "product_name": "x"},
    ],
)
def test_linux_refuses_wrong_or_missing_cpu_evidence(hardware):
    report = linux_valid()
    report["linux_hardware"] = hardware
    with pytest.raises(RuntimeError):
        require_stack(report)
