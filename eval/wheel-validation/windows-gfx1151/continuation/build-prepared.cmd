@echo off
call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat
if errorlevel 1 exit /b %errorlevel%
set MAX_JOBS=8
set libuv_ROOT=C:\b\libuv-install
set CMAKE_BUILD_PARALLEL_LEVEL=8
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set PIP_DEFAULT_TIMEOUT=60
set PIP_RETRIES=2
set HTTPS_PROXY=http://127.0.0.1:7897
set HTTP_PROXY=http://127.0.0.1:7897
set NO_PROXY=files.pythonhosted.org
set PIP_RESUME_RETRIES=8
set PATH=C:\b\be\Scripts;%PATH%
C:\b\be\Scripts\python.exe C:\b\therock\external-builds\pytorch\build_prod_wheels.py build --no-install-rocm --index-url https://repo.amd.com/rocm/whl-multi-arch/ --rocm-sdk-version ==7.14.1 --no-pre --rocm-extras device-gfx1151 --pytorch-rocm-arch gfx1151 --pytorch-dir C:/b/pytorch --pytorch-git-ref release/2.12 --no-build-triton --no-build-pytorch-audio --no-build-pytorch-vision --no-build-apex --enable-pytorch-flash-attention --output-dir C:/b/out/wheels
exit /b %errorlevel%



