@echo off
call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat
if errorlevel 1 exit /b %errorlevel%
set MAX_JOBS=8
set GIT_CONFIG_COUNT=3
set GIT_CONFIG_KEY_0=url.C:/b/dependency-cache/xz.insteadOf
set GIT_CONFIG_VALUE_0=https://github.com/tukaani-project/xz.git
set GIT_CONFIG_KEY_1=url.C:/b/dependency-cache/aotriton.insteadOf
set GIT_CONFIG_VALUE_1=https://github.com/ROCm/aotriton.git
set GIT_CONFIG_KEY_2=submodule.third_party/triton.update
set GIT_CONFIG_VALUE_2=none
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
C:\b\be\Scripts\cmake.exe --build C:/b/pytorch/build --target torch_cpu --parallel 8
exit /b %errorlevel%





