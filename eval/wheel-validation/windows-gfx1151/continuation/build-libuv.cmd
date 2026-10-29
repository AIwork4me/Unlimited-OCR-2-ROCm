@echo off
call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat
if errorlevel 1 exit /b %errorlevel%
set PATH=C:\b\be\Scripts;%PATH%
set CMAKE_BUILD_PARALLEL_LEVEL=8
cmake -S C:\b\libuv -B C:\b\libuv\build -DBUILD_TESTING=OFF
if errorlevel 1 exit /b %errorlevel%
cmake --build C:\b\libuv\build --config Release --parallel 8
if errorlevel 1 exit /b %errorlevel%
cmake --install C:\b\libuv\build --config Release --prefix C:\b\libuv-install
exit /b %errorlevel%
