@echo off
rem Configure + build p100e (CUDA 12.9, Ninja, Release). Usage: build.cmd [arch list, default "60;75;86"]; env JOBS = parallel jobs (default 8; use 2 when several builds run at once); env CMAKE_ARGS = extra CMake arguments
rem 60 = P100, 75 = GTX 1660 (local tests), 86 = RTX 3070 (net node 1); pass "60" alone for a faster P100-only build
setlocal
set "ARCH=%~1"
if "%ARCH%"=="" set "ARCH=60;75;86"
if not defined P100E_VCVARS set "P100E_VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not defined P100E_CUDA_PATH set "P100E_CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
call "%P100E_VCVARS%" >nul || exit /b 1
set "CUDA_PATH=%P100E_CUDA_PATH%"
set "PATH=%CUDA_PATH%\bin;%PATH%"
cmake -S "%~dp0." -B "%~dp0build" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=%ARCH%" -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler %CMAKE_ARGS% || exit /b 1
if "%JOBS%"=="" set "JOBS=8"
cmake --build "%~dp0build" -j %JOBS% || exit /b 1
