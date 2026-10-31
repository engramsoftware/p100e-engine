#!/usr/bin/env bash
# Configure + build p100e on Linux (CUDA 12.9, Ninja, Release). Usage: ./build.sh [arch list, default "60;75;86"]
# (60 = P100, 75 = GTX 1660, 86 = RTX 3070 net node 1; pass "60" alone for a faster P100-only build)
# env: JOBS = parallel jobs (default 8), CUDA_HOME = CUDA toolkit (default /usr/local/cuda-12.9, then /usr/local/cuda),
#      BUILD_DIR = build tree (default ./build), extra CMake options in CMAKE_ARGS (e.g. "-DP100E_FORK_DIR=$HOME/llama.cpp-fork")
set -euo pipefail
ARCH="${1:-60;75;86}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${CUDA_HOME:-}" ]; then
    if [ -x /usr/local/cuda-12.9/bin/nvcc ]; then CUDA_HOME=/usr/local/cuda-12.9; else CUDA_HOME=/usr/local/cuda; fi
fi
[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "nvcc not found in $CUDA_HOME/bin (set CUDA_HOME; CUDA 12.x, not 13)" >&2; exit 1; }
export PATH="$CUDA_HOME/bin:$PATH"
BUILD_DIR="${BUILD_DIR:-$HERE/build}"
# shellcheck disable=SC2086
cmake -S "$HERE" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=$ARCH" \
      "-DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc" ${CMAKE_ARGS:-}
cmake --build "$BUILD_DIR" -j "${JOBS:-8}"
