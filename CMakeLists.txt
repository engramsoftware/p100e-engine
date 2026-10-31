cmake_minimum_required(VERSION 3.24)
project(p100e LANGUAGES CXX CUDA)

# P100 = sm_60 (target). 75 = GTX 1660 for local tests. CUDA 12.x only (13 dropped Pascal).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 75 86)   # 86 = RTX 3070 (net node 1): native SASS, no PTX JIT
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/W3>)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    # Linux: the static engine libraries also end up inside libp100e-vision.so (cmake/vision.cmake, via p100e_nvit)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xcudafe --diag_suppress=177 -Wno-deprecated-gpu-targets")

add_library(p100e_core STATIC
    src/gguf.cpp
    src/types.cu
    third_party/ggml/convert.cu
    src/kernels.cu
    src/model.cu
    src/tp_model.cu
    src/tp_mtp.cu
    src/fn_model.cu
    src/fn_qsa.cu
    src/fn_cpu_pool.cpp
    src/ram_guard.cpp
    src/fn_rexp_net.cpp
    src/sched.cpp
    src/engine.cpp
)   # src/net_link.cpp lives in p100e_pdl (cmake/pdl.cmake), linked below: the transports sit together
target_include_directories(p100e_core PUBLIC src third_party/ggml)
# CPU expert pool (P100E_FN_CPU=1): AVX2 + FMA + F16C kernels in this file only; fn_cpu_pool::cpu_ok() checks at run time
if(MSVC)
    set_source_files_properties(src/fn_cpu_pool.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
    set_source_files_properties(src/fn_cpu_pool.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
target_link_libraries(p100e_core PUBLIC CUDA::cublas CUDA::cudart Threads::Threads)
# network tensor parallel (src/net_link.cpp): Winsock on Windows
if(WIN32)
    target_link_libraries(p100e_core PUBLIC ws2_32)
endif()

# KL divergence vs a llama.cpp perplexity-tool base file (host only): p100e-run --kld-base, tests/kld_test.cpp
add_library(p100e_kld STATIC src/kld.cpp)
target_include_directories(p100e_kld PUBLIC src)
target_link_libraries(p100e_kld PUBLIC CUDA::cublas CUDA::cudart Threads::Threads)
add_executable(p100e-kld-test tests/kld_test.cpp)
target_link_libraries(p100e-kld-test p100e_kld)

add_executable(p100e-run tools/p100e_run.cpp tools/p100e_run_vision.cpp)
target_link_libraries(p100e-run p100e_core p100e_kld)

# several tp_model loads in one process must give identical, finite logits (needs the GPUs + a model)
add_executable(p100e-reinit-test tests/tp_reinit_test.cpp)
target_link_libraries(p100e-reinit-test p100e_core)

# tensor parallel over 1 / 2 / 4 ranks (virtual ranks on one GPU work): logits vs the first config, determinism, MTP
add_executable(p100e-tp-ntp-test tests/tp_ntp_test.cpp)
target_link_libraries(p100e-tp-ntp-test p100e_core)

add_executable(p100e-fn-run tools/p100e_fn_run.cpp)
# node 1 of the remote expert tier: Q4 experts on its GPUs, one request per layer per forward
add_executable(p100e-fn-expert-server tools/p100e_fn_expert_server.cu)
target_link_libraries(p100e-fn-expert-server p100e_core)
target_link_libraries(p100e-fn-run p100e_core)

# the CPU expert pool (P100E_FN_CPU=1) on synthetic Flash-Next experts: vs fp64 reference and vs the GPU path (needs a GPU)
# the host-RAM reserve (src/ram_guard.h; P100E_RAM_RESERVE_MB): check arithmetic and precedence, no GPU
add_executable(p100e-ram-guard-test tests/ram_guard_test.cpp src/ram_guard.cpp)
target_include_directories(p100e-ram-guard-test PRIVATE src)

add_executable(p100e-fn-cpu-pool-test tests/fn_cpu_pool_test.cu)
target_link_libraries(p100e-fn-cpu-pool-test p100e_core)

add_executable(p100e-info tools/p100e_info.cpp)
target_link_libraries(p100e-info p100e_core)

add_executable(p100e-dequant tools/p100e_dequant.cu)
target_link_libraries(p100e-dequant p100e_core)

include(cmake/fast_matvec.cmake)
target_link_libraries(p100e_core PUBLIC p100e_fast_matvec)
include(cmake/fast_attn.cmake)
include(cmake/fast_ssm.cmake)
include(cmake/fast_gemm.cmake)
include(cmake/fast_unpack.cmake)
target_link_libraries(p100e_core PUBLIC p100e_fast_attn p100e_fast_gemm)
include(cmake/fast_fusion.cmake)
include(cmake/prefill.cmake)
include(cmake/server.cmake)
include(cmake/build_stamp.cmake)
include(cmake/vision.cmake)
include(cmake/nvit.cmake)
include(cmake/sessions.cmake)
# pdl = raw-Ethernet all-reduce protocol and zero-copy GPU rings, plus the TCP net_link
include(cmake/pdl.cmake)
target_link_libraries(p100e_core PUBLIC p100e_pdl)   # tp_model's network transport: P100E_NET=tcp|pdl
target_link_libraries(p100e_core PUBLIC p100e_pdl_gpu)   # the zero-copy GPU link of P100E_NET_GPULINK (pdl_gpu.cu)
# the optional node-link key (P100E_NET_KEY_FILE, src/net_auth.h): in p100e_pdl next to net_link; fn_rexp_net.cpp
# (p100e_core) reaches it through the link above. BCryptGenRandom on Windows.
target_sources(p100e_pdl PRIVATE src/net_auth.cpp)
if(WIN32)
    target_link_libraries(p100e_pdl PUBLIC bcrypt)
endif()
# SHA-256 / HMAC test vectors and the key handshake between two threads (no network, no GPU)
add_executable(p100e-net-auth-test tests/net_auth_test.cpp)
target_link_libraries(p100e-net-auth-test p100e_pdl)
