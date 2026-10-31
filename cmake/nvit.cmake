# p100e-native vision encoder (src/nvit, tests/nvit_*, tools/p100e_nvit.cpp)
# Public include path is only src/nvit (nvit.h has no p100e / ggml dependencies), so users that include the llama.cpp
# fork's headers (gguf.h, ggml.h) keep resolving them to the fork.
add_library(p100e_nvit STATIC src/nvit/nvit_pre.cpp src/nvit/nvit_kernels.cu src/nvit/nvit.cu)
target_include_directories(p100e_nvit PUBLIC src/nvit PRIVATE src third_party)
target_link_libraries(p100e_nvit PRIVATE p100e_fast_gemm p100e_core PUBLIC CUDA::cublas CUDA::cudart)

add_executable(p100e-nvit-attn-test tests/nvit_attn_test.cu)
target_include_directories(p100e-nvit-attn-test PRIVATE src)
target_link_libraries(p100e-nvit-attn-test p100e_nvit)

# lend_gate (src/lend_gate.h): tp_model's prefill ring lent to the encoder vs prefills (CPU only)
add_executable(p100e-lend-gate-test tests/lend_gate_test.cpp)
target_include_directories(p100e-lend-gate-test PRIVATE src)

# p100e-nvit: encode / parity / bench tool. With the llama.cpp fork build present it can also run the mtmd reference
# encoder in-process (fork DLLs delay-loaded, only for --ref).
set(P100E_NVIT_FORK_SRC "${CMAKE_SOURCE_DIR}/third_party/llama.cpp-fork" CACHE PATH "llama.cpp fork source (at P100E_FORK_COMMIT)")
set(P100E_NVIT_FORK_BUILD "${CMAKE_SOURCE_DIR}/third_party/llama.cpp-fork/build" CACHE PATH "llama.cpp fork build")
set(_nvit_fork_libs
    "${P100E_NVIT_FORK_BUILD}/tools/mtmd/mtmd.lib"
    "${P100E_NVIT_FORK_BUILD}/src/llama.lib"
    "${P100E_NVIT_FORK_BUILD}/ggml/src/ggml.lib"
    "${P100E_NVIT_FORK_BUILD}/ggml/src/ggml-base.lib")
set(_nvit_have_fork ON)
foreach(f ${_nvit_fork_libs} "${P100E_NVIT_FORK_SRC}/tools/mtmd/mtmd.h" "${P100E_NVIT_FORK_SRC}/include/llama.h")
    if(NOT EXISTS "${f}")
        set(_nvit_have_fork OFF)
    endif()
endforeach()
add_executable(p100e-nvit tools/p100e_nvit.cpp)
target_link_libraries(p100e-nvit p100e_nvit)
if(MSVC AND _nvit_have_fork)
    target_sources(p100e-nvit PRIVATE tools/p100e_nvit_ref.cpp)
    target_include_directories(p100e-nvit BEFORE PRIVATE "${P100E_NVIT_FORK_SRC}/include" "${P100E_NVIT_FORK_SRC}/ggml/include"
                               "${P100E_NVIT_FORK_SRC}/tools/mtmd")
    target_compile_definitions(p100e-nvit PRIVATE P100E_NVIT_REF LLAMA_SHARED GGML_SHARED
                               "P100E_FORK_BIN=\"${P100E_NVIT_FORK_BUILD}/bin\"")
    target_link_libraries(p100e-nvit ${_nvit_fork_libs} delayimp)
    target_link_options(p100e-nvit PRIVATE "/DELAYLOAD:mtmd.dll" "/DELAYLOAD:llama.dll" "/DELAYLOAD:ggml.dll"
                        "/DELAYLOAD:ggml-base.dll")
    message(STATUS "p100e: nvit reference mode ON (fork ${P100E_NVIT_FORK_SRC})")
endif()

# the vision branch's vision_encoder (cmake/vision.cmake) can run this encoder instead of mtmd's (vision_opts::native)
if(TARGET p100e_vision)
    target_link_libraries(p100e_vision PUBLIC p100e_nvit)
    target_compile_definitions(p100e_vision PRIVATE P100E_HAVE_NVIT)
    # detail "low" token counts, both encoders (part 2 needs --mmproj / --vocab and a GPU)
    add_executable(p100e-vision-cap-test tests/vision_cap_test.cpp)
    target_link_libraries(p100e-vision-cap-test p100e_vision)
endif()
# p100e-server's copy of the same encoder (p100e-vision.dll, cmake/vision.cmake): the native encoder too, so the server
# runs nvit (two GPUs, weights in host RAM) by default instead of mtmd on one GPU (P100E_NVIT=0: mtmd)
if(TARGET p100e_vision_dll)
    target_link_libraries(p100e_vision_dll PRIVATE p100e_nvit)
    target_compile_definitions(p100e_vision_dll PRIVATE P100E_HAVE_NVIT)
endif()
