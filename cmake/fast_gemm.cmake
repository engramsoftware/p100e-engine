# fast prefill GEMM for raw quantized weights (src/fast/gemm*, tests/fast_gemm*)
add_library(p100e_fast_gemm STATIC src/fast/gemm.cu)
target_include_directories(p100e_fast_gemm PUBLIC src)
target_link_libraries(p100e_fast_gemm PUBLIC p100e_core CUDA::cuda_driver)   # driver API: the bank-fixed SASS cubin

# src/fast/gemm_sass_fold8.h holds a binary of fold_seg_kernel<8, false> that only
# cc 6.0 GPUs run. `fold_bankfix.py --check` (pure Python, no nvcc) runs before gemm.cu compiles and fails the build
# when the header was built from another kernel text (rerun tools/sass/fold_bankfix.py then); the header itself
# static_asserts seg_tab's layout and the launch arguments. Without Python the check cannot run, so the SASS path is
# then off by default (P100E_PF_SASS_DEFAULT=0; P100E_PF_SASS=1 still turns it on).
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    set(P100E_FOLD_SASS_STAMP ${CMAKE_CURRENT_BINARY_DIR}/fold_sass_check.stamp)
    add_custom_command(OUTPUT ${P100E_FOLD_SASS_STAMP}
        COMMAND ${Python3_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/tools/sass/fold_bankfix.py --check
        COMMAND ${CMAKE_COMMAND} -E touch ${P100E_FOLD_SASS_STAMP}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/fast/gemm.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/fast/gemm_sass_fold8.h
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/sass/fold_bankfix.py
        COMMENT "fold_bankfix.py --check: is the bank-fixed SASS header built from the current fold kernel?"
        VERBATIM)
    add_custom_target(p100e_fold_sass_check DEPENDS ${P100E_FOLD_SASS_STAMP})
    add_dependencies(p100e_fast_gemm p100e_fold_sass_check)
else()
    message(WARNING "Python 3 not found: cannot check src/fast/gemm_sass_fold8.h against the fold kernel; "
                    "the bank-fixed SASS prefill GEMM is off by default (P100E_PF_SASS=1 turns it on)")
    target_compile_definitions(p100e_core PRIVATE P100E_PF_SASS_DEFAULT=0)
endif()

add_executable(p100e-fast-gemm-test tests/fast_gemm_test.cu)
target_link_libraries(p100e-fast-gemm-test p100e_fast_gemm)
# the bank-fixed sm_60 cubins of the fold kernel vs the compiled kernel (exactness, time)
add_executable(p100e-fast-sass-test tests/fast_sass_test.cu)
target_link_libraries(p100e-fast-sass-test p100e_fast_gemm)
# the f16 wire's saturation guard (out16 + sat at +-65504 / 65520) and fast::abs_max (wire statistics)
add_executable(p100e-fast-wire-test tests/fast_wire_test.cu)
target_link_libraries(p100e-fast-wire-test p100e_fast_gemm)
