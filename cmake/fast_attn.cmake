# fused norm/quantize + decode attention (src/fast/fused*, src/fast/attn*, tests/fast_attn*, tests/fast_fused*)
add_library(p100e_fast_attn STATIC src/fast/fused.cu src/fast/attn.cu src/fast/attn_fold.cu src/fast/attn_dec2.cu)
target_include_directories(p100e_fast_attn PUBLIC src)
target_link_libraries(p100e_fast_attn PUBLIC CUDA::cudart)

add_executable(p100e-fast-fused-test tests/fast_fused_test.cu)
target_link_libraries(p100e-fast-fused-test p100e_fast_attn p100e_core)

add_executable(p100e-fast-attn-test tests/fast_attn_test.cu)
target_link_libraries(p100e-fast-attn-test p100e_fast_attn p100e_core)

add_executable(p100e-fast-tinv-test tests/fast_tinv_test.cu)
target_link_libraries(p100e-fast-tinv-test p100e_fast_attn p100e_core)
# fold-GEMM prefill attention (fast/attn_fold.cu) vs attn_prefill
add_executable(p100e-fast-attn-fold-test tests/fast_attn_fold_test.cu)
target_link_libraries(p100e-fast-attn-fold-test p100e_fast_attn p100e_core)
add_executable(p100e-attn-dec-bench tests/attn_dec_bench.cu)
target_link_libraries(p100e-attn-dec-bench p100e_fast_attn p100e_core)
add_executable(p100e-attn-pf-bench tests/attn_pf_bench.cu)
target_link_libraries(p100e-attn-pf-bench p100e_fast_attn p100e_core)
# the reference attention's long-context path and P100E_FN_ATTN_FAST vs the serial kernel
add_executable(p100e-ref-attn-fast-test tests/ref_attn_fast_test.cu)
target_link_libraries(p100e-ref-attn-fast-test p100e_core)
