# fast matvec kernels (src/fast/matvec*, tests/fast_matvec*)
add_library(p100e_fast_matvec STATIC
    src/fast/matvec.cu
    src/fast/matvec_a16.cu
    src/fast/matvec_q4r.cu
    src/fast/matvec_q4kr.cu
    src/fast/matvec_q3r.cu
    src/fast/matvec_q2kr.cu
    src/fast/matvec_iq1.cu
    src/fast/matvec_q10.cu
    src/fast/matvec_q20.cu
    src/fast/matvec_iq2r.cu
    src/fast/matvec_iq3r.cu
    src/fast/matvec_q41.cu
    src/fast/matvec_kq.cu
    src/fast/matvec_generic.cu
    src/fast/matvec_wr.cu
    src/fast/matvec_wrx.cu
    src/fast/matvec_wrm.cu
    src/fast/matvec_wrm_q4kr.cu
    src/fast/matvec_wrm_q3r.cu
    src/fast/matvec_wrm_q5kr.cu
    src/fast/matvec_wrm_q6r3.cu
    src/fast/matvec_wrm_q6h1.cu
    src/fast/matvec_wrm_q6h2.cu
    src/fast/matvec_wrm_q6r2.cu
)
target_include_directories(p100e_fast_matvec PUBLIC src src/fast third_party/ggml)
target_link_libraries(p100e_fast_matvec PUBLIC CUDA::cudart)

add_executable(p100e-fast-matvec-test tests/fast_matvec_test.cu)
target_link_libraries(p100e-fast-matvec-test p100e_fast_matvec p100e_core)

add_executable(p100e-fast-moe-gemv-test tests/fast_moe_gemv_test.cu)
target_link_libraries(p100e-fast-moe-gemv-test p100e_fast_matvec p100e_core)

add_executable(p100e-fast-matvec-bench tests/fast_matvec_bench.cu)
target_link_libraries(p100e-fast-matvec-bench p100e_fast_matvec p100e_core)
# Flash-Next hyper-connection mixer matvecs (hc down Q8_0 10240 -> 320, hc up f16 320 -> 10240), T = 1..4
add_executable(p100e-fn-hc-bench tests/fn_hc_bench.cu)
target_link_libraries(p100e-fn-hc-bench p100e_fast_matvec p100e_core)

# F32 / F16 / BF16 float GEMV and the Q4_1 a16 GEMV on synthetic weights (accuracy, T invariance)
add_executable(p100e-fast-matvec-types-test tests/fast_matvec_types_test.cu)
target_link_libraries(p100e-fast-matvec-types-test p100e_fast_matvec p100e_core)
