# packed weights -> f16 (src/fast/matvec_unpack*, tests/fast_unpack*): lets the prefill fold GEMM run on packed weights
add_library(p100e_fast_unpack STATIC src/fast/matvec_unpack.cu)
target_include_directories(p100e_fast_unpack PUBLIC src src/fast third_party/ggml)
target_link_libraries(p100e_fast_unpack PUBLIC p100e_fast_matvec p100e_core)
target_link_libraries(p100e_fast_gemm PUBLIC p100e_fast_unpack)

add_executable(p100e-fast-unpack-test tests/fast_unpack_test.cu)
target_link_libraries(p100e-fast-unpack-test p100e_fast_gemm p100e_fast_unpack)

add_executable(p100e-fast-fusedq-test tests/fast_fusedq_test.cu)
target_link_libraries(p100e-fast-fusedq-test p100e_fast_gemm p100e_fast_unpack)
