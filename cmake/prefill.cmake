# P2P transfer micro-benchmark for the tensor-parallel exchange
add_executable(p100e-p2p-bw-test tests/p2p_bw_test.cu)
target_link_libraries(p100e-p2p-bw-test CUDA::cudart)
add_executable(p100e-fast-prefill-test tests/fast_prefill_test.cu)
target_link_libraries(p100e-fast-prefill-test p100e_fast_gemm p100e_fast_unpack p100e_fast_attn p100e_core)
