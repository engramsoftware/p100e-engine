# decode fusions (add2_rmsnorm_q, ew_quant, attn_prep, gdn quantize epilogue, matvec_prequantized,
# concatenated weights) checked bit-exact against the per-op path they replace
add_executable(p100e-fast-fusion-test tests/fast_fusion_test.cu)
target_link_libraries(p100e-fast-fusion-test p100e_core)
