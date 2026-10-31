# paged KV cache + multi-sequence kernels (fast/kv_page.cuh, attn_decode_rows, gdn_layer_multi) checked
# bit-exact against the single-sequence calls
add_executable(p100e-fast-sessions-test tests/fast_sessions_test.cu)
target_link_libraries(p100e-fast-sessions-test p100e_fast_attn p100e_core)
