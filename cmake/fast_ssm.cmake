# fast gated-delta-net kernels (src/fast/ssm*, tests/fast_ssm*)
target_sources(p100e_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/fast/ssm.cu)
add_executable(p100e-fast-ssm-test ${CMAKE_CURRENT_SOURCE_DIR}/tests/fast_ssm_test.cu)
target_link_libraries(p100e-fast-ssm-test p100e_core)
# the MTP verify's batched delta net (one T-token call writing the rollback snapshots) vs the
# per-token loop it replaces, bit-exact
add_executable(p100e-fast-gdn-verify-test ${CMAKE_CURRENT_SOURCE_DIR}/tests/fast_gdn_verify_test.cu)
target_link_libraries(p100e-fast-gdn-verify-test p100e_core)
# the fork's chunked gated delta net for prefill (P100E_GDN_CHUNKED), vs the sequential
# scan and an fp64 reference
target_sources(p100e_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/fast/gdn_chunked.cu)
add_executable(p100e-fast-gdn-chunk-test ${CMAKE_CURRENT_SOURCE_DIR}/tests/fast_gdn_chunk_test.cu)
target_link_libraries(p100e-fast-gdn-chunk-test p100e_core)
# a decay that underflowed to 0 must not turn the chunked scan into NaN
add_executable(p100e-gdn-chunk-zero-decay-test ${CMAKE_CURRENT_SOURCE_DIR}/tests/gdn_chunk_zero_decay_test.cu)
target_link_libraries(p100e-gdn-chunk-zero-decay-test p100e_core)
