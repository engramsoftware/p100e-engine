#pragma once

// Test-only switch shared by the Kmic-68 prefill ports (pf-xchg, pf-gemm, pf-attn). One definition for all three
// (REVIEW-ALL B1): header-only and self-contained, so tests/test-pf-xchg-kernels.cu can still compile pf-xchg.cuh
// on its own.
//
//   GGML_PF_TEST_ANY_ARCH  TEST ONLY, default unset (= 0, off). Never set it on a serving machine.
//     >= 1  the Pascal-gated pf paths run on any NVIDIA device with fast fp16, so they can be checked against the
//           CPU on a desktop card: fold GEMM, cuBLAS ALGO6, q4r repack and P100 matmul routing past MMQ (pf-gemm);
//           the FlashAttention dispatcher chooses as for cc 6.0 and the GEMM / fold attention ignores its pre-Volta
//           gate (pf-attn); the exchange runs between virtual devices on one GPU and the sm_6x gate of the f16
//           producer check is lifted (pf-xchg)
//     == 2  as 1, and pf-xchg also skips the producer check (any batched MUL_MAT partial is f16-eligible), so only
//           the probe / verify gate stands between a non-f16-exact partial and the wire: a test of that safety net
//   (q4r-sm60.cu reads the variable directly for its repack gate; any non-zero value turns that on.)

#include <cstdlib>

static inline int ggml_cuda_pf_test_level() {
    static const int v = getenv("GGML_PF_TEST_ANY_ARCH") != nullptr ? atoi(getenv("GGML_PF_TEST_ANY_ARCH")) : 0;
    return v;
}

static inline bool ggml_cuda_pf_test_any_arch() {
    return ggml_cuda_pf_test_level() >= 1;
}

static inline bool ggml_cuda_pf_test_force_eligible() {
    return ggml_cuda_pf_test_level() >= 2;
}
