#pragma once

#include "common.cuh"

// Ported from Kmic-68/llama.cpp 738022bda (MIT), see fattn-gemm.cu for the full commit list.
// Long-prefill attention for pre-Volta NVIDIA as GEMM / fold kernels. OFF by default in this fork:
// GGML_CUDA_FA_GEMM=1 turns it on (cuBLAS GEMM attention), GGML_CUDA_FA_FOLD=1 additionally selects the
// hand-written fold kernels for a q4_0 K/V cache. Taken only for >= 128 query tokens and >= 4096 keys
// (GGML_CUDA_FA_GEMM_MINKV), one sequence per ubatch (mask ne[2] == ne[3] == 1), no ALiBi/softcap/sinks.
bool ggml_cuda_fa_gemm_enabled();
bool ggml_cuda_flash_attn_ext_gemm_supported(const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// enabled && supported && the device is pre-Volta NVIDIA (or GGML_PF_TEST_ANY_ARCH=1).
bool ggml_cuda_flash_attn_ext_gemm_route(int device, const ggml_tensor * dst);

// GGML_PF_TEST_ANY_ARCH=1 (test only, default off): lets the Pascal-gated prefill-attention paths run on a
// newer GPU so they can be checked against the CPU on the desktop GTX 1660: the FlashAttention dispatcher
// picks kernels as it would for cc 6.0, and the GEMM / fold path ignores its pre-Volta gate. The embedded
// sm_60 SASS is still only used on a real cc 6.x device. Never set it on a serving machine.
// ggml_cuda_pf_test_any_arch() lives in pf-test.cuh (shared with pf-xchg and pf-gemm).
#include "pf-test.cuh"
