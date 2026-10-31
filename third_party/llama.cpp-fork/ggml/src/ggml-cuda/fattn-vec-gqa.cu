// Opt-in routing for the GQA-grouped vector FlashAttention kernel (fattn-vec-gqa.cuh).
//
// GGML_P100_FA_GQA=1       quantized-KV (q4_0/q4_0 or q8_0/q8_0) D=256 FlashAttention with 1-4 query tokens on
//                          Pascal (cc 600-699) uses the GQA kernel with the stock q8_1 math, instead of the stock
//                          vector kernel (1-2 tokens) or the tile kernel with a K/V -> f16 conversion (3-4 tokens).
// GGML_P100_FA_GQA_H2=1    with GGML_P100_FA_GQA=1 on cc 600 (P100): half2 math instead of the emulated dp4a.
// GGML_P100_FA_GQA_SM86=1  with GGML_P100_FA_GQA=1: also Turing and newer for 1 query token (stock: vector kernel).
//
// With GGML_P100_FA_GQA unset ggml_cuda_fattn_vec_gqa_route returns false and the stock kernel selection is unchanged.

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-vec-gqa.cuh"

static bool ggml_cuda_fattn_vec_gqa_env(const char * name) {
    const char * val = getenv(name);
    return val && atoi(val) != 0;
}

bool ggml_cuda_fattn_vec_gqa_route(const int cc, const ggml_tensor * dst, const bool gqa_opt_applies) {
    static const bool enabled = ggml_cuda_fattn_vec_gqa_env("GGML_P100_FA_GQA");
    if (!enabled) {
        return false;
    }

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(cc, dst, gqa_opt_applies);
    return false;
#else
    static const bool sm86 = ggml_cuda_fattn_vec_gqa_env("GGML_P100_FA_GQA_SM86");

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * sinks = dst->src[4];

    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    // gqa_opt_applies: mask present, no ALiBi, K/V length a multiple of FATTN_KQ_STRIDE, 16 byte aligned strides.
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !gqa_opt_applies || sinks || logit_softcap != 0.0f) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != V->type || (K->type != GGML_TYPE_Q4_0 && K->type != GGML_TYPE_Q8_0)) {
        return false;
    }
    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256) {
        return false;
    }
    const int64_t gqa_ratio = Q->ne[2] / K->ne[2];
    if (gqa_ratio != 4 && gqa_ratio != 6 && gqa_ratio != 8) {
        return false;
    }
    // 3 tokens run as ncols1 == 4; with several sequences launch_fattn's KV_max pre-pass would read a 4th mask row:
    if (Q->ne[1] == 3 && Q->ne[3] > 1) {
        return false;
    }

    if (cc >= GGML_CUDA_CC_PASCAL && cc < GGML_CUDA_CC_VOLTA) {
        return Q->ne[1] <= 4;
    }
    return sm86 && cc >= GGML_CUDA_CC_TURING && Q->ne[1] == 1;
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

// ncols2 must divide gqa_ratio and ncols1*ncols2 <= 8. 3-4 tokens (speculative decoding verify) run as (4, 2),
// 3 tokens with one padded token.
template <ggml_type type_KV, bool h2>
static void ggml_cuda_flash_attn_ext_vec_gqa_switch_ncols(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];

    const int64_t gqa_ratio = Q->ne[2] / K->ne[2];
    const int64_t ne1       = Q->ne[1];

    switch (gqa_ratio) {
        case 4:
            if (ne1 == 1) {
                ggml_cuda_flash_attn_ext_vec_gqa_case<256, 1, 4, type_KV, h2>(ctx, dst);
                return;
            }
            if (ne1 == 2) {
                ggml_cuda_flash_attn_ext_vec_gqa_case<256, 2, 4, type_KV, h2>(ctx, dst);
                return;
            }
            break;
        case 6:
            if (ne1 == 1) {
                ggml_cuda_flash_attn_ext_vec_gqa_case<256, 1, 6, type_KV, h2>(ctx, dst);
                return;
            }
            if (ne1 == 2) {
                ggml_cuda_flash_attn_ext_vec_gqa_case<256, 2, 3, type_KV, h2>(ctx, dst);
                return;
            }
            break;
        case 8:
            if (ne1 == 1) {
                ggml_cuda_flash_attn_ext_vec_gqa_case<256, 1, 8, type_KV, h2>(ctx, dst);
                return;
            }
            if (ne1 == 2) {
                ggml_cuda_flash_attn_ext_vec_gqa_case<256, 2, 4, type_KV, h2>(ctx, dst);
                return;
            }
            break;
        default:
            GGML_ABORT("fatal error");
    }

    ggml_cuda_flash_attn_ext_vec_gqa_case<256, 4, 2, type_KV, h2>(ctx, dst);
}

void ggml_cuda_flash_attn_ext_vec_gqa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];

    static const bool h2_enabled = ggml_cuda_fattn_vec_gqa_env("GGML_P100_FA_GQA_H2");
    const int  cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool h2 = h2_enabled && cc == GGML_CUDA_CC_PASCAL;

    switch (K->type) {
        case GGML_TYPE_Q4_0:
            if (h2) {
                ggml_cuda_flash_attn_ext_vec_gqa_switch_ncols<GGML_TYPE_Q4_0, true>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_vec_gqa_switch_ncols<GGML_TYPE_Q4_0, false>(ctx, dst);
            }
            break;
        case GGML_TYPE_Q8_0:
            if (h2) {
                ggml_cuda_flash_attn_ext_vec_gqa_switch_ncols<GGML_TYPE_Q8_0, true>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_vec_gqa_switch_ncols<GGML_TYPE_Q8_0, false>(ctx, dst);
            }
            break;
        default:
            GGML_ABORT("fatal error");
    }
}
