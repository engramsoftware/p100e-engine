// Fused Q4_0 prefill GEMM for sm_60: see q4mm-sm60.cuh.

#include "q4mm-sm60.cuh"

#include <cstdlib>

// 0 = off (default), 1 = sm_60 only, 2 = any NVIDIA device (tests)
static int q4mm_mode() {
    static const int mode = getenv("GGML_P100_Q4MM") ? atoi(getenv("GGML_P100_Q4MM")) : 0;
    return mode;
}

bool ggml_cuda_q4mm_enabled(const int cc) {
    const int mode = q4mm_mode();
    if (mode == 2) {
        return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_PASCAL;
    }
    return mode == 1 && cc == GGML_CUDA_CC_PASCAL;
}

bool ggml_cuda_q4mm_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                            ggml_tensor * dst, const bool repacked) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ggml_cuda_q4mm_enabled(cc)) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // plain 2D product only: one weight matrix, contiguous activation rows, contiguous dst
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->ne[0] % 32 != 0 || src1->nb[0] != sizeof(float) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src0->nb[0] != ggml_type_size(GGML_TYPE_Q4_0) || src0->nb[1] % 2 != 0) {
        return false;
    }
    if (repacked && src0->nb[1] % 8 != 0) {
        return false;
    }
    // float4 activation loads
    const int64_t s11 = src1->nb[1] / sizeof(float);
    if (src1->nb[1] % (4*sizeof(float)) != 0 || ((uintptr_t) src1->data) % 16 != 0) {
        return false;
    }
    if (src0->ne[1] > INT_MAX || src1->ne[1] > INT_MAX) {
        return false;
    }

    const int ne00 = (int) src0->ne[0];
    const int ne01 = (int) src0->ne[1];
    const int ne11 = (int) src1->ne[1];
    cudaStream_t stream = ctx.stream();
    if (repacked) {
        q4mm_sm60_launch<true >((const char *) src0->data, (const float *) src1->data, (float *) dst->data,
                                ne00, ne01, ne11, src0->nb[1], s11, dst->ne[0], stream);
    } else {
        q4mm_sm60_launch<false>((const char *) src0->data, (const float *) src1->data, (float *) dst->data,
                                ne00, ne01, ne11, src0->nb[1], s11, dst->ne[0], stream);
    }
    CUDA_CHECK(cudaGetLastError());
    static bool logged[GGML_CUDA_MAX_DEVICES] = {};
    if (!logged[ctx.device]) {
        logged[ctx.device] = true;
        GGML_LOG_INFO("q4mm: fused Q4_0 prefill GEMM active on device %d (first: %s, %lld x %lld x %lld, %s layout)\n",
            ctx.device, dst->name, (long long) ne01, (long long) ne00, (long long) ne11, repacked ? "q4r" : "standard");
    }
    return true;
}
