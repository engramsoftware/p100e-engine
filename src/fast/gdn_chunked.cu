// p100e::fast chunked gated delta net, see gdn_chunked.cuh. Host side of the fork's gdn-chunked.cu (our fork
// at 98ab7e5, from Kmic-68/llama.cpp 77e05601d, MIT) on p100e's buffers: the kernels are
// gdn_chunked_kernels.cuh (the fork's gdn-chunked.cuh), instantiated in fp32 and fp64 flavours as in the fork.
#include "fast/gdn_chunked.cuh"

#include <cstdint>
#include <cstdlib>

namespace p100e::fast {

namespace gdn_f32 {
#include "gdn_chunked_kernels.cuh"
}
#undef TG
#undef TA
#undef TB
#undef TC
#undef TF
#define TG double
#define TC double
namespace gdn_f64 {
#include "gdn_chunked_kernels.cuh"
}

namespace {

__device__ __forceinline__ float warp_sum_c(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float silu_c(float v) { return v / (1.0f + expf(-v)); }

// gated RMSNorm over rows of 128, a warp per row, in place (row = t*Hv + h): the arithmetic of ssm.cu's gdn_norm_k
__global__ void gdn_ch_norm_k(float * out, const float * __restrict__ z, const float * __restrict__ nw, int nrows,
                              float eps, int Hv, int ldz) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= nrows) return;
    const int64_t off = (int64_t) row * 128;
    const float4 w4 = reinterpret_cast<const float4 *>(nw)[lane];
    float4 o4 = reinterpret_cast<float4 *>(out + off)[lane];
    const float4 z4 = reinterpret_cast<const float4 *>(z + (int64_t) (row / Hv) * ldz + (row % Hv) * 128)[lane];
    const float ss = warp_sum_c(o4.x * o4.x + o4.y * o4.y + o4.z * o4.z + o4.w * o4.w);
    const float sc = rsqrtf(ss / 128 + eps);
    o4.x *= sc * w4.x * silu_c(z4.x); o4.y *= sc * w4.y * silu_c(z4.y);
    o4.z *= sc * w4.z * silu_c(z4.z); o4.w *= sc * w4.w * silu_c(z4.w);
    reinterpret_cast<float4 *>(out + off)[lane] = o4;
}

// the fork's gdn_ref_f64 on p100e's layout: one thread per state row j (v dim) of head h, state row in fp64
__global__ void gdn_ref_f64_k(const float * __restrict__ y, const float * __restrict__ decay, const float * __restrict__ beta,
                              float * state, float * out, int T, int C, int Hk, int Hv) {
    const int j = threadIdx.x, h = blockIdx.x, hk = h % Hk;
    double s[128];
    float * st = state + ((int64_t) h * 128 + j) * 128;
    for (int i = 0; i < 128; i++) s[i] = st[i];
    for (int t = 0; t < T; t++) {
        const float * qt = y + (int64_t) t * C + hk * 128;
        const float * kt = y + (int64_t) t * C + Hk * 128 + hk * 128;
        const double eg = decay[(int64_t) t * Hv + h];
        const double bt = beta[(int64_t) t * Hv + h];
        const double vj = y[(int64_t) t * C + 2 * Hk * 128 + h * 128 + j];
        double kv = 0.0;
        for (int i = 0; i < 128; i++) kv = fma(s[i], (double) kt[i], kv);
        const double delta = (vj - eg * kv) * bt;
        double o = 0.0;
        for (int i = 0; i < 128; i++) {
            s[i] = fma(eg, s[i], (double) kt[i] * delta);
            o = fma(s[i], (double) qt[i], o);
        }
        out[((int64_t) t * Hv + h) * 128 + j] = (float) o;
    }
    for (int i = 0; i < 128; i++) st[i] = (float) s[i];
}

bool aligned16(const void * p) { return ((uintptr_t) p & 15) == 0; }

} // namespace

size_t gdn_chunked_scratch_bytes(int T, int Hk, int Hv) {
    return gdn_f32::gdn_ch_scratch_floats(Hv, Hk, 1, T) * sizeof(float);
}

bool gdn_scan_chunked(const float * y, const float * decay, const float * beta, const float * z, const float * norm_w,
                      float * state, float * out, int T, int Hk, int Hv, int D, float eps, void * scratch,
                      size_t scratch_bytes, cudaStream_t s, int mode, int ldz) {
    if (D != 128 || Hk <= 0 || Hv % Hk != 0 || T < GCH) return false;
    const int C = 2 * Hk * D + Hv * D;
    if (ldz <= 0) ldz = Hv * D;
    if (ldz % 4 || !aligned16(y) || !aligned16(state) || !aligned16(out) || !aligned16(z) || !aligned16(scratch))
        return false;
    if (scratch_bytes < gdn_chunked_scratch_bytes(T, Hk, Hv)) return false;
#define GDN_CH_RUN(NS) do { \
        NS::gdn_ch_params p = {}; \
        p.q = y; p.k = y + Hk * D; p.v = y + 2 * Hk * D; p.g = decay; p.beta = beta; \
        p.curr_state = state; p.dst = out; p.state = state; \
        p.H = Hv; p.n_tokens = T; p.n_seqs = 1; p.neqk1 = Hk; p.rq3 = 1; p.K = 1; \
        p.sq1 = D; p.sq2 = C; p.sq3 = 0; p.sv1 = D; p.sv2 = C; p.sv3 = 0; \
        p.sb1 = 1; p.sb2 = Hv; p.sb3 = 0; \
        p.scale = 1.0f; p.state_slot_stride = 0; p.s_idx = nullptr; p.s_row = 0; \
        p.keep = false; p.g_is_decay = true; \
        p.beta_pre = nullptr; p.beta_sig = nullptr; p.gather_dst = nullptr; \
        NS::gdn_ch_launch(p, (float *) scratch, s); \
    } while (0)
    if (mode == 2) GDN_CH_RUN(gdn_f64);
    else GDN_CH_RUN(gdn_f32);
#undef GDN_CH_RUN
    const int nrows = T * Hv;
    gdn_ch_norm_k<<<(nrows + 3) / 4, 128, 0, s>>>(out, z, norm_w, nrows, eps, Hv, ldz);
    return cudaGetLastError() == cudaSuccess;
}

void gdn_scan_ref_f64(const float * y, const float * decay, const float * beta, float * state, float * out, int T,
                      int Hk, int Hv, cudaStream_t s) {
    const int C = 2 * Hk * 128 + Hv * 128;
    gdn_ref_f64_k<<<Hv, 128, 0, s>>>(y, decay, beta, state, out, T, C, Hk, Hv);
}

} // namespace p100e::fast
