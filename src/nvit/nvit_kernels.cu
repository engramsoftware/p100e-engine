// nvit CUDA kernels, see nvit_kernels.cuh.
#include "nvit/nvit_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace p100e::nvit {

namespace {

constexpr int XEXP = 3;   // activation prescale: max|v| -> [8, 16) (fast::gemm's convention)

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("nvit kernel ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
__device__ __forceinline__ float amax4(float m, const float4 v) {
    return fmaxf(m, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
}
__device__ __forceinline__ int prescale_exp(float m) { return m > 0.0f && isfinite(m) ? ilogbf(m) - XEXP : 0; }

// patch index (merge order) -> patch row / column
__device__ __forceinline__ void patch_rc(int p, int pw, int & py, int & px) {
    const int g = p >> 2, w = p & 3, gw = pw >> 1;
    const int gy = g / gw, gx = g - gy * gw;
    py = 2 * gy + (w >> 1);
    px = 2 * gx + (w & 1);
}

// ---------------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) im2col_kernel(const float * __restrict__ img, int nx, int ny, int p0,
                                                     float * __restrict__ A) {
    const int i = blockIdx.x;
    int py, px;
    patch_rc(p0 + i, nx / 16, py, px);
    const int kh = threadIdx.x >> 4, kw = threadIdx.x & 15;
    const size_t plane = (size_t) nx * ny;
    const float * src = img + (size_t) (py * 16 + kh) * nx + px * 16 + kw;
    float * dst = A + (size_t) i * PATCH_K + threadIdx.x;
#pragma unroll
    for (int ic = 0; ic < 3; ++ic) dst[ic * 256] = src[ic * plane];
}

__global__ void __launch_bounds__(288) bias_pos_kernel(float * __restrict__ x, const float * __restrict__ bias,
                                                       const float * __restrict__ pos, int pw, int p0, float sf0,
                                                       float sf1) {
    const int i = blockIdx.x;
    int py, px;
    patch_rc(p0 + i, pw, py, px);
    // ggml-cuda upscale_f32_bilinear, pixel_offset 0 (align corners)
    const float y_src_f = ((float) py + 0.0f) / sf1 - 0.0f;
    int y0 = (int) floorf(y_src_f), y1 = y0 + 1;
    y0 = max(0, min(y0, POS_SIDE - 1));
    y1 = max(0, min(y1, POS_SIDE - 1));
    float dy = y_src_f - (float) y0;
    dy = max(0.0f, min(dy, 1.0f));
    const float x_src_f = ((float) px + 0.0f) / sf0 - 0.0f;
    int x0 = (int) floorf(x_src_f), x1 = x0 + 1;
    x0 = max(0, min(x0, POS_SIDE - 1));
    x1 = max(0, min(x1, POS_SIDE - 1));
    float dx = x_src_f - (float) x0;
    dx = max(0.0f, min(dx, 1.0f));
    const int c = threadIdx.x * 4;
    const float4 va = *(const float4 *) (pos + (size_t) (y0 * POS_SIDE + x0) * E + c);
    const float4 vb = *(const float4 *) (pos + (size_t) (y0 * POS_SIDE + x1) * E + c);
    const float4 vc = *(const float4 *) (pos + (size_t) (y1 * POS_SIDE + x0) * E + c);
    const float4 vd = *(const float4 *) (pos + (size_t) (y1 * POS_SIDE + x1) * E + c);
    const float4 bb = *(const float4 *) (bias + c);
    float4 * xp = (float4 *) (x + (size_t) i * E + c);
    float4 v = *xp;
    auto lerp = [&](float a, float b, float cc, float d) {
        return a * (1.0f - dx) * (1.0f - dy) + b * dx * (1.0f - dy) + cc * (1.0f - dx) * dy + d * dx * dy;
    };
    v.x = (v.x + bb.x) + lerp(va.x, vb.x, vc.x, vd.x);
    v.y = (v.y + bb.y) + lerp(va.y, vb.y, vc.y, vd.y);
    v.z = (v.z + bb.z) + lerp(va.z, vb.z, vc.z, vd.z);
    v.w = (v.w + bb.w) + lerp(va.w, vb.w, vc.w, vd.w);
    *xp = v;
}

// ---------------------------------------------------------------------------------------------------------------------
// ROWS rows per block, 96 threads (3 warps) per row, 3 float4 per thread (E = 1152)
template <int ROWS>
__global__ void __launch_bounds__(96 * ROWS) add_ln_prescale_kernel(float * __restrict__ x, const float * __restrict__ add,
                                                                   const float * __restrict__ badd,
                                                                   const float * __restrict__ w, const float * __restrict__ b,
                                                                   float eps, half * __restrict__ x16, int64_t ldx,
                                                                   float * __restrict__ cs, float * __restrict__ y32) {
    constexpr int TPR = 96, NV = 3, NW = 3 * ROWS;
    const int rr = threadIdx.x / TPR, lt = threadIdx.x - rr * TPR;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int64_t row = (int64_t) blockIdx.x * ROWS + rr;
    float4 * xr = (float4 *) (x + row * E);
    float4 v[NV];
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        const int i = lt + k * TPR;
        float4 a = xr[i];
        if (add) {
            float4 p = ((const float4 *) (add + row * E))[i];
            if (badd) {
                const float4 bb = ((const float4 *) badd)[i];
                p.x += bb.x; p.y += bb.y; p.z += bb.z; p.w += bb.w;
            }
            a.x += p.x; a.y += p.y; a.z += p.z; a.w += p.w;
            xr[i] = a;
        }
        v[k] = a;
        sum += (a.x + a.y) + (a.z + a.w);
    }
    if (!x16 && !y32) return;
    __shared__ float sh[NW];
    __shared__ float stat[ROWS];
    sum = warp_sum(sum);
    if (lane == 0) sh[wid] = sum;
    __syncthreads();
    if (lt == 0) stat[rr] = (sh[rr * 3] + sh[rr * 3 + 1]) + sh[rr * 3 + 2];
    __syncthreads();
    const float mean = stat[rr] / E;
    float sq = 0.0f;
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        const float a = v[k].x - mean, bq = v[k].y - mean, c = v[k].z - mean, d = v[k].w - mean;
        sq += (a * a + bq * bq) + (c * c + d * d);
    }
    sq = warp_sum(sq);
    __syncthreads();
    if (lane == 0) sh[wid] = sq;
    __syncthreads();
    if (lt == 0) stat[rr] = (sh[rr * 3] + sh[rr * 3 + 1]) + sh[rr * 3 + 2];
    __syncthreads();
    const float inv = rsqrtf(stat[rr] / E + eps);
    float mx = 0.0f;
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        const int i = lt + k * TPR;
        const float4 ww = ((const float4 *) w)[i], bb = ((const float4 *) b)[i];
        float4 o;
        o.x = (v[k].x - mean) * inv * ww.x + bb.x;
        o.y = (v[k].y - mean) * inv * ww.y + bb.y;
        o.z = (v[k].z - mean) * inv * ww.z + bb.z;
        o.w = (v[k].w - mean) * inv * ww.w + bb.w;
        v[k] = o;
        mx = amax4(mx, o);
        if (y32) ((float4 *) (y32 + row * E))[i] = o;
    }
    if (!x16) return;
    mx = warp_max(mx);
    __syncthreads();
    if (lane == 0) sh[wid] = mx;
    __syncthreads();
    mx = 0.0f;
#pragma unroll
    for (int k = 0; k < NW; ++k) mx = fmaxf(mx, sh[k]);
    const int e = prescale_exp(mx);
    const float s = ldexpf(1.0f, -e);
    half2 * y = (half2 *) (x16 + (int64_t) blockIdx.x * ldx + rr * E);
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        const int i = lt + k * TPR;
        y[2 * i + 0] = __floats2half2_rn(v[k].x * s, v[k].y * s);
        y[2 * i + 1] = __floats2half2_rn(v[k].z * s, v[k].w * s);
    }
    if (threadIdx.x == 0) cs[blockIdx.x] = ldexpf(1.0f, e);
}

// ---------------------------------------------------------------------------------------------------------------------
// one block per (64-patch tile, head): rotate in registers, stage f16 in shared memory, write the tiled layouts with
// coalesced u32 stores. Work item (pp, j): patch pp of the tile, rotation pair j (dims j and j + 36).
__global__ void __launch_bounds__(256) qkv_rope_kernel(const float * __restrict__ y, const float * __restrict__ bias,
                                                       int T, int p0, int pw, float theta_scale, float qscale,
                                                       uint32_t * __restrict__ qt, int64_t q_hs, int qrow0,
                                                       uint32_t * __restrict__ kt, uint32_t * __restrict__ vt,
                                                       int64_t kv_hs) {
    constexpr int LDQ = TILE + 2;                 // [dim][row] with an odd word stride (conflict-free column writes)
    __shared__ half sq[HD][LDQ], sk[HD][LDQ];
    __shared__ half sv[TILE][HD];
    const int bt = blockIdx.x, h = blockIdx.y;
    for (int idx = threadIdx.x; idx < TILE * 36; idx += 256) {
        const int pp = idx / 36, j = idx - pp * 36;
        const int t = bt * TILE + pp;
        float q0 = 0, q1 = 0, k0 = 0, k1 = 0, v0 = 0, v1 = 0;
        if (t < T) {
            int py, px;
            patch_rc(p0 + t, pw, py, px);
            // ggml rope_vision: sector j < 18 -> position stream 0 (patch row), else stream 1 (patch column)
            const int pos = j < 18 ? py : px;
            const int pe = j < 18 ? j : j - 18;
            const float theta = pos * powf(theta_scale, pe);
            const float cth = cosf(theta) * 1.0f, sth = sinf(theta) * 1.0f;
            const float * yr = y + (size_t) t * (3 * E);
            const int d0 = h * HD + j, d1 = d0 + 36;
            const float a0 = yr[d0] + bias[d0], a1 = yr[d1] + bias[d1];
            const float b0 = yr[E + d0] + bias[E + d0], b1 = yr[E + d1] + bias[E + d1];
            v0 = yr[2 * E + d0] + bias[2 * E + d0];
            v1 = yr[2 * E + d1] + bias[2 * E + d1];
            q0 = (a0 * cth - a1 * sth) * qscale;
            q1 = (a0 * sth + a1 * cth) * qscale;
            k0 = b0 * cth - b1 * sth;
            k1 = b0 * sth + b1 * cth;
        }
        sq[j][pp] = __float2half_rn(q0); sq[j + 36][pp] = __float2half_rn(q1);
        sk[j][pp] = __float2half_rn(k0); sk[j + 36][pp] = __float2half_rn(k1);
        sv[pp][j] = __float2half_rn(v0); sv[pp][j + 36] = __float2half_rn(v1);
    }
    __syncthreads();
    uint32_t * qd = qt + h * q_hs + (int64_t) (qrow0 / TILE + bt) * TILE_U32;
    uint32_t * kd = kt + h * kv_hs + (int64_t) (p0 / TILE + bt) * TILE_U32;
    uint32_t * vd = vt + h * kv_hs + (int64_t) (p0 / TILE + bt) * TILE_U32;
    for (int idx = threadIdx.x; idx < TILE_U32; idx += 256) {
        const int d2 = idx >> 6, r = idx & 63;
        const half2 a = __halves2half2(sq[2 * d2][r], sq[2 * d2 + 1][r]);
        const half2 b = __halves2half2(sk[2 * d2][r], sk[2 * d2 + 1][r]);
        qd[idx] = *(const uint32_t *) &a;
        kd[idx] = *(const uint32_t *) &b;
        const int k2 = idx / HD, d = idx - k2 * HD;
        const half2 c = __halves2half2(sv[2 * k2][d], sv[2 * k2 + 1][d]);
        vd[idx] = *(const uint32_t *) &c;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ float gelu_tanh(float x) {   // ggml_cuda_op_gelu_single
    const float GELU_COEF_A = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

template <int NV>
__global__ void __launch_bounds__(256) bias_gelu_prescale_kernel(const float * __restrict__ y, int64_t ldy,
                                                                 const float * __restrict__ bias, int K, int Kpad,
                                                                 half * __restrict__ x16, int64_t ldx,
                                                                 float * __restrict__ cs) {
    const int K4 = K / 4;
    const float4 * yr = (const float4 *) (y + (int64_t) blockIdx.x * ldy);
    float4 v[NV];
    float mx = 0.0f;
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        const int i = threadIdx.x + 256 * k;
        if (i < K4) {
            const float4 a = yr[i], bb = ((const float4 *) bias)[i];
            v[k] = make_float4(gelu_tanh(a.x + bb.x), gelu_tanh(a.y + bb.y), gelu_tanh(a.z + bb.z), gelu_tanh(a.w + bb.w));
            mx = amax4(mx, v[k]);
        }
    }
    __shared__ float sh[8];
    mx = warp_max(mx);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = mx;
    __syncthreads();
    mx = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) mx = fmaxf(mx, sh[k]);
    const int e = prescale_exp(mx);
    const float s = ldexpf(1.0f, -e);
    half2 * o = (half2 *) (x16 + (int64_t) blockIdx.x * ldx);
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        const int i = threadIdx.x + 256 * k;
        if (i < K4) {
            o[2 * i + 0] = __floats2half2_rn(v[k].x * s, v[k].y * s);
            o[2 * i + 1] = __floats2half2_rn(v[k].z * s, v[k].w * s);
        }
    }
    for (int i = K / 2 + threadIdx.x; i < Kpad / 2; i += 256) o[i] = __floats2half2_rn(0.0f, 0.0f);
    if (threadIdx.x == 0) cs[blockIdx.x] = ldexpf(1.0f, e);
}

__global__ void add_bias_kernel(float4 * __restrict__ y, const float4 * __restrict__ bias, int64_t n4, int c4) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (int64_t) gridDim.x * blockDim.x) {
        float4 a = y[i];
        const float4 b = bias[i % c4];
        a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
        y[i] = a;
    }
}

__global__ void unprescale_kernel(const half * __restrict__ x16, int64_t ldx, const float * __restrict__ cs, int K,
                                  float * __restrict__ y) {
    const int t = blockIdx.x;
    for (int k = threadIdx.x; k < K; k += blockDim.x) y[(int64_t) t * K + k] = __half2float(x16[t * ldx + k]) * cs[t];
}

// ---------------------------------------------------------------------------------------------------------------------
// attention, D = 72, GEMM-style 64 x 64 tiles (included into nvit_kernels.cu).
// Block = 128 threads = one 64-query tile of one head. Thread t: lo = t & 7, hi = t >> 3.
//   S phase:  queries hi*4 .. +3, keys lo*4 .. +3 and 32 + lo*4 .. +3 of the key tile (4 x 8 logits), per dim pair d2:
//             one LDS.128 of Q (4 queries), two LDS.128 of K (8 keys), 32 HFMA2. Two f16 chains of 18 pairs, folded
//             into fp32. Row max over the 8 threads of a query group by shuffles (lanes differ in lo only).
//   P:        p = 2^(s - m) rounded to f16, stored as key pairs [k2][q] over the K tile's smem (XOR-swizzled by k2 so
//             the 16-byte stores of the 8 lanes of a quarter warp hit different banks).
//   PV phase: dims lo + 8i (i < 9) x the same 4 queries: per key pair one LDS.128 of P (4 queries), nine LDS.32 of V,
//             36 HFMA2 (lanes = even / odd key), folded into the fp32 output with the online-softmax factor.
// The thread owns the same 4 queries in both phases, so its row max, rescale factor and partial row sum stay in
// registers; the partial sums of the 8 lanes are added at the end.
__device__ __forceinline__ half2 u2h(uint32_t u) { return *reinterpret_cast<const half2 *>(&u); }
__device__ __forceinline__ uint32_t h2u(half2 h) { return *reinterpret_cast<const uint32_t *>(&h); }
__device__ __forceinline__ int pswz(int k2) { return ((k2 >> 1) & 7) << 2; }
// 2^x via MUFU.EX2 without the denormal fix-ups of exp2f (rel. error ~2^-22; ex2(-inf) = 0): the probabilities are
// rounded to f16 right after
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

// phase 0: whole key range, normalised output; 1: first part (state m, l -> st, unnormalised O -> O); 2: continue
// from that state over another key range and normalise. Key tiles kt0 .. kt1-1.
__global__ void __launch_bounds__(128, 2) attn72_kernel(const uint32_t * __restrict__ Qt, int64_t q_hs, int qt0, int nq,
                                                        const uint32_t * __restrict__ Kt, const uint32_t * __restrict__ Vt,
                                                        int64_t kv_hs, int N, float * __restrict__ O, int64_t ldo,
                                                        int kt0, int kt1, int phase, float * __restrict__ st) {
    __shared__ __align__(16) uint32_t Qs[TILE_U32];
    __shared__ __align__(16) uint32_t KP[TILE_U32];   // K tile [d2][key], then P [k2][query] (swizzled)
    __shared__ __align__(16) uint32_t Vs[TILE_U32];   // [k2][d]
    const int t = threadIdx.x, h = blockIdx.y, qtile = blockIdx.x;
    const int lo = t & 7, hi = t >> 3;
    {
        const uint4 * src = (const uint4 *) (Qt + h * q_hs + (int64_t) (qt0 + qtile) * TILE_U32);
        uint4 * dst = (uint4 *) Qs;
        for (int i = t; i < TILE_U32 / 4; i += 128) dst[i] = src[i];
    }
    float o[9][4];
#pragma unroll
    for (int i = 0; i < 9; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) o[i][j] = 0.0f;
    float m[4], lp[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) { m[j] = __int_as_float(0xff800000); lp[j] = 0.0f; }
    if (phase == 2) {   // resume: running max / sum and the unnormalised output of the first part
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qtile * TILE + hi * 4 + j;
            if (q < nq) {
                m[j] = st[((int64_t) q * NH + h) * 2 + 0];
                lp[j] = lo == 0 ? st[((int64_t) q * NH + h) * 2 + 1] : 0.0f;
                const float * src = O + (int64_t) q * ldo + h * HD + lo;
#pragma unroll
                for (int i = 0; i < 9; ++i) o[i][j] = src[8 * i];
            }
        }
    }
    const int ntk = (N + TILE - 1) / TILE;
    const uint32_t * Kh = Kt + h * kv_hs;
    const uint32_t * Vh = Vt + h * kv_hs;

    // K / V tiles go through registers one tile ahead (loads issued before the S phase, stored after the PV phase)
    constexpr int NLD = (TILE_U32 / 4 + 127) / 128;   // 5 uint4 per thread and operand (576 per tile)
    uint4 pk[NLD], pv[NLD];
    auto gload = [&](int kt) {
        const uint4 * ks = (const uint4 *) (Kh + (int64_t) kt * TILE_U32);
        const uint4 * vs = (const uint4 *) (Vh + (int64_t) kt * TILE_U32);
#pragma unroll
        for (int k = 0; k < NLD; ++k) {
            const int i = t + 128 * k;
            if (i < TILE_U32 / 4) { pk[k] = ks[i]; pv[k] = vs[i]; }
        }
    };
    gload(kt0);
    for (int kt = kt0; kt < kt1; ++kt) {
        __syncthreads();
        {
            uint4 * kd = (uint4 *) KP;
            uint4 * vd = (uint4 *) Vs;
#pragma unroll
            for (int k = 0; k < NLD; ++k) {
                const int i = t + 128 * k;
                if (i < TILE_U32 / 4) { kd[i] = pk[k]; vd[i] = pv[k]; }
            }
        }
        __syncthreads();
        if (kt + 1 < kt1) gload(kt + 1);

        // ---- S = Q K^T (log2 units) ----
        float sf[4][8];
        half2 acc[4][8];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[i][j] = __floats2half2_rn(0.0f, 0.0f);
#pragma unroll
        for (int d2 = 0; d2 < 36; ++d2) {
            const uint4 qv = *(const uint4 *) &Qs[d2 * TILE + hi * 4];
            const uint4 ka = *(const uint4 *) &KP[d2 * TILE + lo * 4];
            const uint4 kb = *(const uint4 *) &KP[d2 * TILE + 32 + lo * 4];
            const half2 qh[4] = {u2h(qv.x), u2h(qv.y), u2h(qv.z), u2h(qv.w)};
            const half2 kh[8] = {u2h(ka.x), u2h(ka.y), u2h(ka.z), u2h(ka.w), u2h(kb.x), u2h(kb.y), u2h(kb.z), u2h(kb.w)};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = __hfma2(qh[i], kh[j], acc[i][j]);
            if (d2 == 17 || d2 == 35) {
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float f = __low2float(acc[i][j]) + __high2float(acc[i][j]);
                        sf[i][j] = d2 == 17 ? f : sf[i][j] + f;
                        acc[i][j] = __floats2half2_rn(0.0f, 0.0f);
                    }
            }
        }
        if (kt == ntk - 1 && (N & (TILE - 1))) {
            const int kb0 = kt * TILE;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int key = kb0 + (j < 4 ? lo * 4 + j : 32 + lo * 4 + j - 4);
                if (key >= N) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) sf[i][j] = __int_as_float(0xff800000);
                }
            }
        }

        // ---- online softmax (rows of this thread's 4 queries; the 8 lanes of a group share them) ----
        float alpha[4];
        half2 p2[4][4];   // [query][key pair]: pairs (0,1), (2,3) = keys lo*4 + 0..3; (4,5), (6,7) = 32 + lo*4 + 0..3
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float mt = sf[i][0];
#pragma unroll
            for (int j = 1; j < 8; ++j) mt = fmaxf(mt, sf[i][j]);
            mt = fmaxf(mt, __shfl_xor_sync(0xffffffff, mt, 1));
            mt = fmaxf(mt, __shfl_xor_sync(0xffffffff, mt, 2));
            mt = fmaxf(mt, __shfl_xor_sync(0xffffffff, mt, 4));
            const float mn = fmaxf(m[i], mt);
            alpha[i] = ex2(m[i] - mn);
            m[i] = mn;
            float ls = 0.0f;
#pragma unroll
            for (int pr = 0; pr < 4; ++pr) {
                const half2 pp = __floats2half2_rn(ex2(sf[i][2 * pr] - mn), ex2(sf[i][2 * pr + 1] - mn));
                p2[i][pr] = pp;
                ls += __low2float(pp) + __high2float(pp);
            }
            lp[i] = lp[i] * alpha[i] + ls;
        }
        __syncthreads();   // every thread is done reading the K tile
#pragma unroll
        for (int pr = 0; pr < 4; ++pr) {
            const int k2 = (pr < 2 ? 0 : 16) + lo * 2 + (pr & 1);
            *(uint4 *) &KP[k2 * TILE + ((hi * 4) ^ pswz(k2))] =
                make_uint4(h2u(p2[0][pr]), h2u(p2[1][pr]), h2u(p2[2][pr]), h2u(p2[3][pr]));
        }
        __syncthreads();

        // ---- O += P V ----
        half2 a2[9][4];
#pragma unroll
        for (int i = 0; i < 9; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) a2[i][j] = __floats2half2_rn(0.0f, 0.0f);
#pragma unroll 8
        for (int k2 = 0; k2 < 32; ++k2) {
            const uint4 pv = *(const uint4 *) &KP[k2 * TILE + ((hi * 4) ^ pswz(k2))];
            const half2 pq[4] = {u2h(pv.x), u2h(pv.y), u2h(pv.z), u2h(pv.w)};
            half2 vv[9];
#pragma unroll
            for (int i = 0; i < 9; ++i) vv[i] = u2h(Vs[k2 * HD + lo + 8 * i]);
#pragma unroll
            for (int i = 0; i < 9; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) a2[i][j] = __hfma2(vv[i], pq[j], a2[i][j]);
        }
#pragma unroll
        for (int i = 0; i < 9; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) o[i][j] = fmaf(o[i][j], alpha[j], __low2float(a2[i][j]) + __high2float(a2[i][j]));
    }

    // ---- normalise, write (phase 1: state + unnormalised output) ----
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float l = lp[j];
        l += __shfl_xor_sync(0xffffffff, l, 1);
        l += __shfl_xor_sync(0xffffffff, l, 2);
        l += __shfl_xor_sync(0xffffffff, l, 4);
        const int q = qtile * TILE + hi * 4 + j;
        if (q < nq) {
            if (phase == 1 && lo == 0) {
                st[((int64_t) q * NH + h) * 2 + 0] = m[j];
                st[((int64_t) q * NH + h) * 2 + 1] = l;
            }
            const float il = phase == 1 ? 1.0f : 1.0f / l;
            float * dst = O + (int64_t) q * ldo + h * HD + lo;
#pragma unroll
            for (int i = 0; i < 9; ++i) dst[8 * i] = o[i][j] * il;
        }
    }
}

// ---- fp32 oracle kernels ----
__global__ void __launch_bounds__(NH * 36) qkv_rope_f32_kernel(const float * __restrict__ y, const float * __restrict__ bias,
                                                               int p0, int pw, float theta_scale, float * __restrict__ q,
                                                               float * __restrict__ k, float * __restrict__ v, int64_t N) {
    const int t = blockIdx.x;
    const int h = threadIdx.x / 36, j = threadIdx.x - h * 36;
    int py, px;
    patch_rc(p0 + t, pw, py, px);
    const int pos = j < 18 ? py : px;
    const int pe = j < 18 ? j : j - 18;
    const float theta = pos * powf(theta_scale, pe);
    const float cth = cosf(theta), sth = sinf(theta);
    const float * yr = y + (size_t) t * (3 * E);
    const int d0 = h * HD + j, d1 = d0 + 36;
    const float a0 = yr[d0] + bias[d0], a1 = yr[d1] + bias[d1];
    const float b0 = yr[E + d0] + bias[E + d0], b1 = yr[E + d1] + bias[E + d1];
    const size_t o = ((size_t) h * N + p0 + t) * HD;
    q[o + j] = a0 * cth - a1 * sth;
    q[o + j + 36] = a0 * sth + a1 * cth;
    k[o + j] = b0 * cth - b1 * sth;
    k[o + j + 36] = b0 * sth + b1 * cth;
    v[o + j] = yr[2 * E + d0] + bias[2 * E + d0];
    v[o + j + 36] = yr[2 * E + d1] + bias[2 * E + d1];
}

__global__ void __launch_bounds__(128) attn_f32_kernel(const float * __restrict__ Q, const float * __restrict__ K,
                                                       const float * __restrict__ V, int N, float * __restrict__ O) {
    const int lane = threadIdx.x & 31;
    const int qi = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int h = blockIdx.y;
    if (qi >= N) return;
    const float scale = 1.0f / sqrtf((float) HD);
    float q[HD], o[HD];
    const float * qp = Q + ((size_t) h * N + qi) * HD;
#pragma unroll
    for (int d = 0; d < HD; ++d) { q[d] = qp[d] * scale; o[d] = 0.0f; }
    float m = -3.0e38f, l = 0.0f;
    for (int key = lane; key < N; key += 32) {
        const float * kp = K + ((size_t) h * N + key) * HD;
        float s = 0.0f;
#pragma unroll
        for (int d = 0; d < HD; ++d) s = fmaf(q[d], kp[d], s);
        const float mn = fmaxf(m, s);
        const float a = expf(m - mn), p = expf(s - mn);
        l = l * a + p;
        const float * vp = V + ((size_t) h * N + key) * HD;
#pragma unroll
        for (int d = 0; d < HD; ++d) o[d] = fmaf(o[d], a, p * vp[d]);
        m = mn;
    }
    float M = m;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffff, M, off));
    const float sc = expf(m - M);
    l *= sc;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) l += __shfl_xor_sync(0xffffffff, l, off);
#pragma unroll
    for (int d = 0; d < HD; ++d) {
        float v = o[d] * sc;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
        o[d] = v;
    }
    if (lane == 0) {
        float * dst = O + (size_t) qi * E + h * HD;
#pragma unroll
        for (int d = 0; d < HD; ++d) dst[d] = o[d] / l;
    }
}

__global__ void bias_gelu_f32_kernel(float * __restrict__ y, const float * __restrict__ bias, int64_t n, int K) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = gelu_tanh(y[i] + bias[i % K]);
}

} // namespace

// =====================================================================================================================
void im2col_merge(const float * img, int nx, int ny, int p0, int np, float * A, cudaStream_t s) {
    if (np <= 0) return;
    im2col_kernel<<<np, 256, 0, s>>>(img, nx, ny, p0, A);
    ck(cudaGetLastError(), "im2col");
}

void add_bias_pos(float * x, const float * bias, const float * pos, int pw, int ph, int p0, int np, cudaStream_t s) {
    if (np <= 0) return;
    // ggml_cuda_op_upscale with GGML_SCALE_FLAG_ALIGN_CORNERS (src 48 x 48 -> dst pw x ph)
    float sf0 = (float) pw / POS_SIDE, sf1 = (float) ph / POS_SIDE;
    sf0 = pw > 1 && POS_SIDE > 1 ? (float) (pw - 1) / (POS_SIDE - 1) : sf0;
    sf1 = ph > 1 && POS_SIDE > 1 ? (float) (ph - 1) / (POS_SIDE - 1) : sf1;
    bias_pos_kernel<<<np, E / 4, 0, s>>>(x, bias, pos, pw, p0, sf0, sf1);
    ck(cudaGetLastError(), "bias_pos");
}

void add_ln_prescale(int ROWS, float * x, const float * add, const float * badd, const float * w, const float * b,
                     float eps, int rows, half * x16, int64_t ldx, float * cs, float * y32, cudaStream_t s) {
    if (rows <= 0) return;
    if (ROWS == 1) {
        add_ln_prescale_kernel<1><<<rows, 96, 0, s>>>(x, add, badd, w, b, eps, x16, ldx, cs, y32);
    } else if (ROWS == 4) {
        if (rows % 4) throw std::runtime_error("nvit add_ln_prescale: rows % 4 != 0");
        add_ln_prescale_kernel<4><<<rows / 4, 384, 0, s>>>(x, add, badd, w, b, eps, x16, ldx, cs, y32);
    } else {
        throw std::runtime_error("nvit add_ln_prescale: ROWS must be 1 or 4");
    }
    ck(cudaGetLastError(), "add_ln_prescale");
}

void qkv_rope(const float * y, const float * bias, int T, int p0, int pw, float qscale, uint32_t * qt, int64_t q_hs,
              int qrow0, uint32_t * kt, uint32_t * vt, int64_t kv_hs, cudaStream_t s) {
    if (T <= 0) return;
    if (p0 % TILE || qrow0 % TILE) throw std::runtime_error("nvit qkv_rope: p0 / qrow0 must be multiples of 64");
    const float theta_scale = powf(10000.0f, -2.0f / 36);   // rope_vision_cuda: powf(freq_base, -2.0f/n_dims)
    qkv_rope_kernel<<<dim3((T + TILE - 1) / TILE, NH), 256, 0, s>>>(y, bias, T, p0, pw, theta_scale, qscale, qt, q_hs, qrow0,
                                                                   kt, vt, kv_hs);
    ck(cudaGetLastError(), "qkv_rope");
}

void bias_gelu_prescale(const float * y, int64_t ldy, const float * bias, int T, int K, int Kpad, half * x16,
                        int64_t ldx, float * cs, cudaStream_t s) {
    if (T <= 0) return;
    if (K % 4 || Kpad % 2 || ldy % 4 || ldx % 8) throw std::runtime_error("nvit bias_gelu_prescale: bad shape");
    const int K4 = K / 4;
    if (K4 <= 256 * 3) bias_gelu_prescale_kernel<3><<<T, 256, 0, s>>>(y, ldy, bias, K, Kpad, x16, ldx, cs);
    else if (K4 <= 256 * 5) bias_gelu_prescale_kernel<5><<<T, 256, 0, s>>>(y, ldy, bias, K, Kpad, x16, ldx, cs);
    else throw std::runtime_error("nvit bias_gelu_prescale: K too large");
    ck(cudaGetLastError(), "bias_gelu_prescale");
}

void add_bias(float * y, const float * bias, int T, int n, cudaStream_t s) {
    if (T <= 0) return;
    const int64_t n4 = (int64_t) T * n / 4;
    const unsigned nb = (unsigned) std::min<int64_t>(4096, (n4 + 255) / 256);
    add_bias_kernel<<<nb, 256, 0, s>>>((float4 *) y, (const float4 *) bias, n4, n / 4);
    ck(cudaGetLastError(), "add_bias");
}

void attn72(const uint32_t * qt, int64_t q_hs, int qrow0, int nq, const uint32_t * kt, const uint32_t * vt, int64_t kv_hs,
            int N, int nh, float * out, int64_t ldo, cudaStream_t s, int kt0, int kt1, int phase, float * st) {
    if (nq <= 0) return;
    if (qrow0 % TILE) throw std::runtime_error("nvit attn72: qrow0 must be a multiple of 64");
    const int ntk = (N + TILE - 1) / TILE;
    if (kt1 < 0) kt1 = ntk;
    if (kt0 < 0 || kt1 > ntk || kt0 >= kt1 || (phase && !st)) throw std::runtime_error("nvit attn72: bad key range / state");
    const dim3 grid((nq + TILE - 1) / TILE, nh);
    attn72_kernel<<<grid, 128, 0, s>>>(qt, q_hs, qrow0 / TILE, nq, kt, vt, kv_hs, N, out, ldo, kt0, kt1, phase, st);
    ck(cudaGetLastError(), "attn72");
}

void unprescale(const half * x16, int64_t ldx, const float * cs, int T, int K, float * y, cudaStream_t s) {
    if (T <= 0) return;
    unprescale_kernel<<<T, 256, 0, s>>>(x16, ldx, cs, K, y);
    ck(cudaGetLastError(), "unprescale");
}

void qkv_rope_f32(const float * y, const float * bias, int T, int p0, int pw, float * q, float * k, float * v, int64_t N,
                  cudaStream_t s) {
    if (T <= 0) return;
    const float theta_scale = powf(10000.0f, -2.0f / 36);
    qkv_rope_f32_kernel<<<T, NH * 36, 0, s>>>(y, bias, p0, pw, theta_scale, q, k, v, N);
    ck(cudaGetLastError(), "qkv_rope_f32");
}

void attn_f32(const float * q, const float * k, const float * v, int N, float * out, cudaStream_t s) {
    attn_f32_kernel<<<dim3((N + 3) / 4, NH), 128, 0, s>>>(q, k, v, N, out);
    ck(cudaGetLastError(), "attn_f32");
}

void bias_gelu_f32(float * y, const float * bias, int T, int K, cudaStream_t s) {
    const int64_t n = (int64_t) T * K;
    bias_gelu_f32_kernel<<<(unsigned) std::min<int64_t>(8192, (n + 255) / 256), 256, 0, s>>>(y, bias, n, K);
    ck(cudaGetLastError(), "bias_gelu_f32");
}

} // namespace p100e::nvit
