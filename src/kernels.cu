#include "kernels.cuh"

#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>

namespace p100e::k {

static __device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// block-wide sum for blockDim.x <= 1024, result broadcast to all threads
static __device__ float block_sum(float v) {
    __shared__ float sh[32];
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    v = threadIdx.x < nw ? sh[threadIdx.x] : 0.0f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

static int nthreads_for(int ncols) { return ncols >= 1024 ? 1024 : ((ncols + 31) / 32) * 32; }

__global__ void rmsnorm_k(const float * x, const float * w, float * y, int ncols, float eps) {
    const float * xr = x + (int64_t) blockIdx.x * ncols;
    float * yr = y + (int64_t) blockIdx.x * ncols;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / ncols + eps);
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) yr[i] = xr[i] * sc * (w ? w[i] : 1.0f);
}

void rmsnorm(const float * x, const float * w, float * y, int64_t nrows, int ncols, float eps, cudaStream_t s) {
    rmsnorm_k<<<(unsigned) nrows, nthreads_for(ncols), 0, s>>>(x, w, y, ncols, eps);
}

__global__ void l2norm_k(const float * x, float * y, int ncols, float eps) {
    const float * xr = x + (int64_t) blockIdx.x * ncols;
    float * yr = y + (int64_t) blockIdx.x * ncols;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss);
    const float sc = 1.0f / fmaxf(sqrtf(ss), eps);
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) yr[i] = xr[i] * sc;
}

void l2norm(const float * x, float * y, int64_t nrows, int ncols, float eps, cudaStream_t s) {
    l2norm_k<<<(unsigned) nrows, nthreads_for(ncols), 0, s>>>(x, y, ncols, eps);
}

static __device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }

__global__ void rmsnorm_gated_k(const float * x, const float * w, const float * z, float * y, int ncols, float eps) {
    const int64_t off = (int64_t) blockIdx.x * ncols;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) ss += x[off + i] * x[off + i];
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / ncols + eps);
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) y[off + i] = x[off + i] * sc * w[i] * silu(z[off + i]);
}

void rmsnorm_gated(const float * x, const float * w, const float * z, float * y, int64_t nrows, int ncols, float eps, cudaStream_t s) {
    rmsnorm_gated_k<<<(unsigned) nrows, nthreads_for(ncols), 0, s>>>(x, w, z, y, ncols, eps);
}

__global__ void add_k(float * x, const float * y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
__global__ void silu_mul_k(float * g, const float * u, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = silu(g[i]) * u[i];
}
__global__ void sigmoid_mul_k(float * x, const float * g, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 1.0f / (1.0f + expf(-g[i]));
}

static unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }
void add(float * x, const float * y, int64_t n, cudaStream_t s) { add_k<<<blocks(n), 256, 0, s>>>(x, y, n); }
void silu_mul(float * g, const float * u, int64_t n, cudaStream_t s) { silu_mul_k<<<blocks(n), 256, 0, s>>>(g, u, n); }
void sigmoid_mul(float * x, const float * g, int64_t n, cudaStream_t s) { sigmoid_mul_k<<<blocks(n), 256, 0, s>>>(x, g, n); }

__global__ void split_q_gate_k(const float * qg, float * q, float * gate, int H, int D) {
    const int t = blockIdx.y, h = blockIdx.x;
    const float * src = qg + ((int64_t) t * H + h) * 2 * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        q[((int64_t) t * H + h) * D + i]    = src[i];
        gate[((int64_t) t * H + h) * D + i] = src[D + i];
    }
}

void split_q_gate(const float * qg, float * q, float * gate, int T, int H, int D, cudaStream_t s) {
    split_q_gate_k<<<dim3(H, T), 256, 0, s>>>(qg, q, gate, H, D);
}

__global__ void rope_neox_k(float * x, const int32_t * pos, int H, int D, int n_rot, float freq_base) {
    const int t = blockIdx.y, h = blockIdx.x;
    float * xr = x + ((int64_t) t * H + h) * D;
    const int half = n_rot / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float theta = (float) pos[t] * powf(freq_base, -2.0f * i / n_rot);
        const float c = cosf(theta), sn = sinf(theta);
        const float a = xr[i], b = xr[i + half];
        xr[i]        = a * c - b * sn;
        xr[i + half] = a * sn + b * c;
    }
}

void rope_neox(float * x, const int32_t * pos, int T, int H, int D, int n_rot, float freq_base, cudaStream_t s) {
    rope_neox_k<<<dim3(H, T), 64, 0, s>>>(x, pos, H, D, n_rot, freq_base);
}

__global__ void rope_mrope_k(float * x, const fast::mrope_pos m, int H, int D, int n_rot, float freq_base) {
    const int t = blockIdx.y, h = blockIdx.x;
    float * xr = x + ((int64_t) t * H + h) * D;
    const int half = n_rot / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float theta = (float) fast::mrope_pick(m, t, i) * powf(freq_base, -2.0f * i / n_rot);
        const float c = cosf(theta), sn = sinf(theta);
        const float a = xr[i], b = xr[i + half];
        xr[i]        = a * c - b * sn;
        xr[i + half] = a * sn + b * c;
    }
}

void rope_mrope(float * x, const fast::mrope_pos & m, int T, int H, int D, int n_rot, float freq_base, cudaStream_t s) {
    rope_mrope_k<<<dim3(H, T), 64, 0, s>>>(x, m, H, D, n_rot, freq_base);
}

// one block per (query t, head h); scores kept in dynamic shared memory (n_kv floats)
__global__ void attention_k(const float * q, const float * kc, const float * vc, float * out,
                            int p0, int Hq, int Hkv, int D, float scale) {
    extern __shared__ float sc[];
    const int t = blockIdx.y, h = blockIdx.x, hk = h / (Hq / Hkv);
    const int n_kv = p0 + t + 1;
    const float * qr = q + ((int64_t) t * Hq + h) * D;
    for (int j = 0; j < n_kv; ++j) {
        const float * kr = kc + ((int64_t) j * Hkv + hk) * D;
        float d = 0.0f;
        for (int i = threadIdx.x; i < D; i += blockDim.x) d += qr[i] * kr[i];
        d = block_sum(d);
        if (threadIdx.x == 0) sc[j] = d * scale;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int j = 0; j < n_kv; ++j) mx = fmaxf(mx, sc[j]);
    float den = 0.0f;
    for (int j = 0; j < n_kv; ++j) den += expf(sc[j] - mx);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < n_kv; ++j) acc += expf(sc[j] - mx) * vc[((int64_t) j * Hkv + hk) * D + i];
        out[((int64_t) t * Hq + h) * D + i] = acc / den;
    }
}

// online-softmax attention: one block per (query t, head h), ATT_W warps over the keys (warp w takes keys
// w*U.. w*U+U-1, then strides by ATT_W*U), lane owns the float4s lane + 32*r of the head (D % 4 == 0, D <= 128*NV).
// Per warp a running max / sum / accumulator, merged across the warps at the end. No barrier in the key loop.
constexpr int ATT_W = 16;
template <int NV, int U>
__global__ void __launch_bounds__(ATT_W * 32) attention_fast_k(const float * q, const float * kc, const float * vc, float * out,
                                                              int p0, int Hq, int Hkv, int D, float scale) {
    extern __shared__ float4 acc_sh[];   // [ATT_W][NV][32] float4, then m [ATT_W], l [ATT_W]
    const int t = blockIdx.y, h = blockIdx.x, hk = h / (Hq / Hkv);
    const int n_kv = p0 + t + 1, D4 = D / 4, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int64_t rs = (int64_t) Hkv * D4;   // row stride in float4
    const float4 * qr = reinterpret_cast<const float4 *>(q + ((int64_t) t * Hq + h) * D);
    const float4 * kb = reinterpret_cast<const float4 *>(kc) + (int64_t) hk * D4;
    const float4 * vb = reinterpret_cast<const float4 *>(vc) + (int64_t) hk * D4;
    float4 qv[NV], acc[NV];
#pragma unroll
    for (int r = 0; r < NV; ++r) {
        const int c = lane + 32 * r;
        qv[r] = c < D4 ? qr[c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        acc[r] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    float m = -INFINITY, l = 0.0f;
    for (int j0 = w * U; j0 < n_kv; j0 += ATT_W * U) {
        float4 kv[U][NV], vv[U][NV];
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int r = 0; r < NV; ++r) {
                const int c = lane + 32 * r;
                const bool ok = j0 + u < n_kv && c < D4;
                kv[u][r] = ok ? kb[(int64_t) (j0 + u) * rs + c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                vv[u][r] = ok ? vb[(int64_t) (j0 + u) * rs + c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        float sv[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            float d = 0.0f;
#pragma unroll
            for (int r = 0; r < NV; ++r) d += qv[r].x * kv[u][r].x + qv[r].y * kv[u][r].y + qv[r].z * kv[u][r].z + qv[r].w * kv[u][r].w;
            sv[u] = d;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
#pragma unroll
            for (int u = 0; u < U; ++u) sv[u] += __shfl_xor_sync(0xffffffff, sv[u], o);
        float mx = m;
#pragma unroll
        for (int u = 0; u < U; ++u) { sv[u] = j0 + u < n_kv ? sv[u] * scale : -INFINITY; mx = fmaxf(mx, sv[u]); }
        const float corr = expf(m - mx);   // m = -inf on the first batch: 0
        l *= corr;
#pragma unroll
        for (int r = 0; r < NV; ++r) { acc[r].x *= corr; acc[r].y *= corr; acc[r].z *= corr; acc[r].w *= corr; }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const float p = expf(sv[u] - mx);
            l += p;
#pragma unroll
            for (int r = 0; r < NV; ++r) {
                acc[r].x += p * vv[u][r].x; acc[r].y += p * vv[u][r].y; acc[r].z += p * vv[u][r].z; acc[r].w += p * vv[u][r].w;
            }
        }
        m = mx;
    }
    // merge: warps that saw no key carry m = -inf, l = 0, acc = 0
    float * ml = reinterpret_cast<float *>(acc_sh + ATT_W * NV * 32);
#pragma unroll
    for (int r = 0; r < NV; ++r) acc_sh[(w * NV + r) * 32 + lane] = acc[r];
    if (lane == 0) { ml[w] = m; ml[ATT_W + w] = l; }
    __syncthreads();
    float M = -INFINITY;
#pragma unroll
    for (int i = 0; i < ATT_W; ++i) M = fmaxf(M, ml[i]);
    float L = 0.0f, f[ATT_W];
#pragma unroll
    for (int i = 0; i < ATT_W; ++i) { f[i] = expf(ml[i] - M); L += ml[ATT_W + i] * f[i]; }
    const float inv = 1.0f / L;
    float4 * orow = reinterpret_cast<float4 *>(out + ((int64_t) t * Hq + h) * D);
    for (int c = threadIdx.x; c < D4; c += ATT_W * 32) {
        const int r = c >> 5, ln = c & 31;
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int i = 0; i < ATT_W; ++i) {
            const float4 a = acc_sh[(i * NV + r) * 32 + ln];
            o.x += a.x * f[i]; o.y += a.y * f[i]; o.z += a.z * f[i]; o.w += a.w * f[i];
        }
        orow[c] = make_float4(o.x * inv, o.y * inv, o.z * inv, o.w * inv);
    }
}

static bool aligned16(const void * p) { return ((uintptr_t) p & 15) == 0; }

bool attention_fast(const float * q, const float * kc, const float * vc, float * out,
                    int T, int p0, int Hq, int Hkv, int D, float scale, cudaStream_t s) {
    if (T < 1 || p0 < 0 || D <= 0 || D % 4 || D > 512 || Hkv <= 0 || Hq % Hkv || !aligned16(q) || !aligned16(kc) || !aligned16(vc) || !aligned16(out)) return false;
    const dim3 grid(Hq, T);
    // smem: per warp NV*32 float4 accumulators + m, l (<= 32 KB at NV = 4)
#define L(NV, U) attention_fast_k<NV, U><<<grid, ATT_W * 32, (size_t) ATT_W * (NV * 32 * 16 + 8), s>>>(q, kc, vc, out, p0, Hq, Hkv, D, scale)
    if (D <= 128) L(1, 4); else if (D <= 256) L(2, 4); else L(4, 2);
#undef L
    return true;
}

void attention(const float * q, const float * kc, const float * vc, float * out,
               int T, int p0, int Hq, int Hkv, int D, float scale, cudaStream_t s) {
    const size_t smem = (size_t)(p0 + T) * sizeof(float);
    // the scores live in shared memory: past the default 48 KB (block_sum's 128 B static included) raise the
    // kernel's limit to the device opt-in maximum (same kernel, same numbers); past that the launch would fail
    // silently and leave `out` stale, so route to attention_fast (online softmax, no score buffer) or refuse
    if (smem + 32 * sizeof(float) > 48 * 1024) {
        static std::mutex mu;
        static int lim[64];   // per device: max dynamic smem now allowed for attention_k (0 = not set yet)
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) throw std::runtime_error("k::attention: no device");
        int cap;
        {
            std::lock_guard<std::mutex> g(mu);
            if (!lim[dev]) {
                int optin = 0;
                cudaFuncAttributes fa{};
                cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
                cudaFuncGetAttributes(&fa, attention_k);
                const int dyn = optin - (int) fa.sharedSizeBytes;
                lim[dev] = (dyn > 48 * 1024 - (int) fa.sharedSizeBytes &&
                            cudaFuncSetAttribute(attention_k, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn) == cudaSuccess)
                         ? dyn : 48 * 1024 - (int) fa.sharedSizeBytes;
                cudaGetLastError();
            }
            cap = lim[dev];
        }
        if (smem > (size_t) cap) {
            if (attention_fast(q, kc, vc, out, T, p0, Hq, Hkv, D, scale, s)) return;
            throw std::runtime_error("k::attention: context too long for the reference kernel's shared-memory scores");
        }
    }
    attention_k<<<dim3(Hq, T), 256, smem, s>>>(q, kc, vc, out, p0, Hq, Hkv, D, scale);
}

__global__ void conv1d_silu_k(const float * x, float * state, const float * w, float * y, int T, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float s0 = state[c * 3 + 0], s1 = state[c * 3 + 1], s2 = state[c * 3 + 2];
    const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < T; ++t) {
        const float xt = x[(int64_t) t * C + c];
        const float v = s0 * w0 + s1 * w1 + s2 * w2 + xt * w3;
        y[(int64_t) t * C + c] = silu(v);
        s0 = s1; s1 = s2; s2 = xt;
    }
    state[c * 3 + 0] = s0; state[c * 3 + 1] = s1; state[c * 3 + 2] = s2;
}

void conv1d_silu(const float * x, float * state, const float * w, float * y, int T, int C, cudaStream_t s) {
    conv1d_silu_k<<<(C + 255) / 256, 256, 0, s>>>(x, state, w, y, T, C);
}

__global__ void gdn_gates_k(float * a, float * b, const float * dt_bias, const float * ssm_a, int n, int Hv) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int h = i % Hv;
    const float z = a[i] + dt_bias[h];
    const float sp = z > 20.0f ? z : log1pf(expf(z));
    a[i] = sp * ssm_a[h];
    b[i] = 1.0f / (1.0f + expf(-b[i]));
}

void gdn_gates(float * a, float * b, const float * dt_bias, const float * ssm_a, int T, int Hv, cudaStream_t s) {
    gdn_gates_k<<<(T * Hv + 255) / 256, 256, 0, s>>>(a, b, dt_bias, ssm_a, T * Hv, Hv);
}

// one block per v-head, one thread per state row j (Dv threads, Dk == Dv). q and k are L2-normalized
// here (eps) before use. Reference speed only.
__global__ void gdn_recurrent_k(const float * conv, const float * g, const float * beta, float * state, float * out,
                                int T, int Hk, int Hv, int Dk, int Dv, float eps) {
    __shared__ float qt[256], kt[256];
    const int h = blockIdx.x, j = threadIdx.x, hk = h % Hk;
    const int C = 2 * Hk * Dk + Hv * Dv;
    float * M = state + (int64_t) h * Dv * Dk + (int64_t) j * Dk;   // row j
    const float scale = rsqrtf((float) Dk);
    for (int t = 0; t < T; ++t) {
        const float qv = conv[(int64_t) t * C + hk * Dk + j];
        const float kv = conv[(int64_t) t * C + Hk * Dk + hk * Dk + j];
        const float qn = 1.0f / fmaxf(sqrtf(block_sum(qv * qv)), eps);
        const float kn = 1.0f / fmaxf(sqrtf(block_sum(kv * kv)), eps);
        qt[j] = qv * qn;
        kt[j] = kv * kn;
        __syncthreads();
        const float vj   = conv[(int64_t) t * C + 2 * Hk * Dk + h * Dv + j];
        const float eg = expf(g[t * Hv + h]);
        const float bt = beta[t * Hv + h];
        float sk = 0.0f;
        for (int i = 0; i < Dk; ++i) { M[i] *= eg; sk += M[i] * kt[i]; }
        const float delta = (vj - sk) * bt;
        float o = 0.0f;
        for (int i = 0; i < Dk; ++i) { M[i] += kt[i] * delta; o += M[i] * qt[i]; }
        out[((int64_t) t * Hv + h) * Dv + j] = o * scale;
        __syncthreads();
    }
}

void gdn_recurrent(const float * conv, const float * g, const float * beta, float * state, float * out,
                   int T, int Hk, int Hv, int Dk, int Dv, float eps, cudaStream_t s) {
    gdn_recurrent_k<<<Hv, Dv, 0, s>>>(conv, g, beta, state, out, T, Hk, Hv, Dk, Dv, eps);
}

void kv_store(const float * kk, const float * v, float * kc, float * vc, int T, int p0, int row, cudaStream_t s) {
    cudaMemcpyAsync(kc + (int64_t) p0 * row, kk, (size_t) T * row * sizeof(float), cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(vc + (int64_t) p0 * row, v,  (size_t) T * row * sizeof(float), cudaMemcpyDeviceToDevice, s);
}

} // namespace p100e::k
