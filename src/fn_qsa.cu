// Flash-Next QSA kernels: see fn_qsa.cuh. Memory-bound glue (pool / combine / select stream NB floats per row) and the
// gathered attention (W cells x 256 floats of K and V per 4 q heads, read once).
#include "fn_qsa.cuh"

#include <cub/block/block_scan.cuh>
#include <cuda_fp16.h>

#include <cstdio>
#include <stdexcept>

namespace p100e::qsa {

#define QCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string("qsa: ") + cudaGetErrorString(e_)); } while (0)

__global__ void round_f16_k(float * x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __half2float(__float2half_rn(x[i]));
}
void round_f16(float * x, int64_t n, cudaStream_t s) { round_f16_k<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(x, n); }

__global__ void pool_k(const float * ring, const float * chunk, int n_past, int b0, int ratio, int ring_n, int D, float * out) {
    const int b = b0 + blockIdx.x;
    const int i = threadIdx.x;
    if (i >= D) return;
    float acc = 0.0f;
    for (int r = 0; r < ratio; ++r) {
        const int p = b * ratio + r;
        const float * k = p < n_past ? ring + (size_t) (p % ring_n) * D : chunk + (size_t) (p - n_past) * D;
        acc += k[i];
    }
    out[(size_t) blockIdx.x * D + i] = acc * (1.0f / (float) ratio);
}
void pool_blocks(const float * ring, const float * chunk, int n_past, int b0, int b1, int ratio, int ring_n, int D, float * out, cudaStream_t s) {
    if (b1 > b0) pool_k<<<b1 - b0, D, 0, s>>>(ring, chunk, n_past, b0, ratio, ring_n, D, out);
}

__global__ void save_tail_k(const float * chunk, int n_past, int p0, int ring_n, int D, float * ring) {
    const int j = blockIdx.x, i = threadIdx.x;
    if (i >= D) return;
    const int p = p0 + j;
    ring[(size_t) (p % ring_n) * D + i] = chunk[(size_t) (p - n_past) * D + i];
}
void save_tail(const float * chunk, int n_past, int T, int ring_n, int D, float * ring, cudaStream_t s) {
    const int p0 = T > ring_n ? n_past + T - ring_n : n_past, n = n_past + T - p0;
    if (n > 0) save_tail_k<<<n, D, 0, s>>>(chunk, n_past, p0, ring_n, D, ring);
}

__global__ void combine_k(const float * S, float * c, const int32_t * pos, int H, int NB, int ratio) {
    const int t = blockIdx.x;
    const int nvis = (pos[t] + 1) / ratio;   // blocks fully before the tail
    const float * Sr = S + (size_t) t * H * NB;
    float * cr = c + (size_t) t * NB;
    for (int b = threadIdx.x; b < NB; b += blockDim.x) {
        float acc = 0.0f;
        for (int h = 0; h < H; ++h) acc += fmaxf(Sr[(size_t) h * NB + b], 0.0f);
        cr[b] = b < nvis ? acc : -INFINITY;
    }
}
void combine(const float * S, float * c, const int32_t * pos, int m, int H, int NB, int ratio, cudaStream_t s) {
    combine_k<<<m, 1024, 0, s>>>(S, c, pos, H, NB, ratio);
}

// float -> order-preserving unsigned key (larger float = larger key)
static __device__ __forceinline__ uint32_t fkey(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

constexpr int SEL_THREADS = 1024;
constexpr int SEL_MAX_K = 1024;   // top_k / ratio + 1 <= this

// one block per row: radix select of the K largest keys (4 passes of 8 bits over the NB scores, lower index wins ties),
// then the cell list. smem: hist[256], list[SEL_MAX_K], scan storage
__global__ void __launch_bounds__(SEL_THREADS) select_k(const float * c, int NB, const int32_t * pos, int ratio, int W, int32_t * sel, int32_t * seln) {
    typedef cub::BlockScan<int, SEL_THREADS> Scan;
    __shared__ typename Scan::TempStorage scan_tmp;
    __shared__ int hist[256];
    __shared__ int list[SEL_MAX_K];
    __shared__ int s_digit, s_krem, s_count, s_base;
    __shared__ unsigned long long s_last;
    const int t = blockIdx.x, tid = threadIdx.x;
    const float * cr = c + (size_t) t * NB;
    const int q = pos[t];
    const int tail_start = (q + 1) / ratio * ratio, tail = q + 1 - tail_start;
    const int nvis = tail_start / ratio, Wr = W - tail;
    int K, rem;
    if (nvis * ratio <= Wr) { K = nvis; rem = ratio; } else { K = (Wr + ratio - 1) / ratio; rem = Wr - (K - 1) * ratio; }
    int32_t * out = sel + (size_t) t * W;
    for (int i = tid; i < tail; i += SEL_THREADS) out[i] = tail_start + i;
    if (K == 0) { if (tid == 0) seln[t] = tail; return; }
    // threshold search over the keys of the visible blocks
    uint32_t prefix = 0, mask = 0;
    int krem = K;
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
        for (int b = tid; b < nvis; b += SEL_THREADS) {
            const uint32_t k = fkey(cr[b]);
            if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 255], 1);
        }
        __syncthreads();
        if (tid == 0) {
            int cum = 0, digit = 255;
            for (int bin = 255; bin >= 0; --bin) {
                cum += hist[bin];
                if (cum >= krem) { digit = bin; krem -= cum - hist[bin]; break; }
            }
            s_digit = digit; s_krem = krem;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift; mask |= 0xffu << shift; krem = s_krem;
    }
    const uint32_t thr = prefix;   // K-th largest key; krem of the keys equal to it are taken (lowest index first)
    if (tid == 0) { s_count = 0; s_base = 0; s_last = ~0ull; }
    __syncthreads();
    for (int b0 = 0; b0 < nvis; b0 += SEL_THREADS) {
        const int b = b0 + tid;
        const bool valid = b < nvis;
        const uint32_t k = valid ? fkey(cr[b]) : 0u;
        const int eq = valid && k == thr ? 1 : 0;
        int pre = 0, tot = 0;
        Scan(scan_tmp).ExclusiveSum(eq, pre, tot);
        const bool take = valid && (k > thr || (eq && s_base + pre < krem));
        if (take) {
            const int slot = atomicAdd(&s_count, 1);
            list[slot] = b;
            atomicMin(&s_last, ((unsigned long long) k << 32) | (unsigned long long) (0xffffffffu - (uint32_t) b));   // min key, then max index
        }
        __syncthreads();
        if (tid == 0) s_base += tot;
        __syncthreads();
    }
    // the last block in (score desc, index asc) order is the partial one: move it to list[K - 1]
    if (tid == 0) {
        const int lastb = (int) (0xffffffffu - (uint32_t) (s_last & 0xffffffffu));
        int at = K - 1;
        for (int i = 0; i < K; ++i) if (list[i] == lastb) { at = i; break; }
        const int tmp = list[K - 1]; list[K - 1] = list[at]; list[at] = tmp;
    }
    __syncthreads();
    for (int i = tid; i < K * ratio; i += SEL_THREADS) {
        const int m = i / ratio, r = i % ratio;
        if (m == K - 1 && r >= rem) continue;
        out[tail + i] = list[m] * ratio + r;
    }
    if (tid == 0) seln[t] = tail + (K - 1) * ratio + rem;
}
void select(const float * c, int NB, const int32_t * pos, int m, int ratio, int W, int32_t * sel, int32_t * seln, cudaStream_t s) {
    if ((W + ratio - 1) / ratio + 1 > SEL_MAX_K) throw std::runtime_error("qsa: top_k too large for select_k");
    select_k<<<m, SEL_THREADS, 0, s>>>(c, NB, pos, ratio, W, sel, seln);
}

constexpr int ATT_HB = 4;   // q heads per block (one K/V read serves them)

static __device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// KV cache row storage (per cell and kv head, D values): MODE 0 f32, 1 f16, 2 q8 (blocks of 32: half scale + 2 pad +
// 32 int8 = 36 bytes, 4-byte aligned loads). kvrow<MODE>: row bytes, 8 consecutive values from dim d0 (d0 % 8 == 0), one value.
template <int MODE> struct kvrow;
template <> struct kvrow<0> {
    static __host__ __device__ constexpr size_t bytes(int D) { return (size_t) D * 4; }
    static __device__ __forceinline__ void load8(const char * row, int d0, float * o) {
        const float4 a = *(const float4 *) (row + (size_t) d0 * 4), b = *(const float4 *) (row + (size_t) d0 * 4 + 16);
        o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w; o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
    }
    static __device__ __forceinline__ float one(const char * row, int d) { return *(const float *) (row + (size_t) d * 4); }
};
template <> struct kvrow<1> {
    static __host__ __device__ constexpr size_t bytes(int D) { return (size_t) D * 2; }
    static __device__ __forceinline__ void load8(const char * row, int d0, float * o) {
        const uint4 u = *(const uint4 *) (row + (size_t) d0 * 2);
        const half2 * h = (const half2 *) &u;
        for (int i = 0; i < 4; ++i) { const float2 f = __half22float2(h[i]); o[2 * i] = f.x; o[2 * i + 1] = f.y; }
    }
    static __device__ __forceinline__ float one(const char * row, int d) { return __half2float(*(const half *) (row + (size_t) d * 2)); }
};
constexpr int Q8B = 36;   // bytes per block of 32
template <> struct kvrow<2> {
    static __host__ __device__ constexpr size_t bytes(int D) { return (size_t) (D / 32) * Q8B; }
    static __device__ __forceinline__ void load8(const char * row, int d0, float * o) {
        const char * blk = row + (size_t) (d0 >> 5) * Q8B;
        const float sc = __half2float(*(const half *) blk);
        uint32_t u[2];   // the block stride (36) keeps these 4-byte aligned only; no 8-byte load
        u[0] = *(const uint32_t *) (blk + 4 + (d0 & 31)); u[1] = *(const uint32_t *) (blk + 8 + (d0 & 31));
        const int8_t * q = (const int8_t *) u;
        for (int i = 0; i < 8; ++i) o[i] = sc * (float) q[i];
    }
    static __device__ __forceinline__ float one(const char * row, int d) {
        const char * blk = row + (size_t) (d >> 5) * Q8B;
        return __half2float(*(const half *) blk) * (float) *(const int8_t *) (blk + 4 + (d & 31));
    }
};

// block per (head group, query row); 256 threads = 8 warps; smem: sc[HB][W] + qs[HB][D]
template <int D, int MODE>
__global__ void __launch_bounds__(256) attn_gather_k(const float * q, const char * kc, const char * vc, const int32_t * sel, const int32_t * seln,
                                                     float * out, int Hq, int Hkv, int W, float scale) {
    constexpr size_t RB = kvrow<MODE>::bytes(D);
    extern __shared__ float smem[];
    float * sc = smem;                      // [HB][W]
    float * qs = smem + (size_t) ATT_HB * W; // [HB][D]
    __shared__ float red[2][8];
    const int t = blockIdx.y, h0 = blockIdx.x * ATT_HB;
    const int nh = min(ATT_HB, Hq - h0);
    const int G = Hq / Hkv, hk = h0 / G;
    const int n = seln[t];
    const int32_t * cells = sel + (size_t) t * W;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (int i = tid; i < nh * D; i += 256) qs[i] = q[((size_t) t * Hq + h0) * D + i];
    __syncthreads();
    // scores: warp per cell, lane owns dims [lane*8, lane*8+8)
    for (int j = warp; j < n; j += 8) {
        float kr[8];
        kvrow<MODE>::load8(kc + ((size_t) cells[j] * Hkv + hk) * RB, lane * 8, kr);
        for (int hh = 0; hh < nh; ++hh) {
            const float * qr = qs + hh * D + lane * 8;
            float d = kr[0] * qr[0] + kr[1] * qr[1] + kr[2] * qr[2] + kr[3] * qr[3] + kr[4] * qr[4] + kr[5] * qr[5] + kr[6] * qr[6] + kr[7] * qr[7];
            d = warp_sum(d);
            if (lane == 0) sc[hh * W + j] = d * scale;
        }
    }
    __syncthreads();
    // softmax per head
    float inv[ATT_HB];
    for (int hh = 0; hh < nh; ++hh) {
        float * s = sc + hh * W;
        float mx = -INFINITY;
        for (int j = tid; j < n; j += 256) mx = fmaxf(mx, s[j]);
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        if (lane == 0) red[0][warp] = mx;
        __syncthreads();
        mx = red[0][0];
        for (int w = 1; w < 8; ++w) mx = fmaxf(mx, red[0][w]);
        float den = 0.0f;
        for (int j = tid; j < n; j += 256) { const float e = expf(s[j] - mx); s[j] = e; den += e; }
        den = warp_sum(den);
        if (lane == 0) red[1][warp] = den;
        __syncthreads();
        den = 0.0f;
        for (int w = 0; w < 8; ++w) den += red[1][w];
        inv[hh] = 1.0f / den;
        __syncthreads();
    }
    // output: thread per dim (D == 256), all heads
    float acc[ATT_HB] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = 0; j < n; ++j) {
        const float v = kvrow<MODE>::one(vc + ((size_t) cells[j] * Hkv + hk) * RB, tid);
        for (int hh = 0; hh < ATT_HB; ++hh) acc[hh] += sc[hh * W + j] * v;
    }
    for (int hh = 0; hh < nh; ++hh) out[((size_t) t * Hq + h0 + hh) * D + tid] = acc[hh] * inv[hh];
}

void attention(const float * q, const void * kc, const void * vc, int kv_mode, const int32_t * sel, const int32_t * seln, float * out,
               int T, int Hq, int Hkv, int D, int W, float scale, cudaStream_t s) {
    if (D != 256) throw std::runtime_error("qsa: attention needs head dim 256");
    if ((Hq / Hkv) % ATT_HB != 0) throw std::runtime_error("qsa: q heads per kv head must be a multiple of 4");
    const size_t smem = (size_t) ATT_HB * (W + D) * sizeof(float);
    static bool attr = false;
    if (!attr) {
        QCK(cudaFuncSetAttribute(attn_gather_k<256, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        QCK(cudaFuncSetAttribute(attn_gather_k<256, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        QCK(cudaFuncSetAttribute(attn_gather_k<256, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        attr = true;
    }
    const dim3 grid((Hq + ATT_HB - 1) / ATT_HB, T);
    const char * k = (const char *) kc, * v = (const char *) vc;
    switch (kv_mode) {
        case 0:  attn_gather_k<256, 0><<<grid, 256, smem, s>>>(q, k, v, sel, seln, out, Hq, Hkv, W, scale); break;
        case 1:  attn_gather_k<256, 1><<<grid, 256, smem, s>>>(q, k, v, sel, seln, out, Hq, Hkv, W, scale); break;
        default: attn_gather_k<256, 2><<<grid, 256, smem, s>>>(q, k, v, sel, seln, out, Hq, Hkv, W, scale); break;
    }
}

size_t kv_row_bytes(int kv_mode, int D) { return kv_mode == 0 ? kvrow<0>::bytes(D) : kv_mode == 1 ? kvrow<1>::bytes(D) : kvrow<2>::bytes(D); }

// q8 store: one thread per block of 32 values of a (row, head) pair
__global__ void kv_store_q8_k(const float * k, const float * v, char * kc, char * vc, int p0, int Hkv, int D) {
    const int t = blockIdx.x;
    const int nblk = Hkv * D / 32;
    for (int b = threadIdx.x; b < nblk; b += blockDim.x) {
        const int head = b / (D / 32), bb = b % (D / 32);
        for (int w = 0; w < 2; ++w) {
            const float * src = (w ? v : k) + ((size_t) t * Hkv + head) * D + bb * 32;
            char * dst = (w ? vc : kc) + ((size_t) (p0 + t) * Hkv + head) * kvrow<2>::bytes(D) + (size_t) bb * Q8B;
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(src[i]));
            const float sc = amax / 127.0f, inv = sc > 0.0f ? 1.0f / sc : 0.0f;
            *(half *) dst = __float2half_rn(sc);
            int8_t * q = (int8_t *) (dst + 4);
            for (int i = 0; i < 32; ++i) { const int r = __float2int_rn(src[i] * inv); q[i] = (int8_t) max(-127, min(127, r)); }
        }
    }
}
__global__ void kv_window_q8_k(const char * kc, const char * vc, float * kf, float * vf, int Hkv, int D, int64_t n) {   // n = N * Hkv * D
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t rowh = i / D; const int d = (int) (i % D);
    const size_t off = (size_t) rowh * kvrow<2>::bytes(D);
    kf[i] = kvrow<2>::one(kc + off, d); vf[i] = kvrow<2>::one(vc + off, d);
}
void kv_store(int kv_mode, const float * k, const float * v, void * kc, void * vc, int T, int p0, int Hkv, int D, cudaStream_t s) {
    if (kv_mode == 1) { kv_store_f16(k, v, kc, vc, T, p0, Hkv * D, s); return; }
    kv_store_q8_k<<<T, 256, 0, s>>>(k, v, (char *) kc, (char *) vc, p0, Hkv, D);
}
void kv_window_f32(int kv_mode, const void * kc, const void * vc, float * kf, float * vf, int N, int Hkv, int D, cudaStream_t s) {
    if (kv_mode == 1) { kv_window_f32(kc, vc, kf, vf, N, Hkv * D, s); return; }
    const int64_t n = (int64_t) N * Hkv * D;
    kv_window_q8_k<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>((const char *) kc, (const char *) vc, kf, vf, Hkv, D, n);
}

__global__ void kv_store_f16_k(const float * k, const float * v, half * kc, half * vc, int p0, int row) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < row; i += blockDim.x) {
        kc[((size_t) p0 + t) * row + i] = __float2half_rn(k[(size_t) t * row + i]);
        vc[((size_t) p0 + t) * row + i] = __float2half_rn(v[(size_t) t * row + i]);
    }
}
void kv_store_f16(const float * k, const float * v, void * kc, void * vc, int T, int p0, int row, cudaStream_t s) {
    kv_store_f16_k<<<T, 256, 0, s>>>(k, v, (half *) kc, (half *) vc, p0, row);
}
__global__ void kv_window_k(const half * kc, const half * vc, float * kf, float * vf, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { kf[i] = __half2float(kc[i]); vf[i] = __half2float(vc[i]); }
}
void kv_window_f32(const void * kc, const void * vc, float * kf, float * vf, int N, int row, cudaStream_t s) {
    const int64_t n = (int64_t) N * row;
    kv_window_k<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>((const half *) kc, (const half *) vc, kf, vf, n);
}

__global__ void block_pos_k(int32_t * pos, int NB, int ratio) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < NB) pos[b] = b * ratio;
}
void block_positions(int32_t * pos, int NB, int ratio, cudaStream_t s) { block_pos_k<<<(NB + 255) / 256, 256, 0, s>>>(pos, NB, ratio); }

} // namespace p100e::qsa
