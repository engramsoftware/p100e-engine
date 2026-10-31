#include "fast/fused.cuh"
#include "fast/a16q.cuh"
#include "fast/kv_q8.cuh"
#include "fast/kv_page.cuh"

#include <cmath>
#include <cstdlib>
#include <type_traits>

namespace p100e::fast {
namespace {

__device__ __forceinline__ unsigned long long global_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// thread 0 of a P2P allreduce wait (flag block layout: fused.cuh p2p_flag_*): spin until the peer's sequence number
// reaches seq. Wrap-safe compare (unsigned difference), so the counters may run through 2^31 (~4 days of 128
// allreduces per token at 50 t/s). Bounded by %globaltimer (flag[P2P_FLAG_TIMEOUT_MS], default 2000 ms; a clock64
// backstop if the timer stalls): on timeout the wait counts it in flag[1], writes the host-mapped status line (if
// set) and returns; results are then invalid and the host aborts after the step (tp_model::check_ar). Once a timeout
// was counted, later waits return at once, so the failing step finishes quickly instead of spinning per allreduce.
__device__ __forceinline__ void p2p_wait_thread(const int * flag, const int seq) {
    if ((int) ((unsigned) *((volatile const int *) flag) - (unsigned) seq) >= 0) return;   // arrived: one load, as before
    if (*((volatile const int *) (flag + 1)) != 0) return;
    const int ms = flag[P2P_FLAG_TIMEOUT_MS];
    const unsigned long long lim = (unsigned long long) (ms > 0 ? ms : 2000) * 1000000ull;
    const unsigned long long t0 = global_ns();
    const long long c0 = clock64();
    while ((int) ((unsigned) *((volatile const int *) flag) - (unsigned) seq) < 0) {
        const unsigned long long dt = global_ns() - t0;
        if (dt > lim || clock64() - c0 > (long long) (2 * lim)) {
            atomicAdd(const_cast<int *>(flag) + 1, 1);
            int * st = *(int * const *) (flag + P2P_FLAG_STATUS_PTR);   // host-mapped status line (may be null)
            if (st) {
                *((volatile int *) st + 1) = seq;
                *((volatile int *) st + 2) = *((volatile const int *) flag);
                *((volatile int *) st + 3) = (int) (dt / 1000000ull);
                __threadfence_system();
                *((volatile int *) st) = 1;
            }
            return;
        }
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// block sum, blockDim.x multiple of 32 and <= 1024; result broadcast to all threads
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float sh[32];
    v = warp_sum(v);
    if (blockDim.x == 32) return v;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = blockDim.x >> 5;
    v = lane < nw ? sh[lane] : 0.0f;
    return warp_sum(v);   // every warp reduces the same values -> broadcast
}

__device__ __forceinline__ float silu_f(float v) { return v / (1.0f + expf(-v)); }

// ---- vectorized row kernel: each thread keeps its NV float4 of the row in registers ----
// ADD2: a second residual r2 is added after r (x = (x + r) + r2). QOUT: y is also quantized into qbuf (a16 layout,
// row = token column, gridDim.x columns); needs n % 32 == 0 so every 32-group sits in 8 lanes of one k step.
// WAIT (needs ADD): one-shot P2P allreduce epilogue: x is loaded, then thread 0 spins until *flag >= seq (the peer's
// p2p_signal), then r / r2 (one of them in the peer GPU's memory) are read with cache-volatile loads. seqp != null:
// seq is *seqp, read on the device (the caller's own signal counter, see p2p_signal_ctr: CUDA-graph replays).
template <bool ADD, bool GATED, bool HOUT, int NV, bool ADD2 = false, bool QOUT = false, bool WAIT = false>
__global__ void norm_vec_k(float * x, const float * r, const float * w, const float * z, float * y, __half * yh,
                           int n, float eps, const float * r2 = nullptr, void * qbuf = nullptr,
                           const int * flag = nullptr, int seq = 0, const int * seqp = nullptr,
                           int * sig_peer = nullptr, int sig_par = -1) {
    const int n4 = n >> 2;
    const int64_t off = (int64_t) blockIdx.x * n;
    float4 * x4 = reinterpret_cast<float4 *>(x + off);
    const float4 * r4 = reinterpret_cast<const float4 *>(r + off);
    const float4 * w4 = reinterpret_cast<const float4 *>(w);
    const float4 * z4 = reinterpret_cast<const float4 *>(z + off);
    float4 * y4 = reinterpret_cast<float4 *>(y + off);
    float4 v[NV];
    float ss = 0.0f;
    if (ADD) {
        // all loads first: with the x store inside one loop the compiler cannot hoist the next k's loads above it
        // (x may alias r), which serialized NV load round trips. Same per-element arithmetic and ss order.
        float4 b[NV], b2[NV];
        if (WAIT) {
            // sig_par >= 0 (P100E_AR_SIG_FUSE): this kernel also does this GPU's own signal of the allreduce it waits
            // for, which p2p_signal_k did as a separate 1-thread launch right before it in the same stream (the same
            // kernel-boundary ordering after the partial's producer). The counter alternates between two words of the
            // own flag block so no block reads a word written in this kernel: seq = word[1 - par] + 1, block 0
            // stores it into word[par] and into the peer's flag (fence / fault injection as p2p_signal_k).
            if (sig_par >= 0 && threadIdx.x == 0) {
                int * own = const_cast<int *>(flag);
                const int c = *((volatile const int *) (own + (sig_par ? P2P_FLAG_CTR0 : P2P_FLAG_CTR1)));
                seq = (int) ((unsigned) c + 1u);
                if (blockIdx.x == 0) {
                    *((volatile int *) (own + (sig_par ? P2P_FLAG_CTR1 : P2P_FLAG_CTR0))) = seq;
                    const int drop = own[P2P_FLAG_TEST_DROP];
                    if (!(drop != 0 && (int) ((unsigned) seq - (unsigned) drop) >= 0)) {
                        if (own[P2P_FLAG_FENCE] != 0) __threadfence_system();
                        *((volatile int *) sig_peer) = seq;
                    }
                }
            }
#pragma unroll
            for (int k = 0; k < NV; ++k) {
                const int i = threadIdx.x + k * blockDim.x;
                if (i < n4) v[k] = x4[i];
            }
            if (threadIdx.x == 0) {
                if (seqp && sig_par < 0) seq = *((volatile const int *) seqp);
                p2p_wait_thread(flag, seq);   // bounded spin; a timeout is counted and reported (results then wrong)
            }
            __syncthreads();
        }
#pragma unroll
        for (int k = 0; k < NV; ++k) {
            const int i = threadIdx.x + k * blockDim.x;
            if (i < n4) {
                if (!WAIT) v[k] = x4[i];
                b[k] = WAIT ? __ldcv(r4 + i) : r4[i];
                if (ADD2) b2[k] = WAIT ? __ldcv(reinterpret_cast<const float4 *>(r2 + off) + i)
                                       : reinterpret_cast<const float4 *>(r2 + off)[i];
            }
        }
#pragma unroll
        for (int k = 0; k < NV; ++k) {
            const int i = threadIdx.x + k * blockDim.x;
            if (i < n4) {
                float4 a = v[k];
                a.x += b[k].x; a.y += b[k].y; a.z += b[k].z; a.w += b[k].w;
                if (ADD2) { a.x += b2[k].x; a.y += b2[k].y; a.z += b2[k].z; a.w += b2[k].w; }
                x4[i] = a;
                v[k] = a;
                ss += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
            }
        }
    } else {
#pragma unroll
        for (int k = 0; k < NV; ++k) {
            const int i = threadIdx.x + k * blockDim.x;
            if (i < n4) {
                const float4 a = x4[i];
                v[k] = a;
                ss += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
            }
        }
    }
    // the norm weights do not depend on the sum: load them before the block reduction (and before any y store,
    // which the compiler could not move the loads above)
    float4 wv[NV];
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (w && i < n4) wv[k] = w4[i];
    }
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / n + eps);
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (i < n4) {
            o = v[k];
            o.x *= sc; o.y *= sc; o.z *= sc; o.w *= sc;
            if (w) { const float4 ww = wv[k]; o.x *= ww.x; o.y *= ww.y; o.z *= ww.z; o.w *= ww.w; }
            if (GATED) {
                const float4 zz = z4[i];
                o.x *= silu_f(zz.x); o.y *= silu_f(zz.y); o.z *= silu_f(zz.z); o.w *= silu_f(zz.w);
            }
            y4[i] = o;
            if (HOUT) {
                __half2 * h2 = reinterpret_cast<__half2 *>(yh + off) + 2 * i;
                h2[0] = __floats2half2_rn(o.x, o.y);
                h2[1] = __floats2half2_rn(o.z, o.w);
            }
        }
        if (QOUT) {
            // whole 8-lane segments are either all < n4 or all >= n4 (n4 % 8 == 0); every lane runs the shuffles
            if (k * blockDim.x < n4) {
                const int g = i >> 3, ng = n >> 5;
                a16_quant8(o, a16_qs(qbuf, blockIdx.x, g, ng), a16_ads(qbuf, blockIdx.x, g, ng, gridDim.x), i < n4);
            }
        }
    }
}

// ---- scalar fallback, any n / alignment ----
template <bool ADD, bool GATED, bool HOUT>
__global__ void norm_gen_k(float * x, const float * r, const float * w, const float * z, float * y, __half * yh,
                           int n, float eps) {
    const int64_t off = (int64_t) blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float a = x[off + i];
        if (ADD) { a += r[off + i]; x[off + i] = a; }
        ss += a * a;
    }
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float o = x[off + i] * sc * (w ? w[i] : 1.0f);
        if (GATED) o *= silu_f(z[off + i]);
        y[off + i] = o;
        if (HOUT) yh[off + i] = __float2half(o);
    }
}

bool aligned16(const void * p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

template <bool ADD, bool GATED, bool HOUT, int NV>
void launch_vec(float * x, const float * r, const float * w, const float * z, float * y, __half * yh,
                int64_t rows, int n, float eps, cudaStream_t s) {
    const int n4 = n >> 2;
    int thr = (n4 + NV - 1) / NV;
    thr = (thr + 31) & ~31;
    norm_vec_k<ADD, GATED, HOUT, NV><<<(unsigned) rows, thr, 0, s>>>(x, r, w, z, y, yh, n, eps);
}

template <bool ADD, bool GATED, bool HOUT>
void dispatch(float * x, const float * r, const float * w, const float * z, float * y, __half * yh,
              int64_t rows, int n, float eps, cudaStream_t s) {
    if (rows <= 0) return;
    const bool ok = (n % 4 == 0) && n <= 4096 * 4 && aligned16(x) && aligned16(y) && (!w || aligned16(w)) &&
                    (!ADD || aligned16(r)) && (!GATED || aligned16(z)) && (!HOUT || aligned16(yh));
    if (!ok) {
        norm_gen_k<ADD, GATED, HOUT><<<(unsigned) rows, n >= 1024 ? 512 : ((n + 31) / 32 * 32), 0, s>>>(x, r, w, z, y, yh, n, eps);
        return;
    }
    const int n4 = n >> 2;
    if      (n4 <= 256)  launch_vec<ADD, GATED, HOUT, 1>(x, r, w, z, y, yh, rows, n, eps, s);   // n=128,256 -> 32/64 threads
    else if (n4 <= 512)  launch_vec<ADD, GATED, HOUT, 2>(x, r, w, z, y, yh, rows, n, eps, s);
    else if (n4 <= 1024) launch_vec<ADD, GATED, HOUT, 4>(x, r, w, z, y, yh, rows, n, eps, s);
    else if (n4 <= 1280) launch_vec<ADD, GATED, HOUT, 5>(x, r, w, z, y, yh, rows, n, eps, s);   // n=5120 -> 256 threads x 5 float4
    else if (n4 <= 2048) launch_vec<ADD, GATED, HOUT, 8>(x, r, w, z, y, yh, rows, n, eps, s);
    else                 launch_vec<ADD, GATED, HOUT, 16>(x, r, w, z, y, yh, rows, n, eps, s);
}

__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }

__global__ void silu_mul_k(float * g, const float * u, int64_t n, int64_t n4) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (int64_t) gridDim.x * blockDim.x) {
        float4 a = reinterpret_cast<float4 *>(g)[i];
        const float4 b = reinterpret_cast<const float4 *>(u)[i];
        a.x = silu_f(a.x) * b.x; a.y = silu_f(a.y) * b.y; a.z = silu_f(a.z) * b.z; a.w = silu_f(a.w) * b.w;
        reinterpret_cast<float4 *>(g)[i] = a;
    }
    if (blockIdx.x == 0 && threadIdx.x < (n & 3)) {
        const int64_t i = n4 * 4 + threadIdx.x;
        g[i] = silu_f(g[i]) * u[i];
    }
}
__global__ void sigmoid_mul_k(float * x, const float * g, int64_t n, int64_t n4) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (int64_t) gridDim.x * blockDim.x) {
        float4 a = reinterpret_cast<float4 *>(x)[i];
        const float4 b = reinterpret_cast<const float4 *>(g)[i];
        a.x *= sigm(b.x); a.y *= sigm(b.y); a.z *= sigm(b.z); a.w *= sigm(b.w);
        reinterpret_cast<float4 *>(x)[i] = a;
    }
    if (blockIdx.x == 0 && threadIdx.x < (n & 3)) {
        const int64_t i = n4 * 4 + threadIdx.x;
        x[i] *= sigm(g[i]);
    }
}
__global__ void silu_mul_s(float * g, const float * u, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) g[i] = silu_f(g[i]) * u[i];
}
__global__ void sigmoid_mul_s(float * x, const float * g, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) x[i] *= sigm(g[i]);
}

unsigned ew_blocks(int64_t n) { int64_t b = (n + 255) / 256; return (unsigned) (b < 1 ? 1 : (b > 4096 ? 4096 : b)); }

// ---- residual adds + rmsnorm (+ a16 quantize): same NV / thread-count choice as dispatch(), so the block sums match ----
template <bool ADD, bool ADD2, bool QOUT, int NV, bool WAIT = false>
void launch_nq(float * x, const float * r1, const float * r2, const float * w, float * y, void * qbuf, int T, int n,
               float eps, cudaStream_t s, const int * flag = nullptr, int seq = 0, const int * seqp = nullptr,
               int * sig_peer = nullptr, int sig_par = -1) {
    const int n4 = n >> 2;
    int thr = (n4 + NV - 1) / NV;
    thr = (thr + 31) & ~31;
    norm_vec_k<ADD, false, false, NV, ADD2, QOUT, WAIT><<<(unsigned) T, thr, 0, s>>>(x, r1, w, nullptr, y, nullptr, n, eps, r2,
                                                                                     qbuf, flag, seq, seqp, sig_peer, sig_par);
}
template <bool ADD, bool ADD2, bool QOUT, bool WAIT = false>
void dispatch_nq(float * x, const float * r1, const float * r2, const float * w, float * y, void * qbuf, int T, int n,
                 float eps, cudaStream_t s, const int * flag = nullptr, int seq = 0, const int * seqp = nullptr,
                 int * sig_peer = nullptr, int sig_par = -1) {
    const int n4 = n >> 2;
    if      (n4 <= 256)  launch_nq<ADD, ADD2, QOUT, 1, WAIT>(x, r1, r2, w, y, qbuf, T, n, eps, s, flag, seq, seqp, sig_peer, sig_par);
    else if (n4 <= 512)  launch_nq<ADD, ADD2, QOUT, 2, WAIT>(x, r1, r2, w, y, qbuf, T, n, eps, s, flag, seq, seqp, sig_peer, sig_par);
    else if (n4 <= 1024) launch_nq<ADD, ADD2, QOUT, 4, WAIT>(x, r1, r2, w, y, qbuf, T, n, eps, s, flag, seq, seqp, sig_peer, sig_par);
    else if (n4 <= 1280) launch_nq<ADD, ADD2, QOUT, 5, WAIT>(x, r1, r2, w, y, qbuf, T, n, eps, s, flag, seq, seqp, sig_peer, sig_par);
    else if (n4 <= 2048) launch_nq<ADD, ADD2, QOUT, 8, WAIT>(x, r1, r2, w, y, qbuf, T, n, eps, s, flag, seq, seqp, sig_peer, sig_par);
    else                 launch_nq<ADD, ADD2, QOUT, 16, WAIT>(x, r1, r2, w, y, qbuf, T, n, eps, s, flag, seq, seqp, sig_peer, sig_par);
}

// The partial was written by the previous kernel in this stream, so it is complete in this GPU's memory before this
// kernel starts (kernel boundary), and the peer reads it from this GPU's memory (P2P loads served by this GPU's memory
// system) only after it has seen the flag: no fence is needed for that ordering. The fork's allreduce fences before
// its arrival token because it stages the data through host memory; here flag[P2P_FLAG_FENCE] != 0
// (P100E_AR_FENCE=1) adds the same __threadfence_system() before the flag store, off by default: it costs ~2 us per
// allreduce (decode 128 per token, -1.4 % t/s measured).
// ctr != null: seq = ++*ctr (this GPU's own signal counter, in its memory, flag[2] of its block), so the value is
// decided on the device; unsigned arithmetic (wraps, see p2p_wait_thread). Fault injection (P100E_AR_TEST_DROP):
// flag[P2P_FLAG_TEST_DROP] of the own block != 0 -> signals from that sequence number on are not stored.
__global__ void p2p_signal_k(int * peer_flag, int seq, int * ctr) {
    bool fence = false;
    if (ctr) {
        const int c = *ctr;
        const int drop = ctr[P2P_FLAG_TEST_DROP - 2];
        fence = ctr[P2P_FLAG_FENCE - 2] != 0;
        seq = (int) ((unsigned) c + 1u);
        *ctr = seq;
        if (drop != 0 && (int) ((unsigned) seq - (unsigned) drop) >= 0) return;
    }
    if (fence) __threadfence_system();
    *((volatile int *) peer_flag) = seq;
}

// the signal of add2_rmsnorm_q_sig_wait alone (same counter words / fence / fault injection), for an allreduce whose
// wait is not a sig_wait launch
__global__ void p2p_sig_flush_k(int * peer_flag, int * own, int par) {
    const int seq = (int) ((unsigned) own[par ? P2P_FLAG_CTR0 : P2P_FLAG_CTR1] + 1u);
    own[par ? P2P_FLAG_CTR1 : P2P_FLAG_CTR0] = seq;
    const int drop = own[P2P_FLAG_TEST_DROP];
    if (drop != 0 && (int) ((unsigned) seq - (unsigned) drop) >= 0) return;
    if (own[P2P_FLAG_FENCE] != 0) __threadfence_system();
    *((volatile int *) peer_flag) = seq;
}

// ---- allreduce over 3..4 GPUs (fused.cuh addn_rmsnorm_q_wait_ctr / p2p_signal_n) ----
// p2p_wait_thread with the sequence number in `word` (one word per sender) instead of word 0; same bound, timeout
// count (flag[1]) and status line
__device__ __forceinline__ void p2p_wait_word(const int * flag, const int * word, const int seq) {
    if ((int) ((unsigned) *((volatile const int *) word) - (unsigned) seq) >= 0) return;
    if (*((volatile const int *) (flag + 1)) != 0) return;
    const int ms = flag[P2P_FLAG_TIMEOUT_MS];
    const unsigned long long lim = (unsigned long long) (ms > 0 ? ms : 2000) * 1000000ull;
    const unsigned long long t0 = global_ns();
    const long long c0 = clock64();
    while ((int) ((unsigned) *((volatile const int *) word) - (unsigned) seq) < 0) {
        const unsigned long long dt = global_ns() - t0;
        if (dt > lim || clock64() - c0 > (long long) (2 * lim)) {
            atomicAdd(const_cast<int *>(flag) + 1, 1);
            int * st = *(int * const *) (flag + P2P_FLAG_STATUS_PTR);
            if (st) {
                *((volatile int *) st + 1) = seq;
                *((volatile int *) st + 2) = *((volatile const int *) word);
                *((volatile int *) st + 3) = (int) (dt / 1000000ull);
                __threadfence_system();
                *((volatile int *) st) = 1;
            }
            return;
        }
    }
}

// norm_vec_k<ADD, false, false, NV, ADD2, QOUT = true, WAIT = true> with NP partials (rank order) instead of two, and a
// wait on every other rank's flag word (flag == null: no wait, the event allreduce's copies): same per-element adds,
// block shape, block sum, scale and a16 epilogue. The sum of squares is spelled with __fmul_rn / __fmaf_rn (x, y, z,
// w): nvcc contracts the plain expression differently per instantiation (norm_vec_k<.., NV = 1, ..> squares y first,
// NV = 5 x first), so the order is pinned here; this kernel is bitwise the same for every NV / wait mode, while
// k::add + add2_rmsnorm_q can differ from it in the last bit of a row's scale (tests/tp_ntp_test.cpp --kernel-check).
template <int NV, int NP>
__global__ void normn_wait_k(float * x, const p2p_ptrs P, const float * w, float * y, void * qbuf, int n, float eps,
                             const int * flag, const int * ctr, int self) {
    const int n4 = n >> 2;
    const int64_t off = (int64_t) blockIdx.x * n;
    float4 * x4 = reinterpret_cast<float4 *>(x + off);
    const float4 * w4 = reinterpret_cast<const float4 *>(w);
    float4 * y4 = reinterpret_cast<float4 *>(y + off);
    float4 v[NV];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < n4) v[k] = x4[i];
    }
    if (flag && threadIdx.x == 0) {
        const int seq = *((volatile const int *) ctr);
        for (int q = 0; q < NP; ++q)
            if (q != self) p2p_wait_word(flag, flag + P2P_FLAG_PEER0 + q, seq);
    }
    __syncthreads();
    float4 b[NP][NV];
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < n4) {
#pragma unroll
            for (int q = 0; q < NP; ++q) b[q][k] = __ldcv(reinterpret_cast<const float4 *>(P.p[q] + off) + i);
        }
    }
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < n4) {
            float4 a = v[k];
#pragma unroll
            for (int q = 0; q < NP; ++q) { a.x += b[q][k].x; a.y += b[q][k].y; a.z += b[q][k].z; a.w += b[q][k].w; }
            x4[i] = a;
            v[k] = a;
            ss = __fadd_rn(ss, __fmaf_rn(a.w, a.w, __fmaf_rn(a.z, a.z, __fmaf_rn(a.y, a.y, __fmul_rn(a.x, a.x)))));
        }
    }
    float4 wv[NV];
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (w && i < n4) wv[k] = w4[i];
    }
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / n + eps);
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (i < n4) {
            o = v[k];
            o.x *= sc; o.y *= sc; o.z *= sc; o.w *= sc;
            if (w) { const float4 ww = wv[k]; o.x *= ww.x; o.y *= ww.y; o.z *= ww.z; o.w *= ww.w; }
            y4[i] = o;
        }
        if (k * blockDim.x < n4) {
            const int g = i >> 3, ng = n >> 5;
            a16_quant8(o, a16_qs(qbuf, blockIdx.x, g, ng), a16_ads(qbuf, blockIdx.x, g, ng, gridDim.x), i < n4);
        }
    }
}

// p2p_signal_k (ctr mode) storing into one word of every other rank's flag block
__global__ void p2p_signal_n_k(const p2p_flags peer, int np, int self, int * ctr) {
    const int c = *ctr;
    const int drop = ctr[P2P_FLAG_TEST_DROP - P2P_FLAG_CTR0];
    const bool fence = ctr[P2P_FLAG_FENCE - P2P_FLAG_CTR0] != 0;
    const int seq = (int) ((unsigned) c + 1u);
    *ctr = seq;
    if (drop != 0 && (int) ((unsigned) seq - (unsigned) drop) >= 0) return;
    if (fence) __threadfence_system();
#pragma unroll
    for (int q = 0; q < P2P_MAX_RANKS; ++q)   // static indices: the pointer struct stays in parameter space (no stack)
        if (q < np && q != self && peer.f[q]) *((volatile int *) (peer.f[q] + P2P_FLAG_PEER0 + self)) = seq;
}

template <int NP>
void launch_normn(float * x, const p2p_ptrs & P, const float * w, float * y, void * qbuf, int T, int n, float eps,
                  const int * flag, const int * ctr, int self, cudaStream_t s) {
    const int n4 = n >> 2;
    auto go = [&](auto nv_tag) {   // NV and thread count exactly as dispatch_nq (same reduction tree)
        constexpr int NV = decltype(nv_tag)::value;
        int thr = (n4 + NV - 1) / NV;
        thr = (thr + 31) & ~31;
        normn_wait_k<NV, NP><<<(unsigned) T, thr, 0, s>>>(x, P, w, y, qbuf, n, eps, flag, ctr, self);
    };
    if      (n4 <= 256)  go(std::integral_constant<int, 1>());
    else if (n4 <= 512)  go(std::integral_constant<int, 2>());
    else if (n4 <= 1024) go(std::integral_constant<int, 4>());
    else if (n4 <= 1280) go(std::integral_constant<int, 5>());
    else if (n4 <= 2048) go(std::integral_constant<int, 8>());
    else                 go(std::integral_constant<int, 16>());
}

__global__ void add_s(float * x, const float * r, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) x[i] += r[i];
}

// ---- elementwise gate op + a16 quantize, one float4 per thread (8 lanes per 32-group) ----
template <int OP>
__global__ void ew_q_k(const float * a, const float * b, float * y, void * qbuf, int n, int lda, int ldb) {
    const int n4 = n >> 2, t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool act = i < n4;
    float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (act) {
        const float4 av = reinterpret_cast<const float4 *>(a + (int64_t) t * lda)[i];
        const float4 bv = reinterpret_cast<const float4 *>(b + (int64_t) t * ldb)[i];
        if (OP == EW_SILU_MUL) {   // k::silu_mul_k: g = silu(g) * u
            o.x = silu_f(av.x) * bv.x; o.y = silu_f(av.y) * bv.y; o.z = silu_f(av.z) * bv.z; o.w = silu_f(av.w) * bv.w;
        } else {                   // k::sigmoid_mul_k: x *= 1 / (1 + exp(-g))
            o.x = av.x * sigm(bv.x); o.y = av.y * sigm(bv.y); o.z = av.z * sigm(bv.z); o.w = av.w * sigm(bv.w);
        }
        reinterpret_cast<float4 *>(y + (int64_t) t * n)[i] = o;
    }
    const int ng = n >> 5;
    a16_quant8(o, a16_qs(qbuf, t, i >> 3, ng), a16_ads(qbuf, t, i >> 3, ng, gridDim.y), act);
}

// ---- attention input prep: one block per (head, token); blockDim = round32(D/4) as fast::rmsnorm uses for n = D ----
// KT: f16, or q8_0 / q4_0 (fast/kv_q8.cuh: one warp per 32-value block, the same device functions as fast::kv_append,
// so the cache bytes are identical)
template <uint32_t KT, bool MR = false>
__global__ void attn_prep_k(const float * __restrict__ qg, const float * __restrict__ k, const float * __restrict__ v,
                            const float * __restrict__ qn, const float * __restrict__ kn, const int32_t * __restrict__ pos,
                            float * __restrict__ q, float * __restrict__ gate, char * kc, char * vc, size_t rowb, int p0,
                            int Hq, int Hkv, int D, int n_rot, float freq_base, float eps, int ldq, int ldk, int ldv,
                            const kv_pg pg, const mrope_pos mr = mrope_pos{}) {
    extern __shared__ float smr[];   // D floats: the normed head, roped in place
    const int t = blockIdx.y, b = blockIdx.x, i = threadIdx.x, D4 = D >> 2;
    const int row = p0 < 0 ? pos[t] : p0 + t;   // cache position of token t
    const size_t roff = kv_off(pg, t, row, rowb);   // its byte offset in the (paged) cache
    const int nb = D / 32, lane = i & 31, wid = i >> 5, nw = blockDim.x >> 5;
    if (b >= Hq + Hkv) {   // v head -> V cache (fast::kv_append)
        const int h = b - Hq - Hkv;
        const float * src = v + (int64_t) t * ldv + (int64_t) h * D;
        if constexpr (KT == KV_Q8_0) {
            char * head = vc + roff + (size_t) h * nb * 34;
            for (int bb = wid; bb < nb; bb += nw) q8_quant_block_warp(src[bb * 32 + lane], lane, head, bb, nb);
        } else if constexpr (KT == KV_Q4_0) {
            char * head = vc + roff + (size_t) h * nb * 18;
            for (int bb = wid; bb < nb; bb += nw) q4_quant_block_warp(src[bb * 32 + lane], lane, head + bb * 18);
        } else {
            __half * dst = reinterpret_cast<__half *>(vc + roff) + (size_t) h * D;
            for (int e = i; e < D; e += blockDim.x) dst[e] = __float2half(src[e]);
        }
        return;
    }
    const bool isq = b < Hq;
    const int h = isq ? b : b - Hq;
    const float * src = isq ? qg + (int64_t) t * ldq + (int64_t) h * 2 * D : k + (int64_t) t * ldk + (int64_t) h * D;
    const float * w = isq ? qn : kn;
    // fast::rmsnorm (norm_vec_k, NV = 1): same expressions in the same order
    float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float ss = 0.0f;
    if (i < D4) {
        a = reinterpret_cast<const float4 *>(src)[i];
        if (isq) reinterpret_cast<float4 *>(gate + ((int64_t) t * Hq + h) * D)[i] = reinterpret_cast<const float4 *>(src + D)[i];
        ss += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
    }
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / D + eps);
    if (i < D4) {
        float4 o = a;
        o.x *= sc; o.y *= sc; o.z *= sc; o.w *= sc;
        if (w) { const float4 ww = reinterpret_cast<const float4 *>(w)[i]; o.x *= ww.x; o.y *= ww.y; o.z *= ww.z; o.w *= ww.w; }
        reinterpret_cast<float4 *>(smr)[i] = o;
    }
    __syncthreads();
    // k::rope_neox_k
    const int half = n_rot / 2;
    for (int j = i; j < half; j += blockDim.x) {
        float theta;
        if constexpr (MR) theta = (float) mrope_pick(mr, t, j) * powf(freq_base, -2.0f * j / n_rot);
        else theta = (float) pos[t] * powf(freq_base, -2.0f * j / n_rot);
        const float c = cosf(theta), sn = sinf(theta);
        const float x0 = smr[j], x1 = smr[j + half];
        smr[j]        = x0 * c - x1 * sn;
        smr[j + half] = x0 * sn + x1 * c;
    }
    __syncthreads();
    if (isq) {
        float4 * dst = reinterpret_cast<float4 *>(q + ((int64_t) t * Hq + h) * D);
        for (int e = i; e < D4; e += blockDim.x) dst[e] = reinterpret_cast<const float4 *>(smr)[e];
    } else if constexpr (KT == KV_Q8_0) {
        char * head = kc + roff + (size_t) h * nb * 34;
        for (int bb = wid; bb < nb; bb += nw) q8_quant_block_warp(smr[bb * 32 + lane], lane, head, bb, nb);
    } else if constexpr (KT == KV_Q4_0) {
        char * head = kc + roff + (size_t) h * nb * 18;
        for (int bb = wid; bb < nb; bb += nw) q4_quant_block_warp(smr[bb * 32 + lane], lane, head + bb * 18);
    } else {
        __half * dst = reinterpret_cast<__half *>(kc + roff) + (size_t) h * D;
        for (int e = i; e < D; e += blockDim.x) dst[e] = __float2half(smr[e]);
    }
}

} // namespace

bool add2_rmsnorm_q(float * x, const float * r1, const float * r2, const float * w, float * y, void * qbuf,
                    int T, int n, float eps, cudaStream_t s) {
    if (T <= 0) return qbuf != nullptr;
    if (!r1) { r1 = r2; r2 = nullptr; }
    const bool ok = (n % 4 == 0) && n <= 4096 * 4 && aligned16(x) && aligned16(y) && (!w || aligned16(w)) &&
                    (!r1 || aligned16(r1)) && (!r2 || aligned16(r2));
    if (!ok) {   // unaligned / odd n: the plain kernels (adds, then the scalar norm); caller quantizes
        if (r1) add_s<<<ew_blocks((int64_t) T * n), 256, 0, s>>>(x, r1, (int64_t) T * n);
        if (r2) add_s<<<ew_blocks((int64_t) T * n), 256, 0, s>>>(x, r2, (int64_t) T * n);
        dispatch<false, false, false>(x, nullptr, w, nullptr, y, nullptr, T, n, eps, s);
        return false;
    }
    const bool q = qbuf && n % 32 == 0;
#define NQ(A, A2) do { if (q) dispatch_nq<A, A2, true>(x, r1, r2, w, y, qbuf, T, n, eps, s); \
                       else   dispatch_nq<A, A2, false>(x, r1, r2, w, y, nullptr, T, n, eps, s); } while (0)
    if (r2)      NQ(true, true);
    else if (r1) NQ(true, false);
    else         NQ(false, false);
#undef NQ
    return q;
}

static bool wait_impl(float * x, const float * r1, const float * r2, const int * flag, int seq, const int * seqp,
                      const float * w, float * y, void * qbuf, int T, int n, float eps, cudaStream_t s,
                      int * sig_peer = nullptr, int sig_par = -1) {
    if (T <= 0) return true;
    const bool ok = (n % 32 == 0) && n <= 4096 * 4 && aligned16(x) && aligned16(y) && (!w || aligned16(w)) &&
                    r1 && r2 && aligned16(r1) && aligned16(r2) && qbuf && flag;
    if (!ok) return false;
    dispatch_nq<true, true, true, true>(x, r1, r2, w, y, qbuf, T, n, eps, s, flag, seq, seqp, sig_peer, sig_par);
    return true;
}
bool add2_rmsnorm_q_wait(float * x, const float * r1, const float * r2, const int * flag, int seq, const float * w, float * y,
                         void * qbuf, int T, int n, float eps, cudaStream_t s) {
    return wait_impl(x, r1, r2, flag, seq, nullptr, w, y, qbuf, T, n, eps, s);
}
bool add2_rmsnorm_q_wait_ctr(float * x, const float * r1, const float * r2, const int * flag, const int * ctr, const float * w,
                             float * y, void * qbuf, int T, int n, float eps, cudaStream_t s) {
    if (!ctr) return false;
    return wait_impl(x, r1, r2, flag, 0, ctr, w, y, qbuf, T, n, eps, s);
}

bool add2_rmsnorm_q_sig_wait(float * x, const float * r1, const float * r2, int * flag, int * peer_flag, int par,
                             const float * w, float * y, void * qbuf, int T, int n, float eps, cudaStream_t s) {
    if (!peer_flag || par < 0 || par > 1) return false;
    return wait_impl(x, r1, r2, flag, 0, nullptr, w, y, qbuf, T, n, eps, s, peer_flag, par);
}
__global__ void p2p_sig_equalize_k(int * own) {
    const int a = own[P2P_FLAG_CTR0], b = own[P2P_FLAG_CTR1];
    const int m = (int) ((unsigned) a - (unsigned) b) >= 0 ? a : b;
    own[P2P_FLAG_CTR0] = m; own[P2P_FLAG_CTR1] = m;
}
void p2p_sig_equalize(int * flag, cudaStream_t s) { p2p_sig_equalize_k<<<1, 1, 0, s>>>(flag); }
void p2p_sig_flush(int * peer_flag, int * flag, int par, cudaStream_t s) { p2p_sig_flush_k<<<1, 1, 0, s>>>(peer_flag, flag, par); }
void p2p_signal(int * peer_flag, int seq, cudaStream_t s) { p2p_signal_k<<<1, 1, 0, s>>>(peer_flag, seq, nullptr); }
void p2p_signal_ctr(int * peer_flag, int * ctr, cudaStream_t s) { p2p_signal_k<<<1, 1, 0, s>>>(peer_flag, 0, ctr); }
int p2p_timeouts(const int * flag) {
    int v[2] = {0, 0};
    if (cudaMemcpy(v, flag, sizeof v, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    return v[1];
}
void p2p_signal_n(const p2p_flags & peer, int np, int self, int * ctr, cudaStream_t s) {
    p2p_signal_n_k<<<1, 1, 0, s>>>(peer, np, self, ctr);
}
bool addn_rmsnorm_q(float * x, const p2p_ptrs & parts, int np, const float * w, float * y, void * qbuf, int T, int n,
                    float eps, cudaStream_t s) {
    return addn_rmsnorm_q_wait_ctr(x, parts, np, 0, nullptr, nullptr, w, y, qbuf, T, n, eps, s);
}
bool addn_rmsnorm_q_wait_ctr(float * x, const p2p_ptrs & parts, int np, int self, const int * flag, const int * ctr,
                             const float * w, float * y, void * qbuf, int T, int n, float eps, cudaStream_t s) {
    if (T <= 0) return true;
    bool ok = (np == 3 || np == 4) && self >= 0 && self < np && (n % 32 == 0) && n <= 4096 * 4 && aligned16(x) &&
              aligned16(y) && (!w || aligned16(w)) && qbuf && (!flag == !ctr);
    for (int q = 0; ok && q < np; ++q) ok = parts.p[q] && aligned16(parts.p[q]);
    if (!ok) return false;
    if (np == 3) launch_normn<3>(x, parts, w, y, qbuf, T, n, eps, flag, ctr, self, s);
    else         launch_normn<4>(x, parts, w, y, qbuf, T, n, eps, flag, ctr, self, s);
    return true;
}

bool ew_quant(ew_op op, const float * a, const float * b, float * y, void * qbuf, int T, int n, cudaStream_t s,
              int lda, int ldb) {
    if (T <= 0 || n <= 0) return true;
    if (lda <= 0) lda = n;
    if (ldb <= 0) ldb = n;
    if (n % 32 || !aligned16(a) || !aligned16(b) || !aligned16(y) || !qbuf || lda < n || ldb < n || lda % 4 || ldb % 4) return false;
    // y may alias a only when dense (a later token's input row would be overwritten otherwise)
    if (T > 1 && y == a && lda != n) return false;
    // threads per block (multiple of 32; P100E_EW_THR to experiment): small blocks spread an 8704-wide row over more SMs
    // (anything else, 0 included, would divide by zero or split a warp under a16_quant8's full-mask shuffles: 128)
    static const int thr = [] {
        const int v = getenv("P100E_EW_THR") ? atoi(getenv("P100E_EW_THR")) : 128;
        return v >= 32 && v <= 1024 && v % 32 == 0 ? v : 128;
    }();
    const dim3 grid((unsigned) ((n / 4 + thr - 1) / thr), (unsigned) T);
    if (op == EW_SILU_MUL) ew_q_k<EW_SILU_MUL><<<grid, thr, 0, s>>>(a, b, y, qbuf, n, lda, ldb);
    else                   ew_q_k<EW_SIGMOID_MUL><<<grid, thr, 0, s>>>(a, b, y, qbuf, n, lda, ldb);
    return true;
}

bool attn_prep(const float * qg, const float * k, const float * v, const float * q_norm, const float * k_norm,
               const int32_t * pos, float * q, float * gate, kv_cache & c, int p0, int T, int Hq, int Hkv, int D,
               int n_rot, float freq_base, float eps, cudaStream_t s, int ldq, int ldk, int ldv, const mrope_pos & mr) {
    if (T <= 0) return true;
    if (ldq <= 0) ldq = Hq * 2 * D;
    if (ldk <= 0) ldk = Hkv * D;
    if (ldv <= 0) ldv = Hkv * D;
    if (ldq < Hq * 2 * D || ldk < Hkv * D || ldv < Hkv * D || ldq % 4 || ldk % 4) return false;
    if ((c.type != KV_F16 && c.type != KV_Q8_0 && c.type != KV_Q4_0) || !c.k || c.D != D || c.Hkv != Hkv || D % 128 ||
        D > 1024 || n_rot > D || n_rot % 2 ||
        !aligned16(qg) || !aligned16(k) || !aligned16(q) || !aligned16(gate) || (q_norm && !aligned16(q_norm)) ||
        (k_norm && !aligned16(k_norm)) || (p0 >= 0 && p0 + T > c.n_ctx))
        return false;
    const int thr = ((D / 4) + 31) & ~31;
    const size_t rowb = kv_row_bytes(c.type, Hkv, D);
    const dim3 grid((unsigned) (Hq + 2 * Hkv), (unsigned) T);
    const size_t smem = (size_t) D * sizeof(float);
    const kv_pg pg = kv_pg_of(c);
    if (mr.p) {   // interleaved M-RoPE positions (images in the context)
        if (mr.sect <= 0) return false;
        if (c.type == KV_Q8_0)
            attn_prep_k<KV_Q8_0, true><<<grid, thr, smem, s>>>(qg, k, v, q_norm, k_norm, pos, q, gate, (char *) c.k,
                                                               (char *) c.v, rowb, p0, Hq, Hkv, D, n_rot, freq_base, eps,
                                                               ldq, ldk, ldv, pg, mr);
        else if (c.type == KV_Q4_0)
            attn_prep_k<KV_Q4_0, true><<<grid, thr, smem, s>>>(qg, k, v, q_norm, k_norm, pos, q, gate, (char *) c.k,
                                                               (char *) c.v, rowb, p0, Hq, Hkv, D, n_rot, freq_base, eps,
                                                               ldq, ldk, ldv, pg, mr);
        else
            attn_prep_k<KV_F16, true><<<grid, thr, smem, s>>>(qg, k, v, q_norm, k_norm, pos, q, gate, (char *) c.k,
                                                              (char *) c.v, rowb, p0, Hq, Hkv, D, n_rot, freq_base, eps,
                                                              ldq, ldk, ldv, pg, mr);
        return true;
    }
    if (c.type == KV_Q8_0)
        attn_prep_k<KV_Q8_0><<<grid, thr, smem, s>>>(qg, k, v, q_norm, k_norm, pos, q, gate, (char *) c.k, (char *) c.v, rowb,
                                                     p0, Hq, Hkv, D, n_rot, freq_base, eps, ldq, ldk, ldv, pg);
    else if (c.type == KV_Q4_0)
        attn_prep_k<KV_Q4_0><<<grid, thr, smem, s>>>(qg, k, v, q_norm, k_norm, pos, q, gate, (char *) c.k, (char *) c.v, rowb,
                                                     p0, Hq, Hkv, D, n_rot, freq_base, eps, ldq, ldk, ldv, pg);
    else
        attn_prep_k<KV_F16><<<grid, thr, smem, s>>>(qg, k, v, q_norm, k_norm, pos, q, gate, (char *) c.k, (char *) c.v, rowb,
                                                    p0, Hq, Hkv, D, n_rot, freq_base, eps, ldq, ldk, ldv, pg);
    return true;
}

void rmsnorm(const float * x, const float * w, float * y, int64_t rows, int n, float eps, cudaStream_t s) {
    dispatch<false, false, false>(const_cast<float *>(x), nullptr, w, nullptr, y, nullptr, rows, n, eps, s);
}
void add_rmsnorm(float * x, const float * r, const float * w, float * y, int T, int n, float eps, cudaStream_t s) {
    dispatch<true, false, false>(x, r, w, nullptr, y, nullptr, T, n, eps, s);
}
void rmsnorm_h(const float * x, const float * w, float * y, __half * yh, int64_t rows, int n, float eps, cudaStream_t s) {
    dispatch<false, false, true>(const_cast<float *>(x), nullptr, w, nullptr, y, yh, rows, n, eps, s);
}
void add_rmsnorm_h(float * x, const float * r, const float * w, float * y, __half * yh, int T, int n, float eps, cudaStream_t s) {
    dispatch<true, false, true>(x, r, w, nullptr, y, yh, T, n, eps, s);
}
void rmsnorm_gated(const float * x, const float * w, const float * z, float * y, int64_t rows, int n, float eps, cudaStream_t s) {
    dispatch<false, true, false>(const_cast<float *>(x), nullptr, w, z, y, nullptr, rows, n, eps, s);
}

void silu_mul(float * g, const float * u, int64_t n, cudaStream_t s) {
    if (n <= 0) return;
    if (aligned16(g) && aligned16(u)) silu_mul_k<<<ew_blocks(n >> 2), 256, 0, s>>>(g, u, n, n >> 2);
    else silu_mul_s<<<ew_blocks(n), 256, 0, s>>>(g, u, n);
}
void sigmoid_mul(float * x, const float * g, int64_t n, cudaStream_t s) {
    if (n <= 0) return;
    if (aligned16(x) && aligned16(g)) sigmoid_mul_k<<<ew_blocks(n >> 2), 256, 0, s>>>(x, g, n, n >> 2);
    else sigmoid_mul_s<<<ew_blocks(n), 256, 0, s>>>(x, g, n);
}

} // namespace p100e::fast
