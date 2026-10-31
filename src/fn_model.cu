// Qwen3.8-Flash-Next (qwen4exp) forward on 2 x P100: see fn_model.h. v0 = correctness build:
// batched over T rows (T = 1 decode, T <= FN_MAX_T prefill chunk), dense weights replicated per GPU, experts split by id.
#include "fn_model.h"
#include "fn_cpu_pool.h"
#include "ram_guard.h"
#include "fn_rexp.h"
#include "fn_hc_kernels.cuh"
#include <cuda_fp16.h>
#include "kernels.cuh"
#include "fn_qsa.cuh"
#include "fast/gemm.cuh"
#include "fast/ssm.cuh"
#include "fast/matvec_internal.cuh"
#include "../third_party/ggml/convert.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#ifdef _WIN32
// PrefetchVirtualMemory (kernel32, Windows 8+) without <windows.h> (it defines `small`, a method name here)
extern "C" {
typedef struct _WIN32_MEMORY_RANGE_ENTRY { void * VirtualAddress; size_t NumberOfBytes; } WIN32_MEMORY_RANGE_ENTRY;
__declspec(dllimport) int __stdcall PrefetchVirtualMemory(void * hProcess, size_t NumberOfEntries, WIN32_MEMORY_RANGE_ENTRY * VirtualAddresses, unsigned long Flags);
__declspec(dllimport) void * __stdcall GetCurrentProcess(void);
typedef struct MSX_ { unsigned long dwLength, dwMemoryLoad; unsigned long long ullTotalPhys, ullAvailPhys, ullTotalPageFile, ullAvailPageFile,
                      ullTotalVirtual, ullAvailVirtual, ullAvailExtendedVirtual; } MSX_;   // MEMORYSTATUSEX (fn pipe byte plan)
__declspec(dllimport) int __stdcall GlobalMemoryStatusEx(MSX_ * lpBuffer);
}
#else
#include <sys/mman.h>   // madvise (PLE row readahead, P100E_FN_PLE_ASYNC)
#include <unistd.h>
#endif
#include "pdl/pdl_link.h"   // net_transport
#include <climits>
#include <numeric>
#include <stdexcept>

namespace p100e {

struct fn_model_expert_set_alloc_tag;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error( \
    std::string("CUDA ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)
#define CB(x) do { cublasStatus_t e_ = (x); if (e_ != CUBLAS_STATUS_SUCCESS) throw std::runtime_error( \
    std::string("cuBLAS error ") + std::to_string((int) e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

static std::string bn(int il, const char * suffix) { return "blk." + std::to_string(il) + "." + suffix; }

// ---------------------------------------------------------------------------------------------------------------------
// glue kernels (f32, exact forms of the llama.cpp reference ops; namespace fnk)
namespace fnk {

static __device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
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
static __device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }
static __device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }

// y[row] = rmsnorm(x[row]) * w[row % nw_rows] : per-row weights (the hyper-connection grouped norm: HC rows of E, gamma [HC][E])
__global__ void rmsnorm_rows_k(const float * x, const float * w, float * y, int ncols, int nw_rows, float eps) {
    const int64_t off = (int64_t) blockIdx.x * ncols;
    const float * wr = w + (int64_t) (blockIdx.x % nw_rows) * ncols;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) ss += x[off + i] * x[off + i];
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / ncols + eps);
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) y[off + i] = x[off + i] * sc * wr[i];
}
// MTP block input: out[(t, c)][0..E) = e[t], out[(t, c)][E..2E) = h[(t, c)]  (concat(enorm(embed), hnorm(h)) per hc stream)
__global__ void mtp_concat_k(const float * e, const float * h, float * out, int E, int HC) {
    const int row = blockIdx.y, t = row / HC;
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < E) out[(size_t) row * 2 * E + i] = e[(size_t) t * E + i];
    else if (i < 2 * E) out[(size_t) row * 2 * E + i] = h[(size_t) row * E + (i - E)];
}
// rmsnorm(x) * w * sigmoid(z) per row (delta-net output gate, Flash-Next: sigmoid, the 27B: silu)
__global__ void rmsnorm_sig_gated_k(const float * x, const float * w, const float * z, float * y, int ncols, float eps) {
    const int64_t off = (int64_t) blockIdx.x * ncols;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) ss += x[off + i] * x[off + i];
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / ncols + eps);
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) y[off + i] = x[off + i] * sc * w[i] * sigm(z[off + i]);
}
// lo = silu(lo * s)
__global__ void silu_scale_k(float * x, float s, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = silu(x[i] * s);
}
// hc_mix tail: mixed[t][i] = (((g0 + g1) + g2) + g3) * (1/HC), g_s = xn[t][s][i] * sigmoid(gate[t][s][i])
__global__ void hc_mixed_k(const float * xn, const float * gate, float * mixed, int E, int HC) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    const int64_t base = (int64_t) t * HC * E;
    float acc = 0.0f;
    for (int s = 0; s < HC; ++s) acc += xn[base + (int64_t) s * E + i] * sigm(gate[base + (int64_t) s * E + i]);
    mixed[(int64_t) t * E + i] = acc * (1.0f / (float) HC);
}
// inject[t][s] = sum_j w[s][j] * xn[t][j]  (w [HC][HC*E] f32, xn [T][HC*E])
__global__ void dot_rows_k(const float * w, const float * xn, float * out, int n) {
    const int s = blockIdx.x, t = blockIdx.y;
    const float * wr = w + (int64_t) s * n;
    const float * xr = xn + (int64_t) t * n;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += wr[i] * xr[i];
    acc = block_sum(acc);
    if (threadIdx.x == 0) out[(int64_t) t * gridDim.x + s] = acc;
}
// res[t][s][i] += out[t][i] * (2 * sigmoid(inject[t][s] / HC))
__global__ void hc_combine_k(float * res, const float * out, const float * inject, int E, int HC) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    const float o = out[(int64_t) t * E + i];
    for (int s = 0; s < HC; ++s) {
        const float w = 2.0f * sigm(inject[(int64_t) t * HC + s] * (1.0f / (float) HC));
        res[((int64_t) t * HC + s) * E + i] += o * w;
    }
}
// res[t][s][i] = x[t][i] for every stream (hc_init)
__global__ void hc_init_k(const float * x, float * res, int E, int HC) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    for (int s = 0; s < HC; ++s) res[((int64_t) t * HC + s) * E + i] = x[(int64_t) t * E + i];
}
// y[t][i] += a[t] * x[t][i]  (MoE weighted accumulate; a = per-row weight)
__global__ void axpy_rows_k(float * y, const float * x, const float * a, int n) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[(int64_t) t * n + i] += a[t] * x[(int64_t) t * n + i];
}
// y[t][i] += x[t][i] * sigmoid(g[t])  (shared expert gate)
__global__ void add_sig_rows_k(float * y, const float * x, const float * g, int n) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[(int64_t) t * n + i] += x[(int64_t) t * n + i] * sigm(g[t]);
}
// gather rows: dst[j] = src[idx[j]]  (n floats per row)
__global__ void gather_rows_k(const float * src, const int32_t * idx, float * dst, int n) {
    const int j = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[(int64_t) j * n + i] = src[(int64_t) idx[j] * n + i];
}
// scatter-add weighted rows: dst[idx[j]] += w[j] * src[j]  (rows of one expert's outputs back to their tokens; the token rows
// of one launch are distinct, so no atomics are needed)
__global__ void scatter_add_rows_k(float * dst, const float * src, const int32_t * idx, const float * w, int n) {
    const int j = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[(int64_t) idx[j] * n + i] += w[j] * src[(int64_t) j * n + i];
}
// MoE router: block t = token t, 512 threads = experts: p = softmax(logits), top-K by p (ties: lower id), w = p / sum(top-K
// p) * scale. Same selection rule as the host version (stable sort by p descending); argmax K times with (p, -id) order.
template <int NE>
__global__ void __launch_bounds__(NE) route_k(const float * logits, int32_t * ids, float * wts, int K, float scale) {
    __shared__ float sp[NE];
    __shared__ float red[32];
    __shared__ int   redi[32];
    const int t = blockIdx.x, e = threadIdx.x, lane = e & 31, warp = e >> 5;
    float v = logits[(size_t) t * NE + e];
    // max
    float m = v;
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    if (warp == 0) { m = lane < NE / 32 ? red[lane] : -INFINITY; for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o)); if (lane == 0) red[0] = m; }
    __syncthreads();
    const float mx = red[0];
    __syncthreads();
    float pe = expf(v - mx);
    float su = pe;
    for (int o = 16; o; o >>= 1) su += __shfl_xor_sync(0xffffffffu, su, o);
    if (lane == 0) red[warp] = su;
    __syncthreads();
    if (warp == 0) { su = lane < NE / 32 ? red[lane] : 0.0f; for (int o = 16; o; o >>= 1) su += __shfl_xor_sync(0xffffffffu, su, o); if (lane == 0) red[0] = su; }
    __syncthreads();
    pe /= red[0];
    sp[e] = pe;
    __syncthreads();
    float ws = 0.0f;
    for (int k = 0; k < K; ++k) {
        // argmax of sp with ties to the lower id: warp reduce then block reduce
        float bv = sp[e]; int bi = e;
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { red[warp] = bv; redi[warp] = bi; }
        __syncthreads();
        if (warp == 0) {
            bv = lane < NE / 32 ? red[lane] : -INFINITY; bi = lane < NE / 32 ? redi[lane] : 0x7fffffff;
            for (int o = 16; o; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) { red[0] = bv; redi[0] = bi; }
        }
        __syncthreads();
        const int sel = redi[0]; const float selp = red[0];
        if (e == 0) { ids[(size_t) t * K + k] = sel; wts[(size_t) t * K + k] = selp; }
        ws += selp;
        if (e == sel) sp[e] = -1.0f;   // taken
        __syncthreads();
    }
    if (e < K) { float w = wts[(size_t) t * K + e] / ws; if (scale != 0.0f) w *= scale; wts[(size_t) t * K + e] = w; }
}
// causal softmax over score rows: s[t][j] (ld = n), valid j <= p0 + t, scaled by `scale`; invalid -> 0
__global__ void causal_softmax_k(float * s, int n, int p0, float scale) {
    const int t = blockIdx.x;
    float * row = s + (size_t) t * n;
    const int nv = p0 + t + 1;
    __shared__ float red[32];
    float m = -INFINITY;
    for (int j = threadIdx.x; j < nv; j += blockDim.x) m = fmaxf(m, row[j] * scale);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) { m = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : -INFINITY; for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o)); if (threadIdx.x == 0) red[0] = m; }
    __syncthreads();
    m = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (int j = threadIdx.x; j < nv; j += blockDim.x) { const float e = expf(row[j] * scale - m); row[j] = e; sum += e; }
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) { sum = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.0f; for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o); if (threadIdx.x == 0) red[0] = sum; }
    __syncthreads();
    const float inv = 1.0f / red[0];
    for (int j = threadIdx.x; j < n; j += blockDim.x) row[j] = j < nv ? row[j] * inv : 0.0f;
}
__global__ void f32_to_f16_k(const float * src, half * dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}
// combine the K gathered rows of token t (rowof[t][k] = row r or -1): out[t] = sum_k w[r] * src[r], fixed k order
__global__ void combine_rows_k(float * out, const float * src, const int32_t * rowof, const float * w, int n, int K) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const int r = rowof[t * K + k];
        if (r >= 0) acc = __fmaf_rn(w[r], src[(int64_t) r * n + i], acc);
    }
    out[(int64_t) t * n + i] = acc;
}
// PLE gate: g[t][s] = sigmoid(sign(d) * sqrt(max(|d|, 1e-6))), d = (sum_i key[t][s][i] * query[t][s][i]) / sqrt(E)
__global__ void ple_gate_k(const float * key, const float * query, float * g, int E) {
    const int s = blockIdx.x, t = blockIdx.y;
    const int64_t off = ((int64_t) t * gridDim.x + s) * E;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) acc += key[off + i] * query[off + i];
    acc = block_sum(acc);
    if (threadIdx.x == 0) {
        const float d = acc * rsqrtf((float) E);
        const float mag = sqrtf(fmaxf(fabsf(d), 1e-6f));
        const float sg = d > 0.0f ? 1.0f : (d < 0.0f ? -1.0f : 0.0f);
        g[(int64_t) t * gridDim.x + s] = sigm(sg * mag);
    }
}
// gated[t][s][i] = value[t][i] * g[t][s]
__global__ void ple_gated_k(const float * value, const float * g, float * gated, int E, int HC) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    for (int s = 0; s < HC; ++s) gated[((int64_t) t * HC + s) * E + i] = value[(int64_t) t * E + i] * g[(int64_t) t * HC + s];
}
// PLE dilated depthwise conv over tokens for one token row t (in order; the history holds the previous hist rows,
// oldest first, and this row's normalized vector is appended before the taps are read):
//   conv_out[t][c] = silu( sum_k w[c*K + k] * X[(t + hist) - (K-1-k)*dil][c] ), then res[t][c] += gated[t][c] + conv_out[t][c]
// X = hist rows followed by the T normalized rows (xall [hist + T][n]); the caller shifts the history afterwards.
__global__ void ple_conv_k(const float * xall, const float * w, const float * gated, float * res, int n, int hist, int K, int dil, int T) {
    const int t = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const int row = t + hist - (K - 1 - k) * dil;
        acc += w[(int64_t) c * K + k] * xall[(int64_t) row * n + c];
    }
    const float o = silu(acc);
    res[(int64_t) t * n + c] += gated[(int64_t) t * n + c] + o;
}

} // namespace fnk

static unsigned nb256(int64_t n) { return (unsigned) ((n + 255) / 256); }
static int nthreads_for(int ncols) { return ncols >= 1024 ? 1024 : ((ncols + 31) / 32) * 32; }

static void dequant_f32(uint32_t type, const void * src, float * dst, int64_t n, cudaStream_t s) {
    if (type == GGML_TYPE_F32) { CK(cudaMemcpyAsync(dst, src, n * 4, cudaMemcpyDeviceToDevice, s)); return; }
    to_fp32_cuda_t f = ggml_get_to_fp32_cuda((ggml_type) type);
    if (!f) throw std::runtime_error(std::string("no dequant for type ") + get_type_info(type).name);
    f(src, dst, n, s);
}

// ---------------------------------------------------------------------------------------------------------------------
// ---- CPU expert pool state (P100E_FN_CPU=1, fn_cpu_pool.h) -----------------------------------------------------------
static_assert(FNCPU_Q4R == fast::LAYOUT_Q4R && FNCPU_Q4KR == fast::LAYOUT_Q4KR && FNCPU_IQ2SR == fast::LAYOUT_IQ2SR && FNCPU_Q20 == fast::LAYOUT_Q20,
              "fn_cpu_pool layout ids");
struct fn_model::cpu_moe {
    std::unique_ptr<fn_cpu_pool> pool;
    int maxrows = 8;          // P100E_FN_CPU_MAXROWS: host experts with more routed rows than this stay on the PCIe path
    float pcie = 0.0f;        // P100E_FN_CPU_PCIE: share of the CPU-eligible experts (the most-routed ones) still staged over PCIe
    bool stats = false;       // P100E_FN_CPU_STATS=1 (or verbose): totals at exit
    float * x = nullptr;      // pinned [max_t][E]: rank 0's FFN input rows (route() copies them)
    struct rank {
        std::vector<char> is_cpu;          // [NEXP] this layer's CPU experts of this rank
        float * y[2] = {}; size_t ycap[2] = {};   // pinned CPU output rows [rows][E] (two sets, alternate layers)
        cudaEvent_t ev[2] = {};            // the H2D that read y[k] is done
        bool ev_rec[2] = {};
        int nrows = 0;                     // CPU rows this layer
        const float * cur = nullptr; float * out = nullptr; int nrg = 0, nr = 0;   // deferred combine (ffn_grouped's row counts)
    };
    std::vector<rank> rk;
    int flip = 0;
    bool layer_on = false;
    std::vector<int32_t> tok;   // routed rows' tokens of every job
    std::vector<fn_cpu_job> jobs;
    int64_t n_layers = 0, n_exp = 0, n_rows = 0, n_gpu_host = 0;
    double t_wait = 0.0;
    std::vector<int> devs;
    ~cpu_moe();
};
// remote expert tier state (the code is at the end of this file)
struct fn_model::rexp_moe {
    rexp_sock k;
    std::vector<int> li_of;                 // [NL_] served layer index (-1 = not served)
    std::vector<std::vector<char>> held;    // [NL_][NEXP] node 1 holds the expert
    std::vector<std::vector<int>> want;     // [NL_] the list sent (best first)
    std::vector<int> caps;                  // node 1's slots per layer per GPU
    std::vector<std::string> gpu_names;
    float * x = nullptr;                    // pinned [max_t][E]: rank 0's FFN input rows (route() copies them)
    float * y[2] = {}; size_t ycap[2] = {}; // pinned reply rows (two sets, alternate layers)
    float * ones = nullptr;                 // pinned [max_pairs] 1.0f (sum mode weights)
    std::vector<char> sendbuf;
    struct rank { std::vector<char> is_rem; int nrows = 0, off = 0; const float * cur = nullptr; float * out = nullptr; int nrg = 0, nr = 0;
                  cudaEvent_t ev[2] = {}; bool ev_rec[2] = {}; };
    std::vector<rank> rk;
    std::vector<int> pair_tok;              // this layer's pairs: token row of each (rank by rank, expert ascending)
    std::vector<int> tok_x;                 // token -> x row (-1)
    int nx = 0, npairs = 0, flip = 0;
    bool layer_on = false, sum = false;
    uint64_t seq = 0;
    int timeout_ms = 120000;
    std::vector<int> devs;
    // stats
    int64_t n_req = 0, n_pairs = 0, n_rows_routed = 0, bytes_out = 0, bytes_in = 0;
    double t_wait = 0, t_wait_max = 0;
    std::map<int, std::pair<int64_t, int64_t>> by_T;   // forward rows -> (requests, pairs)
    struct hit { int64_t fwd = 0, local = 0, remote = 0, miss = 0; };
    std::map<int, hit> hits_T;              // forward rows -> routed rows resident here / on node 1 / neither (host path)
    // node 1's set follows the routing counts (adaptive files): swaps in flight, stats
    std::vector<std::vector<char>> pending; // [NL_][NEXP] a swap to this expert was sent, not landed yet
    std::vector<std::vector<char>> swap_ok; // [NL_][NEXP] held on a GPU of node 1 that takes swaps
    std::vector<int> il_of;                 // served layer index -> il
    bool adapt = false;
    int adapt_max = 4, adapt_cap = 96;      // swaps per layer / per adapt step (P100E_FN_REXP_ADAPT_MAX / _CAP)
    int64_t swaps_sent = 0, swaps_landed = 0;
    bool early_shared = true;               // P100E_FN_REXP_EARLY_SHARED=0: the shared expert after the reply, as before
    int min_pairs = 0;                      // P100E_FN_REXP_MIN_PAIRS: layers with fewer remote pairs skip node 1
    int64_t n_skipped = 0;
    bool half = false;                      // P100E_FN_REXP_HALF=1 (sum mode): f16 reply rows
    std::vector<__half> hbuf;
    double t_send = 0, t_hdr = 0, t_body = 0;   // ms: request send, wait for the reply header, reply payload
    ~rexp_moe();
};

fn_model::fn_model(const std::string & path, const std::vector<int> & devices, const opts & o) : g_(path), o_(o) {
    const std::string a = g_.get_str("general.architecture", "");
    if (a != "qwen4exp") throw std::runtime_error("fn_model: unsupported architecture '" + a + "' (want qwen4exp)");
    auto gi = [&](const char * k, int64_t def) { return (int) g_.get_int(a + "." + k, def); };
    if (const char * e = getenv("P100E_FN_PIN")) o_.pin_host = atoi(e) != 0;
    if (const char * e = getenv("P100E_FN_STAGE_MB")) o_.stage_mb = std::max(8, atoi(e));
    if (const char * e = getenv("P100E_FN_PREFETCH_T")) o_.prefetch_min_t = atoi(e);
    if (const char * e = getenv("P100E_FN_TP")) o_.tp = atoi(e) != 0;
    if (const char * e = getenv("P100E_FN_ADAPT")) o_.adapt = atoi(e) != 0;
    prof_hc_ = getenv("P100E_FN_PROF_HC") && atoi(getenv("P100E_FN_PROF_HC")) != 0;
    if (const char * e = getenv("P100E_FN_ADAPT_CAP")) adapt_cap_ = std::max(0, atoi(e));
    if (const char * e = getenv("P100E_FN_ADAPT_EVERY")) o_.adapt_every = std::max(1, atoi(e));
    if (const char * e = getenv("P100E_FN_ADAPT_MAX")) o_.adapt_max = std::max(0, atoi(e));
    if (const char * e = getenv("P100E_FN_ADAPT_MIN")) o_.adapt_min = (float) atof(e);
    E_ = gi("embedding_length", 0); L_ = gi("block_count", 0);
    H_ = gi("attention.head_count", 0); HKV_ = gi("attention.head_count_kv", 0); HD_ = gi("attention.key_length", 0);
    NROT_ = gi("rope.dimension_count", 0);
    HK_ = gi("ssm.group_count", 0); HV_ = gi("ssm.time_step_rank", 0); DK_ = gi("ssm.state_size", 0);
    DV_ = gi("ssm.inner_size", 0) / HV_;
    CONV_C_ = 2 * HK_ * DK_ + HV_ * DV_;
    NEXP_ = gi("expert_count", 0); NEXPU_ = gi("expert_used_count", 0); FFE_ = gi("expert_feed_forward_length", 0);
    FFS_ = gi("expert_shared_feed_forward_length", 0);
    HC_ = gi("hyper_connection.count", 4); HCLR_ = gi("hyper_connection.low_rank", 0); HCDIM_ = HC_ * E_;
    eps_ = (float) g_.get_float(a + ".attention.layer_norm_rms_epsilon", 1e-6);
    rope_base_ = (float) g_.get_float(a + ".rope.freq_base", 1e7);
    if (const gguf_value * v = g_.get(a + ".rope.dimension_sections"); v && v->arr_num.size() >= 3) {   // interleaved M-RoPE (images)
        msect_ = 0;
        for (size_t k = 0; k < v->arr_num.size(); ++k) { if (k < 3) msec_[k] = (int) v->arr_num[k]; msect_ += (int) v->arr_num[k]; }
    }
    for (auto & v : rp_) v.assign((size_t) o.n_ctx + MTP_ROWS, 0);
    exp_scale_ = (float) g_.get_float(a + ".expert_weights_scale", 0.0);
    full_attn_interval_ = gi("full_attention_interval", 4);
    // QSA: indexer hparams + the compress ratio of the attention layers (one ratio for all of them)
    IDX_H_ = gi("attention.indexer.head_count", 0); IDX_D_ = gi("attention.indexer.key_length", 0); IDX_TOPK_ = gi("attention.indexer.top_k", 0);
    if (const gguf_value * v = g_.get(a + ".attention.compress_ratios"); v && !v->arr_num.empty()) {
        for (size_t il = 0; il < v->arr_num.size(); ++il) {
            const int r = (int) v->arr_num[il];
            if (r > 0 && RATIO_ == 0) RATIO_ = r;
            if (r > 0 && r != RATIO_) throw std::runtime_error("fn_model: mixed QSA compress ratios");
        }
    }
    qsa_ = IDX_H_ > 0 && IDX_D_ > 0 && IDX_TOPK_ > 0 && RATIO_ > 0 && !(getenv("P100E_FN_QSA") && atoi(getenv("P100E_FN_QSA")) == 0);
    QSA_W_ = qsa_ ? IDX_TOPK_ + RATIO_ - 1 : 0;
    {
        const char * kv = getenv("P100E_FN_KV");
        const std::string m = kv ? kv : "";
        kv_mode_ = m == "f32" ? 0 : m == "f16" ? 1 : m == "q8" ? 2 : (o.n_ctx > 32768 ? 2 : 1);
    }
    if (qsa_ && IDX_D_ != 128) throw std::runtime_error("fn_model: indexer head dim 128 expected");
    VOCAB_ = (int) g_.find("token_embd.weight")->ne[1];
    if (DK_ != DV_ || DK_ != 128) throw std::runtime_error("fn_model: delta-net state 128 expected");
    const int n_nextn = (int) g_.get_int(a + ".nextn_predict_layers", 0);
    if (n_nextn > 0) L_ -= n_nextn;
    const int L_file = L_;
    if (o.layer_limit > 0 && o.layer_limit < L_) { printf("fn_model: DEBUG layer limit %d of %d\n", o.layer_limit, L_); L_ = o.layer_limit; }
    // the draft block (MTP) is blk.<L_> when present: loaded unless off (P100E_FN_MTP=0) or a layer limit is active
    // (debug opts::mtp_with_limit keeps it under a limit: logical layer L_, file blk.<L_file>)
    const bool lim_ok = o.layer_limit <= 0 || L_ == L_file || o.mtp_with_limit;
    mtp_il_ = (n_nextn == 1 && o.mtp && lim_ok && !(getenv("P100E_FN_MTP") && atoi(getenv("P100E_FN_MTP")) == 0)
               && g_.find(bn(L_file, "nextn.eh_proj.weight"))) ? L_ : -1;
    if (mtp_il_ >= 0 && L_ != L_file) { mtp_file_il_ = L_file; printf("fn_model: DEBUG draft block blk.%d as layer %d\n", L_file, L_); }
    NL_ = L_ + (mtp_il_ >= 0 ? 1 : 0);
    // PLE
    PLE_LAYER_ = -1; PLE_HEADS_ = 0;
    if (const gguf_value * v = g_.get(a + ".ple.layers"); v && !v->arr_num.empty()) {
        PLE_LAYER_ = (int) v->arr_num[0];
        PLE_NGRAM_ = gi("ple.ngram_size", 3); PLE_PER_ = gi("ple.heads_per_ngram", 8); PLE_KERN_ = gi("ple.conv_kernel", 4);
        PLE_HEADS_ = (PLE_NGRAM_ - 1) * PLE_PER_;
        PLE_DIM_ = gi("embedding_length_per_layer_input", 0);
        PLE_HIST_ = (PLE_KERN_ - 1) * PLE_NGRAM_;
        ple_eos_ = g_.get_int(a + ".ple.eos_token_id", 0);
        img_tok_ = (int32_t) g_.get_int(a + ".ple.image_token_id", 0);
        if (img_tok_ == 0) img_tok_ = (int32_t) ple_eos_;
        for (double d : g_.get(a + ".ple.layer_multipliers")->arr_num) ple_mult_.push_back((uint64_t) d);
        for (double d : g_.get(a + ".ple.head_offsets")->arr_num) ple_off_.push_back((int64_t) d);
        for (double d : g_.get(a + ".ple.head_vocab_sizes")->arr_num) ple_vocab_.push_back((int64_t) d);
        if ((int) ple_mult_.size() < PLE_NGRAM_ || (int) ple_off_.size() < PLE_HEADS_ || PLE_HEADS_ * PLE_DIM_ != E_)
            throw std::runtime_error("fn_model: inconsistent PLE metadata");
    }
    if (const gguf_value * v = g_.get(a + ".expert_hot_count")) for (double d : v->arr_num) hot_count_.push_back((int32_t) d);
    if (hot_count_.empty() && !g_.find("blk.0.ffn_exp_map")) {
        // an unsplit file (every expert in ffn_*_exps, e.g. the all-Q1_0 export with --all-hot): all hot, identity map
        hot_count_.assign(NL_, NEXP_);
        if (o.verbose) printf("fn_model: unsplit expert file: all %d experts hot per layer\n", NEXP_);
    }
    if ((int) hot_count_.size() < NL_) throw std::runtime_error("fn_model: not an expert-split file (no expert_hot_count)");
    // rank_order values index [NEXP] arrays (load_experts, rexp_resident0, rexp_connect): kept only when every full layer
    // of it is a permutation of 0..NEXP-1 (an out-of-range double is dropped before the int conversion)
    if (const gguf_value * v = g_.get("expert_split.rank_order"))
        for (double d : v->arr_num) rank_order_.push_back(d >= 0 && d < NEXP_ ? (int32_t) d : -1);
    if (!rank_order_.empty() && (int64_t) rank_order_.size() < (int64_t) L_ * NEXP_) rank_order_.clear();
    for (size_t l0 = 0; l0 + NEXP_ <= rank_order_.size(); l0 += NEXP_) {
        std::vector<char> seen(NEXP_, 0);
        bool ok = true;
        for (int i = 0; i < NEXP_ && ok; ++i) { const int32_t e = rank_order_[l0 + i]; ok = e >= 0 && !seen[e]; if (ok) seen[e] = 1; }
        if (!ok) { printf("fn_model: expert_split.rank_order layer %zu is not a permutation of the experts: ignored\n", l0 / NEXP_); rank_order_.clear(); break; }
    }
    // ffn_exp_map shape (load_experts' check), for every file block before the planners and rexp_resident0 read it
    for (int il = 0; il <= NL_; ++il) {   // blocks 0..NL-1 and the draft block's file index (debug layer limit)
        const int b = il < NL_ ? il : mtp_file_il_;
        if (const gguf_tensor * m = b < 0 ? nullptr : g_.find(bn(b, "ffn_exp_map")); m && (m->type != GGML_TYPE_I32 || m->ne[0] != NEXP_ || m->ne[1] != 2))
            throw std::runtime_error("fn_model: bad ffn_exp_map " + std::to_string(b));
    }
    printf("fn_model: E %d L %d V %d heads %d/%d x %d  gdn %d/%d x %d  experts %d top %d ff %d (shared %d)  hc %d lr %d  ple layer %d heads %d x %d\n",
           E_, L_, VOCAB_, H_, HKV_, HD_, HK_, HV_, DK_, NEXP_, NEXPU_, FFE_, FFS_, HC_, HCLR_, PLE_LAYER_, PLE_HEADS_, PLE_DIM_);
    lb_ = 0; le_ = L_;   // every layer (pipe mode: pipe_init narrows it to this node's)
    if (o_.plan_only) return;
    if (!o_.rank_slots.empty()) {
        if (o_.rank_slots.size() != devices.size()) throw std::runtime_error("fn_model: rank_slots needs one value per device");
        slots_ = o_.rank_slots;
    }
    if (!o_.net_nodes.empty() || o_.pipe_layers != 0) pipe_init(devices);
    else {   // one process: the byte plan over every layer (auto slots resolved; refused when the pinned RAM would eat the reserve)
        std::vector<double> fr = gpu_free_per_rank(devices);
        node_plan p = plan_node(0, L_, true, fr, slots_, host_free_bytes());
        for (const rank_plan & q : p.rk) { plan_tot_.push_back(q.total); plan_free_.push_back(q.free); }
        if (o_.verbose || !slots_.empty() || !p.fits) print_plan("fn_model", 0, L_, p);
        for (size_t r = 0; r < slots_.size(); ++r) slots_[r] = p.rk[r].slots;
        if (p.host + p.reserve > p.host_free)
            throw std::runtime_error("fn_model: not enough RAM: " + std::to_string((long long) (p.host / 1048576)) + " MiB pinned needed, " +
                                     std::to_string((long long) (p.host_free / 1048576)) + " MiB available, reserve " +
                                     std::to_string((long long) (p.reserve / 1048576)) + " MiB (P100E_RAM_RESERVE_MB): fewer layers / slots, or free RAM");
    }
    rexp_connect((int) devices.size());   // remote expert tier: node 1 loads its experts while this node loads
    load(devices);
    cpu_init();
    rexp_ready();
}

fn_model::~fn_model() {
    if (prof_hc_ && hc_n_)
        printf("fn_model: hc_mix GPU time per call (rank 0, %lld calls): rmsnorm %.1f us, down matvec %.1f, silu_scale %.1f, up matvec %.1f, mixed %.1f, inject dot %.1f; total %.1f us\n",
               (long long) hc_n_, 1e3 * hc_sum_[0] / hc_n_, 1e3 * hc_sum_[1] / hc_n_, 1e3 * hc_sum_[2] / hc_n_, 1e3 * hc_sum_[3] / hc_n_, 1e3 * hc_sum_[4] / hc_n_,
               1e3 * hc_sum_[5] / hc_n_, 1e3 * (hc_sum_[0] + hc_sum_[1] + hc_sum_[2] + hc_sum_[3] + hc_sum_[4] + hc_sum_[5]) / hc_n_);
    if (pipe() && !pipe_worker()) { try { pipe_quit(); } catch (const std::exception & e) { fprintf(stderr, "fn pipe: quit: %s\n", e.what()); } }
    if (pipe_h_) cudaFreeHost(pipe_h_);
    cpu_.reset();   // CPU pool threads and buffers first
    rexp_.reset();  // remote expert tier: QUIT to node 1, stats
    if (ple_th_.joinable()) ple_th_.join();
    if (promo_thread_.joinable()) {
        { std::lock_guard<std::mutex> lk(promo_mu_); promo_stop_ = true; }
        promo_cv_.notify_all();
        promo_thread_.join();
        for (auto & S : rk_) { cudaSetDevice(S.dev); cudaDeviceSynchronize(); }
        for (auto & r : ring_) { if (r.buf) cudaFreeHost(r.buf); for (int k = 0; k < 2; ++k) if (r.done[k]) cudaEventDestroy(r.done[k]); }
    }
    if (o_.verbose) {
        for (auto & S : rk_) {
            int64_t f = 0, d = 0; int n = 0;
            for (auto & [il, st] : S.stage_stats) { f += st.fetches; d += (int64_t) st.seen.size(); ++n; }
            if (n) printf("fn_model: rank %d host-expert fetches %lld over %d layers, distinct experts %lld (%.1f per layer; a cache of that many slots per layer would have served every repeat)\n", (int) (&S - rk_.data()), (long long) f, n, (long long) d, (double) d / n);
            int64_t hits = 0, rows = 0, swaps = 0; bool ad = false;
            for (auto & L : S.lw) { hits += L.hits; rows += L.rows; swaps += L.swaps; ad = ad || L.adaptive; }
            if (ad) printf("fn_model: rank %d adaptive: resident hit rate %.1f%% (%lld of %lld routed rows), %lld swaps\n", (int) (&S - rk_.data()), rows ? 100.0 * hits / rows : 0.0, (long long) hits, (long long) rows, (long long) swaps);
        }
    }
    for (auto & S : rk_) {
        cudaSetDevice(S.dev);
        cudaDeviceSynchronize();
        for (auto & [n, p] : S.f) cudaFree(p);
        for (auto & w : S.dense) if (w.owns) cudaFree(w.d);
        for (auto & L : S.lw) {
            for (auto * es : {&L.hot_gate, &L.hot_up, &L.hot_down}) cudaFree(es->base);
            for (auto * es : {&L.cold_gate, &L.cold_up, &L.cold_down, &L.hhot_gate, &L.hhot_up, &L.hhot_down}) { if (!es->base || !es->owner) continue; if (es->pinned) cudaFreeHost(es->base); else free(es->base); }
        }
        for (float * p : {S.res, S.cur, S.xn, S.a, S.b, S.c, S.d, S.e, S.g, S.part, S.peer, S.inject, S.router_out, S.logits, S.w_scratch, S.ple_hist, S.ple_emb, S.gath, S.gout, S.gw})
            cudaFree(p);
        cudaFree(S.mv_scratch); cudaFree(S.qstage); cudaFree(S.pos); cudaFree(S.rpos); cudaFree(S.gidx); cudaFree(S.gtab); cudaFree(S.q2); cudaFree(S.qa); cudaFree(S.rowof);
        for (int k = 0; k < 2; ++k) if (S.tab_pin[k]) cudaFreeHost(S.tab_pin[k]);
        if (S.rowof_pin) cudaFreeHost(S.rowof_pin);
        if (S.hbounce && cudaFreeHost(S.hbounce) != cudaSuccess) { (void) cudaGetLastError(); free(S.hbounce); }
        cudaFree(S.route_ids); cudaFree(S.route_wts);
        for (int k = 0; k < 2; ++k) { if (S.pf[k]) cudaFree(S.pf[k]); if (S.pf_ready[k]) cudaEventDestroy(S.pf_ready[k]); if (S.pf_free[k]) cudaEventDestroy(S.pf_free[k]); }
        if (S.route_ids_pin) cudaFreeHost(S.route_ids_pin); if (S.route_wts_pin) cudaFreeHost(S.route_wts_pin);
        for (auto * v : {&S.conv_state, &S.ssm_state, &S.kblk, &S.ktail}) for (float * p : *v) if (p) cudaFree(p);
        for (auto * v : {&S.kc, &S.vc}) for (void * p : *v) if (p) cudaFree(p);
        for (void * p : {(void *) S.blk_rpos, (void *) S.qk_raw, (void *) S.qq, (void *) S.qsa_s, (void *) S.qsa_c, (void *) S.sel, (void *) S.seln, (void *) S.kf, (void *) S.vf}) if (p) cudaFree(p);
        if (S.cb) cublasDestroy(S.cb);
        if (S.ev) cudaEventDestroy(S.ev);
        if (S.ev2) cudaEventDestroy(S.ev2);
        for (int k = 0; k < 2; ++k) { if (S.st_ready[k]) cudaEventDestroy(S.st_ready[k]); if (S.st_free[k]) cudaEventDestroy(S.st_free[k]); }
        if (S.hidx_pin) cudaFreeHost(S.hidx_pin);
        if (S.hw_pin) cudaFreeHost(S.hw_pin);
        if (S.cs) cudaStreamDestroy(S.cs);
        if (S.s) cudaStreamDestroy(S.s);
    }
}

// one dense weight (or one expert slice of a 3-D expert tensor) uploaded raw and packed for the fast GEMV
static void load_dbg2(const char * what, size_t bytes) {   // P100E_FN_LOAD_DEBUG=2: VRAM after every dense upload
    static const bool on = getenv("P100E_FN_LOAD_DEBUG") && atoi(getenv("P100E_FN_LOAD_DEBUG")) >= 2;
    if (!on) return;
    size_t f = 0, t = 0; cudaMemGetInfo(&f, &t);
    printf("  upload %-34s %10.3f MiB, free %.1f MiB\n", what, bytes / 1048576.0, f / 1048576.0);
}
fn_model::dw fn_model::upload_weight(rank_state & S, const gguf_tensor * t, const char * what, int64_t expert, int64_t rows0, int64_t nrows) {
    if (!t) throw std::runtime_error(std::string("fn_model: missing tensor ") + what);
    const int64_t n_in = t->ne[0], n_out = nrows < 0 ? t->ne[1] : nrows;
    const size_t row_bytes = (size_t) tensor_nbytes(t->type, t->ne) / ((size_t) t->ne[1] * t->ne[2] * t->ne[3]);
    const uint8_t * src = t->data + (expert >= 0 ? (size_t) expert * row_bytes * (size_t) t->ne[1] : 0) + (size_t) rows0 * row_bytes;
    return upload_raw(S, t->type, src, n_in, n_out, what);
}

fn_model::dw fn_model::upload_rows(rank_state & S, const gguf_tensor * t, const char * what, const std::vector<std::pair<int64_t, int64_t>> & ranges) {
    if (!t) throw std::runtime_error(std::string("fn_model: missing tensor ") + what);
    const int64_t n_in = t->ne[0];
    const size_t row_bytes = (size_t) tensor_nbytes(t->type, t->ne) / ((size_t) t->ne[1] * t->ne[2] * t->ne[3]);
    int64_t n_out = 0;
    for (auto & [r0, nr] : ranges) n_out += nr;
    std::vector<uint8_t> buf(row_bytes * (size_t) n_out);
    size_t off = 0;
    for (auto & [r0, nr] : ranges) { g_.read_at(buf.data() + off, t->data + (size_t) r0 * row_bytes, row_bytes * (size_t) nr); off += row_bytes * (size_t) nr; }
    return upload_raw(S, t->type, buf.data(), n_in, n_out, what);
}

fn_model::dw fn_model::upload_cols(rank_state & S, const gguf_tensor * t, const char * what, const std::vector<std::pair<int64_t, int64_t>> & cols) {
    if (!t) throw std::runtime_error(std::string("fn_model: missing tensor ") + what);
    const int64_t n_in = t->ne[0], n_out = t->ne[1];
    int blck = 0, bsz = 0;
    if (!fast::type_info(t->type, blck, bsz)) throw std::runtime_error(std::string("fn_model: upload_cols: unknown type for ") + what);
    const size_t row_bytes = (size_t) tensor_nbytes(t->type, t->ne) / ((size_t) t->ne[1] * t->ne[2] * t->ne[3]);
    if (row_bytes != (size_t) (n_in / blck) * bsz) throw std::runtime_error(std::string("fn_model: upload_cols: row layout not block-linear for ") + what);
    int64_t ncols = 0;
    for (auto & cr : cols) { if (cr.first % blck || cr.second % blck) throw std::runtime_error(std::string("fn_model: upload_cols: column range not block aligned for ") + what); ncols += cr.second; }
    const size_t nb = (size_t) (ncols / blck) * bsz;
    std::vector<uint8_t> buf(nb * (size_t) n_out);
    std::vector<uint8_t> full(row_bytes * (size_t) n_out);   // the whole tensor in one ReadFile (per-row slices below)
    g_.read_at(full.data(), t->data, full.size());
    for (int64_t r = 0; r < n_out; ++r) {
        size_t o = 0;
        for (auto & cr : cols) { const size_t b0 = (size_t) (cr.first / blck) * bsz, b = (size_t) (cr.second / blck) * bsz; memcpy(buf.data() + (size_t) r * nb + o, full.data() + (size_t) r * row_bytes + b0, b); o += b; }
    }
    return upload_raw(S, t->type, buf.data(), ncols, n_out, what);
}

const float * fn_model::small_f32(rank_state & S, const std::string & key, const std::vector<float> & v) {
    auto it = S.f.find(key);
    if (it != S.f.end()) return it->second;
    float * d = nullptr;
    CK(cudaMalloc(&d, v.size() * 4));
    CK(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    S.f[key] = d;
    return d;
}

void fn_model::h2d(rank_state & S, void * dst, const uint8_t * src, size_t n) {
    if (!g_.mapped(src) || !S.hbounce) { CK(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, S.s)); return; }
    for (size_t off = 0; off < n; off += HBOUNCE_) {
        const size_t c = std::min(HBOUNCE_, n - off);
        g_.read_at(S.hbounce, src + off, c);
        CK(cudaMemcpyAsync((char *) dst + off, S.hbounce, c, cudaMemcpyHostToDevice, S.s));
        CK(cudaStreamSynchronize(S.s));   // the bounce buffer is reused by the next chunk
    }
}

fn_model::dw fn_model::upload_raw(rank_state & S, uint32_t type, const uint8_t * src, int64_t n_in, int64_t n_out, const char * what) {
    int blck = 1, bsz = 0;
    if (!fast::type_info(type, blck, bsz)) { const qtype_info & ti = get_type_info(type); blck = (int) ti.blck; bsz = (int) ti.size; }
    const size_t row_bytes = (size_t) (n_in / blck) * bsz;
    const size_t bytes = row_bytes * (size_t) n_out;
    dw w;
    const bool is_float = type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
    if (!is_float && n_in % 256 != 0) {
        // narrow rows (hc up: n_in 320, PLE: 160): no fast quantized GEMV; dequantize once to f16 (float_gemv<half> for
        // T <= 8, f16 -> f32 + cuBLAS above; gemm_pk)
        const int64_t n = n_in * n_out;
        if (bytes > S.qstage_bytes || (size_t) n * 4 > S.w_scratch_bytes) throw std::runtime_error(std::string("fn_model: narrow weight too large to dequantize: ") + what);
        CK(cudaMalloc(&w.d, (size_t) n * 2));
        h2d(S, S.qstage, src, bytes);
        dequant_f32(type, S.qstage, S.w_scratch, n, S.s);
        fnk::f32_to_f16_k<<<(unsigned) ((n + 255) / 256), 256, 0, S.s>>>(S.w_scratch, (half *) w.d, n);
        CK(cudaStreamSynchronize(S.s));   // qstage / w_scratch are reused by the next upload
        w.bytes = (size_t) n * 2;
        w.raw = false;
        w.pk = fast::pack_weight(GGML_TYPE_F16, w.d, n_in, n_out, S.s);
        if (!w.pk.data) throw std::runtime_error(std::string("fn_model: no f16 GEMV path for ") + what);
        S.dense.push_back(w);
        load_dbg2(what, (size_t) n * 2);
        return w;
    }
    CK(cudaMalloc(&w.d, bytes));
    h2d(S, w.d, src, bytes);
    w.bytes = bytes;
    w.raw = needs_raw(type, n_in);
    if (w.raw) { w.pk.type = type; w.pk.n_in = n_in; w.pk.n_out = n_out; w.pk.data = w.d; w.pk.bytes = bytes; w.pk.layout = fast::LAYOUT_RAW; }
    else {
        w.pk = fast::pack_weight(type, w.d, n_in, n_out, S.s);
        if (!w.pk.data) throw std::runtime_error(std::string("fn_model: no GEMV path for ") + what + " type " + get_type_info(type).name);
    }
    S.dense.push_back(w);
    load_dbg2(what, bytes);
    return w;
}

// types / shapes without a p100e GEMV kernel (IQ4_NL / IQ4_XS / MXFP4 rows need n_in % 256 in the generic kernel)
bool fn_model::needs_raw(uint32_t type, int64_t n_in) {
    if (!fast::has_fast_path(type)) return true;
    if ((type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_MXFP4) && n_in % 256) return true;
    return false;
}

// y[T][n_out] = W x: the fast GEMV (T <= 8), the fold GEMM, cuBLAS f32 for f32 rows, or the dequant + cuBLAS fallback
void fn_model::gemm_pk(rank_state & S, const fast::packed_weight & pk, bool raw, const float * x, float * y, int T) {
    const float one = 1.0f, zero = 0.0f;
    if (raw || (pk.type == GGML_TYPE_F16 && T > 8)) {
        const int64_t n = pk.n_in * pk.n_out;
        if ((size_t) n * 4 > S.w_scratch_bytes) throw std::runtime_error("fn_model: fallback weight larger than w_scratch");
        dequant_f32(pk.type, pk.data, S.w_scratch, n, S.s);
        CB(cublasSgemm(S.cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) pk.n_out, T, (int) pk.n_in, &one, S.w_scratch, (int) pk.n_in, x, (int) pk.n_in, &zero, y, (int) pk.n_out));
        return;
    }
    if (T <= 8) { fast::matvec(pk, x, y, T, S.mv_scratch, S.s); return; }
    if (pk.type == GGML_TYPE_F32) {
        CB(cublasSgemm(S.cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) pk.n_out, T, (int) pk.n_in, &one, (const float *) pk.data, (int) pk.n_in, x, (int) pk.n_in, &zero, y, (int) pk.n_out));
        return;
    }
    // large chunks: f16 cuBLAS with f32 accumulate (P100E_FN_GEMM_T = min rows, default 64; 0 = always the fold GEMM)
    static const int cublas_min_t = getenv("P100E_FN_GEMM_T") ? atoi(getenv("P100E_FN_GEMM_T")) : 0;   // P100: the fold GEMM beat cuBLAS fp16 (maxwell_sgemm_fp16, no tensor cores) by ~4%, so off by default
    if (cublas_min_t > 0 && T >= cublas_min_t && pk.type != GGML_TYPE_F16) {
        fast::gemm_packed_cublas32(pk, x, y, T, S.gemm_scratch, S.gemm_bytes, S.s, S.cb);
        return;
    }
    fast::gemm_packed(pk, x, y, T, S.gemm_scratch, S.gemm_bytes, S.s, S.cb);
}

const float * fn_model::small(rank_state & S, const std::string & name, bool required) {
    auto it = S.f.find(name);
    if (it != S.f.end()) return it->second;
    const gguf_tensor * t = g_.find(name);
    if (!t) { if (required) throw std::runtime_error("fn_model: missing tensor " + name); return nullptr; }
    float * d = nullptr;
    CK(cudaMalloc(&d, t->nelements() * 4));
    if (t->nbytes > S.qstage_bytes) throw std::runtime_error("fn_model: small tensor too large " + name);
    h2d(S, S.qstage, t->data, t->nbytes);
    dequant_f32(t->type, S.qstage, d, t->nelements(), S.s);
    CK(cudaStreamSynchronize(S.s));
    S.f[name] = d;
    load_dbg2(name.c_str(), (size_t) t->nelements() * 4);
    return d;
}

// this rank's share of the layer's experts: hot local ids [r*HOTR, (r+1)*HOTR) resident on the GPU (packed), every other
// hot/cold expert owned by this rank (cold local id % R == r; hot ids past R*HOTR also go here) packed in pinned host RAM
static size_t g_host_pinned_bytes = 0, g_host_pageable_bytes = 0;   // expert host memory by kind (reported at load end)
static int64_t g_ram_reserve_mb = 0;   // host-RAM reserve of the loading model (ram_guard.h; set by fn_model::load)
static void host_alloc_impl(fn_model_expert_set_alloc_tag *, void ** p, size_t bytes, bool pin, bool * pinned) {
    std::string why;   // every host expert allocation, pinned or not, must leave the reserve free (checked strictly, now)
    if (!ram_check(bytes, ram_available_bytes(), g_ram_reserve_mb, "a host expert set", why)) throw std::runtime_error("fn_model: " + why);
    if (pin) { cudaError_t e = cudaMallocHost(p, bytes); if (e == cudaSuccess) { *pinned = true; g_host_pinned_bytes += bytes; return; } (void) cudaGetLastError(); }
    g_host_pageable_bytes += bytes;
    *p = malloc(bytes);
    if (!*p) throw std::runtime_error("fn_model: host allocation failed");
    *pinned = false;
}

void fn_model::load_experts(rank_state & S, int r, int il) {
    layer_w & L = S.lw[il];
    const int R = (int) rk_.size();
    const int fil = file_layer(il);   // the file's block index (the draft block under a debug layer limit: blk.<file L>)
    const int n_hot = hot_count_[fil], n_cold = NEXP_ - n_hot;
    // resident hot local ids of rank r: [h0, h0 + HOTR) with per-rank counts (opts::rank_slots; equal counts = the
    // original r * HOTR ranges); host hot = ids past every rank's range, h % R == r
    int HOTR = o_.hot_per_rank < 0 ? (n_hot + R - 1) / R : o_.hot_per_rank, h0 = r * HOTR, hall = R * HOTR;
    if (!slots_.empty()) {
        h0 = 0; hall = 0;
        for (int q = 0; q < R; ++q) { const int c = slots_[q] < 0 ? HOTR : slots_[q]; if (q < r) h0 += c; hall += c; }
        HOTR = slots_[r] < 0 ? HOTR : slots_[r];
    }
    const gguf_tensor * map = g_.find(bn(fil, "ffn_exp_map"));
    std::vector<int32_t> ident;   // unsplit file: hot local id == expert id, no cold group
    if (!map) {
        if (n_hot != NEXP_) throw std::runtime_error("fn_model: missing ffn_exp_map " + std::to_string(il));
        ident.resize((size_t) 2 * NEXP_, -1);
        for (int e = 0; e < NEXP_; ++e) ident[e] = e;
    } else if (map->type != GGML_TYPE_I32 || map->ne[0] != NEXP_ || map->ne[1] != 2) throw std::runtime_error("fn_model: bad ffn_exp_map " + std::to_string(il));
    const int32_t * mp = map ? (const int32_t *) map->data : ident.data();   // row 0: hot local id, row 1: cold local id (-1 = the other group)
    L.hot_map.assign(NEXP_, -1); L.cold_map.assign(NEXP_, -1);
    // resident set: the hot local ids of this rank; host set: the rest of this rank's hot ids + its cold ids
    std::vector<int> res_hot, host_hot, host_cold;
    bool cold_all = n_hot == NEXP_;
    for (int e = 0; e < NEXP_ && cold_all; ++e) cold_all = mp[NEXP_ + e] >= 0;
    if (cold_all && o_.adapt && !rank_order_.empty()) {
        // every expert exists in both types: this rank owns experts e % R == r; the best-ranked of them start resident
        // (hot local id == e for an all-hot file), the low-bit copies of all its experts stay pinned in RAM
        const int slots = !slots_.empty() && slots_[r] >= 0 ? slots_[r] : o_.hot_per_rank > 0 ? o_.hot_per_rank : 72;
        for (int i = 0; i < NEXP_ && (int) res_hot.size() < slots; ++i) {
            const int e = rank_order_[(size_t) fil * NEXP_ + i];
            if (e % R == r) res_hot.push_back(mp[e]);
        }
        for (int e = 0; e < NEXP_; ++e) if (e % R == r) host_cold.push_back(mp[NEXP_ + e]);
        // promotions (promo_start) read the hot tensors at expert id e: each must hold all NEXP_ experts
        for (const char * tn : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"})
            if (const gguf_tensor * t = g_.find(bn(fil, tn)); t && t->ne[2] < NEXP_) throw std::runtime_error("fn_model: " + bn(fil, tn) + " holds fewer experts than the model");
        L.adaptive = true;
        L.cnt.assign(NEXP_, 0.0f);
    } else {
        for (int h = 0; h < n_hot; ++h) {
            if (h >= h0 && h < h0 + HOTR) res_hot.push_back(h);
            else if (h >= hall && (h % R) == r) host_hot.push_back(h);
        }
        for (int c = 0; c < n_cold; ++c) if ((c % R) == r) host_cold.push_back(c);
    }
    auto al = [](size_t b) { return (b + 255) & ~(size_t) 255; };
    auto dims = [&](expert_set & es, const char * tname, const std::vector<int> & ids) -> const gguf_tensor * {
        const gguf_tensor * t = g_.find(bn(fil, tname));
        if (!t) throw std::runtime_error("fn_model: missing " + bn(fil, tname));
        // the ids (hot / cold local ids from ffn_exp_map) become byte offsets into t: each must name one of its experts
        for (int id : ids) if (id < 0 || id >= t->ne[2]) throw std::runtime_error("fn_model: expert id " + std::to_string(id) + " outside " + bn(fil, tname));
        es.type = t->type; es.n_in = t->ne[0]; es.n_out = t->ne[1]; es.n = (int) ids.size();
        es.row_bytes = (size_t) tensor_nbytes(t->type, t->ne) / ((size_t) t->ne[1] * t->ne[2]);
        es.bytes = es.row_bytes * (size_t) es.n_out;
        es.raw = needs_raw(t->type, es.n_in);
        return t;
    };
    // resident set: device [n][bytes] per projection, packed in place
    // ask the OS to page in the selected expert rows in bulk before they are copied: a cold file cache otherwise faults
    // them in 4 KB at a time (a SATA disk fell to ~38 MB/s after a large copy evicted the cache)
    auto prefetch_ids = [&](const gguf_tensor * t, const std::vector<int> & ids, size_t bytes) {
#ifdef _WIN32
        // off by default: the expert bytes are read with ReadFile (gguf_file::read_at); paging them through the mapping as
        // well doubled the disk traffic and starved those reads
        static const bool pf_on = getenv("P100E_FN_LOAD_PREFETCH") && atoi(getenv("P100E_FN_LOAD_PREFETCH")) != 0;
        if (!pf_on || ids.empty()) return;
        std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
        for (size_t k = 0; k < ids.size(); ++k) {   // consecutive ids coalesce into one range
            if (!ranges.empty() && (char *) ranges.back().VirtualAddress + ranges.back().NumberOfBytes == (char *) t->data + (size_t) ids[k] * bytes) ranges.back().NumberOfBytes += bytes;
            else ranges.push_back({(void *) (t->data + (size_t) ids[k] * bytes), bytes});
        }
        PrefetchVirtualMemory(GetCurrentProcess(), ranges.size(), ranges.data(), 0);   // best effort
#else
        (void) t; (void) ids; (void) bytes;
#endif
    };
    auto load_device = [&](expert_set & es, const char * tname, const std::vector<int> & ids) {
        const gguf_tensor * t = dims(es, tname, ids);
        prefetch_ids(t, ids, es.bytes);
        es.stride = es.bytes;
        if (es.n == 0) return;
        CK(cudaMalloc(&es.base, es.bytes * es.n));
        es.pk.resize(es.n);
        for (int j = 0; j < es.n; ++j) {
            char * dst = es.base + (size_t) j * es.bytes;
            const uint8_t * src = t->data + (size_t) ids[j] * es.bytes;
            if (es.bytes <= HBOUNCE_) { g_.read_at(S.hbounce, src, es.bytes); src = (const uint8_t *) S.hbounce; }
            CK(cudaMemcpyAsync(dst, src, es.bytes, cudaMemcpyHostToDevice, S.s));
            if (src == (const uint8_t *) S.hbounce) CK(cudaStreamSynchronize(S.s));
            if (es.raw) { fast::packed_weight & pk = es.pk[j]; pk.type = es.type; pk.n_in = es.n_in; pk.n_out = es.n_out; pk.data = dst; pk.bytes = es.bytes; pk.layout = fast::LAYOUT_RAW; es.layout = fast::LAYOUT_RAW; continue; }
            es.pk[j] = fast::pack_weight(t->type, dst, es.n_in, es.n_out, S.s);
            if (!es.pk[j].data) throw std::runtime_error(std::string("fn_model: no GEMV path for expert type ") + get_type_info(t->type).name);
            es.layout = es.pk[j].layout;
        }
    };
    // host set: ONE record per expert = gate | up | down (each 256-aligned) in one pinned buffer, so a staged expert is a
    // single PCIe copy (three ~0.5 MB copies ran at ~6 GB/s on the P100s, DMA setup bound); packed via the staging buffer
    auto load_host = [&](expert_set & g, expert_set & u, expert_set & d, const char * tg, const char * tu, const char * td, const std::vector<int> & ids) {
        const gguf_tensor * t3[3] = {dims(g, tg, ids), dims(u, tu, ids), dims(d, td, ids)};
        expert_set * es3[3] = {&g, &u, &d};
        const size_t stride = al(g.bytes) + al(u.bytes) + al(d.bytes);
        if (ids.empty()) return;
        for (int k = 0; k < 3; ++k) prefetch_ids(t3[k], ids, es3[k]->bytes);
        if (stride > S.qstage_bytes / 2) throw std::runtime_error("fn_model: expert record larger than half the staging arena (P100E_FN_STAGE_MB)");
        static const bool ldbg = getenv("P100E_FN_LOAD_DEBUG") != nullptr;
        if (ldbg) { printf("  load_host %s: %zu experts x %zu B, pinned alloc...\n", tg, ids.size(), stride); fflush(stdout); }
        host_alloc_impl(nullptr, (void **) &g.base, stride * ids.size(), o_.pin_host, &g.pinned);
        if (ldbg) { printf("  load_host %s: alloc done (pinned %d)\n", tg, (int) g.pinned); fflush(stdout); }
        g.owner = true; g.stride = stride;
        u.base = g.base + al(g.bytes); u.stride = stride; u.pinned = g.pinned; u.owner = false;
        d.base = u.base + al(u.bytes); d.stride = stride; d.pinned = g.pinned; d.owner = false;
        for (int k = 0; k < 3; ++k) {
            expert_set & es = *es3[k]; const gguf_tensor * t = t3[k];
            for (int j = 0; j < es.n; ++j) {
                char * dst = es.base + (size_t) j * stride;
                const uint8_t * src = t->data + (size_t) ids[j] * es.bytes;
                if (es.raw) { g_.read_at(dst, src, es.bytes); es.layout = fast::LAYOUT_RAW; continue; }
                // pack on the device, keep the packed bytes on the host (the fast layout is a byte permutation of the row set)
                using clk = std::chrono::steady_clock;
                static double t_read = 0, t_h2d = 0, t_pack = 0, t_sync = 0;
                auto t0 = clk::now();
                if (es.bytes <= HBOUNCE_) { g_.read_at(S.hbounce, src, es.bytes); src = (const uint8_t *) S.hbounce; }
                auto t1 = clk::now();
                CK(cudaMemcpyAsync(S.qstage, src, es.bytes, cudaMemcpyHostToDevice, S.s));
                CK(cudaStreamSynchronize(S.s));
                auto t2 = clk::now();
                fast::packed_weight pk = fast::pack_weight(t->type, S.qstage, es.n_in, es.n_out, S.s);
                if (!pk.data) throw std::runtime_error(std::string("fn_model: no GEMV path for host expert type ") + get_type_info(t->type).name);
                es.layout = pk.layout;
                auto t3 = clk::now();
                CK(cudaMemcpyAsync(dst, S.qstage, es.bytes, cudaMemcpyDeviceToHost, S.s));
                CK(cudaStreamSynchronize(S.s));
                auto t4 = clk::now();
                t_read += std::chrono::duration<double, std::milli>(t1 - t0).count(); t_h2d += std::chrono::duration<double, std::milli>(t2 - t1).count();
                t_pack += std::chrono::duration<double, std::milli>(t3 - t2).count(); t_sync += std::chrono::duration<double, std::milli>(t4 - t3).count();
                if (ldbg && (j % 64 == 0 || j == es.n - 1)) {
                    printf("  load_host %s: expert %d/%d  cumulative ms: read %.0f h2d %.0f pack %.0f d2h+sync %.0f\n", k == 0 ? tg : k == 1 ? tu : td, j, es.n, t_read, t_h2d, t_pack, t_sync); fflush(stdout);
                }
            }
        }
    };
    // resident hot experts
    load_device(L.hot_gate, "ffn_gate_exps.weight", res_hot);
    load_device(L.hot_up,   "ffn_up_exps.weight",   res_hot);
    load_device(L.hot_down, "ffn_down_exps.weight", res_hot);
    // host experts: this rank's cold ids (+ hot ids without a resident slot); two record sets (different types)
    if (n_cold > 0 || cold_all)   // an unsplit layer has no cold tensors
        load_host(L.cold_gate, L.cold_up, L.cold_down, "ffn_gate_exps_cold.weight", "ffn_up_exps_cold.weight", "ffn_down_exps_cold.weight", host_cold);
    if (!host_hot.empty()) load_host(L.hhot_gate, L.hhot_up, L.hhot_down, "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", host_hot);
    L.slot_expert.assign(res_hot.size(), -1); L.slot_busy.assign(res_hot.size(), 0);
    for (int e = 0; e < NEXP_; ++e) {
        const int h = mp[e], c = mp[NEXP_ + e];
        if (h >= 0) {
            auto it = std::find(res_hot.begin(), res_hot.end(), h);
            if (it != res_hot.end()) { L.hot_map[e] = (int) (it - res_hot.begin()); L.slot_expert[L.hot_map[e]] = e; }
            else if (!L.adaptive) {
                auto jt = std::find(host_hot.begin(), host_hot.end(), h);
                if (jt != host_hot.end()) L.hhot_map_set(e, (int) (jt - host_hot.begin()));
            }
        }
        if (c >= 0 && (h < 0 || L.adaptive)) {   // adaptive: the low-bit copy of every own expert, resident ones too
            auto it = std::find(host_cold.begin(), host_cold.end(), c);
            if (it != host_cold.end()) L.cold_map[e] = (int) (it - host_cold.begin());
        }
    }
}

// ---- asynchronous promotions ----
static size_t al256(size_t b) { return (b + 255) & ~(size_t) 255; }

void fn_model::promo_start() {
    if (promo_thread_.joinable()) return;
    const layer_w & L0 = rk_[0].lw[lb_];   // this node's first layer (pipe mode: not always layer 0)
    promo_bytes_ = al256(L0.hot_gate.bytes) + al256(L0.hot_up.bytes) + al256(L0.hot_down.bytes);
    const int n = 96;   // ring entries (~270 MB pinned at 2.8 MB per expert)
    ring_.resize(n);
    for (auto & r : ring_) {
        CK(cudaMallocHost(&r.buf, promo_bytes_));
        for (int k = 0; k < (int) rk_.size() && k < 2; ++k) { CK(cudaSetDevice(rk_[k].dev)); CK(cudaEventCreateWithFlags(&r.done[k], cudaEventDisableTiming)); }
    }
    adapt_async_ = getenv("P100E_FN_ADAPT_ASYNC") && atoi(getenv("P100E_FN_ADAPT_ASYNC")) != 0;
    if (adapt_async_)
        for (auto & S : rk_) {
            CK(cudaSetDevice(S.dev));
            int lo = 0, hi = 0;
            CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
            CK(cudaStreamCreateWithPriority(&S.ps, cudaStreamNonBlocking, lo));
            CK(cudaEventCreateWithFlags(&S.ps_ev, cudaEventDisableTiming));
            for (auto & L : S.lw) L.inflight.assign(NEXP_, 0);
        }
    CK(cudaSetDevice(rk_[0].dev));
    promo_stop_ = false;
    promo_thread_ = std::thread([this] { promo_worker(); });
}

// worker: reads the three slices of each queued expert from the memory-mapped GGUF (page faults happen here, off the
// forward's critical path) into a free ring entry, then hands the entry to adapt()
void fn_model::promo_worker() {
    const char * names[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
    while (true) {
        promo p;
        {
            std::unique_lock<std::mutex> lk(promo_mu_);
            promo_cv_.wait(lk, [&] { return promo_stop_ || !promo_todo_.empty(); });
            if (promo_stop_) return;
            p = promo_todo_.front(); promo_todo_.pop_front();
        }
        // a free ring entry: one whose last copy has finished (the main thread records r.done when it issues the copy)
        int ri = -1;
        while (ri < 0) {
            for (int i = 0; i < (int) ring_.size(); ++i) {
                ring_entry & r = ring_[i];
                if (!r.used) { r.used = true; ri = i; break; }
            }
            if (ri < 0) {
                { std::lock_guard<std::mutex> lk(promo_mu_); if (promo_stop_) return; }
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        }
        char * dst = ring_[ri].buf;
        const layer_w & L = rk_[p.rank].lw[p.il];
        const expert_set * sets[3] = {&L.hot_gate, &L.hot_up, &L.hot_down};
        size_t off = 0;
#ifdef _WIN32
        {
            WIN32_MEMORY_RANGE_ENTRY ranges[3];
            for (int k = 0; k < 3; ++k) { const gguf_tensor * t = g_.find(bn(file_layer(p.il), names[k])); ranges[k].VirtualAddress = (void *) (t->data + (size_t) p.e * sets[k]->bytes); ranges[k].NumberOfBytes = sets[k]->bytes; }
            PrefetchVirtualMemory(GetCurrentProcess(), 3, ranges, 0);
        }
#endif
        for (int k = 0; k < 3; ++k) {
            const gguf_tensor * t = g_.find(bn(file_layer(p.il), names[k]));
            g_.read_at(dst + off, t->data + (size_t) p.e * sets[k]->bytes, sets[k]->bytes);   // ReadFile: no mapping faults
            off += al256(sets[k]->bytes);
        }
        p.ring = ri;
        std::lock_guard<std::mutex> lk(promo_mu_);
        promo_ready_.push_back(p);
    }
}

// main thread: ring entry -> slot (copy stream), pack (compute stream, after the copy), maps updated; the ring entry
// is released when its copy completes (event polled by the next landing / the worker through r.done)
void fn_model::promo_land(const promo & p) {
    static const bool dbg = getenv("P100E_FN_ADAPT_DEBUG") && atoi(getenv("P100E_FN_ADAPT_DEBUG")) != 0;
    if (dbg) fprintf(stderr, "[land] r%d L%d slot %d <- e%d ring %d\n", p.rank, p.il, p.slot, p.e, p.ring);
    rank_state & S = rk_[p.rank];
    CK(cudaSetDevice(S.dev));
    layer_w & L = S.lw[p.il];
    expert_set * sets[3] = {&L.hot_gate, &L.hot_up, &L.hot_down};
    ring_entry & r = ring_[p.ring];
    if (adapt_async_) {
        // after everything already queued on the compute stream (the only work that may still read the old expert),
        // copy + pack on the promotion stream; the old expert leaves the maps now, the new one in promo_finalize
        CK(cudaEventRecord(S.ps_ev, S.s));
        CK(cudaStreamWaitEvent(S.ps, S.ps_ev, 0));
        size_t o2 = 0;
        for (int k = 0; k < 3; ++k) {
            expert_set & es = *sets[k];
            char * dst = es.base + (size_t) p.slot * es.stride;
            CK(cudaMemcpyAsync(dst, r.buf + o2, es.bytes, cudaMemcpyHostToDevice, S.ps));
            o2 += al256(es.bytes);
            if (es.raw) es.pk[p.slot].data = dst;
            else es.pk[p.slot] = fast::pack_weight(es.type, dst, es.n_in, es.n_out, S.ps);
        }
        CK(cudaEventRecord(r.done[p.rank], S.ps));
        r.landed = false;   // released by promo_finalize, not by adapt()'s generic loop
        const int old = L.slot_expert[p.slot];
        if (old >= 0) L.hot_map[old] = -1;
        L.slot_expert[p.slot] = -1;
        lands_.push_back(p);
        return;
    }
    size_t off = 0;
    for (int k = 0; k < 3; ++k) {
        expert_set & es = *sets[k];
        char * dst = es.base + (size_t) p.slot * es.stride;
        CK(cudaMemcpyAsync(dst, r.buf + off, es.bytes, cudaMemcpyHostToDevice, S.cs));
        off += al256(es.bytes);
    }
    CK(cudaEventRecord(r.done[p.rank], S.cs));
    r.landed = true; r.landed_rank = p.rank;
    CK(cudaStreamWaitEvent(S.s, r.done[p.rank], 0));
    for (int k = 0; k < 3; ++k) {
        expert_set & es = *sets[k];
        char * dst = es.base + (size_t) p.slot * es.stride;
        if (es.raw) { es.pk[p.slot].data = dst; continue; }
        es.pk[p.slot] = fast::pack_weight(es.type, dst, es.n_in, es.n_out, S.s);
        if (dbg) fprintf(stderr, "[land]   packed %d\n", k);
    }
    const int old = L.slot_expert[p.slot];
    if (old >= 0) L.hot_map[old] = -1;
    L.hot_map[p.e] = p.slot;
    L.slot_expert[p.slot] = p.e;
    L.slot_busy[p.slot] = 0;
    ++L.swaps;
}

void fn_model::promo_finalize() {
    if (lands_.empty()) return;
    std::vector<promo> keep;
    for (const promo & p : lands_) {
        ring_entry & r = ring_[p.ring];
        CK(cudaSetDevice(rk_[p.rank].dev));
        if (cudaEventQuery(r.done[p.rank]) != cudaSuccess) { (void) cudaGetLastError(); keep.push_back(p); continue; }
        layer_w & L = rk_[p.rank].lw[p.il];
        L.hot_map[p.e] = p.slot;
        L.slot_expert[p.slot] = p.e;
        L.slot_busy[p.slot] = 0;
        if (!L.inflight.empty()) L.inflight[p.e] = 0;
        ++L.swaps;
        r.used = false;
    }
    lands_.swap(keep);
    CK(cudaSetDevice(rk_[0].dev));
}

// Between forwards: land the promotions whose bytes arrived, then queue new ones: per rank and layer the least-used
// resident experts (not busy) give their slots to the most-used non-resident ones with >= adapt_min routed rows since
// the last decay; counts decay by half each step. `big` (after a prefill chunk) allows more swaps per layer.
void fn_model::adapt(bool big) {
    static const bool dbg = getenv("P100E_FN_ADAPT_DEBUG") && atoi(getenv("P100E_FN_ADAPT_DEBUG")) != 0;
    bool any = false;
    for (auto & S : rk_) for (auto & L : S.lw) any = any || L.adaptive;
    if (!any) return;
    promo_start();
    promo_finalize();
    const int max_swaps = big ? o_.adapt_max_prefill : o_.adapt_max;
    {   // landed promotions (bytes in pinned ring entries) -> GPU
        std::deque<promo> ready;
        { std::lock_guard<std::mutex> lk(promo_mu_); ready.swap(promo_ready_); }
        if (dbg) fprintf(stderr, "[adapt] enter big=%d ready=%zu\n", (int) big, ready.size());
        for (const promo & p : ready) promo_land(p);
        if (dbg) fprintf(stderr, "[adapt] landed\n");
        // release ring entries whose copies have completed
        for (auto & r : ring_) if (r.used && r.landed && cudaEventQuery(r.done[r.landed_rank]) == cudaSuccess) { r.used = false; r.landed = false; }
        (void) cudaGetLastError();   // cudaEventQuery(cudaErrorNotReady) is not an error
    }
    rexp_adapt();   // remote experts: node 1's set first, from the same (not yet decayed) counts
    std::vector<promo> todo;
    struct pick { float gain; int il, slot, e; };
    for (int rr = 0; rr < (int) rk_.size(); ++rr) {
        rank_state & S = rk_[rr];
        std::vector<pick> picks;   // P100E_FN_ADAPT_CAP: this rank's candidate swaps of every layer, best gain first
        for (int il = 0; il < L_; ++il) {
            layer_w & L = S.lw[il];
            if (!L.adaptive || max_swaps <= 0) continue;
            std::vector<std::pair<float, int>> cand, vict;
            for (int e = 0; e < NEXP_; ++e) {
                if (L.cold_map[e] < 0) continue;   // not this rank's expert
                const int sl = L.hot_map[e];
                // remote experts: node 1's experts are covered (never promoted here), and a resident one node 1 also
                // holds is the first to give its slot away (both are no-ops without the remote tier)
                const bool rem = rexp_covered(il, e);
                if (sl >= 0) { if (!L.slot_busy[sl]) vict.push_back({rem ? 0.0f : L.cnt[e], sl}); }
                else if (!rem && L.cnt[e] >= o_.adapt_min && (!adapt_async_ || !L.inflight[e])) cand.push_back({L.cnt[e], e});
            }
            std::sort(cand.begin(), cand.end(), [](auto & a, auto & b) { return a.first > b.first; });
            std::sort(vict.begin(), vict.end());
            int n = 0;
            for (size_t i = 0; i < cand.size() && i < vict.size() && n < max_swaps; ++i) {
                if (cand[i].first <= vict[i].first) break;   // no clear gain
                if (adapt_cap_ > 0) { picks.push_back({cand[i].first - vict[i].first, il, vict[i].second, cand[i].second}); ++n; continue; }
                L.slot_busy[vict[i].second] = 1;
                if (adapt_async_) L.inflight[cand[i].second] = 1;
                todo.push_back({rr, il, vict[i].second, cand[i].second, -1});
                ++n;
            }
            for (float & c : L.cnt) c *= 0.5f;
        }
        if (adapt_cap_ > 0) {   // the PCIe budget goes to the biggest gains of the step (ties: lower layer)
            std::stable_sort(picks.begin(), picks.end(), [](const pick & a, const pick & b) { return a.gain > b.gain; });
            if ((int) picks.size() > adapt_cap_) picks.resize(adapt_cap_);
            for (const pick & p : picks) {
                layer_w & L = S.lw[p.il];
                L.slot_busy[p.slot] = 1;
                if (adapt_async_) L.inflight[p.e] = 1;
                todo.push_back({rr, p.il, p.slot, p.e, -1});
            }
        }
    }
    if (!todo.empty()) {
        std::lock_guard<std::mutex> lk(promo_mu_);
        for (auto & p : todo) promo_todo_.push_back(p);
    }
    promo_cv_.notify_all();
    if (dbg) fprintf(stderr, "[adapt] queued %zu, exit\n", todo.size());
    since_adapt_ = 0;
}


// host-RAM reserve (ram_guard.h): the host expert records every rank will pin (the same choice as load_experts: an
// adaptive layer keeps the low-bit copy of every expert, else the hot experts past the resident slots + the cold ones),
// the promotion ring and the CPU pool's input rows, checked against the available RAM before anything is allocated
void fn_model::ram_reserve_check(int R) {
    g_ram_reserve_mb = ram_reserve_mb(o_.ram_reserve_mb);
    if (g_ram_reserve_mb <= 0) return;
    auto al = [](size_t b) { return (b + 255) & ~(size_t) 255; };
    auto rec = [&](int il, const char * g, const char * u, const char * d) -> size_t {
        size_t b = 0;
        for (const char * n : {g, u, d}) {
            const gguf_tensor * t = g_.find(bn(il, n));
            if (!t) return 0;
            b += al((size_t) tensor_nbytes(t->type, t->ne) / (size_t) t->ne[2]);
        }
        return b;
    };
    size_t need = 0, hot_rec = 0;
    bool any_adaptive = false;
    for (int il = 0; il < NL_; ++il) {
        const int n_hot = hot_count_[il], n_cold = NEXP_ - n_hot;
        const int HOTR = o_.hot_per_rank < 0 ? (n_hot + R - 1) / R : o_.hot_per_rank;
        const size_t hr = rec(il, "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight");
        const size_t cr = rec(il, "ffn_gate_exps_cold.weight", "ffn_up_exps_cold.weight", "ffn_down_exps_cold.weight");
        hot_rec = std::max(hot_rec, hr);
        const gguf_tensor * map = g_.find(bn(il, "ffn_exp_map"));
        bool cold_all = n_hot == NEXP_ && map != nullptr;
        for (int e = 0; e < NEXP_ && cold_all; ++e) cold_all = ((const int32_t *) map->data)[NEXP_ + e] >= 0;
        if (cold_all && o_.adapt && !rank_order_.empty()) { need += (size_t) NEXP_ * cr; any_adaptive = true; }
        else need += (size_t) std::max(0, n_hot - R * HOTR) * hr + (size_t) n_cold * cr;
    }
    if (any_adaptive) need += 96 * hot_rec;   // promotion ring (promo_start)
    need += (size_t) o_.max_t * E_ * 4;       // CPU pool input rows (P100E_FN_CPU)
    std::string why;
    if (!ram_check(need, ram_available_bytes(), g_ram_reserve_mb, "the host experts of this load", why)) throw std::runtime_error("fn_model: " + why);
}

void fn_model::load(const std::vector<int> & devices) {
    const int R = (int) devices.size();
    if (R < 1 || R > 2) throw std::runtime_error("fn_model: 1 or 2 GPUs");
    ram_reserve_check(R);
    rk_.resize(R);
    const int T = std::max(o_.max_t, 1);
    for (int r = 0; r < R; ++r) {
        rank_state & S = rk_[r];
        S.dev = devices[r];
        CK(cudaSetDevice(S.dev));
        {
            const bool tp = o_.tp && R > 1 && H_ % R == 0 && HKV_ % R == 0 && HK_ % R == 0 && HV_ % R == 0 && (H_ / HKV_) % 1 == 0 &&
                            ((int64_t) (H_ / R) * HD_) % 256 == 0 && ((int64_t) (HV_ / R) * DV_) % 256 == 0 && (HV_ / R) % 4 == 0;
            const int d = tp ? R : 1, rr = tp ? r : 0;
            S.Hr = H_ / d; S.HKVr = HKV_ / d; S.HKr = HK_ / d; S.HVr = HV_ / d; S.CONVr = 2 * S.HKr * DK_ + S.HVr * DV_;
            S.h0 = rr * S.Hr; S.hkv0 = rr * S.HKVr; S.hk0 = rr * S.HKr; S.hv0 = rr * S.HVr;
            if (o_.verbose) printf("fn_model: rank %d dense split: q heads %d..%d kv %d..%d gdn k %d..%d v %d..%d%s\n", r, S.h0, S.h0 + S.Hr - 1, S.hkv0,
                                   S.hkv0 + S.HKVr - 1, S.hk0, S.hk0 + S.HKr - 1, S.hv0, S.hv0 + S.HVr - 1, tp ? "" : " (replicated)");
        }
        for (int o = 0; o < R; ++o) {
            if (o == r) continue;
            int can = 0;
            cudaDeviceCanAccessPeer(&can, S.dev, devices[o]);
            if (can) { const cudaError_t e = cudaDeviceEnablePeerAccess(devices[o], 0); if (e == cudaErrorPeerAccessAlreadyEnabled) (void) cudaGetLastError(); else CK(e); }
        }
        CK(cudaStreamCreateWithFlags(&S.s, cudaStreamNonBlocking));
        CB(cublasCreate(&S.cb)); CB(cublasSetStream(S.cb, S.s)); CB(cublasSetMathMode(S.cb, CUBLAS_PEDANTIC_MATH));
        CK(cudaEventCreateWithFlags(&S.ev, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&S.ev2, cudaEventDisableTiming));
        CK(cudaStreamCreateWithFlags(&S.cs, cudaStreamNonBlocking));
        for (int k = 0; k < 2; ++k) { CK(cudaEventCreateWithFlags(&S.st_ready[k], cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&S.st_free[k], cudaEventDisableTiming)); }
        CK(cudaMallocHost(&S.hidx_pin, (size_t) T * NEXPU_ * 4)); CK(cudaMallocHost(&S.hw_pin, (size_t) T * NEXPU_ * 4));
        S.qstage_bytes = ((size_t) o_.stage_mb << 20) & ~(size_t) 511;   // two 256-byte aligned halves
        CK(cudaMalloc(&S.qstage, S.qstage_bytes));
        if (cudaMallocHost((void **) &S.hbounce, HBOUNCE_) != cudaSuccess) { (void) cudaGetLastError(); S.hbounce = (char *) malloc(HBOUNCE_); }
        S.w_scratch_bytes = (size_t) 64 << 20;
        CK(cudaMalloc(&S.w_scratch, S.w_scratch_bytes));
        const size_t wide = (size_t) T * std::max({2 * H_ * HD_, CONV_C_, HCDIM_, 4 * E_});
        for (float ** p : {&S.a, &S.b, &S.c, &S.d, &S.e, &S.g}) { CK(cudaMalloc(p, wide * 4)); CK(cudaMemset(*p, 0, wide * 4)); }
        CK(cudaMalloc(&S.res, (size_t) T * HCDIM_ * 4)); CK(cudaMalloc(&S.xn, (size_t) T * HCDIM_ * 4));
        CK(cudaMalloc(&S.cur, (size_t) T * E_ * 4)); CK(cudaMalloc(&S.part, (size_t) T * E_ * 4)); CK(cudaMalloc(&S.peer, (size_t) T * E_ * 4));
        CK(cudaMalloc(&S.inject, (size_t) T * HC_ * 4)); CK(cudaMalloc(&S.router_out, (size_t) T * NEXP_ * 4));
        CK(cudaMalloc(&S.logits, (size_t) std::min(T, FN_MAX_LOGITS) * VOCAB_ * 4));
        CK(cudaMalloc(&S.pos, (size_t) T * 4)); CK(cudaMalloc(&S.rpos, (size_t) 3 * T * 4));
        CK(cudaMalloc(&S.gidx, (size_t) T * NEXPU_ * 4)); CK(cudaMalloc(&S.gw, (size_t) T * NEXPU_ * 4));
        const size_t NR = (size_t) T * NEXPU_;   // max gathered rows per layer
        CK(cudaMalloc(&S.gtab, 2 * 3 * NR * sizeof(fast::mv_group)));
        for (int k = 0; k < 2; ++k) CK(cudaMallocHost(&S.tab_pin[k], 3 * NR * sizeof(fast::mv_group)));
        CK(cudaMalloc(&S.q2, fast::matvec_scratch_bytes(FFE_, (int) NR)));
        CK(cudaMalloc(&S.qa, fast::matvec_scratch_bytes(E_, (int) NR)));
        CK(cudaMalloc(&S.rowof, NR * 4)); CK(cudaMallocHost(&S.rowof_pin, NR * 4));
        CK(cudaMalloc(&S.route_ids, NR * 4)); CK(cudaMalloc(&S.route_wts, NR * 4));
        CK(cudaMallocHost(&S.route_ids_pin, NR * 4)); CK(cudaMallocHost(&S.route_wts_pin, NR * 4));
        CK(cudaMalloc(&S.gath, NR * E_ * 4)); CK(cudaMalloc(&S.gout, NR * std::max(E_, FFE_) * 4));
        CK(cudaMalloc(&S.mv_scratch, fast::matvec_scratch_bytes(std::max({HCDIM_, CONV_C_, 2 * H_ * HD_, E_}), 8)));
        // fold-GEMM scratch: the minimum for the widest input plus room for row chunks (the whole-matrix size would be
        // 0.6 GB for the LM head); gemm_q processes the weight in chunks that fit
        S.gemm_bytes = std::max(fast::gemm_scratch_min_bytes(HCDIM_, T), (size_t) 96 << 20);
        CK(cudaMalloc(&S.gemm_scratch, S.gemm_bytes));
        if (PLE_LAYER_ >= 0) {
            CK(cudaMalloc(&S.ple_hist, (size_t) (PLE_HIST_ + T) * HCDIM_ * 4)); CK(cudaMemset(S.ple_hist, 0, (size_t) (PLE_HIST_ + T) * HCDIM_ * 4));
            CK(cudaMalloc(&S.ple_emb, (size_t) T * E_ * 4));
        }
        // ---- weights ----
        // the LM head: every rank in one process (as before); pipe mode: rank 0 of the head node only (the only user)
        if (has_head() && (!pipe() || r == 0)) {
            S.output = upload_weight(S, g_.find("output.weight"), "output.weight");
            S.output_hc_down = upload_weight(S, g_.find("output_hc_down.weight"), "output_hc_down.weight");
            S.output_hc_up = upload_weight(S, g_.find("output_hc_up.weight"), "output_hc_up.weight");
            S.output_hc_norm = small(S, "output_hc_norm.weight");
        }
        S.lw.assign(NL_, layer_w{});
        S.kc.assign(NL_, nullptr); S.vc.assign(NL_, nullptr); S.conv_state.assign(NL_, nullptr); S.ssm_state.assign(NL_, nullptr);
        S.kblk.assign(NL_, nullptr); S.ktail.assign(NL_, nullptr);
        S.conv_snap.assign(NL_, nullptr); S.ssm_snap.assign(NL_, nullptr);
        if (mtp_il_ >= 0) {
            const size_t T = (size_t) o_.max_t;
            CK(cudaMalloc(&S.hn, T * HCDIM_ * 4)); CK(cudaMalloc(&S.mhin, T * HCDIM_ * 4));
            CK(cudaMalloc(&S.mcarry, (size_t) HCDIM_ * 4)); CK(cudaMemset(S.mcarry, 0, (size_t) HCDIM_ * 4));
            CK(cudaMalloc(&S.mself, (size_t) HCDIM_ * 4));
            CK(cudaMalloc(&S.mcat, T * HC_ * 2 * E_ * 4));
            if (PLE_LAYER_ >= 0) CK(cudaMalloc(&S.ple_snap, (size_t) PLE_HIST_ * HCDIM_ * 4));
            if (o_.verbose && &S == &rk_[0]) printf("fn_model: MTP draft block = layer %d (verify batch <= %d rows)\n", mtp_il_, MTP_ROWS);
        }
        if (qsa_) {
            const int NB = o_.n_ctx / RATIO_ + 1, T = o_.max_t;
            CK(cudaMalloc(&S.blk_rpos, (size_t) 3 * NB * 4)); CK(cudaMemset(S.blk_rpos, 0, (size_t) 3 * NB * 4));
            CK(cudaMalloc(&S.qk_raw, (size_t) T * IDX_D_ * 4));
            CK(cudaMalloc(&S.qq, (size_t) T * IDX_H_ * IDX_D_ * 4));
            CK(cudaMalloc(&S.qsa_s, (size_t) QSA_SL_ * IDX_H_ * NB * 4));
            CK(cudaMalloc(&S.qsa_c, (size_t) QSA_SL_ * NB * 4));
            CK(cudaMalloc(&S.sel, (size_t) T * QSA_W_ * 4));
            CK(cudaMalloc(&S.seln, (size_t) T * 4));
            if (o_.verbose && &S == &rk_[0]) printf("fn_model: QSA on: ratio %d, indexer %d x %d, top-k %d (W %d), block keys %d per layer; KV %s\n", RATIO_, IDX_H_, IDX_D_, IDX_TOPK_, QSA_W_, NB, kv_mode_ == 0 ? "f32" : kv_mode_ == 1 ? "f16" : "q8");
        }
        for (int il = 0; il < NL_; ++il) {
            layer_w & L = S.lw[il];
            L.attn = (il + 1) % full_attn_interval_ == 0 || il == mtp_il_;
            if (!owns(il)) continue;   // pipe mode: another node's layer (no weights, caches or state here)
            const int fil = file_layer(il);
            L.hc_attn_norm = small(S, bn(fil, "hc_attn_norm.weight")); L.hc_attn_inject = small(S, bn(fil, "hc_attn_inject.weight"));
            L.hc_ffn_norm = small(S, bn(fil, "hc_ffn_norm.weight")); L.hc_ffn_inject = small(S, bn(fil, "hc_ffn_inject.weight"));
            L.hc_attn_down = upload_weight(S, g_.find(bn(fil, "hc_attn_down.weight")), "hc_attn_down");
            L.hc_attn_up = upload_weight(S, g_.find(bn(fil, "hc_attn_up.weight")), "hc_attn_up");
            L.hc_ffn_down = upload_weight(S, g_.find(bn(fil, "hc_ffn_down.weight")), "hc_ffn_down");
            L.hc_ffn_up = upload_weight(S, g_.find(bn(fil, "hc_ffn_up.weight")), "hc_ffn_up");
            if (L.attn) {
                // tp: this rank's q heads [h0, h0+Hr) (q | gate per head: 2*HD rows each), kv heads [hkv0, hkv0+HKVr), and
                // the matching input columns of attn_output (its partial sums are allreduced)
                L.attn_q = upload_weight(S, g_.find(bn(fil, "attn_q.weight")), "attn_q", -1, (int64_t) S.h0 * 2 * HD_, (int64_t) S.Hr * 2 * HD_);
                L.attn_k = upload_weight(S, g_.find(bn(fil, "attn_k.weight")), "attn_k", -1, (int64_t) S.hkv0 * HD_, (int64_t) S.HKVr * HD_);
                L.attn_v = upload_weight(S, g_.find(bn(fil, "attn_v.weight")), "attn_v", -1, (int64_t) S.hkv0 * HD_, (int64_t) S.HKVr * HD_);
                L.attn_output = S.Hr == H_ ? upload_weight(S, g_.find(bn(fil, "attn_output.weight")), "attn_output")
                                           : upload_cols(S, g_.find(bn(fil, "attn_output.weight")), "attn_output", {{(int64_t) S.h0 * HD_, (int64_t) S.Hr * HD_}});
                L.attn_q_norm = small(S, bn(fil, "attn_q_norm.weight")); L.attn_k_norm = small(S, bn(fil, "attn_k_norm.weight"));
                const size_t kvr = (size_t) S.HKVr * HD_, cell_bytes = (size_t) S.HKVr * qsa::kv_row_bytes(kv_mode_, HD_);
                CK(cudaMalloc(&S.kc[il], (size_t) o_.n_ctx * cell_bytes)); CK(cudaMalloc(&S.vc[il], (size_t) o_.n_ctx * cell_bytes));
                if (kv_mode_ != 0 && !S.kf) {
                    const size_t win = (size_t) ((qsa_ ? QSA_W_ : o_.n_ctx) + o_.max_t) * kvr * 4;
                    CK(cudaMalloc(&S.kf, win)); CK(cudaMalloc(&S.vf, win));
                }
                if (qsa_) {
                    // indexer (replicated on every rank: the cell selection must agree across the head split)
                    L.idx_q = upload_weight(S, g_.find(bn(fil, "indexer.q_proj.weight")), "indexer.q_proj");
                    L.idx_k = upload_weight(S, g_.find(bn(fil, "indexer.k_proj.weight")), "indexer.k_proj");
                    L.idx_q_norm = small(S, bn(fil, "indexer.q_norm.weight")); L.idx_k_norm = small(S, bn(fil, "indexer.k_norm.weight"));
                    const int NB = o_.n_ctx / RATIO_ + 1;
                    CK(cudaMalloc(&S.kblk[il], (size_t) NB * IDX_D_ * 4));
                    CK(cudaMalloc(&S.ktail[il], (size_t) QSA_RING_ * IDX_D_ * 4));
                }
            } else {
                // tp: qkv rows [q heads | k heads | v heads] of this rank, z / beta / alpha rows of its v heads, ssm_out columns
                if (S.HKr == HK_) {
                    L.attn_qkv = upload_weight(S, g_.find(bn(fil, "attn_qkv.weight")), "attn_qkv");
                    L.ssm_out = upload_weight(S, g_.find(bn(fil, "ssm_out.weight")), "ssm_out");
                    L.ssm_conv1d = small(S, bn(fil, "ssm_conv1d.weight"));
                    L.attn_gate = upload_weight(S, g_.find(bn(fil, "attn_gate.weight")), "attn_gate");
                    L.ssm_beta = upload_weight(S, g_.find(bn(fil, "ssm_beta.weight")), "ssm_beta");
                    L.ssm_alpha = upload_weight(S, g_.find(bn(fil, "ssm_alpha.weight")), "ssm_alpha");
                    L.ssm_dt = small(S, bn(fil, "ssm_dt.bias")); L.ssm_a = small(S, bn(fil, "ssm_a"));
                } else {
                    // the delta-net kernels pair v head j with k head j % HK: this rank owns v heads m*HK + hk0 + i
                    // (m < HV/HK, i < HKr), three row segments; local v head m*HKr + i then pairs with local k head i
                    const int M = HV_ / HK_;
                    std::vector<std::pair<int64_t, int64_t>> vrows, vheads, qkv;
                    for (int m = 0; m < M; ++m) {
                        vheads.push_back({(int64_t) m * HK_ + S.hk0, (int64_t) S.HKr});
                        vrows.push_back({((int64_t) m * HK_ + S.hk0) * DV_, (int64_t) S.HKr * DV_});
                    }
                    qkv.push_back({(int64_t) S.hk0 * DK_, (int64_t) S.HKr * DK_});
                    qkv.push_back({(int64_t) HK_ * DK_ + (int64_t) S.hk0 * DK_, (int64_t) S.HKr * DK_});
                    for (auto & vr : vrows) qkv.push_back({(int64_t) 2 * HK_ * DK_ + vr.first, vr.second});
                    L.attn_qkv = upload_rows(S, g_.find(bn(fil, "attn_qkv.weight")), "attn_qkv", qkv);
                    L.ssm_out = upload_cols(S, g_.find(bn(fil, "ssm_out.weight")), "ssm_out", vrows);
                    L.attn_gate = upload_rows(S, g_.find(bn(fil, "attn_gate.weight")), "attn_gate", vrows);
                    L.ssm_beta = upload_rows(S, g_.find(bn(fil, "ssm_beta.weight")), "ssm_beta", vheads);
                    L.ssm_alpha = upload_rows(S, g_.find(bn(fil, "ssm_alpha.weight")), "ssm_alpha", vheads);
                    auto gather_f32 = [&](const std::string & name, const std::vector<std::pair<int64_t, int64_t>> & segs, int64_t w) {
                        const gguf_tensor * t = g_.find(name);
                        if (!t) throw std::runtime_error("fn_model: missing " + name);
                        std::vector<float> full(t->nelements());
                        { const float * d = small(S, name); CK(cudaMemcpy(full.data(), d, full.size() * 4, cudaMemcpyDeviceToHost)); }
                        std::vector<float> part;
                        for (auto & sg : segs) part.insert(part.end(), full.begin() + sg.first * w, full.begin() + (sg.first + sg.second) * w);
                        return small_f32(S, name + ".tp", part);
                    };
                    L.ssm_dt = gather_f32(bn(fil, "ssm_dt.bias"), vheads, 1);
                    L.ssm_a = gather_f32(bn(fil, "ssm_a"), vheads, 1);
                    L.ssm_conv1d = gather_f32(bn(fil, "ssm_conv1d.weight"), qkv, 4);   // [C][4]: channel segments as the qkv rows
                }
                L.ssm_norm = small(S, bn(fil, "ssm_norm.weight"));
                CK(cudaMalloc(&S.conv_state[il], (size_t) S.CONVr * 3 * 4)); CK(cudaMemset(S.conv_state[il], 0, (size_t) S.CONVr * 3 * 4));
                CK(cudaMalloc(&S.ssm_state[il], (size_t) S.HVr * DK_ * DV_ * 4)); CK(cudaMemset(S.ssm_state[il], 0, (size_t) S.HVr * DK_ * DV_ * 4));
                if (mtp_il_ >= 0) {   // rollback snapshots: the state after row t of a verify batch, t = 0..MTP_ROWS-2
                    CK(cudaMalloc(&S.ssm_snap[il], (size_t) (MTP_ROWS - 1) * S.HVr * DK_ * DV_ * 4));
                    CK(cudaMalloc(&S.conv_snap[il], (size_t) (MTP_ROWS - 1) * S.CONVr * 3 * 4));
                }
            }
            L.router = upload_weight(S, g_.find(bn(fil, "ffn_gate_inp.weight")), "ffn_gate_inp");
            L.shexp_gate = small(S, bn(fil, "ffn_gate_inp_shexp.weight"));
            L.shexp_gate_w = upload_weight(S, g_.find(bn(fil, "ffn_gate_shexp.weight")), "ffn_gate_shexp");
            L.shexp_up_w = upload_weight(S, g_.find(bn(fil, "ffn_up_shexp.weight")), "ffn_up_shexp");
            L.shexp_down_w = upload_weight(S, g_.find(bn(fil, "ffn_down_shexp.weight")), "ffn_down_shexp");
            static const bool load_dbg = getenv("P100E_FN_LOAD_DEBUG") != nullptr;
            if (load_dbg) { size_t f_ = 0, t_ = 0; cudaMemGetInfo(&f_, &t_); printf("fn_model: rank %d layer %d dense done, free %.1f MiB\n", r, il, f_ / 1048576.0); fflush(stdout); }
            load_experts(S, r, il);
            if (load_dbg) { size_t f_ = 0, t_ = 0; cudaMemGetInfo(&f_, &t_); printf("fn_model: rank %d layer %d experts done, free %.1f MiB\n", r, il, f_ / 1048576.0); fflush(stdout); }
            L.pf_hh_off = (size_t) L.cold_gate.n * L.cold_gate.stride;
            L.host_bytes = L.pf_hh_off + (size_t) L.hhot_gate.n * L.hhot_gate.stride;
            if (il == mtp_il_) {
                L.enorm = small(S, bn(fil, "nextn.enorm.weight")); L.hnorm = small(S, bn(fil, "nextn.hnorm.weight"));
                L.hc_head_norm = small(S, bn(fil, "nextn.hc_head_norm.weight"));
                L.eh_proj = upload_weight(S, g_.find(bn(fil, "nextn.eh_proj.weight")), "nextn.eh_proj");
                L.hc_head_down = upload_weight(S, g_.find(bn(fil, "nextn.hc_head_down.weight")), "nextn.hc_head_down");
                L.hc_head_up = upload_weight(S, g_.find(bn(fil, "nextn.hc_head_up.weight")), "nextn.hc_head_up");
            }
            if (il == PLE_LAYER_) {
                L.ple_key = upload_weight(S, g_.find(bn(fil, "ple_key.weight")), "ple_key");
                L.ple_value = upload_weight(S, g_.find(bn(fil, "ple_value.weight")), "ple_value");
                L.ple_norm_key = small(S, bn(fil, "ple_norm_key.weight")); L.ple_norm_query = small(S, bn(fil, "ple_norm_query.weight"));
                L.ple_norm_conv = small(S, bn(fil, "ple_norm_conv.weight")); L.ple_conv1d = small(S, bn(fil, "ple_conv1d.weight"));
            }
            if (o_.verbose && (il % 8 == 0 || il == L_ - 1 || il == mtp_il_ || getenv("P100E_FN_LOAD_DEBUG"))) {
                size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
                printf("fn_model: rank %d layer %d loaded, free %.0f MiB\n", r, il, fr / 1048576.0); fflush(stdout);
            }
        }
        CK(cudaStreamSynchronize(S.s));
        size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot);
        if (o_.prefetch_min_t > 0 && T >= o_.prefetch_min_t) {
            for (int il = 0; il < L_; ++il) S.pf_bytes = std::max(S.pf_bytes, S.lw[il].host_bytes);
            if (S.pf_bytes) {
                for (int k = 0; k < 2; ++k) {
                    CK(cudaMalloc(&S.pf[k], S.pf_bytes));
                    CK(cudaEventCreateWithFlags(&S.pf_ready[k], cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&S.pf_free[k], cudaEventDisableTiming));
                }
                if (o_.verbose) printf("fn_model: rank %d prefetch arenas 2 x %.0f MiB (chunks >= %d rows)\n", r, S.pf_bytes / 1048576.0, o_.prefetch_min_t);
            }
        }
        if (r < (int) plan_tot_.size()) {   // the byte plan against what this load really took (arenas included)
            size_t f2 = 0, t2 = 0; cudaMemGetInfo(&f2, &t2);
            const double used = plan_free_[r] - (double) f2;
            printf("fn_model: rank %d byte plan check: planned %.2f GiB (incl. %.2f GiB margin), used %.2f GiB, %.2f GiB free now\n", r,
                   plan_tot_[r] / 1073741824.0, 384.0 / 1024.0, used / 1073741824.0, f2 / 1073741824.0);
        }
        printf("fn_model: rank %d (dev %d) loaded: %.0f MiB free of %.0f; host experts so far: pinned %.0f MiB, pageable %.0f MiB%s\n", r, S.dev,
               fr / 1048576.0, tot / 1048576.0, g_host_pinned_bytes / 1048576.0, g_host_pageable_bytes / 1048576.0,
               g_host_pageable_bytes ? " (PAGEABLE: cudaMallocHost failed or P100E_FN_PIN=0; staging runs at pageable speed)" : "");
    }
}

// y[T][n_out] = W x[T][n_in]: the fast GEMV for T <= 8, the fold GEMM above
void fn_model::mv(rank_state & S, const dw & w, const float * x, float * y, int T) { gemm_pk(S, w.pk, w.raw, x, y, T); }

void fn_model::sync_all() { for (auto & S : rk_) { CK(cudaSetDevice(S.dev)); CK(cudaStreamSynchronize(S.s)); } }

void fn_model::grab(const char * name, int il, int r, const float * x, int64_t n) {
    if (!dump_ || r != 0) return;
    rank_state & S = rk_[r];
    std::vector<float> h((size_t) n);
    CK(cudaMemcpyAsync(h.data(), x, n * 4, cudaMemcpyDeviceToHost, S.s));
    CK(cudaStreamSynchronize(S.s));
    dumped_[il >= 0 ? std::string(name) + "-" + std::to_string(il) : std::string(name)] = std::move(h);
}

// hyper-connection mixer: out[T][E] = mix(res), inject[T][HC] (nullptr: none)
void fn_model::hc_mix(rank_state & S, const float * norm_w, const dw & down, const dw & up, const float * inject_w, float * out, float * inject, int T) {
    cudaEvent_t * ev = nullptr;   // P100E_FN_PROF_HC: events around each kernel (rank 0)
    if (prof_hc_ && &S == &rk_[0]) {
        if (hc_used_ >= hc_ev_.size()) { hc_ev_.emplace_back(7); for (auto & e : hc_ev_.back()) CK(cudaEventCreate(&e)); }
        ev = hc_ev_[hc_used_++].data();
    }
    auto mark = [&](int i) { if (ev) CK(cudaEventRecord(ev[i], S.s)); };
    mark(0);
    fnk::rmsnorm_rows_k<<<(unsigned) (T * HC_), nthreads_for(E_), 0, S.s>>>(S.res, norm_w, S.xn, E_, HC_, eps_);   // xn [T][HC*E]
    mark(1);
    // P100E_FN_HC_TILE=1 (off by default; other summation order): the Q8_0 down matvec as row tiles x K slices with the
    // activations in shared memory + one fixed-order reduce (fn_hc_kernels.cuh; partials in S.gout, free outside the FFN)
    static const bool hc_tile = getenv("P100E_FN_HC_TILE") && atoi(getenv("P100E_FN_HC_TILE")) != 0;
    if (!(hc_tile && !down.raw && down.pk.layout == fast::LAYOUT_Q8R &&
          fnhc::q8r_tile(down.pk.data, S.xn, S.a, S.gout, T, down.pk.n_in, down.pk.n_out, S.s)))
        mv(S, down, S.xn, S.a, T);                                                                               // lo [T][HCLR]
    mark(2);
    fnk::silu_scale_k<<<nb256((int64_t) T * HCLR_), 256, 0, S.s>>>(S.a, 1.0f / (float) HC_, (int64_t) T * HCLR_);
    mark(3);
    mv(S, up, S.a, S.b, T);                                                                                      // gate [T][HC*E]
    mark(4);
    fnk::hc_mixed_k<<<dim3(nb256(E_), T), 256, 0, S.s>>>(S.xn, S.b, out, E_, HC_);
    mark(5);
    if (inject_w) fnk::dot_rows_k<<<dim3(HC_, T), 1024, 0, S.s>>>(inject_w, S.xn, inject, HCDIM_);
    mark(6);
}
void fn_model::hc_prof_flush() {
    if (!prof_hc_ || hc_used_ == 0) return;
    CK(cudaSetDevice(rk_[0].dev));
    CK(cudaEventSynchronize(hc_ev_[hc_used_ - 1][6]));
    for (size_t i = 0; i < hc_used_; ++i)
        for (int k = 0; k < 6; ++k) { float ms = 0; CK(cudaEventElapsedTime(&ms, hc_ev_[i][k], hc_ev_[i][k + 1])); hc_sum_[k] += ms; }
    hc_n_ += (int64_t) hc_used_;
    hc_used_ = 0;
}

void fn_model::hc_combine(rank_state & S, const float * block_out, const float * inject, int T) {
    fnk::hc_combine_k<<<dim3(nb256(E_), T), 256, 0, S.s>>>(S.res, block_out, inject, E_, HC_);
}

// PLE (layer PLE_LAYER_, applied to the residual before the attention mixer); tokens[i] at positions n_past_ + i
// PLE rows -> the OS: every row's page asked for at once, so the disk reads run in parallel instead of one page fault
// at a time. Windows: PrefetchVirtualMemory (as before, P100E_FN_PLE_PREFETCH). Linux: madvise(MADV_WILLNEED) per row
// page, which only queues the reads (P100E_FN_PLE_ASYNC=1; off by default). A SATA SSD (CPU-only
// bench): 64 fresh rows = 16.5 ms through serial faults, 0.75 ms after WILLNEED (0.4 ms to submit).
// Pure scheduling: the bytes read are the same.
static bool ple_async_on() {
    static const bool on = getenv("P100E_FN_PLE_ASYNC") && atoi(getenv("P100E_FN_PLE_ASYNC")) != 0;
    return on;
}
static void ple_willneed(const uint8_t * base, const int64_t * rowid, size_t n, size_t rb) {
#ifdef _WIN32
    std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges(n);
    for (size_t k = 0; k < n; ++k) { ranges[k].VirtualAddress = (void *) (base + (size_t) rowid[k] * rb); ranges[k].NumberOfBytes = rb; }
    if (n) PrefetchVirtualMemory(GetCurrentProcess(), ranges.size(), ranges.data(), 0);   // best effort
#else
    static const uintptr_t pg = (uintptr_t) sysconf(_SC_PAGESIZE);
    uintptr_t last = 0;
    for (size_t k = 0; k < n; ++k) {
        const uintptr_t a = (uintptr_t) (base + (size_t) rowid[k] * rb), a0 = a & ~(pg - 1), a1 = (a + rb + pg - 1) & ~(pg - 1);
        if (a0 == last) continue;   // the previous row's page
        madvise((void *) a0, a1 - a0, MADV_WILLNEED);   // best effort
        last = a0;
    }
#endif
}

void fn_model::ple(rank_state & S, int il, const std::vector<int32_t> & tokens) {
    const int T = (int) tokens.size();
    layer_w & L = S.lw[il];
    // n-gram hash rows (host): ctx[0] = token, ctx[s] = s-th predecessor (EOS when missing or past an EOS)
    const gguf_tensor * tab = g_.find("per_layer_token_embd.weight");
    if (!tab) throw std::runtime_error("fn_model: missing per_layer_token_embd.weight");
    const size_t rb = tensor_nbytes(tab->type, tab->ne) / (size_t) tab->ne[1];
    std::vector<uint8_t> rows((size_t) T * PLE_HEADS_ * rb);
    // the table is a memory-mapped 28.8 GB file: each row is its own page fault (~0.1 ms from disk), 48 rows per token.
    // Pass 1 computes every row; the OS is asked to fetch all of them at once (PrefetchVirtualMemory: batched,
    // parallel disk reads); pass 2 copies. Without this a 512-token chunk spent ~2.2 s here on a SATA disk.
    std::vector<int64_t> rowid;
    ple_rowids(tokens, n_past_, rowid);
    {
        static const bool pf_on = !getenv("P100E_FN_PLE_PREFETCH") || atoi(getenv("P100E_FN_PLE_PREFETCH")) != 0;
#ifdef _WIN32
        if (pf_on) ple_willneed(tab->data, rowid.data(), rowid.size(), rb);
#else
        if (pf_on && ple_async_on()) ple_willneed(tab->data, rowid.data(), rowid.size(), rb);
#endif
    }
    for (size_t k = 0; k < rowid.size(); ++k) memcpy(rows.data() + k * rb, tab->data + (size_t) rowid[k] * rb, rb);
    CK(cudaMemcpyAsync(S.qstage, rows.data(), rows.size(), cudaMemcpyHostToDevice, S.s));
    dequant_f32(tab->type, S.qstage, S.ple_emb, (int64_t) T * E_, S.s);   // [T][heads*dim] = [T][E]
    grab("ple_embd", -1, (int) (&S - rk_.data()), S.ple_emb, (int64_t) T * E_);
    mv(S, L.ple_key, S.ple_emb, S.a, T);                                       // key [T][HC*E]
    mv(S, L.ple_value, S.ple_emb, S.b, T);                                     // value [T][E]
    fnk::rmsnorm_rows_k<<<(unsigned) (T * HC_), nthreads_for(E_), 0, S.s>>>(S.a, L.ple_norm_key, S.a, E_, HC_, eps_);
    fnk::rmsnorm_rows_k<<<(unsigned) (T * HC_), nthreads_for(E_), 0, S.s>>>(S.res, L.ple_norm_query, S.xn, E_, HC_, eps_);
    fnk::ple_gate_k<<<dim3(HC_, T), 1024, 0, S.s>>>(S.a, S.xn, S.inject, E_);          // gate [T][HC] (S.inject as scratch)
    grab("ple_gate", il, (int) (&S - rk_.data()), S.inject, (int64_t) T * HC_);
    fnk::ple_gated_k<<<dim3(nb256(E_), T), 256, 0, S.s>>>(S.b, S.inject, S.c, E_, HC_);   // gated [T][HC*E]
    // normalized rows appended to the conv history (rows PLE_HIST_.. of ple_hist), conv, residual update
    float * xall = S.ple_hist;
    fnk::rmsnorm_rows_k<<<(unsigned) (T * HC_), nthreads_for(E_), 0, S.s>>>(S.c, L.ple_norm_conv, xall + (size_t) PLE_HIST_ * HCDIM_, E_, HC_, eps_);
    fnk::ple_conv_k<<<dim3(nb256(HCDIM_), T), 256, 0, S.s>>>(xall, L.ple_conv1d, S.c, S.res, HCDIM_, PLE_HIST_, PLE_KERN_, PLE_NGRAM_, T);
    // history <- last PLE_HIST_ rows of xall (oldest first)
    CK(cudaMemcpyAsync(S.b, xall + (size_t) T * HCDIM_, (size_t) PLE_HIST_ * HCDIM_ * 4, cudaMemcpyDeviceToDevice, S.s));
    CK(cudaMemcpyAsync(xall, S.b, (size_t) PLE_HIST_ * HCDIM_ * 4, cudaMemcpyDeviceToDevice, S.s));
}

void fn_model::layer_attn(rank_state & S, int il, const float * cur, float * out, int T) {
    layer_w & L = S.lw[il];
    mv(S, L.attn_q, cur, S.a, T);                                               // [T][H][2*HD]
    k::split_q_gate(S.a, S.c, S.d, T, S.Hr, HD_, S.s);                            // q -> c [T][H][HD], gate -> d [T][H*HD]
    k::rmsnorm(S.c, L.attn_q_norm, S.c, (int64_t) T * S.Hr, HD_, eps_, S.s);
    mv(S, L.attn_k, cur, S.e, T);                                               // [T][HKV][HD]
    k::rmsnorm(S.e, L.attn_k_norm, S.e, (int64_t) T * S.HKVr, HD_, eps_, S.s);
    mv(S, L.attn_v, cur, S.g, T);
    rope_rows(S, S.c, T, S.Hr, HD_);
    rope_rows(S, S.e, T, S.HKVr, HD_);
    if (kv_mode_ != 0) qsa::kv_store(kv_mode_, S.e, S.g, S.kc[il], S.vc[il], T, n_past_, S.HKVr, HD_, S.s);
    else k::kv_store(S.e, S.g, (float *) S.kc[il], (float *) S.vc[il], T, n_past_, S.HKVr * HD_, S.s);
    static const int cublas_attn_t = getenv("P100E_FN_ATTN_T") ? atoi(getenv("P100E_FN_ATTN_T")) : 5;
    const int N_all = n_past_ + T;
    static const bool qsa_force = getenv("P100E_FN_QSA_FORCE") && atoi(getenv("P100E_FN_QSA_FORCE")) != 0;   // test: sparse path even when it selects every cell
    const bool sparse = qsa_ && (N_all > QSA_W_ || qsa_force);
    // the dense paths read f32 rows: the cache itself, or (f16 cache) the window [0, N_all) converted into kf / vf
    const float * kd = (const float *) S.kc[il], * vd = (const float *) S.vc[il];
    if (kv_mode_ != 0 && !sparse) { qsa::kv_window_f32(kv_mode_, S.kc[il], S.vc[il], S.kf, S.vf, N_all, S.HKVr, HD_, S.s); kd = S.kf; vd = S.vf; }
    if (qsa_) {
        // indexer keys of this chunk: raw (f16-rounded as the reference caches them); complete blocks [b0, b1) get their
        // pooled, normed, roped key; the raw keys of the incomplete tail are carried for the next chunk
        mv(S, L.idx_k, cur, S.qk_raw, T);
        qsa::round_f16(S.qk_raw, (int64_t) T * IDX_D_, S.s);
        const int b0 = n_past_ / RATIO_, b1 = N_all / RATIO_;
        if (b1 > b0) {
            float * kb = S.kblk[il] + (size_t) b0 * IDX_D_;
            qsa::pool_blocks(S.ktail[il], S.qk_raw, n_past_, b0, b1, RATIO_, QSA_RING_, IDX_D_, kb, S.s);
            k::rmsnorm(kb, L.idx_k_norm, kb, b1 - b0, IDX_D_, eps_, S.s);
            rope_blocks(S, kb, b0, b1 - b0);
        }
        qsa::save_tail(S.qk_raw, n_past_, T, QSA_RING_, IDX_D_, S.ktail[il], S.s);   // the last QSA_RING raw keys (tail + MTP rollback)
        grab("indexer_k", il, (int) (&S - rk_.data()), S.kblk[il], (int64_t) b1 * IDX_D_);
    }
    if (sparse) {
        // sparse: score the full blocks per query (cuBLAS, slabs of QSA_SL rows), select the top cells, attend over them
        mv(S, L.idx_q, cur, S.qq, T);
        k::rmsnorm(S.qq, L.idx_q_norm, S.qq, (int64_t) T * IDX_H_, IDX_D_, eps_, S.s);
        rope_rows(S, S.qq, T, IDX_H_, IDX_D_);
        grab("indexer_q", il, (int) (&S - rk_.data()), S.qq, (int64_t) T * IDX_H_ * IDX_D_);
        const int NB = N_all / RATIO_;   // block keys available (every query's visible blocks are among them)
        const float one = 1.0f, zero = 0.0f;
        CB(cublasSetStream(S.cb, S.s));
        for (int t0 = 0; t0 < T; t0 += QSA_SL_) {
            const int m = std::min(QSA_SL_, T - t0);
            // row-major S[m*H][NB] = Q[m*H][D] . Kb[NB][D]^T  (column-major view: S^T = Kb^T-as-(D x NB) ... Q)
            CB(cublasSgemm(S.cb, CUBLAS_OP_T, CUBLAS_OP_N, NB, m * IDX_H_, IDX_D_, &one, S.kblk[il], IDX_D_, S.qq + (size_t) t0 * IDX_H_ * IDX_D_, IDX_D_, &zero, S.qsa_s, NB));
            qsa::combine(S.qsa_s, S.qsa_c, S.pos + t0, m, IDX_H_, NB, RATIO_, S.s);
            qsa::select(S.qsa_c, NB, S.pos + t0, m, RATIO_, QSA_W_, S.sel + (size_t) t0 * QSA_W_, S.seln + t0, S.s);
        }
        qsa::attention(S.c, S.kc[il], S.vc[il], kv_mode_, S.sel, S.seln, S.b, T, S.Hr, S.HKVr, HD_, QSA_W_, 1.0f / sqrtf((float) HD_), S.s);
    } else if (cublas_attn_t > 0 && T >= cublas_attn_t) {
        // prefill: per q head two f32 GEMMs through cuBLAS + a causal softmax (the reference kernel is O(T^2) serial per
        // query and takes seconds at 2048 rows); scores [T][N] in S.gout (free until the FFN). cuBLAS column-major view:
        // Q_h^T is HD x T with ld H*HD, K_g^T is HD x N with ld HKV*HD -> S^T (N x T) = K Q^T; O^T (HD x T) = V^T P^T.
        const int N = n_past_ + T, G = S.Hr / S.HKVr;
        if ((size_t) T * N > (size_t) o_.max_t * NEXPU_ * std::max(E_, FFE_)) throw std::runtime_error("fn_model: attention scratch too small for this context");
        const float one = 1.0f, zero = 0.0f;
        float * sc = S.gout;
        CB(cublasSetStream(S.cb, S.s));
        for (int h = 0; h < S.Hr; ++h) {
            const int g = h / G;
            const float * q = S.c + (size_t) h * HD_;
            const float * k = kd + (size_t) g * HD_;
            const float * v = vd + (size_t) g * HD_;
            CB(cublasSgemm(S.cb, CUBLAS_OP_T, CUBLAS_OP_N, N, T, HD_, &one, k, S.HKVr * HD_, q, S.Hr * HD_, &zero, sc, N));
            fnk::causal_softmax_k<<<T, 256, 0, S.s>>>(sc, N, n_past_, 1.0f / sqrtf((float) HD_));
            CB(cublasSgemm(S.cb, CUBLAS_OP_N, CUBLAS_OP_N, HD_, T, N, &one, v, S.HKVr * HD_, sc, N, &zero, S.b + (size_t) h * HD_, S.Hr * HD_));
        }
    } else {
        // decode / MTP verify: the serial reference kernel (default), or P100E_FN_ATTN_FAST=1 its online-softmax
        // warp-per-key twin (not bit-identical; falls back to the reference for a shape it does not take)
        static const bool attn_fast = getenv("P100E_FN_ATTN_FAST") && atoi(getenv("P100E_FN_ATTN_FAST")) != 0;
        if (!attn_fast || !k::attention_fast(S.c, kd, vd, S.b, T, n_past_, S.Hr, S.HKVr, HD_, 1.0f / sqrtf((float) HD_), S.s))
            k::attention(S.c, kd, vd, S.b, T, n_past_, S.Hr, S.HKVr, HD_, 1.0f / sqrtf((float) HD_), S.s);
    }
    k::sigmoid_mul(S.b, S.d, (int64_t) T * S.Hr * HD_, S.s);
    mv(S, L.attn_output, S.b, out, T);
}

void fn_model::layer_gdn(rank_state & S, int il, const float * cur, float * out, int T) {
    layer_w & L = S.lw[il];
    mv(S, L.attn_qkv, cur, S.a, T);                                             // [T][CONV_C]
    mv(S, L.attn_gate, cur, S.b, T);                                            // z [T][HV*DV]
    mv(S, L.ssm_beta, cur, S.d, T);                                             // [T][HV]
    mv(S, L.ssm_alpha, cur, S.e, T);                                            // [T][HV]
    static const bool gdn_ref = getenv("P100E_FN_GDN_REF") && atoi(getenv("P100E_FN_GDN_REF")) != 0;
    if (gdn_ref) {   // reference per-op kernels (src/kernels.cu) + the sigmoid-gated norm
        k::gdn_gates(S.e, S.d, L.ssm_dt, L.ssm_a, T, S.HVr, S.s);
        k::conv1d_silu(S.a, S.conv_state[il], L.ssm_conv1d, S.c, T, S.CONVr, S.s);
        k::gdn_recurrent(S.c, S.e, S.d, S.ssm_state[il], S.g, T, S.HKr, S.HVr, DK_, DV_, eps_, S.s);   // [T][HV][DV]
        grab("attn_output", il, (int) (&S - rk_.data()), S.g, (int64_t) T * S.HVr * DV_);
        fnk::rmsnorm_sig_gated_k<<<(unsigned) (T * S.HVr), nthreads_for(DV_), 0, S.s>>>(S.g, L.ssm_norm, S.b, S.g, DV_, eps_);
    } else {         // fast fused delta net (src/fast/ssm.cu) with the sigmoid output gate
        fast::GdnWeights w; w.conv_w = L.ssm_conv1d; w.dt_bias = L.ssm_dt; w.ssm_a = L.ssm_a; w.norm_w = L.ssm_norm; w.gate = 1;
        const bool snap = S.ssm_snap[il] && T >= 2 && T <= MTP_ROWS;   // MTP verify batch: per-row state snapshots for rollback
        fast::gdn_layer(S.a, S.e, S.d, S.b, w, S.conv_state[il], S.ssm_state[il], S.c, S.e, S.d, S.g, T, S.HKr, S.HVr, DK_, eps_, S.s,
                        nullptr, snap ? S.conv_snap[il] : nullptr, snap ? S.ssm_snap[il] : nullptr);
    }
    mv(S, L.ssm_out, S.g, out, T);
}

// router on rank 0 (the mixed input is replicated): probs = softmax(logits), top-k by prob (ties: lower id), weights
// normalized to sum 1 (norm_w) and scaled by expert_weights_scale when set. ids/wts: [T][NEXPU]
void fn_model::route(int il, int T, std::vector<int> & ids, std::vector<float> & wts) {
    rank_state & S = rk_[0];
    CK(cudaSetDevice(S.dev));
    mv(S, S.lw[il].router, S.cur, S.router_out, T);
    if (cpu_) cpu_stage_x(S, T);   // P100E_FN_CPU: the FFN input rows for the CPU pool (lands with the sync below)
    if (rexp_ && rexp_->x && T <= o_.rexp_max_t && rexp_->li_of[il] >= 0)   // remote experts: the same rows for node 1
        CK(cudaMemcpyAsync(rexp_->x, S.cur, (size_t) T * E_ * 4, cudaMemcpyDeviceToHost, S.s));
    ids.assign((size_t) T * NEXPU_, -1); wts.assign((size_t) T * NEXPU_, 0.0f);
    static const bool host_route = getenv("P100E_FN_HOST_ROUTE") && atoi(getenv("P100E_FN_HOST_ROUTE")) != 0;
    if (!host_route && NEXP_ == 512 && NEXPU_ <= 32) {   // GPU softmax + top-K; only T x K ids / weights cross PCIe
        fnk::route_k<512><<<T, 512, 0, S.s>>>(S.router_out, S.route_ids, S.route_wts, NEXPU_, exp_scale_);
        CK(cudaMemcpyAsync(S.route_ids_pin, S.route_ids, ids.size() * 4, cudaMemcpyDeviceToHost, S.s));
        CK(cudaMemcpyAsync(S.route_wts_pin, S.route_wts, wts.size() * 4, cudaMemcpyDeviceToHost, S.s));
        CK(cudaStreamSynchronize(S.s));
        memcpy(ids.data(), S.route_ids_pin, ids.size() * 4);
        memcpy(wts.data(), S.route_wts_pin, wts.size() * 4);
        return;
    }
    std::vector<float> lg((size_t) T * NEXP_);
    CK(cudaMemcpyAsync(lg.data(), S.router_out, lg.size() * 4, cudaMemcpyDeviceToHost, S.s));
    CK(cudaStreamSynchronize(S.s));
    std::vector<float> p(NEXP_);
    std::vector<int> ord(NEXP_);
    for (int t = 0; t < T; ++t) {
        const float * l = lg.data() + (size_t) t * NEXP_;
        float mx = l[0];
        for (int e = 1; e < NEXP_; ++e) mx = std::max(mx, l[e]);
        float sum = 0.0f;
        for (int e = 0; e < NEXP_; ++e) { p[e] = expf(l[e] - mx); sum += p[e]; }
        for (int e = 0; e < NEXP_; ++e) p[e] /= sum;
        std::iota(ord.begin(), ord.end(), 0);
        std::stable_sort(ord.begin(), ord.end(), [&](int x, int y) { return p[x] > p[y]; });
        float ws = 0.0f;
        for (int k = 0; k < NEXPU_; ++k) { ids[(size_t) t * NEXPU_ + k] = ord[k]; ws += p[ord[k]]; }
        for (int k = 0; k < NEXPU_; ++k) {
            float w = p[ord[k]] / ws;
            if (exp_scale_ != 0.0f) w *= exp_scale_;
            wts[(size_t) t * NEXPU_ + k] = w;
        }
    }
}


static bool grouped_layout(int layout) {
    return layout == fast::LAYOUT_Q4KR || layout == fast::LAYOUT_Q4R || layout == fast::LAYOUT_Q10 || layout == fast::LAYOUT_Q20 ||
           layout == fast::LAYOUT_IQ2SR;
}
static bool grouped_nc_layout(int layout) {   // layouts whose grouped kernel takes NC = 4 columns per group
    return layout == fast::LAYOUT_Q4KR || layout == fast::LAYOUT_Q4R || layout == fast::LAYOUT_IQ2SR;
}
static void launch_grouped(int layout, const fast::mv_group * tab, int G, const fast::a16_args & a, cudaStream_t s, int nc = 1) {
    if (G <= 0) return;
    switch (layout) {
        case fast::LAYOUT_Q4KR:  fast::launch_q4kr_grouped_nc(tab, G, a, nc, s); break;
        case fast::LAYOUT_Q4R:   fast::launch_q4r_grouped_nc(tab, G, a, nc, s); break;
        case fast::LAYOUT_IQ2SR: fast::launch_iq2sr_grouped_nc(tab, G, a, nc, s); break;
        case fast::LAYOUT_Q10:   fast::launch_q10_grouped(tab, G, a, s); break;
        default:                 fast::launch_q20_grouped(tab, G, a, s); break;
    }
}

// Large chunks use (nearly) every expert of a layer, so routing is not needed to start the copies: the layer's whole
// host set (cold records, then hhot records) goes into arena il & 1 on the copy stream, one copy per set, while the
// previous layer computes. pf_free[k] (recorded after the compute that read arena k) gates the refill.
void fn_model::prefetch_layer(rank_state & S, int il) {
    const int k = il & 1;
    layer_w & L = S.lw[il];
    if (!S.pf[k] || L.host_bytes == 0) { S.pf_layer[k] = -1; return; }
    CK(cudaStreamWaitEvent(S.cs, S.pf_free[k], 0));
    if (L.cold_gate.n) CK(cudaMemcpyAsync(S.pf[k], L.cold_gate.base, (size_t) L.cold_gate.n * L.cold_gate.stride, cudaMemcpyHostToDevice, S.cs));
    if (L.hhot_gate.n) CK(cudaMemcpyAsync(S.pf[k] + L.pf_hh_off, L.hhot_gate.base, (size_t) L.hhot_gate.n * L.hhot_gate.stride, cudaMemcpyHostToDevice, S.cs));
    CK(cudaEventRecord(S.pf_ready[k], S.cs));
    S.pf_layer[k] = il;
}

// Any T: this rank's (expert, token) pairs as grouped launches. The routed rows are gathered once in expert order and
// a16-quantized once (the bytes matvec would make); experts are processed in batches: a batch is the run of experts
// whose host-staged weights fit one half of the staging arena (resident experts take no arena). Per batch and weight
// layout: ONE grouped launch per projection (gate | up on the gathered rows, silu, a16 of the batch's rows, down), i.e.
// a few launches per batch instead of 7 per expert. The copies of batch i+1 (copy stream) overlap the compute of batch i
// (two arena halves, st_free / st_ready events). Finally out[t] = sum over the token's rows in ascending expert id
// (combine_rows_k): a fixed order per token, so the sum is the same bits whatever the chunking (T-invariant).
bool fn_model::ffn_grouped(rank_state & S, int il, const float * cur, float * out, int T, const std::map<int, std::vector<std::pair<int, float>>> & by_e,
                           const std::vector<char> * cpu_e, int * nr) {
    layer_w & L = S.lw[il];
    auto ok = [&](const expert_set & g, const expert_set & u, const expert_set & d) {
        return g.n == 0 || (grouped_layout(g.layout) && g.layout == u.layout && grouped_layout(d.layout));
    };
    if (!ok(L.hot_gate, L.hot_up, L.hot_down) || !ok(L.hhot_gate, L.hhot_up, L.hhot_down) || !ok(L.cold_gate, L.cold_up, L.cold_down)) return false;
    // rows in expert order; rowof[t][k] = the k-th row of token t (k runs in expert order)
    struct ex { int e, hl, hh, cl, r0, n; };
    std::vector<ex> xs;
    std::vector<int> cnt(T, 0);
    for (int i = 0; i < T * NEXPU_; ++i) S.rowof_pin[i] = -1;
    // CPU experts (cpu_e): their rows come after every GPU row, so the GPU batches below see rows 0..NRg-1 only
    auto on_cpu = [&](int e) { return cpu_e && (*cpu_e)[e]; };
    int NRg = 0;
    if (cpu_e) for (auto & [e, lst] : by_e) if (on_cpu(e)) continue; else if (L.hot_map[e] >= 0 || L.hhot_map(e) >= 0 || L.cold_map[e] >= 0) NRg += (int) lst.size();
    int NR = 0, NRc = NRg;
    for (auto & [e, lst] : by_e) {
        const int hl = L.hot_map[e], hh = L.hhot_map(e), cl = L.cold_map[e];
        if (hl < 0 && hh < 0 && cl < 0) continue;   // another rank's expert
        int & R0 = on_cpu(e) ? NRc : NR;
        xs.push_back({e, hl, hh, cl, R0, (int) lst.size()});
        for (auto & [t, w] : lst) { S.hidx_pin[R0] = t; S.hw_pin[R0] = w; S.rowof_pin[t * NEXPU_ + cnt[t]++] = R0; ++R0; }
    }
    if (cpu_e) { nr[0] = NRg; nr[1] = NRc; NR = NRc; }
    if (NR == 0) return true;   // out is already zero
    if (L.adaptive) for (const ex & x : xs) { L.cnt[x.e] += (float) x.n; L.rows += x.n; if (x.hl >= 0) L.hits += x.n; }
    if (o_.verbose) {   // host-expert reuse statistics: distinct host experts touched per layer vs fetches
        auto & st = S.stage_stats[il];
        for (const ex & x : xs) if (x.hl < 0 && !on_cpu(x.e)) { st.fetches += 1; st.seen.insert(x.e); }
    }
    CK(cudaMemcpyAsync(S.gidx, S.hidx_pin, (size_t) NR * 4, cudaMemcpyHostToDevice, S.s));
    CK(cudaMemcpyAsync(S.gw, S.hw_pin, (size_t) NR * 4, cudaMemcpyHostToDevice, S.s));
    CK(cudaMemcpyAsync(S.rowof, S.rowof_pin, (size_t) T * NEXPU_ * 4, cudaMemcpyHostToDevice, S.s));
    if (cpu_e) {
        xs.erase(std::remove_if(xs.begin(), xs.end(), [&](const ex & x) { return on_cpu(x.e); }), xs.end());
        if (NRg == 0) return true;   // every row of this rank is on the CPU: the caller combines
    }
    const int NRG = cpu_e ? NRg : NR;   // rows the GPU computes
    fnk::gather_rows_k<<<dim3(nb256(E_), NRG), 256, 0, S.s>>>(cur, S.gidx, S.gath, E_);
    fast::matvec_quantize(S.gath, E_, NRG, S.qa, S.s);
    const int ng = E_ / 32, ngd = FFE_ / 32;
    const size_t arena = S.qstage_bytes / 2, TABCAP = (size_t) o_.max_t * NEXPU_ * 3;
    auto al = [](size_t b) { return (b + 255) & ~(size_t) 255; };
    struct ltab { std::vector<fast::mv_group> g, u, g4, u4; int rb = 0; };   // singles and quads (NC = 4 groups)
    struct dtab { std::vector<fast::mv_group> d, d4; int rb = 0; };
    static const int fold_min = getenv("P100E_FN_FOLD_MIN") ? atoi(getenv("P100E_FN_FOLD_MIN")) : 16;   // rows from which an expert uses the fold GEMM (0 = never)
    static const int fold_max = getenv("P100E_FN_FOLD_MAX") ? atoi(getenv("P100E_FN_FOLD_MAX")) : 48;   // ... and up to which (P100, 2048-row chunks: the grouped quads won above ~48 rows)
    int set = 0;
    bool first = true;   // the layer's first batch holds few staged experts: its copies land fast and the GPU starts early
    constexpr int FIRST_BATCH = 12;
    // prefetched layer: every host expert already sits in arena il & 1 (no copies, one batch)
    const bool pf = S.pf_layer[il & 1] == il;
    const char * pfa = pf ? S.pf[il & 1] : nullptr;
    if (pf) { CK(cudaStreamWaitEvent(S.s, S.pf_ready[il & 1], 0)); first = false; }
    for (size_t i = 0; i < xs.size();) {
        // the host may refill set's pinned tables / arena once the compute that used it is done
        CK(cudaEventSynchronize(S.st_free[set]));
        CK(cudaStreamWaitEvent(S.cs, S.st_free[set], 0));
        char * ar = (char *) S.qstage + (size_t) set * arena;
        size_t ap = 0;
        int nstaged = 0;
        bool any_copy = false;
        std::map<int, ltab> gu; std::map<int, dtab> dn;
        struct fold_job { ex x; fast::packed_weight pg, pu; const void * wd; int ld, rbd; };
        std::vector<fold_job> fold_jobs;
        const int rb0 = xs[i].r0;
        size_t j = i;
        for (; j < xs.size(); ++j) {
            const ex & x = xs[j];
            const void * wg; const void * wu; const void * wd;
            int lg, ld, rbg, rbd;
            if (x.hl >= 0) {
                wg = L.hot_gate.pk[x.hl].data; wu = L.hot_up.pk[x.hl].data; wd = L.hot_down.pk[x.hl].data;
                lg = L.hot_gate.layout; ld = L.hot_down.layout;
                rbg = (int) (L.hot_gate.bytes / (size_t) L.hot_gate.n_out); rbd = (int) (L.hot_down.bytes / (size_t) L.hot_down.n_out);
            } else {
                expert_set & G = x.hh >= 0 ? L.hhot_gate : L.cold_gate;
                expert_set & U = x.hh >= 0 ? L.hhot_up : L.cold_up;
                expert_set & D = x.hh >= 0 ? L.hhot_down : L.cold_down;
                const int idx = x.hh >= 0 ? x.hh : x.cl;
                const size_t need = G.stride;   // the expert's gate | up | down record
                char * dst;
                if (pf) {
                    dst = (char *) pfa + (x.hh >= 0 ? L.pf_hh_off : 0) + (size_t) idx * G.stride;
                } else {
                    if (ap + need > arena || (first && nstaged >= FIRST_BATCH)) {
                        if (ap == 0) throw std::runtime_error("fn_model: staging arena smaller than one expert (P100E_FN_STAGE_MB)");
                        break;   // next batch
                    }
                    dst = ar + ap; ap += need; ++nstaged;
                    CK(cudaMemcpyAsync(dst, G.base + (size_t) idx * G.stride, need, cudaMemcpyHostToDevice, S.cs));
                    any_copy = true;
                }
                wg = dst; wu = dst + (U.base - G.base); wd = dst + (D.base - G.base);
                lg = G.layout; ld = D.layout;
                rbg = (int) (G.bytes / (size_t) G.n_out); rbd = (int) (D.bytes / (size_t) D.n_out);
            }
            if (fold_min > 0 && x.n >= fold_min && x.n <= fold_max) {
                // many rows: fold GEMM per projection on the expert's contiguous rows (x = gathered rows, y = S.a / S.b rows);
                // its down projection reads the f32 silu rows of S.a (after the batch's silu_mul), so it is deferred
                fast::packed_weight pg, pu;
                pg.type = x.hl >= 0 ? L.hot_gate.type : (x.hh >= 0 ? L.hhot_gate.type : L.cold_gate.type);
                pg.n_in = E_; pg.n_out = FFE_; pg.data = (void *) wg; pg.bytes = (size_t) rbg * FFE_; pg.layout = lg;
                pu = pg; pu.data = (void *) wu;
                fold_jobs.push_back({x, pg, pu, wd, ld, rbd});
                continue;
            }
            ltab & tg = gu[lg]; dtab & tdn = dn[ld];
            tg.rb = rbg; tdn.rb = rbd;
            // the expert's rows are contiguous (r0 .. r0+n-1): quads where the layout's kernel takes 4 columns, then singles
            int r = x.r0;
            const int r1 = x.r0 + x.n;
            if (grouped_nc_layout(lg)) for (; r + 4 <= r1; r += 4) { tg.g4.push_back({wg, r, r}); tg.u4.push_back({wu, r, r}); }
            for (; r < r1; ++r) { tg.g.push_back({wg, r, r}); tg.u.push_back({wu, r, r}); }
            r = x.r0;
            if (grouped_nc_layout(ld)) for (; r + 4 <= r1; r += 4) tdn.d4.push_back({wd, r - rb0, r});
            for (; r < r1; ++r) tdn.d.push_back({wd, r - rb0, r});
        }
        const int rb1 = xs[j - 1].r0 + xs[j - 1].n, nb = rb1 - rb0;
        if (any_copy) { CK(cudaEventRecord(S.st_ready[set], S.cs)); CK(cudaStreamWaitEvent(S.s, S.st_ready[set], 0)); }
        // tables of this batch -> pinned set -> device set (one copy)
        fast::mv_group * hp = (fast::mv_group *) S.tab_pin[set];
        fast::mv_group * dp = (fast::mv_group *) S.gtab + set * TABCAP;
        size_t pos = 0;
        auto up = [&](const std::vector<fast::mv_group> & v) { memcpy(hp + pos, v.data(), v.size() * sizeof(fast::mv_group)); const fast::mv_group * d = dp + pos; pos += v.size(); return d; };
        struct dptr { const fast::mv_group *g = nullptr, *u = nullptr, *g4 = nullptr, *u4 = nullptr, *d = nullptr, *d4 = nullptr; };
        std::map<int, dptr> dgu, ddn;
        for (auto & [lay, t] : gu) { dptr & p = dgu[lay]; p.g = up(t.g); p.u = up(t.u); p.g4 = up(t.g4); p.u4 = up(t.u4); }
        for (auto & [lay, t] : dn) { dptr & p = ddn[lay]; p.d = up(t.d); p.d4 = up(t.d4); }
        if (pos > TABCAP) throw std::runtime_error("fn_model: grouped table overflow");
        CK(cudaMemcpyAsync(dp, hp, pos * sizeof(fast::mv_group), cudaMemcpyHostToDevice, S.s));
        // gate | up: columns = gathered rows (absolute), outputs = rows of S.a / S.b [NR][FFE]
        fast::a16_args a;
        a.aq = (const half2 *) S.qa; a.ads = (const half2 *) S.qa + (size_t) NRG * ng * 16;
        a.ngroups = ng; a.nrows = (int) FFE_;
        for (auto & [lay, t] : gu) {
            a.row_bytes = t.rb;
            const dptr & p = dgu[lay];
            a.dst = S.a; launch_grouped(lay, p.g4, (int) t.g4.size(), a, S.s, 4); launch_grouped(lay, p.g, (int) t.g.size(), a, S.s, 1);
            a.dst = S.b; launch_grouped(lay, p.u4, (int) t.u4.size(), a, S.s, 4); launch_grouped(lay, p.u, (int) t.u.size(), a, S.s, 1);
        }
        for (const fold_job & fj : fold_jobs) {   // gate | up through the fold GEMM on the expert's rows
            const float * xr = S.gath + (size_t) fj.x.r0 * E_;
            fast::gemm_packed(fj.pg, xr, S.a + (size_t) fj.x.r0 * FFE_, fj.x.n, S.gemm_scratch, S.gemm_bytes, S.s, S.cb);
            fast::gemm_packed(fj.pu, xr, S.b + (size_t) fj.x.r0 * FFE_, fj.x.n, S.gemm_scratch, S.gemm_bytes, S.s, S.cb);
        }
        k::silu_mul(S.a + (size_t) rb0 * FFE_, S.b + (size_t) rb0 * FFE_, (int64_t) nb * FFE_, S.s);
        // down: the batch's nb silu rows as a16 columns (col = r - rb0), outputs = rows of S.gout [NR][E]
        fast::matvec_quantize(S.a + (size_t) rb0 * FFE_, FFE_, nb, S.q2, S.s);
        fast::a16_args d;
        d.aq = (const half2 *) S.q2; d.ads = (const half2 *) S.q2 + (size_t) nb * ngd * 16;
        d.ngroups = ngd; d.nrows = (int) E_; d.dst = S.gout;
        for (auto & [lay, t] : dn) {
            d.row_bytes = t.rb;
            const dptr & p = ddn[lay];
            launch_grouped(lay, p.d4, (int) t.d4.size(), d, S.s, 4); launch_grouped(lay, p.d, (int) t.d.size(), d, S.s, 1);
        }
        for (const fold_job & fj : fold_jobs) {   // down through the fold GEMM: f32 silu rows of S.a -> S.gout rows
            fast::packed_weight pd;
            pd.type = fj.x.hl >= 0 ? L.hot_down.type : (fj.x.hh >= 0 ? L.hhot_down.type : L.cold_down.type);
            pd.n_in = FFE_; pd.n_out = E_; pd.data = (void *) fj.wd; pd.bytes = (size_t) fj.rbd * E_; pd.layout = fj.ld;
            fast::gemm_packed(pd, S.a + (size_t) fj.x.r0 * FFE_, S.gout + (size_t) fj.x.r0 * E_, fj.x.n, S.gemm_scratch, S.gemm_bytes, S.s, S.cb);
        }
        CK(cudaEventRecord(S.st_free[set], S.s));
        set ^= 1;
        first = false;
        i = j;
    }
    if (pf) { CK(cudaEventRecord(S.pf_free[il & 1], S.s)); S.pf_layer[il & 1] = -1; }
    if (cpu_e) return true;   // the caller adds the CPU rows, then combines
    fnk::combine_rows_k<<<dim3(nb256(E_), T), 256, 0, S.s>>>(out, S.gout, S.rowof, S.gw, E_, NEXPU_);
    return true;
}

// this rank's share of the MoE FFN of T rows: out[T][E] = sum over its selected experts (+ the shared expert on rank 0).
// Prefill-friendly: the rows routed to one expert are gathered once and run through its three weights as one T_e-row call.
void fn_model::layer_ffn(rank_state & S, int r, int il, const float * cur, float * out, int T, const std::vector<int> & ids, const std::vector<float> & wts) {
    layer_w & L = S.lw[il];
    CK(cudaMemsetAsync(out, 0, (size_t) T * E_ * 4, S.s));
    static const bool grouped_on = !getenv("P100E_FN_GROUPED") || atoi(getenv("P100E_FN_GROUPED")) != 0;
    // group (token, weight) pairs by expert
    std::map<int, std::vector<std::pair<int, float>>> by_e;
    ffn_by_e(S, il, T, ids, wts, by_e);
    if (rexp_) {   // remote experts: rank 0 picks and sends every rank's node-1 experts, the last rank combines
        if (r == 0) rexp_begin(il, T, ids, wts);
        if (rexp_->layer_on) {
            auto & q = rexp_->rk[r];
            int nr[2] = {0, 0};
            ffn_grouped(S, il, cur, out, T, by_e, &q.is_rem, nr);   // rexp_begin checked the layouts: returns true
            q.cur = cur; q.out = out; q.nrg = nr[0]; q.nr = nr[1];
            // the shared expert's matvecs run on the GPU while the host waits for node 1 (into S.c / S.d, which the
            // remote rows' upload and the combine do not touch); only its add waits for the combine (same bits)
            if (r == 0 && rexp_->early_shared) ffn_shared_pre(S, il, cur, T);
            if (r == (int) rk_.size() - 1) rexp_finish(il, T);
            return;
        }
    }
    if (cpu_) {   // P100E_FN_CPU: rank 0 picks and submits the CPU experts of every rank, the last rank combines
        if (r == 0) cpu_begin(il, T, ids, wts);
        if (cpu_->layer_on) {
            auto & cr = cpu_->rk[r];
            int nr[2] = {0, 0};
            ffn_grouped(S, il, cur, out, T, by_e, &cr.is_cpu, nr);   // cpu_begin checked the layouts: returns true
            cr.cur = cur; cr.out = out; cr.nrg = nr[0]; cr.nr = nr[1];
            if (r == (int) rk_.size() - 1) cpu_finish(il, T);
            return;
        }
    }
    const bool grouped = grouped_on && ffn_grouped(S, il, cur, out, T, by_e);
    if (grouped) by_e.clear();   // the per-expert path below runs only for layouts without grouped kernels
    // Host-staged experts: two staging sets (3 slots each) so the PCIe copy of the next host expert (copy stream S.cs)
    // overlaps this one's compute on S.s; st_free[k] (recorded after the compute that used set k) gates the refill of set
    // k, st_ready[k] (recorded after its three copies) gates the compute. Gather rows / weights go through pinned
    // per-layer arrays (a pageable cudaMemcpyAsync would stall the host until the stream drained). The previous layer's
    // copies are long done: route() synchronized the stream to read the router output.
    const size_t stride = (S.qstage_bytes / 6) & ~(size_t) 255;   // 256-byte aligned staging slots
    int nst = 0, off = 0;   // host-staged experts so far (set = nst & 1), rows used in hidx_pin / gidx
    for (auto & [e, lst] : by_e) {
        // the expert's weights: resident hot, host hot leftover, or host cold (this rank only)
        fast::packed_weight g, u, d;
        bool rg = false, rd = false;
        const int hl = L.hot_map[e], hh = L.hhot_map(e), cl = L.cold_map[e];
        const int set = nst & 1;
        auto stage = [&](expert_set & es, int j, int slot) -> fast::packed_weight {
            char * dst = (char *) S.qstage + (size_t) (set * 3 + slot) * stride;
            if (es.bytes > stride) throw std::runtime_error("fn_model: expert slice larger than a staging slot");
            if (slot == 0) CK(cudaStreamWaitEvent(S.cs, S.st_free[set], 0));
            CK(cudaMemcpyAsync(dst, es.base + (size_t) j * es.stride, es.bytes, cudaMemcpyHostToDevice, S.cs));
            if (slot == 2) { CK(cudaEventRecord(S.st_ready[set], S.cs)); CK(cudaStreamWaitEvent(S.s, S.st_ready[set], 0)); }
            fast::packed_weight pk; pk.type = es.type; pk.n_in = es.n_in; pk.n_out = es.n_out; pk.data = dst; pk.bytes = es.bytes; pk.layout = es.layout;
            return pk;
        };
        if (hl >= 0) { g = L.hot_gate.pk[hl]; u = L.hot_up.pk[hl]; d = L.hot_down.pk[hl]; rg = L.hot_gate.raw; rd = L.hot_down.raw; }
        else if (hh >= 0) { g = stage(L.hhot_gate, hh, 0); u = stage(L.hhot_up, hh, 1); d = stage(L.hhot_down, hh, 2); rg = L.hhot_gate.raw; rd = L.hhot_down.raw; }
        else if (cl >= 0) { g = stage(L.cold_gate, cl, 0); u = stage(L.cold_up, cl, 1); d = stage(L.cold_down, cl, 2); rg = L.cold_gate.raw; rd = L.cold_down.raw; }
        else continue;   // another rank's expert
        const bool staged = hh >= 0 || cl >= 0;
        const int Te = (int) lst.size();
        int32_t * hidx = S.hidx_pin + off; float * hw = S.hw_pin + off;
        int32_t * gidx = S.gidx + off; float * gw = S.gw + off;
        for (int j = 0; j < Te; ++j) { hidx[j] = lst[j].first; hw[j] = lst[j].second; }
        off += Te;
        CK(cudaMemcpyAsync(gidx, hidx, Te * 4, cudaMemcpyHostToDevice, S.s));
        CK(cudaMemcpyAsync(gw, hw, Te * 4, cudaMemcpyHostToDevice, S.s));
        const float * x = cur;
        if (!(Te == T && std::is_sorted(hidx, hidx + Te) && hidx[0] == 0)) {
            fnk::gather_rows_k<<<dim3(nb256(E_), Te), 256, 0, S.s>>>(cur, gidx, S.gath, E_);
            x = S.gath;
        }
        gemm_pk(S, g, rg, x, S.a, Te); gemm_pk(S, u, rg, x, S.b, Te);
        k::silu_mul(S.a, S.b, (int64_t) Te * FFE_, S.s);
        gemm_pk(S, d, rd, S.a, S.gout, Te);
        fnk::scatter_add_rows_k<<<dim3(nb256(E_), Te), 256, 0, S.s>>>(out, S.gout, gidx, gw, E_);
        if (staged) { CK(cudaEventRecord(S.st_free[set], S.s)); ++nst; }
    }
    ffn_shared(S, r, il, cur, out, T);
}

// (token, weight) pairs by expert. P100E_FN_TAIL_SKIP=K (off by default): in a decode /
// verify forward (T <= MTP_ROWS) an expert of this rank that is not resident and that every one of its tokens routes
// at slot K or later (slots in descending router weight, 0-based) is dropped: neither copied nor computed (changes output)
void fn_model::ffn_by_e(rank_state & S, int il, int T, const std::vector<int> & ids, const std::vector<float> & wts,
                        std::map<int, std::vector<std::pair<int, float>>> & by_e) {
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < NEXPU_; ++k) by_e[ids[(size_t) t * NEXPU_ + k]].push_back({t, wts[(size_t) t * NEXPU_ + k]});
    static const int tail = getenv("P100E_FN_TAIL_SKIP") ? atoi(getenv("P100E_FN_TAIL_SKIP")) : 0;
    if (tail <= 0 || tail >= NEXPU_ || T > MTP_ROWS) return;
    const layer_w & L = S.lw[il];
    std::map<int, int> first;   // expert -> its earliest slot over the forward's tokens
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < NEXPU_; ++k) { const int e = ids[(size_t) t * NEXPU_ + k]; auto it = first.find(e); if (it == first.end() || k < it->second) first[e] = k; }
    for (auto & [e, k] : first)
        if (k >= tail && L.hot_map[e] < 0 && (L.hhot_map(e) >= 0 || L.cold_map[e] >= 0)) by_e.erase(e);
}

// ffn_shared in two halves (remote tier): the matvecs (S.a / S.b / S.c / S.d) early, the add after the combine
void fn_model::ffn_shared_pre(rank_state & S, int il, const float * cur, int T) {
    layer_w & L = S.lw[il];
    mv(S, L.shexp_gate_w, cur, S.a, T); mv(S, L.shexp_up_w, cur, S.b, T);
    k::silu_mul(S.a, S.b, (int64_t) T * FFS_, S.s);
    mv(S, L.shexp_down_w, S.a, S.c, T);                                   // [T][E]
    fnk::dot_rows_k<<<dim3(1, T), 1024, 0, S.s>>>(L.shexp_gate, cur, S.d, E_);   // gate logit [T]
}
void fn_model::ffn_shared_post(rank_state & S, int il, float * out, int T) {
    grab("ffn_moe_out", il, 0, out, (int64_t) T * E_);
    fnk::add_sig_rows_k<<<dim3(nb256(E_), T), 256, 0, S.s>>>(out, S.c, S.d, E_);
}
void fn_model::ffn_shared(rank_state & S, int r, int il, const float * cur, float * out, int T) {
    if (r != 0) return;
    layer_w & L = S.lw[il];
    grab("ffn_moe_out", il, r, out, (int64_t) T * E_);   // rank 0's partial only (the full sum needs the allreduce)
    mv(S, L.shexp_gate_w, cur, S.a, T); mv(S, L.shexp_up_w, cur, S.b, T);
    k::silu_mul(S.a, S.b, (int64_t) T * FFS_, S.s);
    mv(S, L.shexp_down_w, S.a, S.c, T);                                   // [T][E]
    fnk::dot_rows_k<<<dim3(1, T), 1024, 0, S.s>>>(L.shexp_gate, cur, S.d, E_);   // gate logit [T]
    fnk::add_sig_rows_k<<<dim3(nb256(E_), T), 256, 0, S.s>>>(out, S.c, S.d, E_);
}

// ---- CPU expert pool --------------------------------------------------------------------------------------------------
void fn_model::cpu_init() {
    const char * e = getenv("P100E_FN_CPU");
    if (!e || atoi(e) == 0) return;
    if (!fn_cpu_pool::cpu_ok()) { printf("fn_model: P100E_FN_CPU=1 ignored: the CPU lacks AVX2 / FMA / F16C\n"); return; }
    cpu_ = std::make_unique<cpu_moe>();
    cpu_moe & C = *cpu_;
    if (const char * v = getenv("P100E_FN_CPU_MAXROWS")) C.maxrows = std::max(0, atoi(v));
    if (const char * v = getenv("P100E_FN_CPU_PCIE")) C.pcie = std::min(1.0f, std::max(0.0f, (float) atof(v)));
    C.stats = o_.verbose || (getenv("P100E_FN_CPU_STATS") && atoi(getenv("P100E_FN_CPU_STATS")) != 0);
    C.pool = std::make_unique<fn_cpu_pool>();
    CK(cudaSetDevice(rk_[0].dev));
    CK(cudaMallocHost(&C.x, (size_t) o_.max_t * E_ * 4));
    C.rk.resize(rk_.size());
    for (size_t r = 0; r < rk_.size(); ++r) {
        CK(cudaSetDevice(rk_[r].dev));
        C.devs.push_back(rk_[r].dev);
        for (int k = 0; k < 2; ++k) CK(cudaEventCreateWithFlags(&C.rk[r].ev[k], cudaEventDisableTiming));
        C.rk[r].is_cpu.assign(NEXP_, 0);
    }
    CK(cudaSetDevice(rk_[0].dev));
    printf("fn_model: CPU expert pool on (P100E_FN_CPU=1): %s; experts with <= %d routed rows, PCIe share %.2f\n",
           C.pool->placement(), C.maxrows, C.pcie);
}

fn_model::cpu_moe::~cpu_moe() {
    pool.reset();
    if (stats) printf("fn_model: CPU pool: %lld layers, %lld experts, %lld rows on the CPU (%.2f experts per layer); %lld host experts kept on PCIe; host waited %.1f ms\n",
                      (long long) n_layers, (long long) n_exp, (long long) n_rows, n_layers ? (double) n_exp / n_layers : 0.0, (long long) n_gpu_host, t_wait);
    for (size_t r = 0; r < rk.size(); ++r) {
        cudaSetDevice(devs[r]);
        cudaDeviceSynchronize();
        for (int k = 0; k < 2; ++k) { if (rk[r].y[k]) cudaFreeHost(rk[r].y[k]); if (rk[r].ev[k]) cudaEventDestroy(rk[r].ev[k]); }
    }
    if (x) cudaFreeHost(x);
}

void fn_model::cpu_stage_x(rank_state & S, int T) {
    CK(cudaMemcpyAsync(cpu_->x, S.cur, (size_t) T * E_ * 4, cudaMemcpyDeviceToHost, S.s));
}

// rank 0's call of layer_ffn: pick every rank's CPU experts of the layer and start the pool. A rank's non-resident
// expert goes to the CPU when its record's layouts have CPU kernels and it has at most maxrows routed rows; a prefetched
// layer (every host expert already in VRAM) keeps the GPU path. Every rank's layouts must have grouped kernels (the CPU
// rows join ffn_grouped's combine), else the layer runs as before.
void fn_model::cpu_begin(int il, int T, const std::vector<int> & ids, const std::vector<float> & wts) {
    cpu_moe & C = *cpu_;
    C.layer_on = false;
    static const bool grouped_on = !getenv("P100E_FN_GROUPED") || atoi(getenv("P100E_FN_GROUPED")) != 0;
    if (!grouped_on || C.maxrows <= 0) return;
    const int R = (int) rk_.size();
    for (int r = 0; r < R; ++r) {
        const layer_w & L = rk_[r].lw[il];
        auto ok = [&](const expert_set & g, const expert_set & u, const expert_set & d) {
            return g.n == 0 || (grouped_layout(g.layout) && g.layout == u.layout && grouped_layout(d.layout));
        };
        if (!ok(L.hot_gate, L.hot_up, L.hot_down) || !ok(L.hhot_gate, L.hhot_up, L.hhot_down) || !ok(L.cold_gate, L.cold_up, L.cold_down)) return;
    }
    struct pick { int r, e; const std::vector<std::pair<int, float>> * lst; };
    std::vector<pick> picks;
    std::vector<std::map<int, std::vector<std::pair<int, float>>>> by(R);
    int rows = 0;
    for (int r = 0; r < R; ++r) {
        rank_state & S = rk_[r];
        const layer_w & L = S.lw[il];
        auto & cr = C.rk[r];
        std::fill(cr.is_cpu.begin(), cr.is_cpu.end(), (char) 0);
        cr.nrows = 0; cr.nr = cr.nrg = 0;
        if (S.pf_layer[il & 1] == il) continue;
        ffn_by_e(S, il, T, ids, wts, by[r]);
        std::vector<std::pair<int, int>> cand;   // (rows, expert)
        for (auto & [e, lst] : by[r]) {
            if (L.hot_map[e] >= 0) continue;
            const int hh = L.hhot_map(e), cl = L.cold_map[e];
            if (hh < 0 && cl < 0) continue;   // another rank's expert
            const expert_set & G = hh >= 0 ? L.hhot_gate : L.cold_gate;
            const expert_set & D = hh >= 0 ? L.hhot_down : L.cold_down;
            if ((int) lst.size() > C.maxrows || !fn_cpu_pool::supports(G.layout, D.layout) || G.n_in != E_ || D.n_in != FFE_ || G.n_out != FFE_ || D.n_out != E_) { ++C.n_gpu_host; continue; }
            cand.push_back({(int) lst.size(), e});
        }
        // the PCIe share: the most-routed candidates stay on the GPU path (ties: lower id)
        const int keep = (int) std::lround(C.pcie * (double) cand.size());
        if (keep > 0) {
            std::stable_sort(cand.begin(), cand.end(), [](const std::pair<int, int> & a, const std::pair<int, int> & b) { return a.first > b.first; });
            C.n_gpu_host += keep;
            cand.erase(cand.begin(), cand.begin() + keep);
        }
        for (auto & [n, e] : cand) cr.is_cpu[e] = 1;
        for (auto & [e, lst] : by[r]) if (cr.is_cpu[e]) { picks.push_back({r, e, &lst}); cr.nrows += (int) lst.size(); }
        rows += cr.nrows;
    }
    if (picks.empty()) return;
    // jobs (rank by rank, ascending expert id: the order ffn_grouped numbers the CPU rows in)
    C.tok.resize(rows);
    C.jobs.clear();
    const int k = C.flip;
    for (int r = 0; r < R; ++r) {
        auto & cr = C.rk[r];
        if (cr.nrows == 0) continue;
        const size_t need = (size_t) cr.nrows * E_ * 4;
        if (cr.ev_rec[k]) { CK(cudaEventSynchronize(cr.ev[k])); cr.ev_rec[k] = false; }   // the H2D that last read y[k]
        if (cr.ycap[k] < need) {
            std::string why;
            if (!ram_check(std::max(need, (size_t) 64 * E_ * 4), ram_available_bytes(), g_ram_reserve_mb, "the CPU pool's output rows", why)) {
                static bool said = false;
                if (!said) { printf("fn_model: CPU pool layer skipped (GPU path): %s\n", why.c_str()); said = true; }
                return;
            }
            if (cr.y[k]) CK(cudaFreeHost(cr.y[k]));
            cr.ycap[k] = std::max(need, (size_t) 64 * E_ * 4);
            CK(cudaMallocHost(&cr.y[k], cr.ycap[k]));
        }
    }
    int toff = 0;
    std::vector<int> yoff(R, 0);
    for (const pick & p : picks) {
        const layer_w & L = rk_[p.r].lw[il];
        const int hh = L.hhot_map(p.e), cl = L.cold_map[p.e];
        const expert_set & G = hh >= 0 ? L.hhot_gate : L.cold_gate;
        const expert_set & U = hh >= 0 ? L.hhot_up : L.cold_up;
        const expert_set & D = hh >= 0 ? L.hhot_down : L.cold_down;
        const uint8_t * rec = (const uint8_t *) G.base + (size_t) (hh >= 0 ? hh : cl) * G.stride;
        fn_cpu_job j;
        j.gate = rec; j.up = rec + (U.base - G.base); j.down = rec + (D.base - G.base);
        j.rb_gu = G.bytes / (size_t) G.n_out; j.rb_d = D.bytes / (size_t) D.n_out;
        j.lay_gu = G.layout; j.lay_d = D.layout;
        j.n = (int) p.lst->size();
        for (int i = 0; i < j.n; ++i) C.tok[toff + i] = (*p.lst)[i].first;
        j.tok = C.tok.data() + toff;
        j.y = C.rk[p.r].y[k] + (size_t) yoff[p.r] * E_;
        toff += j.n; yoff[p.r] += j.n;
        C.jobs.push_back(j);
    }
    C.pool->submit(C.x, T, E_, FFE_, C.jobs.data(), (int) C.jobs.size());
    C.layer_on = true;
    C.n_layers += 1; C.n_exp += (int64_t) picks.size(); C.n_rows += rows;
}

// the last rank's call of layer_ffn: every rank's GPU experts are queued; the host helps the pool, then per rank the
// CPU rows go up in one copy behind the GPU's rows (S.gout) and ffn_grouped's combine + the shared expert run
void fn_model::cpu_finish(int il, int T) {
    cpu_moe & C = *cpu_;
    const auto t0 = std::chrono::steady_clock::now();
    C.pool->wait();
    C.t_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const int k = C.flip;
    for (int r = 0; r < (int) rk_.size(); ++r) {
        rank_state & S = rk_[r];
        auto & cr = C.rk[r];
        CK(cudaSetDevice(S.dev));
        if (cr.nr > cr.nrg) {
            if (cr.nr - cr.nrg != cr.nrows) throw std::runtime_error("fn_model: CPU row count mismatch");
            CK(cudaMemcpyAsync(S.gout + (size_t) cr.nrg * E_, cr.y[k], (size_t) cr.nrows * E_ * 4, cudaMemcpyHostToDevice, S.s));
            CK(cudaEventRecord(cr.ev[k], S.s)); cr.ev_rec[k] = true;
        }
        if (cr.nr > 0) fnk::combine_rows_k<<<dim3(nb256(E_), T), 256, 0, S.s>>>(cr.out, S.gout, S.rowof, S.gw, E_, NEXPU_);
        ffn_shared(S, r, il, cr.cur, cr.out, T);
    }
    C.flip ^= 1;
    C.layer_on = false;
}

// both ranks: out = part + the other rank's part (same bits on both: fp32 add is commutative)
void fn_model::allreduce(int T) {
    const int R = (int) rk_.size();
    if (R == 1) return;
    const size_t bytes = (size_t) T * E_ * 4;
    for (auto & S : rk_) { CK(cudaSetDevice(S.dev)); CK(cudaEventRecord(S.ev, S.s)); }
    for (int r = 0; r < R; ++r) {
        rank_state & S = rk_[r], & O = rk_[1 - r];
        CK(cudaSetDevice(S.dev));
        CK(cudaStreamWaitEvent(S.s, O.ev, 0));
        CK(cudaMemcpyPeerAsync(S.peer, S.dev, O.part, O.dev, bytes, S.s));
        CK(cudaEventRecord(S.ev2, S.s));
    }
    for (int r = 0; r < R; ++r) {
        rank_state & S = rk_[r], & O = rk_[1 - r];
        CK(cudaSetDevice(S.dev));
        CK(cudaStreamWaitEvent(S.s, O.ev2, 0));   // the peer has read my part before I touch it again
        k::add(S.part, S.peer, (int64_t) T * E_, S.s);
    }
}

// token ids index the mapped embedding table (stage_embed): reject ids outside [0, vocab) before any state changes
static void fn_check_tokens(const std::vector<int32_t> & tokens, int vocab) {
    for (int32_t t : tokens) if (t < 0 || t >= vocab) throw std::runtime_error("fn_model: token id " + std::to_string(t) + " out of range");
}

std::vector<float> fn_model::forward_n(const std::vector<int32_t> & tokens, int n_logits) {
    const int T = (int) tokens.size();
    if (T < 1 || T > o_.max_t) throw std::runtime_error("fn_model: 1..max_t tokens per forward");
    fn_check_tokens(tokens, VOCAB_);
    if (n_past_ + T > o_.n_ctx) throw std::runtime_error("fn_model: context full");
    const int R = (int) rk_.size();
    dumped_.clear();
    static const bool time_env = getenv("P100E_FN_TIME") && atoi(getenv("P100E_FN_TIME")) != 0;
    time_on_ = time_env;
    if (time_on_) tm_start();
    // positions (cell index + rotary t/h/w) and embedding -> hc_init on every rank. Pipe driver (node 1): the command
    // goes out first (node 0 starts at once), the residual rows of node 0's last layer come back into S.res.
    if (adapt_async_) promo_finalize();   // landed promotions join the maps before this forward routes
    if (pipe() && !pipe_worker()) pipe_cmd(1, tokens);
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        stage_pos(S, n_past_, T);
        if (has_embed()) {
            stage_embed(S, tokens, n_past_);
            fnk::hc_init_k<<<dim3(nb256(E_), T), 256, 0, S.s>>>(S.cur, S.res, E_, HC_);
        }
    }
    for (int i = 0; i < T; ++i) for (int s = 0; s < 3; ++s) rp_[s][(size_t) n_past_ + i] = rot(n_past_ + i, s);   // the rows' rotary positions
    if (has_embed()) grab("model.input_embed", -1, 0, rk_[0].cur, (int64_t) T * E_);
    std::vector<int> ids; std::vector<float> wts;
    const bool prefetch = o_.prefetch_min_t > 0 && T >= o_.prefetch_min_t && rk_[0].pf[0];
    if (prefetch) for (auto & S : rk_) { CK(cudaSetDevice(S.dev)); prefetch_layer(S, lb_); }
    if (mtp_il_ >= 0 && T <= MTP_ROWS && PLE_LAYER_ >= 0 && owns(PLE_LAYER_))   // a verify batch may be rolled back: keep the PLE history
        for (auto & S : rk_) { CK(cudaSetDevice(S.dev)); CK(cudaMemcpyAsync(S.ple_snap, S.ple_hist, (size_t) PLE_HIST_ * HCDIM_ * 4, cudaMemcpyDeviceToDevice, S.s)); }
    if (pipe() && !has_embed()) pipe_recv_rows(T);   // blocks for node 0's rows (its layers run meanwhile)
    if (time_on_) tm_mark("embed");
    for (int il = lb_; il < le_; ++il) run_layer(il, T, tokens, prefetch, ids, wts);
    std::vector<float> logits;
    if (!has_head()) {   // pipe worker: rank 0's wide residual rows -> node 1 (every rank holds the same bits)
        pipe_send_rows(T);
    } else {
    if (mtp_il_ >= 0)   // h_nextn: the wide residual of every row, the draft block's input at the next position
        for (auto & S : rk_) { CK(cudaSetDevice(S.dev)); CK(cudaMemcpyAsync(S.hn, S.res, (size_t) T * HCDIM_ * 4, cudaMemcpyDeviceToDevice, S.s)); }
    // output mixer + LM head on rank 0 (the last n_logits rows)
    rank_state & S = rk_[0];
    CK(cudaSetDevice(S.dev));
    hc_mix(S, S.output_hc_norm, S.output_hc_down, S.output_hc_up, nullptr, S.cur, nullptr, T);
    grab("result_norm", -1, 0, S.cur, (int64_t) T * E_);
    const int n = std::min({std::max(n_logits, 1), T, FN_MAX_LOGITS});
    const float * last = S.cur + (size_t) (T - n) * E_;
    mv(S, S.output, last, S.logits, n);
    logits.resize((size_t) n * VOCAB_);
    CK(cudaMemcpyAsync(logits.data(), S.logits, logits.size() * 4, cudaMemcpyDeviceToHost, S.s));
    sync_all();
    }
    if (time_on_) {
        tm_mark("head");
        tm_flush();
        double tot = 0; for (auto & kv : tm_) tot += kv.second;
        printf("fn time T=%d: total %.0f ms:", T, tot);
        for (auto & kv : tm_) printf(" %s %.0f", kv.first.c_str(), kv.second);
        printf("\n"); fflush(stdout);
        time_on_ = false;
    }
    hc_prof_flush();
    if (dump_) dumped_["result_output"] = logits;
    for (int i = 0; i < T; ++i) toks_.push_back(tokens[i]);
    n_past_ += T;
    since_adapt_ += T;
    if (o_.adapt && (T >= o_.prefetch_min_t || since_adapt_ >= o_.adapt_every)) adapt(T >= o_.prefetch_min_t);
    return logits;
}

void fn_model::run_layer(int il, int T, const std::vector<int32_t> & tokens, bool prefetch, std::vector<int> & ids, std::vector<float> & wts) {
    const int R = (int) rk_.size();
    if (time_on_) {
        // timing (P100E_FN_TIME): every rank's mixer, mark, then every rank's attention, mark. The one-loop order
        // below puts rank 0's attention before the 'hc_mix' mark, so 'hc_mix' used to include it. Same per-stream
        // work in the same order either way (same bits).
        for (int r = 0; r < R; ++r) {
            rank_state & S = rk_[r];
            CK(cudaSetDevice(S.dev));
            layer_w & L = S.lw[il];
            if (prefetch && il + 1 < le_) prefetch_layer(S, il + 1);
            if (il == PLE_LAYER_) ple(S, il, tokens);
        }
        if (il == PLE_LAYER_) tm_mark("ple");
        for (int r = 0; r < R; ++r) {
            rank_state & S = rk_[r];
            CK(cudaSetDevice(S.dev));
            layer_w & L = S.lw[il];
            hc_mix(S, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject, S.cur, S.inject, T);
            grab("hc_mixed", il, r, S.cur, (int64_t) T * E_);
        }
        tm_mark("hc_mix");
        for (int r = 0; r < R; ++r) {
            rank_state & S = rk_[r];
            CK(cudaSetDevice(S.dev));
            layer_w & L = S.lw[il];
            if (L.attn) layer_attn(S, il, S.cur, S.part, T); else layer_gdn(S, il, S.cur, S.part, T);
        }
        tm_mark(rk_[0].lw[il].attn ? "attn" : "gdn");
    } else
    for (int r = 0; r < R; ++r) {
        rank_state & S = rk_[r];
        CK(cudaSetDevice(S.dev));
        layer_w & L = S.lw[il];
        if (prefetch && il + 1 < le_) prefetch_layer(S, il + 1);   // lands while this layer computes
        if (il == PLE_LAYER_) ple(S, il, tokens);
        hc_mix(S, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject, S.cur, S.inject, T);
        grab("hc_mixed", il, r, S.cur, (int64_t) T * E_);
        if (L.attn) layer_attn(S, il, S.cur, S.part, T); else layer_gdn(S, il, S.cur, S.part, T);
    }
    if (rk_[0].Hr != H_) allreduce(T);   // dense split: the mixer's output is a sum over the ranks' heads
    if (time_on_) tm_mark("allreduce");
    for (int r = 0; r < R; ++r) {
        rank_state & S = rk_[r];
        CK(cudaSetDevice(S.dev));
        layer_w & L = S.lw[il];
        grab(L.attn ? "attn_output" : "linear_attn_out", il, r, S.part, (int64_t) T * E_);
        hc_combine(S, S.part, S.inject, T);
        hc_mix(S, L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject, S.cur, S.inject, T);
        grab("hc_mixed_ffn", il, r, S.cur, (int64_t) T * E_);
    }
    if (time_on_) tm_mark("hc_mix");
    route(il, T, ids, wts);
    if (time_on_) tm_mark("route");
    for (int r = 0; r < R; ++r) { CK(cudaSetDevice(rk_[r].dev)); layer_ffn(rk_[r], r, il, rk_[r].cur, rk_[r].part, T, ids, wts); }
    if (time_on_) tm_mark("ffn");
    allreduce(T);
    if (time_on_) tm_mark("allreduce");
    for (int r = 0; r < R; ++r) {
        rank_state & S = rk_[r];
        CK(cudaSetDevice(S.dev));
        grab("ffn_out", il, r, S.part, (int64_t) T * E_);
        hc_combine(S, S.part, S.inject, T);
        grab("l_last", il, r, S.res, (int64_t) T * HCDIM_);
    }
}

void fn_model::rollback(int T, int n_keep) {
    if (n_keep < 1 || n_keep > T || T > n_past_) throw std::runtime_error("fn_model: rollback out of range");
    if (n_keep == T) return;
    if (T > MTP_ROWS || mtp_il_ < 0) throw std::runtime_error("fn_model: rollback needs the MTP build and T <= MTP_ROWS");
    if (pipe() && !pipe_worker()) {   // driver: node 0 rolls back its layers with the next command
        if (pend_rb_T_ || pend_reset_) pipe_cmd(0, {});   // (never two queued: flush the older one)
        pend_rb_T_ = T; pend_rb_keep_ = n_keep;
    }
    local_rollback(T, n_keep);
}

void fn_model::local_rollback(int T, int n_keep) {
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        const size_t ssz = (size_t) S.HVr * DK_ * DV_, csz = (size_t) S.CONVr * 3;
        for (int il = 0; il < L_; ++il) if (S.ssm_snap[il]) {   // the state after row n_keep - 1
            CK(cudaMemcpyAsync(S.ssm_state[il], S.ssm_snap[il] + (size_t) (n_keep - 1) * ssz, ssz * 4, cudaMemcpyDeviceToDevice, S.s));
            CK(cudaMemcpyAsync(S.conv_state[il], S.conv_snap[il] + (size_t) (n_keep - 1) * csz, csz * 4, cudaMemcpyDeviceToDevice, S.s));
        }
        if (PLE_LAYER_ >= 0 && S.ple_snap) {
            // history = the last PLE_HIST rows of (snapshot ++ the forward's first n_keep normalized rows, still in ple_hist)
            const int H = PLE_HIST_, q = n_keep;
            const float * nw = S.ple_hist + (size_t) H * HCDIM_;
            if (q >= H) CK(cudaMemcpyAsync(S.ple_hist, nw + (size_t) (q - H) * HCDIM_, (size_t) H * HCDIM_ * 4, cudaMemcpyDeviceToDevice, S.s));
            else {
                CK(cudaMemcpyAsync(S.ple_hist, S.ple_snap + (size_t) q * HCDIM_, (size_t) (H - q) * HCDIM_ * 4, cudaMemcpyDeviceToDevice, S.s));
                CK(cudaMemcpyAsync(S.ple_hist + (size_t) (H - q) * HCDIM_, nw, (size_t) q * HCDIM_ * 4, cudaMemcpyDeviceToDevice, S.s));
            }
        }
    }
    n_past_ -= T - n_keep;
    toks_.resize(n_past_);
}

void fn_model::mtp_h_zero() {
    for (auto & S : rk_) { CK(cudaSetDevice(S.dev)); CK(cudaMemsetAsync(S.mcarry, 0, (size_t) HCDIM_ * 4, S.s)); }
}

// the draft block over T rows (tokens at p0.., h rows in S.mhin on every rank); returns the last row's greedy token (-1 when
// the head is skipped). Runs the block's attention at positions p0.. with n_past = p0 (its own KV / indexer caches).
static int32_t host_argmax(const float * l, int n) { int b = 0; for (int i = 1; i < n; ++i) if (l[i] > l[b]) b = i; return b; }

int32_t fn_model::mtp_block(const std::vector<int32_t> & tokens, int p0, bool want_logits) {
    const int T = (int) tokens.size(), il = mtp_il_;
    if (il < 0) throw std::runtime_error("fn_model: no MTP block");
    if (T < 1 || T > o_.max_t) throw std::runtime_error("fn_model: mtp_block: 1..max_t rows");
    fn_check_tokens(tokens, VOCAB_);
    if (p0 + T > o_.n_ctx) throw std::runtime_error("fn_model: context full (mtp)");
    const int saved = n_past_;
    n_past_ = p0;
    if (adapt_async_) promo_finalize();
    static const bool time_env = getenv("P100E_FN_TIME") && atoi(getenv("P100E_FN_TIME")) != 0;
    time_on_ = time_env;   // P100E_FN_TIME=1: the draft block's phases too ('fn time mtp T=..')
    if (time_on_) tm_start();
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        layer_w & L = S.lw[il];
        stage_pos(S, p0, T);
        stage_embed(S, tokens, p0);
        k::rmsnorm(S.cur, L.enorm, S.a, T, E_, eps_, S.s);                                                       // e_norm [T][E]
        fnk::rmsnorm_rows_k<<<(unsigned) (T * HC_), nthreads_for(E_), 0, S.s>>>(S.mhin, L.hnorm, S.xn, E_, HC_, eps_);   // h_norm [T][HC][E]
        fnk::mtp_concat_k<<<dim3(nb256(2 * E_), T * HC_), 256, 0, S.s>>>(S.a, S.xn, S.mcat, E_, HC_);         // [T*HC][2E]
        for (int r0 = 0; r0 < T * HC_; r0 += o_.max_t) {   // eh_proj in slabs of <= max_t rows (the GEMM scratch is sized for max_t)
            const int n = std::min(o_.max_t, T * HC_ - r0);
            mv(S, L.eh_proj, S.mcat + (size_t) r0 * 2 * E_, S.res + (size_t) r0 * E_, n);                      // res [T][HC][E]
        }
    }
    std::vector<int> ids; std::vector<float> wts;
    if (time_on_) tm_mark("mtp_in");
    run_layer(il, T, tokens, false, ids, wts);
    int32_t best = -1;
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        CK(cudaMemcpyAsync(S.mself, S.res + (size_t) (T - 1) * HCDIM_, (size_t) HCDIM_ * 4, cudaMemcpyDeviceToDevice, S.s));
    }
    if (want_logits) {
        rank_state & S = rk_[0];
        CK(cudaSetDevice(S.dev));
        layer_w & L = S.lw[il];
        hc_mix(S, L.hc_head_norm, L.hc_head_down, L.hc_head_up, nullptr, S.cur, nullptr, T);
        mv(S, S.output, S.cur + (size_t) (T - 1) * E_, S.logits, 1);
        std::vector<float> l((size_t) VOCAB_);
        CK(cudaMemcpyAsync(l.data(), S.logits, l.size() * 4, cudaMemcpyDeviceToHost, S.s));
        sync_all();
        best = host_argmax(l.data(), VOCAB_);
    } else sync_all();
    if (time_on_) {
        tm_mark("mtp_head");
        tm_flush();
        double tot = 0; for (auto & kv : tm_) tot += kv.second;
        printf("fn time mtp T=%d: total %.1f ms:", T, tot);
        for (auto & kv : tm_) printf(" %s %.1f", kv.first.c_str(), kv.second);
        printf("\n"); fflush(stdout);
        time_on_ = false;
    }
    hc_prof_flush();
    n_past_ = saved;
    return best;
}

int32_t fn_model::mtp_run(const std::vector<int32_t> & tokens, int p0, int hsrc) {
    const int T = (int) tokens.size();
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        if (hsrc == MTP_H_MAIN) CK(cudaMemcpyAsync(S.mhin, S.hn, (size_t) T * HCDIM_ * 4, cudaMemcpyDeviceToDevice, S.s));
        else CK(cudaMemcpyAsync(S.mhin, hsrc == MTP_H_SELF ? S.mself : S.mcarry, (size_t) HCDIM_ * 4, cudaMemcpyDeviceToDevice, S.s));
    }
    return mtp_block(tokens, p0, true);
}

void fn_model::mtp_catchup(const std::vector<int32_t> & tokens, int p0) {
    const int T = (int) tokens.size();
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        CK(cudaMemcpyAsync(S.mhin, S.mcarry, (size_t) HCDIM_ * 4, cudaMemcpyDeviceToDevice, S.s));
        if (T > 1) CK(cudaMemcpyAsync(S.mhin + HCDIM_, S.hn, (size_t) (T - 1) * HCDIM_ * 4, cudaMemcpyDeviceToDevice, S.s));
    }
    mtp_block(tokens, p0, false);
    for (auto & S : rk_) {   // the carry for the next chunk / the first draft: h of the chunk's last row
        CK(cudaSetDevice(S.dev));
        CK(cudaMemcpyAsync(S.mcarry, S.hn + (size_t) (T - 1) * HCDIM_, (size_t) HCDIM_ * 4, cudaMemcpyDeviceToDevice, S.s));
    }
}

std::vector<int32_t> fn_model::generate_mtp(const std::vector<int32_t> & prompt, int n_gen, int chunk, int k, mtp_stats & st) {
    using clk = std::chrono::steady_clock;
    if (mtp_il_ < 0) throw std::runtime_error("fn_model: generate_mtp needs the MTP block");
    if (k < 1 || k > MTP_ROWS - 1) throw std::runtime_error("fn_model: mtp k must be 1..3");
    chunk = std::max(1, std::min(chunk, o_.max_t));
    std::vector<int32_t> out;
    if (n_past_ == 0) mtp_h_zero();
    std::vector<float> L;
    ple_prefetch(prompt);
    for (size_t c0 = 0; c0 < prompt.size(); c0 += chunk) {
        const std::vector<int32_t> ch(prompt.begin() + c0, prompt.begin() + std::min(prompt.size(), c0 + chunk));
        const int p0 = n_past_;
        auto a = clk::now();
        L = forward_n(ch, 1);
        st.t_prompt += std::chrono::duration<double>(clk::now() - a).count();
        a = clk::now();
        mtp_catchup(ch, p0);
        st.t_catchup += std::chrono::duration<double>(clk::now() - a).count();
    }
    if (L.empty() || n_gen < 1) return out;
    int32_t last = host_argmax(L.data(), VOCAB_);
    out.push_back(last);
    int N = n_past_;   // position of `last` (not yet through the trunk)
    auto drun = [&](const std::vector<int32_t> & t, int p0, int src) {
        const auto a = clk::now();
        const int32_t r = mtp_run(t, p0, src);
        st.t_draft += std::chrono::duration<double>(clk::now() - a).count();
        st.n_draft++;
        return r;
    };
    std::vector<int32_t> drafts;
    // P100E_FN_PLE_ASYNC=1: the next verify batch is [last, d0, d1, ..]; each token's PLE rows are asked from the disk
    // as soon as it is known (while the remaining drafts run), so the verify's PLE layer finds them in the page cache
    static const bool early_ple = ple_async_on() && (!getenv("P100E_FN_PLE_PREFETCH") || atoi(getenv("P100E_FN_PLE_PREFETCH")) != 0);
    auto pf_batch = [&](int32_t lt) {
        if (!early_ple) return;
        std::vector<int32_t> b{lt};
        b.insert(b.end(), drafts.begin(), drafts.end());
        ple_prefetch(b);   // rows of positions n_past_.. (n_past_ = the position of lt here)
    };
    auto chain = [&](int32_t d0, int N0) {
        drafts.assign(1, d0);
        pf_batch(last);
        for (int j = 1; j < k; ++j) { drafts.push_back(drun({drafts.back()}, N0 + j, MTP_H_SELF)); pf_batch(last); }
    };
    chain(drun({last}, N, MTP_H_READY), N);
    while ((int) out.size() < n_gen) {
        std::vector<int32_t> batch{last};
        batch.insert(batch.end(), drafts.begin(), drafts.end());
        const int T = (int) batch.size();
        auto a = clk::now();
        const std::vector<float> lg = forward_n(batch, T);   // T rows of logits
        st.t_verify += std::chrono::duration<double>(clk::now() - a).count();
        int acc = 0;
        while (acc < (int) drafts.size() && host_argmax(lg.data() + (size_t) acc * VOCAB_, VOCAB_) == drafts[acc]) ++acc;
        // test hook (pipe / rollback identity runs on layer-limited models, where drafts rarely match): accept counts
        // cycle through the given list (clamped to the drafts) whatever matched; the output is then NOT greedy
        static const std::vector<int> force_acc = [] { std::vector<int> v; if (const char * e = getenv("P100E_FN_MTP_FORCE_ACC")) for (const char * p = e; *p;) { v.push_back(atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } return v; }();
        if (!force_acc.empty()) acc = std::min((int) drafts.size(), std::max(0, force_acc[(size_t) st.steps % force_acc.size()]));
        const int32_t nt = host_argmax(lg.data() + (size_t) acc * VOCAB_, VOCAB_);
        st.steps++; st.drafted += (int) drafts.size(); st.accepted += acc;
        std::vector<int32_t> acct(drafts.begin(), drafts.begin() + acc);
        acct.push_back(nt);
        const int q = std::min((int) acct.size(), n_gen - (int) out.size());
        rollback(T, q);
        for (int i = 0; i < q; ++i) out.push_back(acct[i]);
        if ((int) out.size() >= n_gen) break;
        // draft block catch-up over the accepted rows (positions N+1..N+acc+1, h = the verify's hn rows), then new drafts
        const int32_t d0 = drun(acct, N + 1, MTP_H_MAIN);
        N += acc + 1;
        last = nt;
        chain(d0, N);
    }
    return out;
}

std::vector<float> fn_model::forward1(int32_t token) { return forward_n({token}, 1); }

int32_t fn_model::forward1_argmax(int32_t token) {
    std::vector<float> l = forward1(token);
    int best = 0;
    for (int i = 1; i < VOCAB_; ++i) if (l[i] > l[best]) best = i;
    return best;
}

void fn_model::reset() {
    if (pipe() && !pipe_worker()) { pend_reset_ = true; pend_rb_T_ = 0; }   // node 0 resets with the next command
    local_reset();
}

void fn_model::local_reset() {
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        for (int il = 0; il < NL_; ++il) {
            if (S.conv_state[il]) CK(cudaMemsetAsync(S.conv_state[il], 0, (size_t) S.CONVr * 3 * 4, S.s));
            if (S.ssm_state[il]) CK(cudaMemsetAsync(S.ssm_state[il], 0, (size_t) S.HVr * DK_ * DV_ * 4, S.s));
        }
        if (S.ple_hist) CK(cudaMemsetAsync(S.ple_hist, 0, (size_t) (PLE_HIST_ + o_.max_t) * HCDIM_ * 4, S.s));
        if (S.mcarry) CK(cudaMemsetAsync(S.mcarry, 0, (size_t) HCDIM_ * 4, S.s));
    }
    sync_all();
    n_past_ = 0; toks_.clear();
    rope_delta_ = 0; mm_seen_ = false; mm_ = nullptr;
}

void fn_model::ple_rowids(const std::vector<int32_t> & tokens, int p0, std::vector<int64_t> & rowid) const {
    const int T = (int) tokens.size();
    rowid.assign((size_t) T * PLE_HEADS_, 0);
    std::vector<int64_t> ctx(PLE_NGRAM_);
    for (int i = 0; i < T; ++i) {
        ctx[0] = tokens[i];
        bool cut = false;
        for (int s = 1; s < PLE_NGRAM_; ++s) {
            const int p = p0 + i - s;   // absolute position of the predecessor
            int64_t t = -1;
            if (!cut && p >= 0) t = p < p0 ? (p < (int) toks_.size() ? toks_[p] : -1) : tokens[p - p0];
            cut = cut || t < 0 || t == ple_eos_;
            ctx[s] = cut ? ple_eos_ : t;
        }
        for (int n = 2; n <= PLE_NGRAM_; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * ple_mult_[0];
            for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * ple_mult_[j];
            const int base = (n - 2) * PLE_PER_;
            for (int g = 0; g < PLE_PER_; ++g) {
                const int h = base + g;
                rowid[(size_t) i * PLE_HEADS_ + h] = (int64_t) (mixed % (uint64_t) ple_vocab_[h]) + ple_off_[h];
            }
        }
    }
}

void fn_model::ple_prefetch(const std::vector<int32_t> & tokens) {
    if (PLE_LAYER_ < 0 || tokens.empty() || !owns(PLE_LAYER_)) return;   // pipe driver: the PLE layer is on node 0
    static const bool on = !getenv("P100E_FN_PLE_PREFETCH") || atoi(getenv("P100E_FN_PLE_PREFETCH")) != 0;
    if (!on) return;
    const gguf_tensor * tab = g_.find("per_layer_token_embd.weight");
    if (!tab || !tab->data) return;
    if (ple_th_.joinable()) ple_th_.join();
    std::vector<int64_t> rowid;
    ple_rowids(tokens, n_past_, rowid);
    const size_t rb = tensor_nbytes(tab->type, tab->ne) / (size_t) tab->ne[1];
    const uint8_t * base = tab->data;
    ple_th_ = std::thread([rowid = std::move(rowid), base, rb]() {
        constexpr size_t BATCH = 2048;   // rows per call: the first chunk's rows are queued first
#ifndef _WIN32
        if (!ple_async_on()) return;   // Linux: madvise readahead only with P100E_FN_PLE_ASYNC=1
#endif
        for (size_t k0 = 0; k0 < rowid.size(); k0 += BATCH) ple_willneed(base, rowid.data() + k0, std::min(BATCH, rowid.size() - k0), rb);
    });
}

void fn_model::tm_start() {
    tm_.clear();
    tm_ev_.clear();
    static const int mode = getenv("P100E_FN_TIME") ? atoi(getenv("P100E_FN_TIME")) : 0;
    time_mode_ = mode;
    tm_t0_ = std::chrono::steady_clock::now();
    if (time_mode_ == 2) tm_mark("");   // the start event
}
void fn_model::tm_flush() {
    if (time_mode_ != 2 || tm_ev_.empty()) return;
    CK(cudaSetDevice(rk_[0].dev));
    CK(cudaEventSynchronize(tm_ev_.back().second));
    for (size_t i = 1; i < tm_ev_.size(); ++i) {
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, tm_ev_[i - 1].second, tm_ev_[i].second));
        tm_[tm_ev_[i].first] += ms;
    }
    tm_ev_.clear();
}
void fn_model::tm_mark(const char * phase) {
    if (time_mode_ == 2) {
        const size_t i = tm_ev_.size();
        int prev = 0; CK(cudaGetDevice(&prev));   // callers keep launching on their rank's device
        CK(cudaSetDevice(rk_[0].dev));
        if (i >= tm_pool_.size()) { cudaEvent_t e; CK(cudaEventCreate(&e)); tm_pool_.push_back(e); }
        CK(cudaEventRecord(tm_pool_[i], rk_[0].s));
        tm_ev_.push_back({phase, tm_pool_[i]});
        CK(cudaSetDevice(prev));
        return;
    }
    sync_all();
    const auto t = std::chrono::steady_clock::now();
    tm_[phase] += std::chrono::duration<double, std::milli>(t - tm_t0_).count();
    tm_t0_ = t;
}

// ---- multimodal rows ----
// rotary position stream s (0 t, 1 h, 2 w) of a cell: the staged chunk's row, a fed row's record, or (rows not fed yet:
// text continuing after the context) cell - rope_delta_
int fn_model::rot(int cell, int s) const {
    if (mm_ && cell >= mm_p0_ && cell < mm_p0_ + mm_n_) { const mm_row & r = mm_[cell - mm_p0_]; return s == 0 ? r.t : s == 1 ? r.h : r.w; }
    if (cell < n_past_ && cell < (int) rp_[s].size()) return rp_[s][cell];
    return cell - rope_delta_;
}

// S.pos = cell index (QSA visibility), S.rpos = rotary streams of rows p0..p0+T-1, S.blk_rpos = the first cell's rotary
// streams of every block completed by these rows
void fn_model::stage_pos(rank_state & S, int p0, int T) {
    std::vector<int32_t> pos(T), rp((size_t) 3 * T);
    for (int i = 0; i < T; ++i) { pos[i] = p0 + i; for (int s = 0; s < 3; ++s) rp[(size_t) s * T + i] = rot(p0 + i, s); }
    CK(cudaMemcpyAsync(S.pos, pos.data(), (size_t) T * 4, cudaMemcpyHostToDevice, S.s));
    for (int s = 0; s < 3; ++s) CK(cudaMemcpyAsync(S.rpos + (size_t) s * o_.max_t, rp.data() + (size_t) s * T, (size_t) T * 4, cudaMemcpyHostToDevice, S.s));
    if (qsa_) {
        const int NB = o_.n_ctx / RATIO_ + 1, b0 = p0 / RATIO_, b1 = (p0 + T) / RATIO_, n = b1 - b0;
        if (n > 0) {
            std::vector<int32_t> bp((size_t) 3 * n);
            for (int b = b0; b < b1; ++b) for (int s = 0; s < 3; ++s) bp[(size_t) s * n + (b - b0)] = rot(b * RATIO_, s);
            for (int s = 0; s < 3; ++s) CK(cudaMemcpyAsync(S.blk_rpos + (size_t) s * NB + b0, bp.data() + (size_t) s * n, (size_t) n * 4, cudaMemcpyHostToDevice, S.s));
        }
    }
}

// token rows: embedding table rows dequantized into S.cur; the staged chunk's image rows: the encoder's f32 rows
void fn_model::stage_embed(rank_state & S, const std::vector<int32_t> & tokens, int p0) {
    const int T = (int) tokens.size();
    const gguf_tensor * emb = g_.find("token_embd.weight");
    const size_t erow = emb->nbytes / emb->ne[1];
    for (int i = 0; i < T; ++i)
        CK(cudaMemcpyAsync((char *) S.qstage + i * erow, emb->data + (size_t) tokens[i] * erow, erow, cudaMemcpyHostToDevice, S.s));
    dequant_f32(emb->type, S.qstage, S.cur, (int64_t) T * E_, S.s);
    if (mm_ && p0 == mm_p0_ && T <= mm_n_)
        for (int i = 0; i < T; ++i) if (mm_[i].embd) CK(cudaMemcpyAsync(S.cur + (size_t) i * E_, mm_[i].embd, (size_t) E_ * 4, cudaMemcpyHostToDevice, S.s));
}

// rope of T rows x [H][D]: plain NEOX at the t stream until an image has been fed (bit-identical to the text-only
// engine), then interleaved M-RoPE over the three streams (text rows: t = h = w, the same rotation)
void fn_model::rope_rows(rank_state & S, float * x, int T, int H, int D) {
    if (!mm_seen_) { k::rope_neox(x, S.rpos, T, H, D, NROT_, rope_base_, S.s); return; }
    fast::mrope_pos m; m.p = S.rpos; m.stride = o_.max_t; m.s0 = msec_[0]; m.s1 = msec_[1]; m.s2 = msec_[2]; m.sect = msect_;
    k::rope_mrope(x, m, T, H, D, NROT_, rope_base_, S.s);
}
void fn_model::rope_blocks(rank_state & S, float * kb, int b0, int n) {
    const int NB = o_.n_ctx / RATIO_ + 1;
    if (!mm_seen_) { k::rope_neox(kb, S.blk_rpos + b0, n, 1, IDX_D_, NROT_, rope_base_, S.s); return; }
    fast::mrope_pos m; m.p = S.blk_rpos + b0; m.stride = NB; m.s0 = msec_[0]; m.s1 = msec_[1]; m.s2 = msec_[2]; m.sect = msect_;
    k::rope_mrope(kb, m, n, 1, IDX_D_, NROT_, rope_base_, S.s);
}

std::vector<float> fn_model::forward_mm(const std::vector<mm_in> & pieces, const std::function<void(int, const std::vector<int32_t> &)> & after_chunk) {
    if (pipe()) throw std::runtime_error("fn pipe: images (forward_mm) are not supported over the pipeline yet");
    std::vector<mm_row> rows;
    std::vector<int> next_text;   // rotary position of the next text token if the call ended after e rows (-1 inside an image)
    int rn = n_past_ - rope_delta_;
    next_text.push_back(rn);
    for (const auto & p : pieces) {
        if (p.embd) {
            if (p.nx < 1 || p.ny < 1) throw std::runtime_error("fn_model: forward_mm: empty image");
            if (p.n_embd != E_)
                throw std::runtime_error("fn_model: forward_mm: image rows are " + std::to_string(p.n_embd) + " wide, the model's are " +
                                         std::to_string(E_) + " (an mmproj of another model?)");
            const int n = p.nx * p.ny, rn_after = rn + std::max(p.nx, p.ny);
            for (int i = 0; i < n; ++i) {
                rows.push_back(mm_row{img_tok_, p.embd + (size_t) i * E_, rn, rn + i / p.nx, rn + i % p.nx});
                next_text.push_back(i + 1 < n ? -1 : rn_after);
            }
            rn = rn_after;
        } else {
            for (int i = 0; i < p.n_tok; ++i) { rows.push_back(mm_row{p.tokens[i], nullptr, rn, rn, rn}); ++rn; next_text.push_back(rn); }
        }
    }
    const int N = (int) rows.size();
    if (N < 1) throw std::runtime_error("fn_model: forward_mm: empty prompt");
    if (n_past_ + N > o_.n_ctx) throw std::runtime_error("fn_model: context full");
    std::vector<float> L;
    {
        std::vector<int32_t> all(N);
        for (int i = 0; i < N; ++i) all[i] = rows[i].tok;
        ple_prefetch(all);
    }
    for (int c0 = 0; c0 < N; c0 += o_.max_t) {
        const int c = std::min(o_.max_t, N - c0), p0 = n_past_;
        std::vector<int32_t> toks(c);
        for (int i = 0; i < c; ++i) toks[i] = rows[(size_t) c0 + i].tok;
        mm_ = rows.data() + c0; mm_p0_ = p0; mm_n_ = c; mm_seen_ = true;
        try {
            L = forward_n(toks, 1);
            if (next_text[(size_t) c0 + c] >= 0) rope_delta_ = n_past_ - next_text[(size_t) c0 + c];
            if (after_chunk) after_chunk(p0, toks);
        } catch (...) { mm_ = nullptr; throw; }
        mm_ = nullptr;
    }
    return L;
}


// =====================================================================================================================
// Hybrid pipeline over two machines (see fn_model.h): byte plan, connect / agree, driver commands,
// worker loop, residual handoff.
// =====================================================================================================================

double fn_model::ram_reserve_bytes() const {
    if (o_.ram_reserve_mb >= 0) return o_.ram_reserve_mb * 1048576.0;
    if (const char * e = getenv("P100E_RAM_RESERVE_MB"); e && *e) return std::max(0.0, atof(e)) * 1048576.0;
#ifdef _WIN32
    return 12288 * 1048576.0;
#else
    return 6144 * 1048576.0;
#endif
}

double fn_model::host_free_bytes() {
    if (const char * e = getenv("P100E_FN_HOST_FREE_MB")) return atof(e) * 1048576.0;   // planning override (tests)
#ifdef _WIN32
    MSX_ m{};
    m.dwLength = sizeof m;
    if (GlobalMemoryStatusEx(&m)) return (double) m.ullAvailPhys;
    return 0;
#else
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    double kb = 0;
    while (fgets(line, sizeof line, f)) if (sscanf(line, "MemAvailable: %lf kB", &kb) == 1) break;
    fclose(f);
    return kb * 1024.0;
#endif
}

std::vector<double> fn_model::gpu_free_per_rank(const std::vector<int> & devices) {
    std::vector<double> fr(devices.size(), 0.0);
    const char * ov = getenv("P100E_FN_GPU_FREE_MB");   // planning override (tests): MiB per rank, comma separated
    for (size_t r = 0; r < devices.size(); ++r) {
        CK(cudaSetDevice(devices[r]));
        size_t f = 0, t = 0;
        CK(cudaMemGetInfo(&f, &t));
        const int share = (int) std::count(devices.begin(), devices.end(), devices[r]);   // ranks on one GPU (tests) split it
        fr[r] = (double) f / share;
    }
    if (ov) {
        std::string s = ov;
        size_t r = 0, p = 0;
        while (r < fr.size() && p <= s.size()) {
            const size_t q = s.find(',', p);
            const std::string v = s.substr(p, q == std::string::npos ? std::string::npos : q - p);
            if (!v.empty()) fr[r] = std::min(fr[r], atof(v.c_str()) * 1048576.0);
            ++r;
            if (q == std::string::npos) break;
            p = q + 1;
        }
    }
    return fr;
}

namespace {
bool fn_small_suffix(const std::string & s) {   // tensors that load() reads through small() (f32 copies)
    static const char * k[] = {"hc_attn_norm.weight", "hc_attn_inject.weight", "hc_ffn_norm.weight", "hc_ffn_inject.weight",
                               "attn_q_norm.weight", "attn_k_norm.weight", "indexer.q_norm.weight", "indexer.k_norm.weight",
                               "ssm_conv1d.weight", "ssm_dt.bias", "ssm_a", "ssm_norm.weight", "ffn_gate_inp_shexp.weight",
                               "nextn.enorm.weight", "nextn.hnorm.weight", "nextn.hc_head_norm.weight", "ple_norm_key.weight",
                               "ple_norm_query.weight", "ple_norm_conv.weight", "ple_conv1d.weight"};
    for (const char * x : k) if (s == x) return true;
    return false;
}
bool fn_upload_suffix(const std::string & s) {   // tensors that load() packs with upload_weight / _rows / _cols
    static const char * k[] = {"hc_attn_down.weight", "hc_attn_up.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", "attn_q.weight",
                               "attn_k.weight", "attn_v.weight", "attn_output.weight", "indexer.q_proj.weight", "indexer.k_proj.weight",
                               "attn_qkv.weight", "ssm_out.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight",
                               "ffn_gate_inp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
                               "nextn.eh_proj.weight", "nextn.hc_head_down.weight", "nextn.hc_head_up.weight", "ple_key.weight",
                               "ple_value.weight"};
    for (const char * x : k) if (s == x) return true;
    return false;
}
bool fn_tp_suffix(const std::string & s) {   // split by heads across the ranks when the dense split is on
    static const char * k[] = {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_qkv.weight",
                               "ssm_out.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight"};
    for (const char * x : k) if (s == x) return true;
    return false;
}
// device bytes upload_raw makes of a tensor: narrow quantized rows are dequantized to f16
double fn_upload_bytes(const gguf_tensor * t) {
    const bool fl = t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16;
    if (!fl && t->ne[0] % 256 != 0) return (double) t->nelements() * 2;
    return (double) t->nbytes;
}
}

// Per rank r of a node running layers [l0, l1) (+ head + draft block when l1 == L): what load() allocates on its GPU
// (the same formulas), the resident expert slots (adaptive layers: request -1 = auto = what fits that GPU after
// everything else), and the node's pinned host RAM (host expert records of its layers, staging, the promotion ring).
// The PLE table and the token embedding are read from the memory-mapped file (page cache: not counted).
fn_model::node_plan fn_model::plan_node(int l0, int l1, bool embed, const std::vector<double> & gpu_free,
                                        const std::vector<int> & slots_req, double host_free, bool pipe_mode, double reserve) const {
    (void) embed;
    node_plan P;
    const int R = (int) gpu_free.size();
    const bool head = l1 == L_;
    const bool tp = o_.tp && R > 1 && H_ % R == 0 && HKV_ % R == 0 && HK_ % R == 0 && HV_ % R == 0 &&
                    ((int64_t) (H_ / R) * HD_) % 256 == 0 && ((int64_t) (HV_ / R) * DV_) % 256 == 0 && (HV_ / R) % 4 == 0;
    const int d = tp ? R : 1;
    const int HKVr = HKV_ / d, HVr = HV_ / d, CONVr = 2 * (HK_ / d) * DK_ + HVr * DV_;
    const size_t T = (size_t) std::max(o_.max_t, 1), NR = T * NEXPU_;
    const double MiB = 1048576.0;
    auto al = [](double b) { return std::ceil(b / 256.0) * 256.0; };
    std::vector<int> layers;
    for (int il = l0; il < l1; ++il) layers.push_back(il);
    if (head && mtp_il_ >= 0) layers.push_back(mtp_il_);
    // fixed buffers (load(): staging, scratch, activations at max_t rows)
    const double wide = (double) T * std::max({2 * H_ * HD_, CONV_C_, HCDIM_, 4 * E_}) * 4;
    double fixed = (double) ((((size_t) o_.stage_mb) << 20) & ~(size_t) 511) + 64 * MiB + 6 * wide + 2.0 * T * HCDIM_ * 4 +
                   3.0 * T * E_ * 4 + (double) T * HC_ * 4 + (double) T * NEXP_ * 4 + (double) std::min<size_t>(T, FN_MAX_LOGITS) * VOCAB_ * 4 +
                   4.0 * T * 4 + 2.0 * NR * 4 + 2.0 * 3 * NR * sizeof(fast::mv_group) + (double) fast::matvec_scratch_bytes(FFE_, (int) NR) +
                   (double) fast::matvec_scratch_bytes(E_, (int) NR) + 3.0 * NR * 4 + (double) NR * E_ * 4 + (double) NR * std::max(E_, FFE_) * 4 +
                   (double) fast::matvec_scratch_bytes(std::max({HCDIM_, CONV_C_, 2 * H_ * HD_, E_}), 8) +
                   (double) std::max(fast::gemm_scratch_min_bytes(HCDIM_, (int) T), (size_t) 96 << 20);
    if (PLE_LAYER_ >= 0) fixed += (double) (PLE_HIST_ + T) * HCDIM_ * 4 + (double) T * E_ * 4;
    if (mtp_il_ >= 0) fixed += 2.0 * T * HCDIM_ * 4 + 2.0 * HCDIM_ * 4 + (double) T * HC_ * 2 * E_ * 4 + (PLE_LAYER_ >= 0 ? (double) PLE_HIST_ * HCDIM_ * 4 : 0);
    int n_att = 0;
    for (int il : layers) if ((il + 1) % full_attn_interval_ == 0 || il == mtp_il_) ++n_att;
    if (qsa_) {
        const double NB = o_.n_ctx / RATIO_ + 1;
        fixed += 3 * NB * 4 + (double) T * IDX_D_ * 4 + (double) T * IDX_H_ * IDX_D_ * 4 + (double) QSA_SL_ * IDX_H_ * NB * 4 + (double) QSA_SL_ * NB * 4 +
                 (double) T * QSA_W_ * 4 + (double) T * 4;
    }
    if (kv_mode_ != 0 && n_att > 0) fixed += 2.0 * ((qsa_ ? QSA_W_ : o_.n_ctx) + o_.max_t) * HKVr * HD_ * 4;
    const double margin = 384 * MiB;   // cuBLAS workspace, the CUDA runtime's own allocations, fragmentation
    // cudaMalloc hands out whole 2 MiB pages for allocations of a MiB and more (smaller ones share pages): measured
    // on the 3070, every dense upload of a layer took its size rounded up to 2 MiB, ~12-24 MiB per layer
    // above the file bytes, i.e. ~0.6-1.1 GiB over 49 layers -- the overrun behind the auto-slot OOMs at the prefetch
    // arenas (the arenas themselves are counted below)
    auto pg = [](double b) { return b >= 1048576.0 ? std::ceil(b / 2097152.0) * 2097152.0 : b; };
    P.rk.assign(R, rank_plan{});
    double host = 0, ring = 0;
    for (int r = 0; r < R; ++r) {
        rank_plan & q = P.rk[r];
        q.buf = fixed + margin;
        q.free = gpu_free[r];
        double per_slot = 0;   // bytes of one resident slot in every adaptive layer of this rank
        double host_r = 0, pf_max = 0, fixed_res = 0;
        int max_own = 0, n_adapt = 0;
        const int dflt = o_.hot_per_rank > 0 ? o_.hot_per_rank : 72;
        const int req = r < (int) slots_req.size() ? slots_req[r] : dflt;   // -1 = auto
        for (int il : layers) {
            const int fil = file_layer(il);
            const std::string p = "blk." + std::to_string(fil) + ".";
            const bool attn = (il + 1) % full_attn_interval_ == 0 || il == mtp_il_;
            for (const auto & t : g_.tensors) {
                if (t.name.compare(0, p.size(), p) != 0) continue;
                const std::string s = t.name.substr(p.size());
                if (s.find("_exps") != std::string::npos || s == "ffn_exp_map") continue;
                if (s.compare(0, 6, "nextn.") == 0 && il != mtp_il_) continue;
                if (s.compare(0, 4, "ple_") == 0 && il != PLE_LAYER_) continue;
                if (fn_small_suffix(s)) {
                    q.dense += (double) t.nelements() * 4;
                    if (tp && (s == "ssm_conv1d.weight" || s == "ssm_dt.bias" || s == "ssm_a")) q.dense += (double) t.nelements() * 4 / R;
                } else if (fn_upload_suffix(s)) {
                    if (!qsa_ && s.compare(0, 8, "indexer.") == 0) continue;
                    double b = fn_upload_bytes(&t);
                    if (tp && fn_tp_suffix(s)) b /= R;
                    q.dense += pg(b);
                }
            }
            if (attn) {
                q.kv += 2.0 * pg((double) o_.n_ctx * HKVr * (double) qsa::kv_row_bytes(kv_mode_, HD_));
                if (qsa_) q.kv += pg((double) (o_.n_ctx / RATIO_ + 1) * IDX_D_ * 4) + (double) QSA_RING_ * IDX_D_ * 4;
            } else {
                const double conv = (double) CONVr * 3 * 4, ssm = (double) HVr * DK_ * DV_ * 4;   // + MTP_ROWS - 1 snapshots each
                q.state += pg(conv) + pg(ssm) + (mtp_il_ >= 0 ? pg((MTP_ROWS - 1) * conv) + pg((MTP_ROWS - 1) * ssm) : 0.0);
            }
            fixed_res += 3 * 2097152.0;   // the layer's three resident expert arrays (gate / up / down): page rounding
            // experts of this layer (load_experts' rules)
            const int n_hot = hot_count_[fil], n_cold = NEXP_ - n_hot;
            const gguf_tensor * map = g_.find(p + "ffn_exp_map");
            bool cold_all = n_hot == NEXP_ && map;
            for (int e = 0; e < NEXP_ && cold_all; ++e) cold_all = ((const int32_t *) map->data)[NEXP_ + e] >= 0;
            const bool adaptive = cold_all && o_.adapt && !rank_order_.empty();
            auto per_expert = [&](const char * a, const char * b, const char * c, bool align) {
                double s = 0;
                for (const char * n : {a, b, c}) {
                    const gguf_tensor * t = g_.find(p + n);
                    if (!t) return 0.0;
                    const double x = (double) t->nbytes / (double) t->ne[2];
                    s += align ? al(x) : x;
                }
                return s;
            };
            const double hot_dev = per_expert("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", false);
            const double hot_rec = per_expert("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", true);
            const double cold_rec = per_expert("ffn_gate_exps_cold.weight", "ffn_up_exps_cold.weight", "ffn_down_exps_cold.weight", true);
            double lay_host = 0;
            if (adaptive) {
                int own = 0;
                for (int e = 0; e < NEXP_; ++e) if (e % R == r) ++own;
                max_own = std::max(max_own, own);
                lay_host = own * cold_rec;
                per_slot += hot_dev;
                ++n_adapt;
                if (ring == 0) ring = 96.0 * hot_rec;   // promo_start's pinned ring
            } else {
                const int HOTR = o_.hot_per_rank < 0 ? (n_hot + R - 1) / R : o_.hot_per_rank;
                int h0 = r * HOTR, hall = R * HOTR, cnt = HOTR;
                if (!slots_req.empty()) {
                    h0 = 0; hall = 0;
                    for (int qq = 0; qq < R; ++qq) { const int c = slots_req[qq] < 0 ? HOTR : slots_req[qq]; if (qq < r) h0 += c; hall += c; }
                    cnt = slots_req[r] < 0 ? HOTR : slots_req[r];
                }
                int res = 0, hh = 0, cc = 0;
                for (int h = 0; h < n_hot; ++h) { if (h >= h0 && h < h0 + cnt) ++res; else if (h >= hall && h % R == r) ++hh; }
                for (int c = 0; c < n_cold; ++c) if (c % R == r) ++cc;
                fixed_res += res * hot_dev;
                lay_host = hh * hot_rec + cc * cold_rec;
                P.resident += (double) res / NEXP_;
            }
            host_r += lay_host;
            pf_max = std::max(pf_max, lay_host);
        }
        if (o_.prefetch_min_t > 0 && o_.max_t >= o_.prefetch_min_t) q.buf += 2 * pg(pf_max);   // the two prefetch arenas (load())
        if (head && (r == 0 || !pipe_mode)) {
            for (const char * n : {"output.weight", "output_hc_down.weight", "output_hc_up.weight"})
                if (const gguf_tensor * t = g_.find(n)) q.head += pg(fn_upload_bytes(t));
            q.head += (double) HCDIM_ * 4;
        }
        const double other = q.dense + q.kv + q.state + q.head + q.buf + fixed_res;
        q.max_slots = max_own;
        if (n_adapt == 0) q.slots = req;   // no adaptive layer: the slot count is the non-adaptive range (in fixed_res)
        else if (req < 0) {                // auto: fill this GPU
            const double avail = q.free - other;
            q.slots = avail > 0 ? std::min(max_own, (int) std::floor(avail / per_slot)) : 0;
        } else q.slots = std::min(req, max_own);
        q.slots_b = fixed_res + (n_adapt ? per_slot * q.slots : 0);
        if (n_adapt) P.resident += (double) n_adapt * q.slots / NEXP_;
        q.total = q.dense + q.kv + q.state + q.head + q.buf + q.slots_b;
        host += host_r + (double) T * NEXPU_ * 4 * 2 + (double) HBOUNCE_ + 2.0 * NR * 4 * 2 + NR * 4 + 2.0 * 3 * NR * sizeof(fast::mv_group);
    }
    host += ring + (double) (sizeof(uint64_t) * 8 + T * HCDIM_ * 4);
    P.host = host;
    P.host_free = host_free;
    P.reserve = reserve >= 0 ? reserve : ram_reserve_bytes();
    const double host_margin = P.reserve;   // P100E_RAM_RESERVE_MB: RAM the OS and other programs keep (pinned RAM cannot be paged)
    P.fits = host + host_margin <= host_free;
    for (const rank_plan & q : P.rk) P.fits = P.fits && q.total <= q.free && !(q.max_slots > 0 && q.slots <= 0);
    return P;
}

void fn_model::print_plan(const char * tag, int l0, int l1, const node_plan & p) const {
    auto gib = [](double b) { return b / 1073741824.0; };
    printf("%s: layers [%d, %d)%s%s: host RAM %.2f GiB pinned of %.2f GiB available (reserve %.1f GiB): %s; resident experts %.1f layer-equivalents of %d\n",
           tag, l0, l1, l0 == 0 ? " + embedding" : "", l1 == L_ ? (mtp_il_ >= 0 ? " + head + draft block" : " + head") : "",
           gib(p.host), gib(p.host_free), gib(p.reserve), p.host + p.reserve <= p.host_free ? "fits" : "DOES NOT FIT", p.resident,
           l1 - l0 + (l1 == L_ && mtp_il_ >= 0 ? 1 : 0));
    for (size_t r = 0; r < p.rk.size(); ++r) {
        const rank_plan & q = p.rk[r];
        printf("%s:   rank %zu: dense %.2f + experts %.2f (%d slots per layer) + KV %.2f + state %.2f + head %.2f + buffers %.2f = %.2f GiB, free %.2f GiB: %s\n",
               tag, r, gib(q.dense), gib(q.slots_b), q.slots, gib(q.kv), gib(q.state), gib(q.head), gib(q.buf), gib(q.total), gib(q.free),
               q.total <= q.free && !(q.max_slots > 0 && q.slots <= 0) ? "fits" : "DOES NOT FIT");
    }
    fflush(stdout);
}

int fn_model::plan_split(const std::vector<double> & free0, double host0, const std::vector<int> & slots0,
                         const std::vector<double> & free1, double host1, const std::vector<int> & slots1, int k_req, bool print,
                         double res0, double res1) const {
    const int kmin = std::max(1, PLE_LAYER_ + 1);   // the PLE layer (it reads the 28.8 GB table) stays on node 0
    int best = -1;
    double best_s = -1;
    for (int k = kmin; k < L_; ++k) {
        if (k_req > 0 && k != k_req) continue;
        const node_plan a = plan_node(0, k, true, free0, slots0, host0, true, res0), b = plan_node(k, L_, false, free1, slots1, host1, true, res1);
        if (!a.fits || !b.fits) continue;
        // the summed resident share is nearly flat in k (VRAM is VRAM); what differs is where the misses land. Pick the
        // k with the most even hit rate (resident share per layer) on the two nodes, so neither stage is the one
        // stalled on PCIe staging; ties: more layers on node 0
        const double s = std::min(a.resident / k, b.resident / (L_ - k + (mtp_il_ >= 0 ? 1 : 0)));
        if (s >= best_s - 1e-9) { best_s = s; best = k; }
    }
    if (print) {
        const int k = best > 0 ? best : (k_req > 0 ? k_req : (kmin + L_) / 2);
        printf("pipe: byte plan for k = %d%s: node 0 layers [0, %d), node 1 layers [%d, %d) + head%s; ctx %d, max_t %d, KV %s\n", k,
               best > 0 ? "" : " (NO split fits: shown for reference)", k, k, L_, mtp_il_ >= 0 ? " + draft block" : "", o_.n_ctx, o_.max_t,
               kv_mode_ == 0 ? "f32" : kv_mode_ == 1 ? "f16" : "q8");
        print_plan("pipe: node 0", 0, k, plan_node(0, k, true, free0, slots0, host0, true, res0));
        print_plan("pipe: node 1", k, L_, plan_node(k, L_, false, free1, slots1, host1, true, res1));
    }
    return best;
}

// ---- connect, agree, plan ----
namespace {
constexpr uint32_t FNP_MAGIC = 0x464e5031u;   // "1PNF"
struct fnp_cmd { uint32_t magic, op; int32_t T, n_past; uint64_t step; int32_t rb_T, rb_keep, reset, pad; };   // + max_t tokens
struct fnp_hdr { uint32_t magic, status; int32_t T, n_past; uint64_t step; };
static_assert(sizeof(fnp_cmd) == 40 && sizeof(fnp_hdr) == 24, "fn pipe layouts");
const char * const kFnPipeEnv[] = {"P100E_NET", "P100E_FN_QSA", "P100E_FN_QSA_FORCE", "P100E_FN_KV", "P100E_FN_MTP"};
}

void fn_model::pipe_init(const std::vector<int> & devices) {
    auto envi = [](const char * n, int d) { const char * e = getenv(n); return e && *e ? atoi(e) : d; };
    const int connect_s = envi("P100E_NET_CONNECT_S", o_.net_connect_s), io_s = envi("P100E_NET_TIMEOUT_S", o_.net_timeout_s);
    if (o_.net_nodes.size() != 2) throw std::runtime_error("fn pipe: exactly 2 nodes (--net-nodes a,b), got " + std::to_string(o_.net_nodes.size()));
    if (o_.pipe_layers == 0) throw std::runtime_error("fn pipe: --net-nodes needs --pipe-layers k|auto (fn_model has no tensor parallel over the network)");
    const std::string list = o_.net_nodes[0] + "," + o_.net_nodes[1];
    printf("fn pipe: node %d of %s (%s), %zu local GPU(s); waiting up to %d s for the other node\n", o_.net_node, list.c_str(),
           o_.net_node == 0 ? "worker: layers [0, k) + embedding" : "driver: layers [k, L) + head + draft block", devices.size(), connect_s);
    fflush(stdout);
    pnet_ = net_transport_open(o_.net_nodes, o_.net_node, connect_s * 1000, io_s * 1000);
    printf("fn pipe: transport %s\n", pnet_->kind());
    pnet_->agree("the multi-node mode (fn_model pipe, protocol 1)", 0xF0001);
    pnet_->agree("the model file size", g_.file_size);
    pnet_->agree("the model metadata (GGUF keys)", g_.meta_hash());
    pnet_->agree("the model tensor table", g_.tensor_table_hash());
    pnet_->agree("the model tensor data (sample)", g_.tensor_sample_hash());
    pnet_->agree("n_ctx (--ctx)", (uint64_t) o_.n_ctx);
    pnet_->agree("max_t (--max-t / chunk)", (uint64_t) o_.max_t);
    pnet_->agree("the layer count (--layers)", (uint64_t) L_);
    pnet_->agree("the KV cache type", (uint64_t) kv_mode_);
    pnet_->agree("MTP on (the draft block; node 0 keeps rollback snapshots)", (uint64_t) (mtp_il_ >= 0));
    pnet_->agree("the run arguments", o_.net_args_hash);
    pnet_->agree("the pipe layers (--pipe-layers / --pipe-split)", (uint64_t) (int64_t) o_.pipe_layers);
    pnet_->agree("the pipe total (--pipe-split a,b: a + b)", (uint64_t) o_.pipe_total);
    for (const char * n : kFnPipeEnv) {
        const char * e = getenv(n);
        const std::string v = e ? e : "<unset>";
        pnet_->agree(std::string("the switch ") + n + " (" + v + " here)", hash_bytes(kHashSeed, v.data(), v.size()));
    }
    if (o_.pipe_total && o_.pipe_total != L_)
        throw std::runtime_error("--pipe-split: the parts add up to " + std::to_string(o_.pipe_total) + " layers, the model has " + std::to_string(L_));
    // both nodes' resources -> the same plan on both
    const std::vector<double> fr = gpu_free_per_rank(devices);
    constexpr int MAXR = 8;
    if (devices.size() > (size_t) MAXR) throw std::runtime_error("fn pipe: at most 8 local ranks");
    const std::vector<uint64_t> nr = pnet_->allgather((uint64_t) devices.size());
    const std::vector<uint64_t> ns = pnet_->allgather((uint64_t) slots_.size());
    const std::vector<uint64_t> hf = pnet_->allgather((uint64_t) host_free_bytes());
    const std::vector<uint64_t> rs = pnet_->allgather((uint64_t) ram_reserve_bytes());   // each node's own reserve
    std::vector<std::vector<double>> frs(2);
    std::vector<std::vector<int>> rqs(2);
    for (int i = 0; i < MAXR; ++i) {
        const std::vector<uint64_t> a = pnet_->allgather(i < (int) fr.size() ? (uint64_t) fr[i] : 0);
        const std::vector<uint64_t> b = pnet_->allgather(i < (int) slots_.size() ? (uint64_t) (int64_t) slots_[i] : 0);
        for (int j = 0; j < 2; ++j) {
            if (i < (int) nr[j]) frs[j].push_back((double) a[j]);
            if (i < (int) ns[j]) rqs[j].push_back((int) (int64_t) b[j]);
        }
    }
    const int k = plan_split(frs[0], (double) hf[0], rqs[0], frs[1], (double) hf[1], rqs[1], o_.pipe_layers > 0 ? o_.pipe_layers : -1, true,
                                (double) rs[0], (double) rs[1]);
    {   // RAM is refused by name (needed vs available vs reserve) before anything is pinned
        const int self0 = pnet_->self();
        const int kk = k > 0 ? k : (o_.pipe_layers > 0 ? o_.pipe_layers : -1);
        if (kk > 0) {
            const node_plan p = self0 == 0 ? plan_node(0, kk, true, frs[0], rqs[0], (double) hf[0], true, (double) rs[0])
                                           : plan_node(kk, L_, false, frs[1], rqs[1], (double) hf[1], true, (double) rs[1]);
            if (p.host + p.reserve > p.host_free)
                fprintf(stderr, "fn pipe: this node lacks RAM for k = %d: %lld MiB pinned needed, %lld MiB available, reserve %lld MiB\n", kk,
                        (long long) (p.host / 1048576), (long long) (p.host_free / 1048576), (long long) (p.reserve / 1048576));
        }
    }
    if (k < 1) throw std::runtime_error(o_.pipe_layers > 0 ? "fn pipe: --pipe-layers " + std::to_string(o_.pipe_layers) + " does not fit both nodes (see the plan above)"
                                                           : std::string("fn pipe: no layer split fits both nodes (see the plan above): a shorter --ctx, fewer slots or more RAM"));
    pipe_k_ = k;
    const int self = pnet_->self();
    lb_ = self == 0 ? 0 : k;
    le_ = self == 0 ? k : L_;
    if (!slots_.empty()) {   // this node's resolved slots (auto -> what its plan gives)
        const node_plan p = self == 0 ? plan_node(0, k, true, frs[0], rqs[0], (double) hf[0], true, (double) rs[0]) : plan_node(k, L_, false, frs[1], rqs[1], (double) hf[1], true, (double) rs[1]);
        for (size_t r = 0; r < slots_.size(); ++r) slots_[r] = p.rk[r].slots;
    }
    printf("fn pipe: k = %d: this node (%d) runs layers [%d, %d)%s\n", k, self, lb_, le_, has_head() ? " + head + draft block" : " + embedding");
    pipe_cap_ = (size_t) o_.max_t * HCDIM_ * 4;
    CK(cudaMallocHost((void **) &pipe_h_, pipe_cap_));
}

// driver: one command; the pending rollback / reset ride on it
void fn_model::pipe_cmd(int op, const std::vector<int32_t> & tokens) {
    if ((int) tokens.size() > o_.max_t) throw std::runtime_error("fn pipe: command too long");
    fnp_cmd c{FNP_MAGIC, (uint32_t) op, (int32_t) tokens.size(), n_past_, pipe_step_, pend_rb_T_, pend_rb_keep_, pend_reset_ ? 1 : 0, 0};
    std::vector<int32_t> tk((size_t) o_.max_t, 0);
    std::copy(tokens.begin(), tokens.end(), tk.begin());
    fnp_hdr ack{};
    std::vector<std::vector<net_link::seg>> rcv(2);
    rcv[0] = {{&ack, sizeof ack}};
    pnet_->exchange(net_link::T_PIPE, {{&c, sizeof c}, {tk.data(), tk.size() * 4}}, rcv);
    if (ack.magic != FNP_MAGIC || ack.step != pipe_step_)
        throw std::runtime_error("fn pipe: lockstep lost at command " + std::to_string(pipe_step_) + " (node 0 is at " + std::to_string(ack.step) + ")");
    if (ack.status != 0) throw std::runtime_error("fn pipe: node 0 reported an error (see its log)");
    ++pipe_step_;
    pend_rb_T_ = pend_rb_keep_ = 0;
    pend_reset_ = false;
}

// driver: node 0's rows of the command just sent -> S.res of every local rank (H2D on each rank's stream; the staging
// is rewritten only by the next forward, after this one's final sync)
void fn_model::pipe_recv_rows(int T) {
    const uint64_t st = pipe_step_ - 1;
    fnp_hdr mine{FNP_MAGIC, 0, T, n_past_, st}, h{};
    std::vector<std::vector<net_link::seg>> rcv(2);
    rcv[0] = {{&h, sizeof h}, {pipe_h_, (size_t) T * HCDIM_ * 4}};
    pnet_->exchange(net_link::T_PIPE, {{&mine, sizeof mine}}, rcv);
    if (h.magic != FNP_MAGIC || h.T != T || h.n_past != n_past_ || h.step != st)
        throw std::runtime_error("fn pipe: lockstep lost at the rows of command " + std::to_string(st) + ": " + std::to_string(T) +
                                 " rows at n_past " + std::to_string(n_past_) + " here, node 0 sent " + std::to_string(h.T) + " at " + std::to_string(h.n_past));
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        CK(cudaMemcpyAsync(S.res, pipe_h_, (size_t) T * HCDIM_ * 4, cudaMemcpyHostToDevice, S.s));
    }
}

// worker: rank 0's wide residual rows after this node's last layer -> node 1
void fn_model::pipe_send_rows(int T) {
    rank_state & S = rk_[0];
    CK(cudaSetDevice(S.dev));
    CK(cudaMemcpyAsync(pipe_h_, S.res, (size_t) T * HCDIM_ * 4, cudaMemcpyDeviceToHost, S.s));
    sync_all();
    const uint64_t st = pipe_step_ - 1;
    fnp_hdr h{FNP_MAGIC, 0, T, n_past_, st}, peer{};
    std::vector<std::vector<net_link::seg>> rcv(2);
    rcv[1] = {{&peer, sizeof peer}};
    pnet_->exchange(net_link::T_PIPE, {{&h, sizeof h}, {pipe_h_, (size_t) T * HCDIM_ * 4}}, rcv);
    if (peer.magic != FNP_MAGIC || peer.T != T || peer.n_past != n_past_ || peer.step != st)
        throw std::runtime_error("fn pipe: lockstep lost at the rows of command " + std::to_string(st));
}

void fn_model::pipe_serve() {
    if (!pipe_worker()) throw std::runtime_error("fn pipe: pipe_serve() runs on node 0 only");
    printf("fn pipe: node 0 serving layers [%d, %d)\n", lb_, le_);
    fflush(stdout);
    std::vector<int32_t> tk((size_t) o_.max_t, 0);
    for (;;) {
        fnp_hdr ack{FNP_MAGIC, 0, 0, n_past_, pipe_step_};
        fnp_cmd c{};
        std::vector<std::vector<net_link::seg>> rcv(2);
        rcv[1] = {{&c, sizeof c}, {tk.data(), tk.size() * 4}};
        pnet_->exchange(net_link::T_PIPE, {{&ack, sizeof ack}}, rcv, INT_MAX);   // idle: wait for the driver as long as it lives
        if (c.magic != FNP_MAGIC || c.step != pipe_step_)
            throw std::runtime_error("fn pipe: lockstep lost: command " + std::to_string(c.step) + ", expected " + std::to_string(pipe_step_));
        ++pipe_step_;
        if (c.reset) local_reset();
        if (c.rb_T) {
            // the driver's rollback() limits: n_keep < T <= MTP_ROWS (the snapshots hold MTP_ROWS - 1 rows) and the MTP build
            if (c.rb_keep < 1 || c.rb_keep >= c.rb_T || c.rb_T > n_past_ || c.rb_T > MTP_ROWS || mtp_il_ < 0)
                throw std::runtime_error("fn pipe: bad rollback from the driver");
            local_rollback(c.rb_T, c.rb_keep);
        }
        if (c.op == 2) { printf("fn pipe: the driver quit after %llu commands\n", (unsigned long long) pipe_step_); fflush(stdout); return; }
        if (c.n_past != n_past_)
            throw std::runtime_error("fn pipe: lockstep lost: the driver is at n_past " + std::to_string(c.n_past) + ", node 0 at " + std::to_string(n_past_));
        if (c.op == 1) {
            if (c.T < 1 || c.T > o_.max_t) throw std::runtime_error("fn pipe: bad row count");
            forward_n(std::vector<int32_t>(tk.begin(), tk.begin() + c.T), 1);
        }
    }
}

void fn_model::pipe_quit() {
    if (!pipe() || pipe_worker() || quit_sent_) return;
    quit_sent_ = true;
    pipe_cmd(2, {});
}
// =====================================================================================================================
// Remote expert tier (src/fn_rexp.h): node 1 (p100e-fn-expert-server) holds Q4 experts this node has
// not resident. Per layer of a forward of <= rexp_max_t rows, rank 0's layer_ffn picks every rank's non-resident
// experts that node 1 holds, sends their rows in one message and runs the GPU experts meanwhile; the last rank's call
// receives the outputs, uploads them behind each rank's GPU rows (the CPU pool's mechanism in ffn_grouped) and runs the
// combine. Exact mode: each pair's expert output comes back, so the combine is the local path's (same expert order,
// same fma chain); sum mode: one weighted sum per row (added as one row of weight 1).
// =====================================================================================================================

fn_model::rexp_moe::~rexp_moe() {
    try { if (k.ok()) { rexp_hdr h; h.op = REXP_QUIT; k.send_msg(h, nullptr, 0, 5000); } } catch (...) {}
    printf("fn_model: remote experts: %lld requests, %lld pairs (%.1f%% of the non-resident routed rows of those layers), %.1f MB sent, %.1f MB received; "
           "host waited %.1f ms for replies after queueing its own experts (max %.2f ms per layer)\n",
           (long long) n_req, (long long) n_pairs, n_rows_routed ? 100.0 * n_pairs / n_rows_routed : 0.0, bytes_out / 1e6, bytes_in / 1e6, t_wait, t_wait_max);
    for (auto & [T, v] : by_T)
        printf("fn_model: remote experts: forwards of %d rows: %lld requests, %.1f pairs per request\n", T, (long long) v.first, v.first ? (double) v.second / v.first : 0.0);
    if (n_req)
        printf("fn_model: remote experts: per request ms: send %.3f, wait for the reply header %.3f, reply payload %.3f (%.1f KB out, %.1f KB in per request%s)\n",
               t_send / n_req, t_hdr / n_req, t_body / n_req, bytes_out / 1024.0 / n_req, bytes_in / 1024.0 / n_req, half ? ", f16 replies" : "");
    hit all;
    for (auto & [T, h] : hits_T) {
        const double n = (double) std::max<int64_t>(1, h.local + h.remote + h.miss);
        printf("fn_model: remote experts: hit rate, forwards of %d rows%s (%lld layer passes): resident here %.1f%% + node 1 %.1f%% = combined %.1f%%, host path %.1f%%\n",
               T, T == 1 ? " (decode / MTP drafts)" : T <= 4 ? " (MTP verify)" : "", (long long) h.fwd, 100.0 * h.local / n, 100.0 * h.remote / n,
               100.0 * (h.local + h.remote) / n, 100.0 * h.miss / n);
        all.local += h.local; all.remote += h.remote; all.miss += h.miss;
    }
    {
        const double n = (double) std::max<int64_t>(1, all.local + all.remote + all.miss);
        printf("fn_model: remote experts: hit rate, all forwards of <= max rows: resident here %.1f%% + node 1 %.1f%% = combined %.1f%%; node 1 swaps %lld sent, %lld landed; %lld layer passes skipped (< %d pairs)\n",
               100.0 * all.local / n, 100.0 * all.remote / n, 100.0 * (all.local + all.remote) / n, (long long) swaps_sent, (long long) swaps_landed, (long long) n_skipped, min_pairs);
    }
    for (size_t r = 0; r < rk.size(); ++r) {
        cudaSetDevice(devs[r]);
        cudaDeviceSynchronize();
        for (int j = 0; j < 2; ++j) if (rk[r].ev[j]) cudaEventDestroy(rk[r].ev[j]);
    }
    for (int j = 0; j < 2; ++j) if (y[j]) cudaFreeHost(y[j]);
    if (x) cudaFreeHost(x);
    if (ones) cudaFreeHost(ones);
}

bool fn_model::rexp_held(int il, int e) const { return rexp_ && il < (int) rexp_->held.size() && !rexp_->held[il].empty() && rexp_->held[il][e]; }
bool fn_model::rexp_covered(int il, int e) const {
    return rexp_held(il, e) || (rexp_ && il < (int) rexp_->pending.size() && !rexp_->pending[il].empty() && rexp_->pending[il][e]);
}

// adapt(), before this node's own swaps (the counts are not decayed yet): per served layer the most-routed experts
// that neither node holds replace node 1's least-routed ones (an expert resident here counts as unused there). One
// SWAP message for the whole step; node 1 loads in the background and reports landings in later replies.
void fn_model::rexp_adapt() {
    if (!rexp_ || !rexp_->adapt || !rexp_->x) return;
    rexp_moe & X = *rexp_;
    std::vector<int32_t> msg;
    int total = 0;
    std::vector<float> cnt(NEXP_);
    std::vector<char> res(NEXP_);
    for (int il = 0; il < NL_ && total < X.adapt_cap; ++il) {
        if (X.li_of[il] < 0) continue;
        bool adaptive = true;
        std::fill(cnt.begin(), cnt.end(), 0.0f); std::fill(res.begin(), res.end(), (char) 0);
        for (auto & S : rk_) {
            const layer_w & L = S.lw[il];
            adaptive = adaptive && L.adaptive;
            if (!L.adaptive) break;
            for (int e = 0; e < NEXP_; ++e) {
                if (L.cold_map[e] >= 0) cnt[e] = L.cnt[e];   // the owner rank's count
                if (L.hot_map[e] >= 0) res[e] = 1;
            }
        }
        if (!adaptive) continue;
        std::vector<std::pair<float, int>> cand, vict;
        for (int e = 0; e < NEXP_; ++e) {
            if (X.pending[il][e]) continue;
            if (X.held[il][e]) { if (X.swap_ok[il][e]) vict.push_back({res[e] ? -1.0f : cnt[e], e}); }
            else if (!res[e] && cnt[e] >= o_.adapt_min) cand.push_back({cnt[e], e});
        }
        std::sort(cand.begin(), cand.end(), [](auto & a, auto & b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        std::sort(vict.begin(), vict.end());
        int n = 0;
        for (size_t i = 0; i < cand.size() && i < vict.size() && n < X.adapt_max && total < X.adapt_cap; ++i) {
            if (cand[i].first <= vict[i].first) break;
            X.held[il][vict[i].second] = 0;   // not sent from now on
            X.swap_ok[il][vict[i].second] = 0;
            X.pending[il][cand[i].second] = 1;
            X.swap_ok[il][cand[i].second] = 1;
            msg.push_back(X.li_of[il]); msg.push_back(vict[i].second); msg.push_back(cand[i].second);
            ++n; ++total;
        }
    }
    if (msg.empty()) return;
    rexp_hdr h; h.op = REXP_SWAP; h.a = (int) msg.size() / 3;
    X.k.send_msg(h, msg.data(), msg.size() * 4, X.timeout_ms);
    X.swaps_sent += h.a;
}

// load_experts()'s resident choice for every rank (adaptive file: rank_order's best of each rank's own experts; else
// the hot local ids [h0, h0 + HOTR) of each rank), without touching a GPU
std::vector<char> fn_model::rexp_resident0(int il, int R) const {
    std::vector<char> res(NEXP_, 0);
    const int fil = file_layer(il);
    const int n_hot = hot_count_[fil];
    const gguf_tensor * map = g_.find(bn(fil, "ffn_exp_map"));
    std::vector<int32_t> ident;
    if (!map) { ident.assign((size_t) 2 * NEXP_, -1); for (int e = 0; e < NEXP_; ++e) ident[e] = e; }
    const int32_t * mp = map ? (const int32_t *) map->data : ident.data();
    bool cold_all = n_hot == NEXP_;
    for (int e = 0; e < NEXP_ && cold_all; ++e) cold_all = mp[NEXP_ + e] >= 0;
    for (int r = 0; r < R; ++r) {
        int HOTR = o_.hot_per_rank < 0 ? (n_hot + R - 1) / R : o_.hot_per_rank, h0 = r * HOTR;
        if (!slots_.empty()) {
            h0 = 0;
            for (int q = 0; q < R; ++q) { const int c = slots_[q] < 0 ? HOTR : slots_[q]; if (q < r) h0 += c; }
            HOTR = slots_[r] < 0 ? HOTR : slots_[r];
        }
        if (cold_all && o_.adapt && !rank_order_.empty()) {
            const int slots = !slots_.empty() && slots_[r] >= 0 ? slots_[r] : o_.hot_per_rank > 0 ? o_.hot_per_rank : 72;
            int n = 0;
            for (int i = 0; i < NEXP_ && n < slots; ++i) { const int e = rank_order_[(size_t) fil * NEXP_ + i]; if (e % R == r) { res[e] = 1; ++n; } }
        } else {
            for (int e = 0; e < NEXP_; ++e) { const int h = mp[e]; if (h >= h0 && h < h0 + HOTR) res[e] = 1; }
        }
    }
    return res;
}

void fn_model::rexp_connect(int R) {
    if (o_.rexp_listen.empty()) return;
    if (pipe()) throw std::runtime_error("fn_model: the remote expert tier and the hybrid pipeline are exclusive");
    if (getenv("P100E_FN_CPU") && atoi(getenv("P100E_FN_CPU")) != 0) throw std::runtime_error("fn_model: the remote expert tier and the CPU pool (P100E_FN_CPU) are exclusive");
    rexp_ = std::make_unique<rexp_moe>();
    rexp_moe & X = *rexp_;
    if (const char * e = getenv("P100E_FN_REXP_MAX_T")) o_.rexp_max_t = atoi(e);
    if (const char * e = getenv("P100E_FN_REXP_SUM")) o_.rexp_sum = atoi(e) != 0;
    o_.rexp_max_t = std::max(1, std::min(o_.rexp_max_t, o_.max_t));
    X.sum = o_.rexp_sum;
    X.half = X.sum && getenv("P100E_FN_REXP_HALF") && atoi(getenv("P100E_FN_REXP_HALF")) != 0;
    X.adapt = o_.adapt && !(getenv("P100E_FN_REXP_ADAPT") && atoi(getenv("P100E_FN_REXP_ADAPT")) == 0);
    if (const char * e = getenv("P100E_FN_REXP_ADAPT_MAX")) X.adapt_max = std::max(0, atoi(e));
    if (const char * e = getenv("P100E_FN_REXP_ADAPT_CAP")) X.adapt_cap = std::max(0, atoi(e));
    X.timeout_ms = std::max(1, o_.rexp_timeout_s) * 1000;
    X.early_shared = !(getenv("P100E_FN_REXP_EARLY_SHARED") && atoi(getenv("P100E_FN_REXP_EARLY_SHARED")) == 0);
    if (const char * e = getenv("P100E_FN_REXP_MIN_PAIRS")) X.min_pairs = std::max(0, atoi(e));
    std::string host; int port = 0;
    rexp_parse_addr(o_.rexp_listen, host, port);
    printf("fn_model: remote experts: waiting up to %d s for p100e-fn-expert-server on port %d\n", o_.rexp_connect_s, port);
    fflush(stdout);
    X.k = rexp_sock::accept_one(host, port, o_.rexp_connect_s * 1000);
    rexp_hdr h = X.k.recv_hdr(REXP_HELLO, 60000);
    rexp_hello he;
    if (h.bytes != sizeof he) throw std::runtime_error("rexp: bad HELLO size");
    X.k.recv_all(&he, sizeof he, 60000);
    if (he.proto != REXP_PROTO) throw std::runtime_error("rexp: protocol " + std::to_string(he.proto) + " (want " + std::to_string(REXP_PROTO) + ")");
    const gguf_tensor * t0 = g_.find("blk.0.ffn_gate_exps.weight");
    const uint64_t th = g_.tensor_table_hash();
    if (he.file_bytes != g_.file_size || he.table_hash != th || he.E != E_ || he.FFE != FFE_ || he.NEXP != NEXP_ || !t0)
        throw std::runtime_error("rexp: node 1 has a different model file (size " + std::to_string(he.file_bytes) + " vs " + std::to_string(g_.file_size) +
                                 ", tensor table hash " + (he.table_hash != th ? "differs" : "equal") + ")");
    for (int q = 0; q < he.n_gpu && q < REXP_MAX_GPU; ++q) X.gpu_names.push_back(he.gpu_name[q]);
    // served layers: every trunk layer of this node + the draft block
    X.li_of.assign(NL_, -1);
    std::vector<int32_t> fl;
    for (int il = 0; il < NL_; ++il) { if (il >= L_ && il != mtp_il_) continue; X.li_of[il] = (int) fl.size(); X.il_of.push_back(il); fl.push_back(file_layer(il)); }
    X.pending.assign(NL_, std::vector<char>(NEXP_, 0));
    rexp_serve sv;
    sv.n_layers = (int) fl.size(); sv.max_x = o_.rexp_max_t; sv.max_pairs = o_.rexp_max_t * NEXPU_;
    sv.fold_min = getenv("P100E_FN_FOLD_MIN") ? atoi(getenv("P100E_FN_FOLD_MIN")) : 16;   // ffn_grouped's values
    sv.fold_max = getenv("P100E_FN_FOLD_MAX") ? atoi(getenv("P100E_FN_FOLD_MAX")) : 48;
    sv.sum_mode = X.sum ? 1 : 0;
    sv.reply_f16 = X.half ? 1 : 0;
    std::vector<char> pl(sizeof sv + fl.size() * 4);
    memcpy(pl.data(), &sv, sizeof sv); memcpy(pl.data() + sizeof sv, fl.data(), fl.size() * 4);
    rexp_hdr s; s.op = REXP_SERVE;
    X.k.send_msg(s, pl.data(), pl.size(), 60000);
    h = X.k.recv_hdr(REXP_CAPS, 120000);
    std::vector<int32_t> cf((size_t) h.a * 2);
    if (h.bytes != cf.size() * 4) throw std::runtime_error("rexp: bad CAPS size");
    X.k.recv_all(cf.data(), cf.size() * 4, 60000);
    X.caps.assign(cf.begin(), cf.begin() + h.a);
    std::vector<int> fixed(cf.begin() + h.a, cf.end());
    int cap = 0, cap_fixed = 0;
    for (int q = 0; q < h.a; ++q) { cap += X.caps[q]; if (fixed[q]) cap_fixed += X.caps[q]; }
    // node 1's set per layer: the best experts (expert_split.rank_order, else id order) this node does not make resident
    std::vector<int32_t> set;
    X.held.assign(NL_, {}); X.want.assign(NL_, {}); X.swap_ok.assign(NL_, {});
    for (int il = 0; il < NL_; ++il) {
        if (X.li_of[il] < 0) continue;
        const std::vector<char> res = rexp_resident0(il, R);
        const int fil = file_layer(il);
        std::vector<int> & w = X.want[il];
        for (int i = 0; i < NEXP_ && (int) w.size() < cap; ++i) {
            const int e = rank_order_.empty() ? i : rank_order_[(size_t) fil * NEXP_ + i];
            if (!res[e]) w.push_back(e);
        }
        X.held[il].assign(NEXP_, 0); X.swap_ok[il].assign(NEXP_, 0);
        for (int e : w) X.held[il][e] = 1;
        // the fixed GPUs take the best-ranked part of the list (they are never swapped), the swappable ones the rest
        size_t pos_f = 0, pos_s = std::min<size_t>(w.size(), (size_t) cap_fixed);
        for (int q = 0; q < h.a; ++q) {
            size_t & pos = fixed[q] ? pos_f : pos_s;
            const size_t lim = fixed[q] ? std::min<size_t>(w.size(), (size_t) cap_fixed) : w.size();
            const size_t n = std::min<size_t>((size_t) X.caps[q], lim - std::min(lim, pos));
            set.push_back((int32_t) n);
            for (size_t i = 0; i < n; ++i) { const int e = w[pos + i]; set.push_back(e); if (!fixed[q]) X.swap_ok[il][e] = 1; }
            pos += n;
        }
    }
    rexp_hdr st; st.op = REXP_SET;
    X.k.send_msg(st, set.data(), set.size() * 4, 60000);
    std::string caps_s;
    for (size_t q = 0; q < X.caps.size(); ++q) caps_s += (q ? " + " : "") + std::to_string(X.caps[q]) + " (" + (q < X.gpu_names.size() ? X.gpu_names[q] : "?") + (fixed[q] ? ", fixed" : ", swaps") + ")";
    printf("fn_model: remote experts: node %s holds %d experts per layer = %s over %d layers; forwards of <= %d rows; %s mode; node 1 loads while this node loads\n",
           X.k.peer().c_str(), cap, caps_s.c_str(), sv.n_layers, o_.rexp_max_t, X.sum ? "sum" : "exact");
    fflush(stdout);
}

void fn_model::rexp_ready() {
    if (!rexp_) return;
    rexp_moe & X = *rexp_;
    const auto t0 = std::chrono::steady_clock::now();
    const rexp_hdr h = X.k.recv_hdr(REXP_READY, o_.rexp_connect_s * 1000);
    if (h.a != 0) throw std::runtime_error("rexp: node 1 failed to load its experts");
    const double w = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const int R = (int) rk_.size();
    X.devs.clear();
    for (auto & S : rk_) X.devs.push_back(S.dev);
    X.rk.resize(R);
    for (int r = 0; r < R; ++r) {
        CK(cudaSetDevice(rk_[r].dev));
        for (int j = 0; j < 2; ++j) CK(cudaEventCreateWithFlags(&X.rk[r].ev[j], cudaEventDisableTiming));
        X.rk[r].is_rem.assign(NEXP_, 0);
    }
    CK(cudaSetDevice(rk_[0].dev));
    const size_t NP = (size_t) o_.rexp_max_t * NEXPU_;
    CK(cudaMallocHost(&X.x, (size_t) o_.rexp_max_t * E_ * 4));
    CK(cudaMallocHost(&X.ones, NP * 4));
    for (size_t i = 0; i < NP; ++i) X.ones[i] = 1.0f;
    for (int j = 0; j < 2; ++j) { X.ycap[j] = NP * E_ * 4; CK(cudaMallocHost(&X.y[j], X.ycap[j])); }
    // coverage at load: resident here vs node 1 (per layer, of NEXP)
    int64_t res = 0, both = 0, rem = 0;
    for (int il = 0; il < NL_; ++il) {
        if (X.li_of[il] < 0) continue;
        for (int e = 0; e < NEXP_; ++e) {
            bool here = false;
            for (auto & S : rk_) here = here || S.lw[il].hot_map[e] >= 0;
            res += here; rem += X.held[il][e]; both += here && X.held[il][e];
        }
    }
    const int nl = (int) std::count_if(X.li_of.begin(), X.li_of.end(), [](int v) { return v >= 0; });
    printf("fn_model: remote experts ready (waited %.1f s for node 1 after this load): per layer %.1f resident here + %.1f on node 1 (%.1f on both) of %d\n",
           w, (double) res / nl, (double) rem / nl, (double) both / nl, NEXP_);
    if (both) printf("fn_model: remote experts: WARNING: this node's resident set differs from the prediction (%lld experts on both nodes)\n", (long long) both);
    fflush(stdout);
}

// rank 0's call of layer_ffn: every rank's non-resident experts that node 1 holds, rows sent in one message
void fn_model::rexp_begin(int il, int T, const std::vector<int> & ids, const std::vector<float> & wts) {
    rexp_moe & X = *rexp_;
    X.layer_on = false;
    if (T > o_.rexp_max_t || X.li_of[il] < 0 || !X.x) return;
    static const bool grouped_on = !getenv("P100E_FN_GROUPED") || atoi(getenv("P100E_FN_GROUPED")) != 0;
    if (!grouped_on) return;
    const int R = (int) rk_.size();
    for (int r = 0; r < R; ++r) {
        const layer_w & L = rk_[r].lw[il];
        auto ok = [&](const expert_set & g, const expert_set & u, const expert_set & d) {
            return g.n == 0 || (grouped_layout(g.layout) && g.layout == u.layout && grouped_layout(d.layout));
        };
        if (!ok(L.hot_gate, L.hot_up, L.hot_down) || !ok(L.hhot_gate, L.hhot_up, L.hhot_down) || !ok(L.cold_gate, L.cold_up, L.cold_down)) return;
    }
    const std::vector<char> & held = X.held[il];
    std::vector<int32_t> jobs, xrow; std::vector<float> w;
    X.tok_x.assign(T, -1); X.pair_tok.clear();
    X.nx = 0;
    int64_t routed = 0;
    auto & hs = X.hits_T[T];
    hs.fwd += 1;
    for (int r = 0; r < R; ++r) {
        rank_state & S = rk_[r];
        const layer_w & L = S.lw[il];
        auto & q = X.rk[r];
        std::fill(q.is_rem.begin(), q.is_rem.end(), (char) 0);
        q.nrows = 0; q.nr = q.nrg = 0; q.off = (int) X.pair_tok.size();
        std::map<int, std::vector<std::pair<int, float>>> by_e;
        ffn_by_e(S, il, T, ids, wts, by_e);
        for (auto & [e, lst] : by_e) {
            if (L.hot_map[e] >= 0) { hs.local += (int64_t) lst.size(); continue; }
            if (L.hhot_map(e) < 0 && L.cold_map[e] < 0) continue;   // another rank's expert
            routed += (int64_t) lst.size();
            if (!held[e]) { hs.miss += (int64_t) lst.size(); continue; }
            hs.remote += (int64_t) lst.size();
            q.is_rem[e] = 1;
            jobs.push_back(e); jobs.push_back((int32_t) lst.size());
            for (auto & [t, wt] : lst) {
                if (X.tok_x[t] < 0) X.tok_x[t] = X.nx++;
                xrow.push_back(X.tok_x[t]); w.push_back(wt); X.pair_tok.push_back(t);
            }
            q.nrows += (int) lst.size();
        }
    }
    X.n_rows_routed += routed;
    X.npairs = (int) X.pair_tok.size();
    if (X.npairs == 0) return;
    if (X.npairs < X.min_pairs) {   // P100E_FN_REXP_MIN_PAIRS: too few pairs to pay a round trip: the host path instead
        for (auto & q : X.rk) { std::fill(q.is_rem.begin(), q.is_rem.end(), (char) 0); q.nrows = 0; }
        hs.remote -= X.npairs; hs.miss += X.npairs;
        X.n_skipped += 1;
        return;
    }
    std::vector<int> xs(X.nx);   // x row i = token xs[i] (first-use order)
    for (int t = 0; t < T; ++t) if (X.tok_x[t] >= 0) xs[X.tok_x[t]] = t;
    const size_t bx = (size_t) X.nx * E_ * 4, bj = jobs.size() * 4, bp = xrow.size() * 4;
    X.sendbuf.resize(bx + bj + 2 * bp);
    for (int i = 0; i < X.nx; ++i) memcpy(X.sendbuf.data() + (size_t) i * E_ * 4, X.x + (size_t) xs[i] * E_, (size_t) E_ * 4);
    memcpy(X.sendbuf.data() + bx, jobs.data(), bj);
    memcpy(X.sendbuf.data() + bx + bj, xrow.data(), bp);
    memcpy(X.sendbuf.data() + bx + bj + bp, w.data(), bp);
    rexp_hdr h; h.op = REXP_FFN; h.seq = ++X.seq; h.a = X.li_of[il]; h.b = X.nx; h.c = (int) jobs.size() / 2; h.d = X.npairs;
    const auto ts0 = std::chrono::steady_clock::now();
    X.k.send_msg(h, X.sendbuf.data(), X.sendbuf.size(), X.timeout_ms);
    X.t_send += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts0).count();
    X.layer_on = true;
    X.n_req += 1; X.n_pairs += X.npairs; X.bytes_out += (int64_t) (X.sendbuf.size() + sizeof h);
    auto & bt = X.by_T[T]; bt.first += 1; bt.second += X.npairs;
}

// the last rank's call of layer_ffn: every rank's GPU experts are queued; receive node 1's rows, upload them behind
// each rank's GPU rows (S.gout), then ffn_grouped's combine + the shared expert
void fn_model::rexp_finish(int il, int T) {
    rexp_moe & X = *rexp_;
    const int R = (int) rk_.size();
    const int k = X.flip;
    for (int r = 0; r < R; ++r) if (X.rk[r].ev_rec[k]) { CK(cudaSetDevice(rk_[r].dev)); CK(cudaEventSynchronize(X.rk[r].ev[k])); X.rk[r].ev_rec[k] = false; }
    const auto t0 = std::chrono::steady_clock::now();
    const rexp_hdr h = X.k.recv_hdr(REXP_FFN_R, X.timeout_ms);
    const auto th = std::chrono::steady_clock::now();
    const size_t nrow = X.sum ? (size_t) X.nx : (size_t) X.npairs, eb = X.half ? 2 : 4;
    if (h.seq != X.seq || (size_t) h.d != nrow || h.c < 0 || h.bytes != nrow * E_ * eb + (size_t) h.c * 8) throw std::runtime_error("rexp: reply does not match the request (seq / rows)");
    if (X.half) {
        X.hbuf.resize(nrow * E_);
        X.k.recv_all(X.hbuf.data(), nrow * E_ * 2, X.timeout_ms);
        for (size_t i = 0; i < nrow * (size_t) E_; ++i) X.y[k][i] = __half2float(X.hbuf[i]);
    } else X.k.recv_all(X.y[k], nrow * E_ * 4, X.timeout_ms);
    if (h.c > 0) {   // swaps node 1 has landed: those experts may be sent from the next layer on
        std::vector<int32_t> ld((size_t) h.c * 2);
        X.k.recv_all(ld.data(), ld.size() * 4, X.timeout_ms);
        for (int i = 0; i < h.c; ++i) {
            const int li = ld[2 * i], e = ld[2 * i + 1];
            if (li < 0 || li >= (int) X.il_of.size() || e < 0 || e >= NEXP_) throw std::runtime_error("rexp: bad landed swap");
            const int l = X.il_of[li];
            X.held[l][e] = 1; X.pending[l][e] = 0;
            ++X.swaps_landed;
        }
    }
    const auto tb = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(tb - t0).count();
    X.t_wait += ms; X.t_wait_max = std::max(X.t_wait_max, ms);
    X.t_hdr += std::chrono::duration<double, std::milli>(th - t0).count();
    X.t_body += std::chrono::duration<double, std::milli>(tb - th).count();
    X.bytes_in += (int64_t) (h.bytes + sizeof h);
    const float * src = X.y[k];
    // sum mode: the first remote pair of each token (rank order) carries the token's sum, the others 0, weights 1. The
    // rows are zeroed on the device and only the sum rows go up, straight from y[k] (the receive into y[k] two layers on
    // waits for ev[k]): nx row copies instead of npairs, and no host wait for the uploads
    std::vector<char> first;
    if (X.sum) {
        first.assign(X.npairs, 0);
        std::vector<char> done(T, 0);
        for (int i = 0; i < X.npairs; ++i) { const int t = X.pair_tok[i]; if (!done[t]) { first[i] = 1; done[t] = 1; } }
    }
    for (int r = 0; r < R; ++r) {
        rank_state & S = rk_[r];
        auto & q = X.rk[r];
        CK(cudaSetDevice(S.dev));
        if (q.nr > q.nrg) {
            if (q.nr - q.nrg != q.nrows) throw std::runtime_error("fn_model: remote row count mismatch");
            float * g = S.gout + (size_t) q.nrg * E_;
            if (X.sum) {
                CK(cudaMemsetAsync(g, 0, (size_t) q.nrows * E_ * 4, S.s));
                for (int i = 0; i < q.nrows; ++i) if (first[(size_t) q.off + i]) {
                    const int t = X.pair_tok[(size_t) q.off + i];
                    CK(cudaMemcpyAsync(g + (size_t) i * E_, src + (size_t) X.tok_x[t] * E_, (size_t) E_ * 4, cudaMemcpyHostToDevice, S.s));
                }
                CK(cudaMemcpyAsync(S.gw + q.nrg, X.ones, (size_t) q.nrows * 4, cudaMemcpyHostToDevice, S.s));
            } else CK(cudaMemcpyAsync(g, src + (size_t) q.off * E_, (size_t) q.nrows * E_ * 4, cudaMemcpyHostToDevice, S.s));
            CK(cudaEventRecord(q.ev[k], S.s)); q.ev_rec[k] = true;
        }
        if (q.nr > 0) fnk::combine_rows_k<<<dim3(nb256(E_), T), 256, 0, S.s>>>(q.out, S.gout, S.rowof, S.gw, E_, NEXPU_);
        if (r == 0 && X.early_shared) ffn_shared_post(S, il, q.out, T);
        else ffn_shared(S, r, il, q.cur, q.out, T);
    }
    X.flip ^= 1;
    X.layer_on = false;
}

} // namespace p100e
