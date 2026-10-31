// Ported from Kmic-68/llama.cpp 77e05601d (MIT): "cuda: chunked gated delta net for prefill".
//
// Alternative gated delta net paths for prefill (scalar gate, S_v = 128), kept out of
// gated_delta_net.cu so a change here rebuilds one small translation unit.
//   GGML_CUDA_GDN_CHUNKED=1  chunked form (gdn-chunked.cuh), all fp32
//   GGML_CUDA_GDN_CHUNKED=2  chunked form, K K^T / Q K^T and the state update in fp64
//   GGML_CUDA_GDN_CHUNKED=0  (or unset) the shipped recurrence
//   GGML_CUDA_GDN_REF=1      sequential recurrence with an fp64 state: an accuracy reference only
//
// pf-gdn adaptations (this fork):
//   - default is 0 here (Kmic: 1): with no env set the shipped recurrence runs, untouched
//   - the args carry our fusions (beta sigmoid fold, state GET_ROWS fold); see gdn-chunked.cuh
//   - float4 alignment is checked before taking the chunked path (falls back to the recurrence)
//   - GGML_PF_GDN_LOG=1 logs the path chosen for the first 64 calls (ours; for model-level checks)
#include "common.cuh"
#include "gated_delta_net.cuh"

#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

namespace gdn_f32 {
#include "gdn-chunked.cuh"
}
#undef TG
#undef TA
#undef TB
#undef TC
#undef TF
#define TG double
#define TC double
namespace gdn_f64 {
#include "gdn-chunked.cuh"
}

// one thread per (column, head, seq); state column in fp64 local memory
static __global__ void gdn_ref_f64(const ggml_cuda_gdn_alt_args p) {
    const int     j   = threadIdx.x;
    const int64_t h   = blockIdx.x;
    const int64_t seq = blockIdx.y;
    const int64_t iq1 = h % p.neqk1;
    const int64_t iq3 = seq / p.rq3;
    const bool keep = p.K > 1;

    const float * cs = p.s_idx ? p.curr_state + (int64_t) p.s_idx[seq] * p.s_row + h * 16384
                               : p.curr_state + (seq * p.H + h) * 16384;
    double s[128];
    for (int i = 0; i < 128; i++) {
        s[i] = cs[j * 128 + i];
    }
    // pf-gdn: folded GET_ROWS stays node-preserving (owning thread writes what it read)
    if (p.gather_dst != nullptr) {
        float * gd = p.gather_dst + (seq * p.H + h) * 16384 + j * 128;
        for (int i = 0; i < 128; i++) {
            gd[i] = (float) s[i];
        }
    }
    float * st = p.state + (seq * p.H + h) * 16384 + j * 128;
    for (int64_t t = 0; t < p.n_tokens; t++) {
        const float * kt = p.k + iq3 * p.sq3 + iq1 * p.sq1 + t * p.sq2;
        const float * qt = p.q + iq3 * p.sq3 + iq1 * p.sq1 + t * p.sq2;
        const int64_t gbo = seq * p.sb3 + t * p.sb2 + h * p.sb1;
        const double eg = exp((double) p.g[gbo]);
        const double bt = p.beta[gbo];
        const double vj = p.v[seq * p.sv3 + t * p.sv2 + h * p.sv1 + j];
        double kv = 0.0;
        for (int i = 0; i < 128; i++) {
            kv = fma(s[i], (double) kt[i], kv);
        }
        const double delta = (vj - eg * kv) * bt;
        double o = 0.0;
        for (int i = 0; i < 128; i++) {
            s[i] = fma(eg, s[i], (double) kt[i] * delta);
            o = fma(s[i], (double) qt[i], o);
        }
        p.dst[((seq * p.n_tokens + t) * p.H + h) * 128 + j] = (float) (o * p.scale);
        const int64_t slot = p.n_tokens - 1 - t;
        if ((keep && slot < p.K) || (!keep && slot == 0)) {
            for (int i = 0; i < 128; i++) {
                st[slot * p.state_slot_stride + i] = (float) s[i];
            }
        }
    }
}

// pf-gdn: the reference kernel reads beta from p.beta; with the sigmoid folded in, fill the
// SIGMOID dst first (same expression as op_sigmoid) in its own launch.
static __global__ void gdn_ref_beta_sigmoid(const float * pre, float * out, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = 1.0f / (1.0f + expf(-pre[i]));
    }
}

static bool gdn_aligned16(const void * ptr) {
    return ptr == nullptr || ((uintptr_t) ptr & 15) == 0;
}

// returns false when the shipped kernel should run
bool ggml_cuda_gdn_alt(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_alt_args & a) {
    static const int  mode = getenv("GGML_CUDA_GDN_CHUNKED") ? atoi(getenv("GGML_CUDA_GDN_CHUNKED")) : 0;
    static const bool ref  = getenv("GGML_CUDA_GDN_REF") && atoi(getenv("GGML_CUDA_GDN_REF")) != 0;
    static const bool do_log = getenv("GGML_PF_GDN_LOG") && atoi(getenv("GGML_PF_GDN_LOG")) != 0;
    static int n_logged = 0;

    if (mode == 0 && !ref) {
        return false;   // default: nothing here runs, the caller launches the shipped recurrence
    }

    cudaStream_t stream = ctx.stream();
    const bool keep = a.K > 1;
    const int64_t L = keep ? a.n_tokens - (a.K - 1) : a.n_tokens;

    const char * path = nullptr;
    if (ref) {
        if (a.beta_pre != nullptr) {
            // beta (p.beta) is the SIGMOID dst; contiguous (asserted by the caller), sb3 * n_seqs elements
            const int64_t n = a.sb3 * a.n_seqs;
            gdn_ref_beta_sigmoid<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(a.beta_pre, a.beta_sig, n);
            CUDA_CHECK(cudaGetLastError());
        }
        gdn_ref_f64<<<dim3(a.H, a.n_seqs), 128, 0, stream>>>(a);
        CUDA_CHECK(cudaGetLastError());
        path = "ref-f64";
    } else {
        // Kmic's gate: chunked only with >= one full chunk left after the K-1 snapshot tail.
        // pf-gdn: plus the layout assumptions of the float4 loads (true for the model's views).
        const bool layout_ok =
            a.H % a.neqk1 == 0 &&
            a.sq1 % 4 == 0 && a.sq2 % 4 == 0 && a.sq3 % 4 == 0 &&
            a.sv1 % 4 == 0 && a.sv2 % 4 == 0 && a.sv3 % 4 == 0 &&
            a.s_row % 4 == 0 && a.state_slot_stride % 4 == 0 &&
            gdn_aligned16(a.q) && gdn_aligned16(a.k) && gdn_aligned16(a.v) &&
            gdn_aligned16(a.curr_state) && gdn_aligned16(a.dst) && gdn_aligned16(a.state) &&
            gdn_aligned16(a.gather_dst);
        if (L < GCH || !layout_ok) {
            if (do_log && n_logged < 64) {
                n_logged++;
                GGML_LOG_INFO("%s: GDN recurrence (n_tokens %lld, K %d, H %lld, n_seqs %lld, layout_ok %d)\n", __func__,
                              (long long) a.n_tokens, a.K, (long long) a.H, (long long) a.n_seqs, (int) layout_ok);
            }
            return false;
        }
        const size_t n = gdn_f32::gdn_ch_scratch_floats(a.H, a.neqk1, a.n_seqs, L);
        ggml_cuda_pool_alloc<float> scratch(ctx.pool(), n);
#define GDN_CH_RUN(NS) do { \
        NS::gdn_ch_params p = {}; \
        p.q = a.q; p.k = a.k; p.v = a.v; p.g = a.g; p.beta = a.beta; \
        p.curr_state = a.curr_state; p.dst = a.dst; p.state = a.state; \
        p.H = a.H; p.n_tokens = a.n_tokens; p.n_seqs = a.n_seqs; p.neqk1 = a.neqk1; p.rq3 = a.rq3; p.K = a.K; \
        p.sq1 = a.sq1; p.sq2 = a.sq2; p.sq3 = a.sq3; p.sv1 = a.sv1; p.sv2 = a.sv2; p.sv3 = a.sv3; \
        p.sb1 = a.sb1; p.sb2 = a.sb2; p.sb3 = a.sb3; \
        p.scale = a.scale; p.state_slot_stride = a.state_slot_stride; p.s_idx = a.s_idx; p.s_row = a.s_row; \
        p.keep = keep; \
        p.beta_pre = a.beta_pre; p.beta_sig = a.beta_sig; p.gather_dst = a.gather_dst; \
        NS::gdn_ch_launch(p, scratch.get(), stream); \
    } while (0)
        if (mode == 2) {
            GDN_CH_RUN(gdn_f64);
            path = "chunked-f64";
        } else {
            GDN_CH_RUN(gdn_f32);
            path = "chunked-f32";
        }
#undef GDN_CH_RUN
        CUDA_CHECK(cudaGetLastError());
    }
    if (do_log && n_logged < 64) {
        n_logged++;
        // cache = the snapshot-cpy fusion is active (state written into the cache, not the dst tail)
        const bool cache_fused = a.state != a.dst + 128 * a.H * a.n_tokens * a.n_seqs;
        GGML_LOG_INFO("%s: GDN %s (n_tokens %lld, K %d, L %lld, H %lld, neqk1 %lld, n_seqs %lld, beta_fold %d, gather %d, cache %d)\n",
                      __func__, path, (long long) a.n_tokens, a.K, (long long) L, (long long) a.H, (long long) a.neqk1,
                      (long long) a.n_seqs, (int) (a.beta_pre != nullptr), (int) (a.s_idx != nullptr), (int) cache_fused);
    }
    return true;
}
