#include "tp_model.h"
#include "kernels.cuh"
#include "fast/ssm.cuh"
#include "fast/fused.cuh"
#include "fast/gemm.cuh"
#include "fast/attn_fold.cuh"
#include "fast/attn_dec2.cuh"
#include "fast/unpack.cuh"
#include "fast/gdn_chunked.cuh"
#include "pdl/pdl_gpu.cuh"
#include "../third_party/ggml/convert.cuh"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace p100e {

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error( \
    std::string("CUDA ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)
#define CB(x) do { cublasStatus_t e_ = (x); if (e_ != CUBLAS_STATUS_SUCCESS) throw std::runtime_error( \
    std::string("cuBLAS error ") + std::to_string((int) e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

void tp_model_pf_free(pf_rank * P);   // prefill pipeline resources, defined below

static std::string bn(int il, const char * s) { return "blk." + std::to_string(il) + "." + s; }
using ranges = std::vector<std::pair<int64_t, int64_t>>;

// token ids index token_embd rows on the host: one compare per row before any gather (a peer node's ids, a caller's)
static void check_tokens(const std::vector<int32_t> & tokens, int vocab, const char * what) {
    for (const int32_t t : tokens)
        if (t < 0 || t >= vocab)
            throw std::runtime_error(std::string(what) + ": token id " + std::to_string(t) + " outside the vocabulary (" +
                                     std::to_string(vocab) + ")");
}

// debug (P100E_WHASH): order-sensitive fingerprint of a device buffer. Each 32-bit word is mixed with its index and
// the mixes are summed with integer atomics, so the result is exact and independent of the launch schedule.
static __global__ void dev_hash_k(const uint32_t * p, size_t n, unsigned long long * out) {
    unsigned long long acc = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        unsigned long long z = ((unsigned long long) p[i] << 32 | (i & 0xffffffffu)) + (i >> 32) * 0x9e3779b97f4a7c15ull;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        acc += z ^ (z >> 31);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, acc);
}

static unsigned long long dev_hash(const void * d, size_t bytes) {
    unsigned long long * acc = nullptr, h = 0;
    CK(cudaMalloc(&acc, 8));
    CK(cudaMemset(acc, 0, 8));
    if (bytes >= 4) dev_hash_k<<<1024, 256>>>((const uint32_t *) d, bytes / 4, acc);
    CK(cudaGetLastError());
    CK(cudaMemcpy(&h, acc, 8, cudaMemcpyDeviceToHost));
    CK(cudaFree(acc));
    if (bytes % 4) {   // tail bytes
        uint32_t t = 0;
        CK(cudaMemcpy(&t, (const char *) d + bytes / 4 * 4, bytes % 4, cudaMemcpyDeviceToHost));
        h = (h ^ t) * 1099511628211ull;
    }
    return h;
}

// P100E_PF_TINY_ATTN=0: a T = 5..8 classic chunk runs attention as a 4-query + (T-4)-query call pair (reads the cache twice)
// (not graph-captured: the T = 5..8 chunks run eagerly, so reload_env just re-reads it)
static int g_pf_tiny_attn = -1;
static bool pf_tiny_attn() {
    if (g_pf_tiny_attn < 0) g_pf_tiny_attn = !getenv("P100E_PF_TINY_ATTN") || atoi(getenv("P100E_PF_TINY_ATTN")) != 0;
    return g_pf_tiny_attn != 0;
}

static bool whash_on() { static const bool on = getenv("P100E_WHASH") && atoi(getenv("P100E_WHASH")); return on; }

static void dequant_f32(uint32_t type, const void * src, float * dst, int64_t n, cudaStream_t s) {
    if (type == GGML_TYPE_F32) { CK(cudaMemcpyAsync(dst, src, n * 4, cudaMemcpyDeviceToDevice, s)); return; }
    to_fp32_cuda_t f = ggml_get_to_fp32_cuda((ggml_type) type);
    if (!f) throw std::runtime_error(std::string("no dequant for ") + get_type_info(type).name);
    f(src, dst, n, s);
}

// greedy pick of logit rows on the GPU: one block per row, (value, index) with the larger value, the lower index on
// ties - the same token as the host loop `if (l[i] > l[b]) b = i` from b = 0 (NaN never wins; all -inf -> index 0)
struct amax_pair { float v; int32_t i; };
static __global__ void argmax_rows_k(const float * __restrict__ l, int n, amax_pair * __restrict__ out) {
    const float * row = l + (size_t) blockIdx.x * n;
    float bv = -INFINITY;
    int bi = INT_MAX;
    // increasing i per thread: strict > keeps the first max. U loads in flight per thread (one block per row is
    // latency-bound otherwise: 60 us for a 62080-wide row on the P100), compared in the same index order.
    constexpr int U = 8;
    const int B = (int) blockDim.x;
    int i = threadIdx.x;
    for (; i + (U - 1) * B < n; i += U * B) {
        float v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) v[u] = row[i + u * B];
#pragma unroll
        for (int u = 0; u < U; ++u) if (v[u] > bv) { bv = v[u]; bi = i + u * B; }
    }
    for (; i < n; i += B) {
        const float v = row[i];
        if (v > bv) { bv = v; bi = i; }
    }
    auto better = [](float ov, int oi, float bv, int bi) { return ov > bv || (ov == bv && oi < bi); };
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffff, bv, o);
        const int oi = __shfl_xor_sync(0xffffffff, bi, o);
        if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if (lane == 0) { sv[wid] = bv; si[wid] = bi; }
    __syncthreads();
    if (wid == 0) {
        bv = lane < nw ? sv[lane] : -INFINITY;
        bi = lane < nw ? si[lane] : INT_MAX;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, bv, o);
            const int oi = __shfl_xor_sync(0xffffffff, bi, o);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) out[blockIdx.x] = amax_pair{bv, bi == INT_MAX ? 0 : bi};
    }
}

tp_model::tp_model(const std::string & path, const std::vector<int> & devices, int n_ctx, int max_batch, bool load_mtp,
                   uint32_t kv_type)
    : g_(path) {
    opts o;
    o.n_ctx = n_ctx; o.max_batch = max_batch; o.load_mtp = load_mtp; o.kv_type = kv_type;
    init(path, devices, o);
}

tp_model::tp_model(const std::string & path, const std::vector<int> & devices, const opts & o) : g_(path) {
    init(path, devices, o);
}

void tp_model::init(const std::string & path, const std::vector<int> & devices, const opts & o) {
    (void) path;
    // the batched decode step (forward_batch_ids, tp_mtp::run_multi) stages up to MAX_ROWS rows in buffers of max_batch rows
    n_ctx_ = o.n_ctx; max_batch_ = std::max(o.max_batch, (int) MAX_ROWS); kv_type_ = o.kv_type;
    const bool load_mtp = o.load_mtp;
    max_slots_ = std::max(1, o.max_seqs);
    if (const char * e = getenv("P100E_PAGED")) paged_ = atoi(e) != 0;
    psh_ = o.page_shift;
    if (const char * e = getenv("P100E_PAGE_SHIFT")) psh_ = atoi(e);
    if (psh_ < 5 || psh_ > 12) throw std::runtime_error("page shift must be 5..12 (pages of 32..4096 positions)");
    if (!paged_ && max_slots_ > 1) throw std::runtime_error("P100E_PAGED=0 (contiguous KV caches) allows one sequence only");
    max_pages_ = (n_ctx_ + (1 << psh_) - 1) >> psh_;
    if (kv_type_ != fast::KV_F16 && kv_type_ != fast::KV_Q8_0 && kv_type_ != fast::KV_Q4_0)
        throw std::runtime_error("unsupported KV cache type " + std::to_string(kv_type_));
    const std::string a = g_.get_str("general.architecture", "");
    if (a != "qwen35") throw std::runtime_error("unsupported architecture " + a);
    E_ = (int) g_.get_int(a + ".embedding_length", 0);
    L_ = (int) (g_.get_int(a + ".block_count", 0) - g_.get_int(a + ".nextn_predict_layers", 0));
    n_nextn_ = load_mtp ? (int) g_.get_int(a + ".nextn_predict_layers", 0) : 0;
    if (n_nextn_ > 1) n_nextn_ = 1;   // only the first nextn block is used
    FF_ = (int) g_.get_int(a + ".feed_forward_length", 0);
    H_ = (int) g_.get_int(a + ".attention.head_count", 0);
    HKV_ = (int) g_.get_int(a + ".attention.head_count_kv", 0);
    HD_ = (int) g_.get_int(a + ".attention.key_length", 0);
    NROT_ = (int) g_.get_int(a + ".rope.dimension_count", 0);
    CONV_K_ = (int) g_.get_int(a + ".ssm.conv_kernel", 4);
    HK_ = (int) g_.get_int(a + ".ssm.group_count", 0);
    HV_ = (int) g_.get_int(a + ".ssm.time_step_rank", 0);
    DK_ = (int) g_.get_int(a + ".ssm.state_size", 0);
    DV_ = HV_ > 0 ? (int) (g_.get_int(a + ".ssm.inner_size", 0) / HV_) : 0;   // HV_ 0: refused below
    eps_ = (float) g_.get_float(a + ".attention.layer_norm_rms_epsilon", 1e-6);
    rope_base_ = (float) g_.get_float(a + ".rope.freq_base", 1e7);
    // M-RoPE sections (images): [11, 11, 10, 0] for Qwen3.8 (llama.cpp reads rope.dimension_sections)
    if (const gguf_value * v = g_.get(a + ".rope.dimension_sections"); v && v->arr_num.size() >= 3) {
        msect_ = 0;
        for (size_t i = 0; i < v->arr_num.size(); ++i) {
            if (i < 3) msec_[i] = (int) v->arr_num[i];
            msect_ += (int) v->arr_num[i];
        }
        if (msect_ <= 0) { msec_[0] = 11; msec_[1] = 11; msec_[2] = 10; msect_ = 32; }
    }
    full_attn_interval_ = (int) g_.get_int(a + ".full_attention_interval", 4);
    const gguf_tensor * temb = g_.find("token_embd.weight");
    if (!temb) throw std::runtime_error("missing tensor token_embd.weight");
    VOCAB_ = (int) temb->ne[1];
    {
        // the metadata sizes drive every slice and buffer below: refuse zero / inconsistent ones (divides, empty
        // splits) and weights whose shapes do not match them (slice / load_small also bound-check every range)
        if (E_ <= 0 || L_ <= 0 || H_ <= 0 || HKV_ <= 0 || HD_ <= 0 || VOCAB_ <= 0 || full_attn_interval_ <= 0)
            throw std::runtime_error("model metadata: embedding / block / head / vocab sizes and full_attention_interval must be > 0");
        bool any_gdn = false;
        for (int il = 0; il < L_; ++il) any_gdn = any_gdn || (il + 1) % full_attn_interval_ != 0;
        if (any_gdn && (HK_ <= 0 || HV_ <= 0 || HV_ % HK_ != 0 || DK_ <= 0 || DV_ <= 0))
            throw std::runtime_error("model metadata: ssm group_count / time_step_rank / state_size / inner_size must be > 0, "
                                     "time_step_rank a multiple of group_count");
        auto shape = [&](const std::string & name, int64_t ne0, int64_t ne1) {   // a missing tensor fails at its load
            const gguf_tensor * t = g_.find(name);
            if (t && (t->ne[0] != ne0 || t->ne[1] != ne1))
                throw std::runtime_error(name + " is [" + std::to_string(t->ne[0]) + ", " + std::to_string(t->ne[1]) +
                                         "], the metadata says [" + std::to_string(ne0) + ", " + std::to_string(ne1) + "]");
        };
        shape("token_embd.weight", E_, VOCAB_);
        shape("output.weight", E_, VOCAB_);
        for (int il = 0; il < L_; ++il) {
            shape(bn(il, "ffn_gate.weight"), E_, FF_);
            shape(bn(il, "ffn_up.weight"), E_, FF_);
            shape(bn(il, "ffn_down.weight"), FF_, E_);
            if ((il + 1) % full_attn_interval_ == 0) {
                shape(bn(il, "attn_q.weight"), E_, 2LL * H_ * HD_);   // q and gate
                shape(bn(il, "attn_k.weight"), E_, (int64_t) HKV_ * HD_);
                shape(bn(il, "attn_v.weight"), E_, (int64_t) HKV_ * HD_);
                shape(bn(il, "attn_output.weight"), (int64_t) H_ * HD_, E_);
            } else {
                shape(bn(il, "attn_qkv.weight"), E_, 2LL * HK_ * DK_ + (int64_t) HV_ * DV_);
                shape(bn(il, "attn_gate.weight"), E_, (int64_t) HV_ * DV_);
                shape(bn(il, "ssm_beta.weight"), E_, HV_);
                shape(bn(il, "ssm_alpha.weight"), E_, HV_);
                shape(bn(il, "ssm_out.weight"), (int64_t) HV_ * DV_, E_);
            }
        }
    }

    const int R = (int) devices.size();   // this process's ranks (net mode: R_ counts every node's)
    if (R < 1 || R > fast::P2P_MAX_RANKS)
        throw std::runtime_error("tensor parallel over " + std::to_string(R) + " GPUs: 1.." +
                                 std::to_string(fast::P2P_MAX_RANKS) + " supported");
    R_ = R; r0_ = 0;
    node_r0_.assign(1, 0); node_nr_.assign(1, R);
    lb_ = 0; le_ = L_;   // every decoder layer (pipe mode: pipe_init narrows it to this node's)
    if (o.pipe_layers != 0 && o.net_nodes.empty())
        throw std::runtime_error("the hybrid pipeline (--pipe-layers / --pipe-split) needs --net-nodes (two nodes)");
    if (!o.net_nodes.empty() && o.pipe_layers != 0) pipe_init(devices, o);   // the split stays local
    else if (!o.net_nodes.empty()) net_init(devices, o);   // connect, agree, R_ / r0_ from every node's rank count
    {
        // ---- the tensor-parallel split (src/tp_split.h): every rank's ranges of the split dimensions.
        // Equal weights + a dimension that divides by R: dim / R each (the old layout, R = 1 / 2 / 4 on the 27B);
        // otherwise whole units apportioned by weight. The units keep every column split block aligned in the file's
        // quant types: ssm_out columns per k head are (HV/HK) runs of DV, so a q4_K / q6_K ssm_out (256-column
        // blocks, DV 128) moves k heads in pairs (and the beta / alpha rows make it fours, below); ffn_down columns
        // in 256 blocks (one K-quant superblock); vocab and eh_proj in 256 blocks (the fast GEMV / GEMM need
        // n_out % 4, the a16 activations n_in % 256).
        tp_split_in si;
        si.R = R_;   // every node's ranks (one process: R)
        si.w = o.split;
        if (si.w.empty())
            if (const char * e = getenv("P100E_SPLIT"); e && *e) si.w = tp_parse_split(e);
        si.H = H_; si.HKV = HKV_; si.HK = HK_; si.HV = HV_; si.FF = FF_; si.VOCAB = VOCAB_; si.E = E_;
        auto blck = [&](const std::string & name) {
            const gguf_tensor * t = g_.find(name);
            return t ? (int) get_type_info(t->type).blck : 1;
        };
        int ssm_b = 1, down_b = 1, n_att = 0, n_gdn = 0;
        for (int il = 0; il < L_; ++il) {
            down_b = std::max(down_b, blck(bn(il, "ffn_down.weight")));
            if ((il + 1) % full_attn_interval_ == 0) ++n_att;
            else { ++n_gdn; ssm_b = std::max(ssm_b, blck(bn(il, "ssm_out.weight"))); }
        }
        // and ssm_beta / ssm_alpha have one row per local v head: the prefill GEMM (fold / gemm_packed) needs
        // n_out % 4, so HVl = HKl * HV/HK must be a multiple of 4 (27B: HV/HK = 3 -> k heads in fours)
        const int rv4 = 4 / std::gcd(4, std::max(1, HK_ > 0 ? HV_ / HK_ : 1));
        si.kh_unit = std::lcm(ssm_b / std::gcd(ssm_b, std::max(1, DV_)), rv4);
        si.ff_unit = FF_ % 256 == 0 ? 256 : down_b;
        si.voc_unit = VOCAB_ % 256 == 0 ? 256 : VOCAB_ % 4 == 0 ? 4 : 1;
        si.eh_unit = (2 * E_) % 256 == 0 ? 256 : std::max(1, blck(bn(L_, "nextn.eh_proj.weight")));
        // parameters per unit (the remainder tie-break balances them; the nextn block is left out, so the split of
        // the main layers does not depend on whether MTP is loaded)
        const double gqa = HKV_ > 0 ? (double) H_ / HKV_ : 1, rv = HK_ > 0 ? (double) HV_ / HK_ : 1;
        si.c_kv = (double) n_att * E_ * HD_ * (3 * gqa + 2);
        si.c_kh = (double) n_gdn * E_ * si.kh_unit * (2.0 * DK_ + 3 * rv * DV_ + 2 * rv);
        si.c_ff = (double) L_ * 3 * E_ * si.ff_unit;
        si.c_voc = (double) E_ * si.voc_unit;
        split_ = tp_make_split(si);
        if (!split_.even) {
            std::string ws;
            for (int r = 0; r < R_; ++r) {
                char b[32];
                snprintf(b, sizeof b, "%s%g", r ? "," : "", si.w.empty() ? 1.0 : si.w[r]);
                ws += b;
            }
            printf("tensor parallel split over %d GPUs (weights %s%s): %s\n", R_, ws.c_str(),
                   si.w.empty() ? ", equal" : "", split_.str().c_str());
        }
        if (net_) net_check(o, si.w);
    }
    rk_.resize(R);
    mv_scratch_bytes_.assign(R, 0);
    whash_total_.assign(R, 1469598103934665603ull);
    if (const char * e = getenv("P100E_FAST")) use_fast_ = atoi(e) != 0;
    if (const char * e = getenv("P100E_FUSE")) fuse_ = atoi(e) != 0;
    if (const char * e = getenv("P100E_CONCAT")) concat_ = atoi(e) != 0;
    if (const char * e = getenv("P100E_AR_SAFE")) ar_safe_ = atoi(e) != 0;
    if (const char * e = getenv("P100E_P2P_AR")) p2p_ar_ = atoi(e) != 0;
    if (const char * e = getenv("P100E_AR_SIG_FUSE")) sig_fuse_ = atoi(e) != 0 && R <= 2;   // R > 2: separate signals
    if (const char * e = getenv("P100E_Q8_FORK")) q8_fork_ = atoi(e) != 0;
    if (const char * e = getenv("P100E_MERGE_INPROJ")) merge_in_ = atoi(e) != 0;
    if (const char * e = getenv("P100E_GDN_FUSE")) gdn_fuse_ = atoi(e) != 0;
    if (const char * e = getenv("P100E_GRAPH")) graph_maxt_ = atoi(e);
    if (const char * e = getenv("P100E_GDN_BATCH")) gdn_batch_ = atoi(e) != 0;
    // T > 1 concatenated in-projections (fast::matvec_prequantized_parts): P100E_CONCAT_T=0 or P100E_VERIFY_CAT=0 -> off
    if (const char * e = getenv("P100E_CONCAT_T")) concat_t_ = atoi(e) != 0;
    if (const char * e = getenv("P100E_VERIFY_CAT")) concat_t_ = concat_t_ && atoi(e) != 0;
    peer_stride_ = (size_t) max_batch_ * E_;

    scratch_elems_ = 64ull << 20;   // 256 MiB f32 dequant scratch per GPU

    for (int r = 0; r < R; ++r) {
        rank_state & S = rk_[r];
        S.dev = devices[r];
        // this rank's shares (every buffer, slice and launch below uses them; equal split = the old dim / R)
        const tp_rank_dims & D = split_.rk[r0_ + r];   // its global rank's ranges (one process: r0_ = 0)
        S.Hl = (int) D.q.n; S.HKVl = (int) D.kv.n; S.HKl = (int) D.kh.n; S.HVl = S.HKl * (HK_ > 0 ? HV_ / HK_ : 0);
        S.Cl = 2 * S.HKl * DK_ + S.HVl * DV_;
        S.FFl = (int) D.ff.n; S.VL = (int) D.voc.n; S.v0 = D.voc.b;
        const int Hl = S.Hl, HKVl = S.HKVl, HKl = S.HKl, HVl = S.HVl, FFl = S.FFl, Cl = S.Cl;
        CK(cudaSetDevice(S.dev));
        for (int o = 0; o < R; ++o) {
            if (o == r) continue;
            int can = 0;
            cudaDeviceCanAccessPeer(&can, S.dev, devices[o]);
            if (can) {
                // already enabled (another component of this process did it, e.g. ggml with GGML_CUDA_P2P set) is fine,
                // but that error would stay the thread's last error and fail the next CK(cudaGetLastError())
                const cudaError_t e = cudaDeviceEnablePeerAccess(devices[o], 0);
                if (e == cudaErrorPeerAccessAlreadyEnabled) (void) cudaGetLastError();
                else CK(e);
            }
            printf("GPU %d -> GPU %d peer access: %s\n", S.dev, devices[o], can ? "yes" : "no");
        }
        {
            // main stream at the highest priority (P100E_STREAM_PRIO=0: default priority): another component's kernels
            // on the same GPU (the vision encoder's ggml stream, default priority) then hand freed SMs to the decode
            // step's blocks first instead of queueing them behind their own
            static const bool sprio = !getenv("P100E_STREAM_PRIO") || atoi(getenv("P100E_STREAM_PRIO")) != 0;
            int lo = 0, hi = 0;
            CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
            CK(cudaStreamCreateWithPriority(&S.s, cudaStreamNonBlocking, sprio ? hi : 0));
        }
        CB(cublasCreate(&S.cb));
        CB(cublasSetStream(S.cb, S.s));
        CB(cublasSetMathMode(S.cb, CUBLAS_PEDANTIC_MATH));
        CK(cudaMalloc(&S.w_scratch, scratch_elems_ * 4));
        const size_t wide = (size_t) max_batch_ * std::max({FFl, 2 * Hl * HD_, Cl, E_});
        const size_t emb = (size_t) max_batch_ * E_;
        CK(cudaMalloc(&S.x, emb * 4)); CK(cudaMalloc(&S.cur, emb * 4));
        // peer: double buffer x one slot per sender (R - 1; one for R <= 2; net mode: every node's ranks, R_ - 1)
        CK(cudaMalloc(&S.part, emb * 4)); CK(cudaMalloc(&S.peer, emb * 4 * 2 * std::max(1, R_ - 1)));
        for (float ** p : {&S.a, &S.b, &S.c, &S.d, &S.e, &S.g}) CK(cudaMalloc(p, wide * 4));
        // the fast matvec kernels may read (and multiply by zero) columns past T: keep buffers finite
        for (float * p : {S.a, S.b, S.c, S.d, S.e, S.g}) CK(cudaMemset(p, 0, wide * 4));
        for (float * p : {S.x, S.cur, S.part}) CK(cudaMemset(p, 0, emb * 4));
        // step descriptor: per-row page-table offsets, per-sequence delta-net rows, positions (S.pos)
        CK(cudaMalloc(&S.desc, (size_t) (DESC_POS + std::max(max_batch_, MAX_ROWS)) * 4));
        CK(cudaMemset(S.desc, 0, (size_t) (DESC_POS + std::max(max_batch_, MAX_ROWS)) * 4));
        S.rtab = S.desc + DESC_RT; S.gseq = S.desc + DESC_GS; S.pos = S.desc + DESC_POS;
        CK(cudaMalloc(&S.ptab, (size_t) max_slots_ * max_pages_ * 4));
        CK(cudaMemset(S.ptab, 0, (size_t) max_slots_ * max_pages_ * 4));
        CK(cudaMalloc(&S.rpos, (size_t) 3 * max_batch_ * 4));
        CK(cudaMemset(S.rpos, 0, (size_t) 3 * max_batch_ * 4));
        // P100E_GDN_FUSE single-writer counters (self-resetting), one per local k head (gdn_layer's fuse_cnt contract)
        CK(cudaMalloc(&S.gdn_cnt, (size_t) std::max(1, HKl) * sizeof(unsigned int)));
        CK(cudaMemset(S.gdn_cnt, 0, (size_t) std::max(1, HKl) * sizeof(unsigned int)));
        CK(cudaMalloc(&S.logits, (size_t) 8 * S.VL * 4));
        CK(cudaMalloc(&S.hn, emb * 4));
        CK(cudaMemset(S.hn, 0, emb * 4));
        CK(cudaMalloc(&S.amax, 8 * sizeof(amax_pair)));
        CK(cudaEventCreateWithFlags(&S.ev_part, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&S.ev_copied, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&S.ev_join, cudaEventDisableTiming));
        CK(cudaStreamCreateWithFlags(&S.s2, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&S.ev_q8f, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&S.ev_q8j, cudaEventDisableTiming));
        if (r == 0) CK(cudaEventCreateWithFlags(&ev_fork_, cudaEventDisableTiming));

        // ---- weights ----
        const ranges all;   // empty = full
        auto rows_of = [](int64_t b, int64_t n) { return ranges{{b, b + n}}; };
        if (has_head()) {   // (pipe node 0: no LM head; its node 1 computes the logits)
            load_rows(r, "output.weight", rows_of(D.voc.b, D.voc.n), {});
            load_small(r, "output_norm.weight", all);
        }
        S.lp.assign(L_, layer_plan{});
        std::vector<float *> po;
        for (int il = lb_; il < le_; ++il) {   // (pipe mode: this node's layers only)
            layer_plan & P = S.lp[il];
            P.attn = (il + 1) % full_attn_interval_ == 0;
            load_small(r, bn(il, "attn_norm.weight"), all);
            load_small(r, bn(il, "post_attention_norm.weight"), all);
            size_t yo = 0;   // decode (T = 1) outputs of one input's projections, back to back in S.a
            P.in_ffn = load_set(r, {{bn(il, "ffn_gate.weight"), rows_of(D.ff.b, FFl), {}},
                                    {bn(il, "ffn_up.weight"), rows_of(D.ff.b, FFl), {}}}, yo, po);
            P.fg = po[0]; P.fu = po[1];
            load_rows(r, bn(il, "ffn_down.weight"), all, {{D.ff.b, D.ff.e()}});
            yo = 0;
            if ((il + 1) % full_attn_interval_ == 0) {
                P.in_attn = load_set(r, {{bn(il, "attn_q.weight"), rows_of(D.q.b * 2 * HD_, (int64_t) Hl * 2 * HD_), {}},
                                         {bn(il, "attn_k.weight"), rows_of(D.kv.b * HD_, (int64_t) HKVl * HD_), {}},
                                         {bn(il, "attn_v.weight"), rows_of(D.kv.b * HD_, (int64_t) HKVl * HD_), {}}}, yo, po);
                P.q = po[0]; P.k = po[1]; P.v = po[2];
                load_rows(r, bn(il, "attn_output.weight"), all, {{D.q.b * HD_, D.q.e() * HD_}});
                load_small(r, bn(il, "attn_q_norm.weight"), all);
                load_small(r, bn(il, "attn_k_norm.weight"), all);
            } else {
                // local k heads [k0, k0 + HKl); local v heads: groups g of HK heads, take [g*HK + k0, +HKl)
                const int64_t k0 = D.kh.b;
                ranges qkv, vh, vrows, vcols;
                qkv.push_back({k0 * DK_, (k0 + HKl) * DK_});                                                   // q
                qkv.push_back({(int64_t) HK_ * DK_ + k0 * DK_, (int64_t) HK_ * DK_ + (k0 + HKl) * DK_});       // k
                for (int g = 0; g < HV_ / HK_; ++g) {
                    const int64_t h0 = (int64_t) g * HK_ + k0;
                    qkv.push_back({2LL * HK_ * DK_ + h0 * DV_, 2LL * HK_ * DK_ + (h0 + HKl) * DV_});            // v
                    vh.push_back({h0, h0 + HKl});
                    vrows.push_back({h0 * DV_, (h0 + HKl) * DV_});
                }
                std::vector<col_range> vc;
                for (auto & p : vrows) vc.push_back({p.first, p.second});
                P.in_attn = load_set(r, {{bn(il, "attn_qkv.weight"), qkv, {}}, {bn(il, "attn_gate.weight"), vrows, {}},
                                         {bn(il, "ssm_beta.weight"), vh, {}}, {bn(il, "ssm_alpha.weight"), vh, {}}}, yo, po);
                P.qkv = po[0]; P.z = po[1]; P.beta = po[2]; P.alpha = po[3];
                load_small(r, bn(il, "ssm_conv1d.weight"), qkv);       // conv weight rows = channels
                load_small(r, bn(il, "ssm_dt.bias"), vh);
                load_small(r, bn(il, "ssm_a"), vh);
                load_small(r, bn(il, "ssm_norm.weight"), all);
                load_rows(r, bn(il, "ssm_out.weight"), all, vc);
            }
        }
        if (n_nextn_ > 0) {
            // MTP (nextn) block: a full-attention decoder block sliced like the main ones, plus eh_proj
            // row-parallel over its 2E input columns (concat order [enorm(embed), hnorm(h)]): rank r takes columns
            // split().rk[r].eh, never weighted: [r, r + 1) * 2E/R when 2E divides by R (R = 2: rank 0 the embedding
            // half, rank 1 the hidden half; R = 4: two quarters each), else 256-column blocks (R = 3 on the 27B:
            // 14/13/13 blocks, rank 1 straddles the halves; tp_mtp::eh_slice)
            if (net_ || pnet_) throw std::runtime_error("MTP over the network (--net-nodes) is not supported yet: run without --mtp / --load-mtp");
            if (R < 2 || !use_fast_) throw std::runtime_error("MTP needs 2 to 4 GPUs and the fast path");
            const int il = L_;
            load_small(r, bn(il, "attn_norm.weight"), all);
            load_small(r, bn(il, "post_attention_norm.weight"), all);
            // q/k/v and gate/up as concatenated sets like the main blocks (fused draft head, S.lp[L_]); S.w keeps a
            // view per weight for the per-op path
            S.lp.resize(L_ + 1);
            layer_plan & P = S.lp[il];
            P.attn = true;
            size_t yo = 0;
            P.in_ffn = load_set(r, {{bn(il, "ffn_gate.weight"), rows_of(D.ff.b, FFl), {}},
                                    {bn(il, "ffn_up.weight"), rows_of(D.ff.b, FFl), {}}}, yo, po);
            P.fg = po[0]; P.fu = po[1];
            load_rows(r, bn(il, "ffn_down.weight"), all, {{D.ff.b, D.ff.e()}});
            yo = 0;
            P.in_attn = load_set(r, {{bn(il, "attn_q.weight"), rows_of(D.q.b * 2 * HD_, (int64_t) Hl * 2 * HD_), {}},
                                     {bn(il, "attn_k.weight"), rows_of(D.kv.b * HD_, (int64_t) HKVl * HD_), {}},
                                     {bn(il, "attn_v.weight"), rows_of(D.kv.b * HD_, (int64_t) HKVl * HD_), {}}}, yo, po);
            P.q = po[0]; P.k = po[1]; P.v = po[2];
            load_rows(r, bn(il, "attn_output.weight"), all, {{D.q.b * HD_, D.q.e() * HD_}});
            load_small(r, bn(il, "attn_q_norm.weight"), all);
            load_small(r, bn(il, "attn_k_norm.weight"), all);
            load_rows(r, bn(il, "nextn.eh_proj.weight"), all, {{D.eh.b, D.eh.e()}});
            load_small(r, bn(il, "nextn.enorm.weight"), all);
            load_small(r, bn(il, "nextn.hnorm.weight"), all);
            if (g_.find(bn(il, "nextn.shared_head_norm.weight"))) load_small(r, bn(il, "nextn.shared_head_norm.weight"), all);
            if (g_.find(bn(il, "nextn.embed_tokens.weight")) || g_.find(bn(il, "nextn.shared_head_head.weight")))
                throw std::runtime_error("nextn.embed_tokens / shared_head_head not supported yet");
        }

        // ---- state ----
        // attention KV: paged (default) = views into the pool allocated by alloc_pool (one page = every KV layer's K and
        // V for 2^psh positions); P100E_PAGED=0 = one contiguous [n_ctx] cache per layer (the pre-'sessions' layout).
        // delta-net states: one slot per resident sequence in two pools ([slot][gdn layer][...]).
        kvl_of_.assign(L_ + 1, -1); gdn_of_.assign(L_, -1);
        n_kvl_ = 0; n_gdn_ = 0;
        for (int il = 0; il < L_; ++il) {
            if (!own_layer(il)) continue;   // pipe mode: another node's layer (no KV pages, no state)
            if ((il + 1) % full_attn_interval_ == 0) kvl_of_[il] = n_kvl_++;
            else gdn_of_[il] = n_gdn_++;
        }
        if (n_nextn_ > 0) kvl_of_[L_] = n_kvl_++;
        auto kv_make = [&](int il) {
            fast::kv_cache c;
            if (!paged_) {
                c = fast::kv_alloc(kv_type_, n_ctx_, HKVl, HD_);
                if (!c.k) throw std::runtime_error(il == L_ ? "kv_alloc (mtp) failed" : "kv_alloc failed");
                return c;
            }
            c.type = kv_type_; c.n_ctx = n_ctx_; c.Hkv = HKVl; c.D = HD_;   // k / v / pt: alloc_pool
            return c;
        };
        if (n_nextn_ > 0) {
            S.kv.resize(L_ + 1);
            S.kv[L_] = kv_make(L_);
        }
        S.kc.assign(L_, nullptr); S.vc.assign(L_, nullptr); S.conv_state.assign(L_, nullptr); S.ssm_state.assign(L_, nullptr);
        for (int il = 0; il < L_; ++il) {
            if (own_layer(il) && (il + 1) % full_attn_interval_ == 0) {
                if (use_fast_) {
                    if (S.kv.size() < (size_t) L_) S.kv.resize(L_);
                    S.kv[il] = kv_make(il);
                } else {
                    if (max_slots_ > 1) throw std::runtime_error("P100E_FAST=0 allows one sequence only");
                    CK(cudaMalloc(&S.kc[il], (size_t) n_ctx_ * HKVl * HD_ * 4));
                    CK(cudaMalloc(&S.vc[il], (size_t) n_ctx_ * HKVl * HD_ * 4));
                }
            }
        }
        S.csz = (size_t) Cl * (CONV_K_ - 1);
        S.ssz = (size_t) HVl * DV_ * DK_;
        CK(cudaMalloc(&S.conv_pool, (size_t) max_slots_ * n_gdn_ * S.csz * 4));
        CK(cudaMemset(S.conv_pool, 0, (size_t) max_slots_ * n_gdn_ * S.csz * 4));
        CK(cudaMalloc(&S.ssm_pool, (size_t) max_slots_ * n_gdn_ * S.ssz * 4));
        CK(cudaMemset(S.ssm_pool, 0, (size_t) max_slots_ * n_gdn_ * S.ssz * 4));

        // ---- fused decode plan: remaining per-layer pointers, the shared a16 activation buffer ----
        bool ok_main = use_fast_ && (!has_head() || S.w["output.weight"].pk.data);
        bool nok = n_nextn_ > 0 && use_fast_ && S.w[bn(L_, "nextn.eh_proj.weight")].pk.data;
        for (int il = 0; il < (int) S.lp.size(); ++il) {   // the main blocks, then the nextn block (il == L_) if loaded
            if (il < L_ && !own_layer(il)) continue;   // pipe mode: another node's layer
            layer_plan & P = S.lp[il];
            bool & ok = il < L_ ? ok_main : nok;
            P.attn_norm = S.f[bn(il, "attn_norm.weight")];
            P.post_norm = S.f[bn(il, "post_attention_norm.weight")];
            P.down = S.w[bn(il, "ffn_down.weight")].pk;
            if (P.attn) {
                P.q_norm = S.f[bn(il, "attn_q_norm.weight")];
                P.k_norm = S.f[bn(il, "attn_k_norm.weight")];
                P.o = S.w[bn(il, "attn_output.weight")].pk;
                ok = ok && !S.kv.empty() && (S.kv[il].k || paged_);   // paged: the pool comes with alloc_pool
            } else {
                P.gw = fast::GdnWeights{S.f[bn(il, "ssm_conv1d.weight")], S.f[bn(il, "ssm_dt.bias")], S.f[bn(il, "ssm_a")],
                                        S.f[bn(il, "ssm_norm.weight")]};
                P.o = S.w[bn(il, "ssm_out.weight")].pk;
            }
            // T > 1 (MTP verify): the T = 1 launches (concatenated weights) with each part's rows written to its own
            // [T][n] buffer (fast::matvec_prequantized_parts: same bits as one launch per weight). P100E_VERIFY_CAT=0 /
            // P100E_CONCAT_T=0: one launch per weight (the S.w entries are views into the concatenated allocations)
            const bool verify_cat = concat_t_;
            auto pk = [&](const char * n) { return S.w[bn(il, n)].pk; };
            float * const ffn_t[] = {S.a, S.b};
            float * const att_t[] = {S.a, S.e, S.g};
            float * const gdn_t[] = {S.a, S.b, S.d, S.e};
            auto with_parts = [&](const std::vector<mv_launch> & in, float * const * tbuf) {
                std::vector<mv_launch> res;
                for (const mv_launch & m : in) {
                    mv_launch t;
                    t.pk = m.pk;
                    t.y = tbuf[m.parts.at(0).first];
                    if (m.parts.size() > 1) for (auto & [i, n] : m.parts) t.outs.push_back({tbuf[i], n});
                    res.push_back(t);
                }
                return res;
            };
            if (verify_cat) {
                P.in_ffn_t = with_parts(P.in_ffn, ffn_t);
                P.in_attn_t = with_parts(P.in_attn, P.attn ? att_t : gdn_t);
            } else {
                P.in_ffn_t = {{pk("ffn_gate.weight"), S.a}, {pk("ffn_up.weight"), S.b}};
                if (P.attn) P.in_attn_t = {{pk("attn_q.weight"), S.a}, {pk("attn_k.weight"), S.e}, {pk("attn_v.weight"), S.g}};
                else P.in_attn_t = {{pk("attn_qkv.weight"), S.a}, {pk("attn_gate.weight"), S.b}, {pk("ssm_beta.weight"), S.d},
                                    {pk("ssm_alpha.weight"), S.e}};
            }
            ok = ok && P.down.data && P.o.data;
            for (auto * v : {&P.in_attn, &P.in_ffn, &P.in_attn_t, &P.in_ffn_t}) for (auto & m : *v) ok = ok && m.pk.data;
        }
        bool ok = ok_main;
        // projection outputs of one input (T = 1) must fit in S.a
        const size_t wide1 = std::max({(size_t) 2 * FFl, (size_t) Hl * 2 * HD_ + 2 * HKVl * HD_, (size_t) Cl + HVl * DV_ + 2 * HVl});
        ok = ok && wide1 <= wide;
        const size_t aq_bytes = fast::matvec_scratch_bytes(std::max({E_, FFl, Hl * HD_, HVl * DV_}), 8);
        CK(cudaMalloc(&S.aq, aq_bytes));
        CK(cudaMemset(S.aq, 0, aq_bytes));
        plan_ok_ = r == 0 ? ok : plan_ok_ && ok;
        nextn_plan_ok_ = r == 0 ? nok && ok : nextn_plan_ok_ && nok && ok;
        CK(cudaDeviceSynchronize());
        if (whash_on()) printf("WHASH_TOTAL r%d %016llx\n", r, whash_total_[r]);
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        size_t kvb = 0;
        for (const auto & c : S.kv) if (c.k && !c.pt && !paged_) kvb += 2 * fast::kv_row_bytes(c.type, c.Hkv, c.D) * (size_t) c.n_ctx;
        printf("GPU %d: weights %.2f GiB, KV %s %s, %d state slot(s) %.0f MiB, free %.0f MiB of %.0f MiB\n", S.dev,
               weight_bytes(r) / 1073741824.0, kv_type_ == fast::KV_F16 ? "f16" : kv_type_ == fast::KV_Q8_0 ? "q8_0" : "q4_0",
               paged_ ? "paged (pool below)" : (std::to_string((int) (kvb >> 20)) + " MiB contiguous (" + std::to_string(n_ctx_) +
                                               " positions)").c_str(),
               max_slots_, (double) max_slots_ * n_gdn_ * (S.csz + S.ssz) * 4 / 1048576.0, fr / 1048576.0, tot / 1048576.0);
    }
    // one-shot P2P allreduce for the fused decode: needs 2+ GPUs with peer access between every pair (a full mesh;
    // else the event path, whose cudaMemcpyPeerAsync also works without peer access)
    if (net_) {
        // every node must take the same decode / prefill paths (each launches the same collectives): a plan that one
        // node's ranks cannot run is off on every node
        const uint64_t mine = (uint64_t) plan_ok_;
        uint64_t all = 1;
        for (uint64_t v : net_->allgather(mine, 3600 * 1000)) all &= v;   // first collective after the load: long wait
        plan_ok_ = all != 0;
        nextn_plan_ok_ = false;
        p2p_ar_ = false;     // the event allreduce (the P2P flag kernels spin on device memory a remote rank cannot write)
        sig_fuse_ = false;
    }
    p2p_ar_ = p2p_ar_ && use_fast_ && fuse_ && plan_ok_ && R >= 2;
    if (p2p_ar_) {
        bool can = true;
        // P100E_P2P_SAME_DEV=1 (local tests only, e.g. --devices 0,0 on one GPU): both ranks on one device reach each
        // other's memory without peer access, so the one-shot P2P allreduce and the CUDA graphs run there too
        static const bool same_dev = getenv("P100E_P2P_SAME_DEV") && atoi(getenv("P100E_P2P_SAME_DEV")) != 0;
        for (int r = 0; can && r < R; ++r)
            for (int o = 0; can && o < R; ++o) {
                if (o == r) continue;
                int a = 0;
                if (same_dev && rk_[r].dev == rk_[o].dev) a = 1;
                else cudaDeviceCanAccessPeer(&a, rk_[r].dev, rk_[o].dev);
                can = a != 0;
                if (!can && R > 2) printf("P2P allreduce: GPU %d cannot access GPU %d (no full peer mesh): event allreduce\n",
                                          rk_[r].dev, rk_[o].dev);
            }
        p2p_ar_ = can;
        for (int r = 0; can && r < R; ++r) {
            CK(cudaSetDevice(rk_[r].dev));
            CK(cudaMalloc(&rk_[r].part2, (size_t) max_batch_ * E_ * 4));
            CK(cudaMemset(rk_[r].part2, 0, (size_t) max_batch_ * E_ * 4));
            CK(cudaMalloc(&rk_[r].flag, 256));
            CK(cudaMemset(rk_[r].flag, 0, 256));
            CK(cudaDeviceSynchronize());
        }
        if (can) {
            // robustness (fork GGML_CUDA_AR_SPIN_TIMEOUT_MS / AR_PASCAL status): bounded spin (default
            // 2000 ms, 500 on a device with a kernel watchdog), a host-mapped status line per GPU that the host checks
            // after every step (check_ar), and fault injection: P100E_AR_TEST_DROP=N makes GPU P100E_AR_TEST_DROP_RANK
            // (default 1) store no allreduce signal from its N-th on (the peer must time out and the step abort)
            CK(cudaHostAlloc((void **) &ar_status_h_, (size_t) R * 16 * sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable));
            memset(ar_status_h_, 0, (size_t) R * 16 * sizeof(int));
            const char * te = getenv("P100E_AR_SPIN_TIMEOUT_MS");
            const int drop = getenv("P100E_AR_TEST_DROP") ? atoi(getenv("P100E_AR_TEST_DROP")) : 0;
            const int drop_rank = getenv("P100E_AR_TEST_DROP_RANK") ? atoi(getenv("P100E_AR_TEST_DROP_RANK")) : 1;
            for (int r = 0; r < R; ++r) {
                CK(cudaSetDevice(rk_[r].dev));
                int wd = 0;
                CK(cudaDeviceGetAttribute(&wd, cudaDevAttrKernelExecTimeout, rk_[r].dev));
                int ms = te ? atoi(te) : wd ? 500 : 2000;
                ms = std::max(10, std::min(ms, wd ? 750 : 60000));
                int * st_dev = nullptr;
                CK(cudaHostGetDevicePointer((void **) &st_dev, ar_status_h_ + (size_t) r * 16, 0));
                static const int fence = getenv("P100E_AR_FENCE") ? atoi(getenv("P100E_AR_FENCE")) : 0;
                int words[8] = {0, 0, 0, r == drop_rank ? drop : 0, ms, fence, 0, 0};
                memcpy(&words[fast::P2P_FLAG_STATUS_PTR], &st_dev, sizeof st_dev);
                CK(cudaMemcpy(rk_[r].flag, words, sizeof words, cudaMemcpyHostToDevice));
                if (drop && r == drop_rank) printf("P2P allreduce: TEST: GPU %d drops its signals from #%d on\n", rk_[r].dev, drop);
            }
        }
    }
    CK(cudaMallocHost(&logits_h_, (size_t) 8 * VOCAB_ * 4));   // up to 8 logit rows (forward_n)
    CK(cudaMallocHost(&desc_h_, (size_t) (DESC_POS + std::max(max_batch_, MAX_ROWS)) * 4));
    memset(desc_h_, 0, (size_t) (DESC_POS + std::max(max_batch_, MAX_ROWS)) * 4);
    pos_h_ = desc_h_ + DESC_POS;
    CK(cudaMallocHost(&ptab_h_, (size_t) max_slots_ * max_pages_ * 4));
    memset(ptab_h_, 0, (size_t) max_slots_ * max_pages_ * 4);
    free_slots_.clear();
    for (int s = max_slots_ - 1; s >= 0; --s) free_slots_.push_back(s);
    CK(cudaMallocHost(&rpos_h_, (size_t) 3 * max_batch_ * 4));
    CK(cudaMallocHost(&amax_h_, (size_t) R * 8 * sizeof(amax_pair)));
    // graphs need every launch of the body to be capturable: the P2P allreduce (kernels only), or no allreduce
    if (!(use_fast_ && fuse_ && plan_ok_ && (p2p_ar_ || R == 1))) graph_maxt_ = 0;
    if (net_) {
        graph_maxt_ = 0;   // the allreduce is host code (sync, TCP): nothing to capture
        // pinned staging of the network allreduce / prefill exchange: one [max_batch][E] f32 slot per rank
        net_slot_ = (size_t) max_batch_ * E_ * 4;
        CK(cudaHostAlloc((void **) &net_tx_, net_slot_ * R, cudaHostAllocPortable));
        CK(cudaHostAlloc((void **) &net_rx_, net_slot_ * R_, cudaHostAllocPortable));
        amax_net_.assign((size_t) R_ * 8 * sizeof(amax_pair), 0);
        // speed switches: pre-add (P100E_NET_PREADD, agreed: it sets the message sizes), graphs
        {
            const char * pe = getenv("P100E_NET_PREADD");
            net_preadd_ = !(pe && *pe && atoi(pe) == 0) && node_nr_[0] >= 2;
            net_->agree("pre-add (P100E_NET_PREADD, effective)", (uint64_t) net_preadd_);
            const char * ge = getenv("P100E_NET_GRAPH");
            net_graph_ = !(ge && *ge && atoi(ge) == 0) && use_fast_ && fuse_ && plan_ok_;
            const size_t emb = (size_t) max_batch_ * E_;
            if (net_preadd_ && net_->self() == 0) {
                CK(cudaSetDevice(rk_[0].dev));
                CK(cudaMalloc(&pre_buf_, emb * 4));
            } else if (net_preadd_) {
                pre_k_ = node_nr_[0];
                const std::vector<uint32_t> nz(emb, 0x80000000u);   // -0.0f: x + -0.0f == x for every x
                for (auto & S : rk_) {
                    float * p = nullptr;
                    CK(cudaSetDevice(S.dev));
                    CK(cudaMalloc(&p, emb * 4));
                    CK(cudaMemcpy(p, nz.data(), emb * 4, cudaMemcpyHostToDevice));
                    pad_.push_back(p);
                }
            }
        }
        gl_init();   // P100E_NET_GPULINK (pdl only; off by default)
        // the prefill pipeline verdict (per instance) must be the same on every node: AND of every node's
        bool ok = prefill_ok();
        for (uint64_t v : net_->allgather((uint64_t) ok)) ok = ok && v != 0;
        pf_ok_ = ok ? 1 : 0;
        const std::string gls = gl_ ? "on (T <= " + std::to_string(gl_maxt_) + ")" : std::string("off");
        printf("net: node %d of %d, global ranks %d..%d of %d, decode %s, prefill %s; pre-add %s, segment graphs %s, "
               "busy-poll %s, gpu link %s\n", net_->self(), net_->n_nodes(), r0_, r0_ + R - 1, R_, plan_ok_ ? "fused" : "per-op",
               ok ? "pipeline" : "classic", net_preadd_ ? "on" : "off", net_graph_ ? (gl_ ? "on (one per step)" : "on") : "off",
               net_->spin_us() > 0 ? "on" : "off", gls.c_str());
    }
    if (pnet_) {
        // hybrid pipeline: pinned staging of the handoff rows (node 0: D2H source of the message; node
        // 1: the receive buffer, H2D into S.x). The first collective after the load waits long: the nodes load
        // different amounts of weights (node 0 typically 3x node 1's)
        pipe_cap_ = (size_t) max_batch_ * E_ * 4;
        CK(cudaHostAlloc((void **) &pipe_h_, pipe_cap_, cudaHostAllocPortable));
        (void) pnet_->allgather(1, 3600 * 1000);
        printf("pipe: node %d of 2 runs layers [%d, %d) of %d%s%s on %d local GPU(s) (R = %d, its own split); decode %s, "
               "prefill %s; handoff %zu bytes per row (f32) + result, transport %s\n", pnet_->self(), lb_, le_, L_,
               has_embd() ? " + token embedding" : "", has_head() ? " + output norm + LM head" : "", R, R,
               use_fast_ && fuse_ && plan_ok_ ? "fused" : "per-op", prefill_ok() ? "pipeline" : "classic",
               (size_t) E_ * 4, pnet_->kind());
    }
    const std::string gs = graph_maxt_ > 0 ? "T <= " + std::to_string(graph_maxt_) : std::string("off");
    printf("decode T=1: %s (concat %s, T>1 concat %s, gdn verify %s, allreduce %s, CUDA graph %s)\n",
           use_fast_ && fuse_ && plan_ok_ ? "fused" : "per-op", concat_ ? "on" : "off",
           concat_t_ ? "parts" : "off", gdn_batch_ ? "batched" : "per-token",
           p2p_ar_ ? "P2P one-shot" : ar_safe_ ? "2-wait" : "double-buffered", gs.c_str());
    if (const char * e = getenv("P100E_HOST_BUDGET_MB")) host_budget_ = (size_t) atoll(e) << 20;
    identity_init();
    if (!paged_) pool_ok_ = true;
    else if (o.pool_tokens >= 0) alloc_pool(o.pool_tokens > 0 ? o.pool_tokens : (int64_t) n_ctx_ * max_slots_);
    if (o.default_seq) seq_select(seq_create());
}

tp_model::~tp_model() {
    for (auto & S : rk_) if (S.rb_s) { cudaSetDevice(S.dev); cudaStreamSynchronize(S.rb_s); }
    if (gl_) {   // the spin thread first (it uses net_'s engine); then the rings (no kernel may still use them)
        for (auto & S : rk_) { cudaSetDevice(S.dev); cudaStreamSynchronize(S.s); }
        try { gl_->stop(); } catch (...) {}
        delete gl_;
        gl_ = nullptr;
    }
    drop_graphs();
    if (ev_fork_) { cudaSetDevice(rk_[0].dev); cudaEventDestroy(ev_fork_); }
    if (logits_h_) cudaFreeHost(logits_h_);
    if (desc_h_) cudaFreeHost(desc_h_);
    if (ptab_h_) cudaFreeHost(ptab_h_);
    for (auto & q : seqs_) for (char * h : q.host) if (h) cudaFreeHost(h);
    if (rpos_h_) cudaFreeHost(rpos_h_);
    if (emb_h_) cudaFreeHost(emb_h_);
    if (amax_h_) cudaFreeHost(amax_h_);
    if (ar_status_h_) cudaFreeHost(ar_status_h_);
    if (net_tx_) cudaFreeHost(net_tx_);
    if (net_rx_) cudaFreeHost(net_rx_);
    if (pipe_h_) cudaFreeHost(pipe_h_);
    for (size_t r = 0; r < pad_.size() && r < rk_.size(); ++r) { cudaSetDevice(rk_[r].dev); cudaFree(pad_[r]); }
    if (pre_buf_) { cudaSetDevice(rk_[0].dev); cudaFree(pre_buf_); }
    for (auto & S : rk_) {
        tp_model_pf_free(S.pf);
        S.pf = nullptr;
        cudaSetDevice(S.dev);
        if (S.flag) {
            const int n = fast::p2p_timeouts(S.flag);
            if (n) printf("P2P allreduce: %d wait timeouts on GPU %d (results invalid)\n", n, S.dev);
        }
        for (auto & [n, w] : S.w) if (w.owns) cudaFree(w.d);
        for (auto & w : S.cat) cudaFree(w.d);
        if (S.aq) cudaFree(S.aq);
        if (S.part2) cudaFree(S.part2);
        if (S.flag) cudaFree(S.flag);
        for (auto & [n, p] : S.f) cudaFree(p);
        for (auto * v : {&S.kc, &S.vc}) for (float * p : *v) if (p) cudaFree(p);
        if (!paged_) for (auto & c : S.kv) if (c.k) fast::kv_free(c);
        for (float * p : {S.conv_pool, S.ssm_pool, S.conv_snp, S.ssm_snp}) if (p) cudaFree(p);
        if (S.kv_pool) cudaFree(S.kv_pool);
        if (S.ptab) cudaFree(S.ptab);
        for (float * p : {S.x, S.cur, S.part, S.peer, S.a, S.b, S.c, S.d, S.e, S.g, S.logits, S.hn, S.w_scratch}) cudaFree(p);
        if (S.mv_scratch) cudaFree(S.mv_scratch);
        if (S.lrows) cudaFree(S.lrows);
        if (S.amax) cudaFree(S.amax);
        cudaFree(S.desc);
        cudaFree(S.gdn_cnt);
        if (S.rpos) cudaFree(S.rpos);
        if (S.ev_part) cudaEventDestroy(S.ev_part);
        if (S.ev_copied) cudaEventDestroy(S.ev_copied);
        if (S.ev_join) cudaEventDestroy(S.ev_join);
        if (S.ev_q8f) cudaEventDestroy(S.ev_q8f);
        if (S.ev_q8j) cudaEventDestroy(S.ev_q8j);
        if (S.s2) { cudaStreamSynchronize(S.s2); cudaStreamDestroy(S.s2); }
        if (S.rb_s) { cudaStreamSynchronize(S.rb_s); cudaStreamDestroy(S.rb_s); }
        if (S.ck_s) { cudaStreamSynchronize(S.ck_s); cudaStreamDestroy(S.ck_s); }
        if (S.ev_ck_in) cudaEventDestroy(S.ev_ck_in);
        if (S.ev_ck) cudaEventDestroy(S.ev_ck);
        if (S.ck_stage) cudaFree(S.ck_stage);
        if (S.ev_rb_in) cudaEventDestroy(S.ev_rb_in);
        if (S.ev_rb) cudaEventDestroy(S.ev_rb);
        if (S.cb) cublasDestroy(S.cb);
        if (S.s) cudaStreamDestroy(S.s);
    }
}

size_t tp_model::weight_bytes(int r) const {
    size_t n = 0;
    for (const auto & [k, w] : rk_[r].w) n += w.row_bytes * w.n_out;
    return n;
}

// slice a quantized [n_out][n_in] tensor: selected output rows x selected input-column ranges
void tp_model::load_rows(int r, const std::string & name, const ranges & rows, const std::vector<col_range> & cols) {
    upload(r, name, slice(name, rows, cols));
}

tp_model::host_slice tp_model::slice(const std::string & name, const ranges & rows, const std::vector<col_range> & cols) {
    const gguf_tensor * t = g_.find(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    const qtype_info & ti = get_type_info(t->type);
    const int64_t n_in = t->ne[0], n_out = t->ne[1] * t->ne[2] * t->ne[3];
    const size_t src_row = (size_t) (n_in / ti.blck) * ti.size;
    std::vector<col_range> cr = cols.empty() ? std::vector<col_range>{{0, n_in}} : cols;
    ranges rr = rows.empty() ? ranges{{0, n_out}} : rows;
    int64_t in_local = 0;
    for (auto & c : cr) {
        if (c.c0 % ti.blck || c.c1 % ti.blck) throw std::runtime_error(name + ": column split not block aligned");
        if (c.c0 < 0 || c.c0 > c.c1 || c.c1 > n_in) throw std::runtime_error(name + ": column split outside the tensor");
        in_local += c.c1 - c.c0;
    }
    int64_t out_local = 0;
    for (auto & p : rr) {
        if (p.first < 0 || p.first > p.second || p.second > n_out) throw std::runtime_error(name + ": row split outside the tensor");
        out_local += p.second - p.first;
    }
    const size_t dst_row = (size_t) (in_local / ti.blck) * ti.size;
    std::vector<uint8_t> h((size_t) out_local * dst_row);
    size_t o = 0;
    for (auto & p : rr) {
        for (int64_t row = p.first; row < p.second; ++row) {
            const uint8_t * src = t->data + (size_t) row * src_row;
            for (auto & c : cr) {
                const size_t b0 = (size_t) (c.c0 / ti.blck) * ti.size, b1 = (size_t) (c.c1 / ti.blck) * ti.size;
                memcpy(h.data() + o, src + b0, b1 - b0);
                o += b1 - b0;
            }
        }
    }
    host_slice hs;
    hs.type = t->type; hs.n_in = in_local; hs.n_out = out_local; hs.row_bytes = dst_row; hs.h = std::move(h);
    return hs;
}

// Q6_K rows whose raw stride is not 8-byte aligned (the 2-GPU ffn_down shard: n_in 8704, 7140 B) get the padded stride
// of fast::pack_row_stride (7144 B with P100E_Q6_PAD=1), so they pack to q6r3 and run on the warp-rows kernels. The
// rows keep their raw bytes at the start of each stride; the pad bytes are zero and never read as weights. Every user
// of the weight takes the stride from dev_weight::row_bytes / fast::row_stride (views, packed_to_f16, the classic path).
void tp_model::pad_slice(host_slice & hs) const {
    if (!use_fast_ || !fast::has_fast_path(hs.type)) return;
    const size_t want = fast::pack_row_stride(hs.type, hs.n_in, hs.n_out);
    if (want == 0 || want == hs.row_bytes || hs.n_out == 0) return;
    if (want < hs.row_bytes || hs.h.size() != hs.row_bytes * (size_t) hs.n_out)
        throw std::runtime_error("pad_slice: unexpected slice shape");
    std::vector<uint8_t> p(want * (size_t) hs.n_out, 0);
    for (int64_t i = 0; i < hs.n_out; ++i) memcpy(p.data() + (size_t) i * want, hs.h.data() + (size_t) i * hs.row_bytes, hs.row_bytes);
    hs.h = std::move(p);
    hs.row_bytes = want;
}

void tp_model::upload(int r, const std::string & name, host_slice && hs) {
    pad_slice(hs);
    const std::vector<uint8_t> & h = hs.h;
    dev_weight w;
    w.type = hs.type; w.n_in = hs.n_in; w.n_out = hs.n_out; w.row_bytes = hs.row_bytes;
    static const size_t pad = getenv("P100E_WPAD") ? (size_t) atoll(getenv("P100E_WPAD")) : 0;
    CK(cudaMalloc(&w.d, h.size() + pad));
    if (pad) CK(cudaMemset((char *) w.d + h.size(), 0, pad));
    CK(cudaMemcpy(w.d, h.data(), h.size(), cudaMemcpyHostToDevice));
    // pageable H2D cudaMemcpy may return before the DMA lands; the in-place repack below runs on a
    // non-blocking stream, so wait for the copy first
    CK(cudaDeviceSynchronize());
    const unsigned long long raw_hash = whash_on() ? dev_hash(w.d, h.size()) : 0;
    static const bool fast_check = getenv("P100E_FAST_CHECK") && atoi(getenv("P100E_FAST_CHECK"));
    float *ck_x = nullptr, *ck_ref = nullptr, *ck_y = nullptr;
    if (fast_check && use_fast_ && fast::has_fast_path(w.type)) {
        // debug: reference result on the raw bytes, compared after packing
        rank_state & S = rk_[r];
        std::vector<float> hx((size_t) w.n_in * 2);
        for (size_t i = 0; i < hx.size(); ++i) hx[i] = ((float) (((i * 2654435761u) >> 8) % 2001) / 1000.0f - 1.0f) * (i % 97 == 0 ? 30.0f : 1.0f);
        CK(cudaMalloc(&ck_x, hx.size() * 4)); CK(cudaMalloc(&ck_ref, (size_t) w.n_out * 2 * 4)); CK(cudaMalloc(&ck_y, (size_t) w.n_out * 2 * 4));
        CK(cudaMemcpy(ck_x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());   // pageable H2D may return before the DMA lands; matmul runs on the non-blocking S.s
        S.w[name] = w;
        matmul(r, name, ck_x, ck_ref, 1);
        CK(cudaStreamSynchronize(S.s));
    }
    if (use_fast_ && fast::has_fast_path(w.type)) {
        w.pk = fast::pack_weight(w.type, w.d, w.n_in, w.n_out, rk_[r].s, w.row_bytes);
        CK(cudaStreamSynchronize(rk_[r].s));
        w.packed = w.pk.layout != fast::LAYOUT_RAW && w.pk.layout != fast::LAYOUT_KQRAW && w.pk.layout != fast::LAYOUT_IQ1S && w.pk.layout != fast::LAYOUT_IQ1M;
        if (w.pk.stride && (!w.packed || !w.pk.bytes))   // a padded weight must pack (pack_row_stride matched the packer)
            throw std::runtime_error(name + ": padded rows were not packed (layout " + std::to_string(w.pk.layout) + ")");
        const size_t need = fast::matvec_scratch_bytes(w.n_in, 8);
        if (need > mv_scratch_bytes_[r]) {
            cudaFree(rk_[r].mv_scratch);
            CK(cudaMalloc(&rk_[r].mv_scratch, need));
            CK(cudaMemset(rk_[r].mv_scratch, 0, need));
            mv_scratch_bytes_[r] = need;
        }
    }
    rk_[r].w[name] = w;
    if (whash_on()) {
        CK(cudaDeviceSynchronize());
        const unsigned long long ph = dev_hash(w.d, h.size());
        whash_total_[r] = (whash_total_[r] ^ ph) * 1099511628211ull;
        printf("WHASH r%d %-30s raw %016llx packed %016llx layout %d\n", r, name.c_str(), raw_hash, ph, w.pk.layout);
    }
    if (ck_x) {
        rank_state & S = rk_[r];
        CK(cudaMemset(ck_y, 0xff, (size_t) w.n_out * 2 * 4));
        matmul(r, name, ck_x, ck_y, 1);
        CK(cudaStreamSynchronize(S.s)); CK(cudaGetLastError());
        std::vector<float> a((size_t) w.n_out), b(a.size());
        CK(cudaMemcpy(a.data(), ck_ref, a.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), ck_y, b.size() * 4, cudaMemcpyDeviceToHost));
        double md = 0, mr = 0; int nan = 0;
        for (size_t i = 0; i < a.size(); ++i) { if (!std::isfinite(b[i])) ++nan; md = std::max(md, (double) std::fabs(a[i] - b[i])); mr = std::max(mr, (double) std::fabs(a[i])); }
        printf("FASTCHECK r%d %-28s type %2u %6lld x %6lld layout %d %-8s rel %.3e nan %d\n", r, name.c_str(), w.type, (long long) w.n_in,
               (long long) w.n_out, w.pk.layout, fast::kernel_name(w.pk), md / (mr + 1e-30), nan);
        printf("  ref %g %g %g | fast %g %g %g | x %p y %p scr %p dev %d\n", a[0], a[1], a[w.n_out], b[0], b[1], b[w.n_out],
               (void *) ck_x, (void *) ck_y, S.mv_scratch, S.dev);
        cudaFree(ck_x); cudaFree(ck_ref); cudaFree(ck_y);
        (void) S;
    }
}

// Weights that read the same input. With the fast path and concat_, parts of the same type and n_in (n_out % 4 == 0
// each) are stored back to back in one allocation and packed as one weight: every layout is row-local and every
// kernel's per-row arithmetic depends only on (type, n_in) (fast/matvec.cu: the warp-rows family, q8r), so one launch
// over the concatenation gives each row exactly the value of a launch on its own weight. S.w[name] still holds each part (a view) for the per-op path.
std::vector<mv_launch> tp_model::load_set(int r, const std::vector<set_part> & parts, size_t & y_off, std::vector<float *> & out) {
    rank_state & S = rk_[r];
    std::vector<host_slice> sl;
    for (auto & p : parts) { sl.push_back(slice(p.name, p.rows, p.cols)); pad_slice(sl.back()); }
    std::vector<std::vector<int>> groups;
    for (int i = 0; i < (int) sl.size(); ++i) {
        int gi = -1;
        const bool can = use_fast_ && concat_ && fast::has_fast_path(sl[i].type) && sl[i].n_out % 4 == 0;
        for (int g = 0; can && g < (int) groups.size() && gi < 0; ++g) {
            const host_slice & a = sl[groups[g][0]];
            if (a.type == sl[i].type && a.n_in == sl[i].n_in && a.n_out % 4 == 0) gi = g;
        }
        // a part only joins when its byte offset keeps the 16-byte alignment the packers check
        if (gi >= 0) {
            size_t off = 0;
            for (int j : groups[gi]) off += sl[j].h.size();
            if (off % 16) gi = -1;
        }
        if (gi >= 0) groups[gi].push_back(i); else groups.push_back({i});
    }
    std::vector<mv_launch> res;
    out.assign(parts.size(), nullptr);
    for (auto & g : groups) {
        mv_launch m;
        m.y = S.a + y_off;
        if (g.size() == 1) {
            const int i = g[0];
            const int64_t n_out = sl[i].n_out;
            upload(r, parts[i].name, std::move(sl[i]));
            m.pk = S.w[parts[i].name].pk;
            out[i] = S.a + y_off;
            m.parts.push_back({i, n_out});
            y_off += (size_t) (n_out + 3) / 4 * 4;
            res.push_back(m);
            continue;
        }
        const host_slice & a0 = sl[g[0]];
        int64_t n_tot = 0;
        size_t bytes = 0;
        for (int i : g) { n_tot += sl[i].n_out; bytes += sl[i].h.size(); }
        dev_weight own;
        own.type = a0.type; own.n_in = a0.n_in; own.n_out = n_tot; own.row_bytes = a0.row_bytes;
        static const size_t pad = getenv("P100E_WPAD") ? (size_t) atoll(getenv("P100E_WPAD")) : 0;
        CK(cudaMalloc(&own.d, bytes + pad));
        if (pad) CK(cudaMemset((char *) own.d + bytes, 0, pad));
        size_t o = 0;
        for (int i : g) { CK(cudaMemcpy((char *) own.d + o, sl[i].h.data(), sl[i].h.size(), cudaMemcpyHostToDevice)); o += sl[i].h.size(); }
        CK(cudaDeviceSynchronize());
        own.pk = fast::pack_weight(own.type, own.d, own.n_in, own.n_out, S.s, own.row_bytes);
        CK(cudaStreamSynchronize(S.s));
        own.packed = own.pk.layout != fast::LAYOUT_RAW && own.pk.layout != fast::LAYOUT_KQRAW && own.pk.layout != fast::LAYOUT_IQ1S && own.pk.layout != fast::LAYOUT_IQ1M;
        if (own.pk.stride && (!own.packed || !own.pk.bytes))
            throw std::runtime_error(parts[g[0]].name + ": padded concatenated rows were not packed");
        const size_t need = fast::matvec_scratch_bytes(own.n_in, 8);
        if (need > mv_scratch_bytes_[r]) {
            cudaFree(S.mv_scratch);
            CK(cudaMalloc(&S.mv_scratch, need));
            CK(cudaMemset(S.mv_scratch, 0, need));
            mv_scratch_bytes_[r] = need;
        }
        int64_t row0 = 0;
        for (int i : g) {
            dev_weight v = own;
            v.owns = false;
            v.n_out = sl[i].n_out;
            v.d = (char *) own.d + (size_t) row0 * own.row_bytes;
            v.pk.n_out = v.n_out;
            // the views share the owner's packed layout; when the packer refused the concatenation (pk.data null, e.g.
            // a float weight with n_in % 4 != 0) they stay null too, so the per-op path takes the classic route for
            // them as for a refused single weight (a view never claims a fast path its owner does not have)
            v.pk.data = own.pk.data ? v.d : nullptr;
            v.pk.bytes = own.pk.data && own.pk.bytes ? own.row_bytes * (size_t) v.n_out : 0;   // 0: matvec refuses
            S.w[parts[i].name] = v;
            out[i] = S.a + y_off + row0;
            m.parts.push_back({i, sl[i].n_out});
            row0 += sl[i].n_out;
        }
        m.pk = own.pk;
        y_off += (size_t) n_tot;
        S.cat.push_back(own);
        res.push_back(m);
    }
    return res;
}

// f32 copy of a small tensor; for 2-D tensors `rows` selects rows, for 1-D tensors it selects elements
void tp_model::load_small(int r, const std::string & name, const ranges & rows) {
    const gguf_tensor * t = g_.find(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    rank_state & S = rk_[r];
    const bool vec = t->ne[1] == 1;
    const int64_t row_el = vec ? 1 : t->ne[0];
    const int64_t n_rows = vec ? t->ne[0] : t->ne[1];
    if (t->nbytes > scratch_elems_ * 4) throw std::runtime_error(name + ": too large for the load scratch");
    for (auto & p : rows)
        if (p.first < 0 || p.first > p.second || p.second > n_rows) throw std::runtime_error(name + ": row split outside the tensor");
    // dequantize the full tensor once into scratch, then gather
    CK(cudaMemcpy(S.w_scratch, t->data, t->nbytes, cudaMemcpyHostToDevice));
    // a pageable H2D cudaMemcpy may return before the DMA lands and the dequant below runs on the non-blocking
    // stream S.s: without this sync it can read the previous tensor's bytes (load-order dependent)
    CK(cudaDeviceSynchronize());
    float * full = nullptr;
    CK(cudaMalloc(&full, (size_t) t->nelements() * 4));
    dequant_f32(t->type, S.w_scratch, full, t->nelements(), S.s);
    ranges rr = rows.empty() ? ranges{{0, n_rows}} : rows;
    int64_t n = 0;
    for (auto & p : rr) n += (p.second - p.first) * row_el;
    float * d = nullptr;
    CK(cudaMalloc(&d, (size_t) n * 4));
    int64_t o = 0;
    for (auto & p : rr) {
        CK(cudaMemcpyAsync(d + o, full + p.first * row_el, (size_t) (p.second - p.first) * row_el * 4, cudaMemcpyDeviceToDevice, S.s));
        o += (p.second - p.first) * row_el;
    }
    CK(cudaStreamSynchronize(S.s));
    cudaFree(full);
    S.f[name] = d;
    if (whash_on()) {
        CK(cudaDeviceSynchronize());
        const unsigned long long sh = dev_hash(d, (size_t) n * 4);
        whash_total_[r] = (whash_total_[r] ^ sh) * 1099511628211ull;
        printf("SHASH r%d %-30s %016llx\n", r, name.c_str(), sh);
    }
}

void tp_model::norm(const float * x, const float * w, float * y, int64_t rows, int n, cudaStream_t s) {
    if (use_fast_) fast::rmsnorm(x, w, y, rows, n, eps_, s);
    else k::rmsnorm(x, w, y, rows, n, eps_, s);
}

void tp_model::matmul(int r, const std::string & name, const float * x, float * y, int T) {
    rank_state & S = rk_[r];
    auto it = S.w.find(name);
    if (it == S.w.end()) throw std::runtime_error("weight not loaded: " + name);
    const dev_weight & w = it->second;
    const float one = 1.0f, zero = 0.0f;
    static const int fast_maxt = getenv("P100E_FAST_MAXT") ? atoi(getenv("P100E_FAST_MAXT")) : 8;
    static const bool use_gemm = !getenv("P100E_GEMM") || atoi(getenv("P100E_GEMM")) != 0;
    if (use_gemm && w.pk.data && T > fast_maxt && fast::gemm_scratch_min_bytes(w.n_in, T) <= scratch_elems_ * 4) {
        // prefill: f16 dequant (from the packed or raw layout) + fold GEMM (f16 products, f32 accumulation)
        fast::gemm_packed(w.pk, x, y, T, S.w_scratch, scratch_elems_ * 4, S.s, S.cb);
        return;
    }
    if (w.pk.data && (T <= fast_maxt || w.packed)) {
        // fast decode GEMV; repacked weights have no raw bytes, so larger batches go 8 columns at a time
        for (int t0 = 0; t0 < T; t0 += 8) {
            const int nt = std::min(8, T - t0);
            fast::matvec(w.pk, x + (size_t) t0 * w.n_in, y + (size_t) t0 * w.n_out, nt, S.mv_scratch, S.s);
        }
        static const bool dbg_nan = getenv("P100E_DEBUG_NAN") != nullptr;
        if (dbg_nan) {
            CK(cudaStreamSynchronize(S.s));
            std::vector<float> hx((size_t) T * w.n_in), hy((size_t) T * w.n_out);
            CK(cudaMemcpy(hx.data(), x, hx.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hy.data(), y, hy.size() * 4, cudaMemcpyDeviceToHost));
            int bx = 0, by = 0; double mx = 0;
            for (float v : hx) { if (!std::isfinite(v)) ++bx; else mx = std::max(mx, (double) std::fabs(v)); }
            for (float v : hy) if (!std::isfinite(v)) ++by;
            if (bx || by || mx > 1e4) printf("NANDBG r%d %s T=%d x_bad %d x_max %g y_bad %d layout %d x %p y %p\n", r, name.c_str(), T, bx, mx, by, w.pk.layout, (const void *) x, (void *) y);
        }
        return;
    }
    // classic path: raw GGUF rows (the FAST_CHECK reference, P100E_FAST=0, types without a fast layout); a padded
    // weight (pad_slice) is dequantized row by row
    const size_t raw_rb = (size_t) (w.n_in / get_type_info(w.type).blck) * get_type_info(w.type).size;
    to_fp16_cuda_t to16 = ggml_get_to_fp16_cuda((ggml_type) w.type);
    if (T >= 16 && to16) {
        // prefill: dequant to f16 + f16 GEMM with f32 accumulate (x converted to f16 once)
        half * w16 = (half *) S.w_scratch;
        const int64_t rows_per16 = std::max<int64_t>(1, (int64_t) (scratch_elems_ * 2 / w.n_in) - (int64_t) T * 2);
        half * x16 = w16 + (size_t) rows_per16 * w.n_in;
        to_fp16_cuda_t x_to16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
        x_to16(x, x16, (int64_t) T * w.n_in, S.s);
        for (int64_t r0 = 0; r0 < w.n_out; r0 += rows_per16) {
            const int64_t nr = std::min(rows_per16, w.n_out - r0);
            if (w.row_bytes == raw_rb) to16((const char *) w.d + r0 * w.row_bytes, w16, nr * w.n_in, S.s);
            else for (int64_t i = 0; i < nr; ++i)   // padded rows (pad_slice): one row at a time
                to16((const char *) w.d + (r0 + i) * w.row_bytes, w16 + (size_t) i * w.n_in, w.n_in, S.s);
            CB(cublasGemmEx(S.cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) nr, T, (int) w.n_in, &one, w16, CUDA_R_16F,
                            (int) w.n_in, x16, CUDA_R_16F, (int) w.n_in, &zero, y + r0, CUDA_R_32F, (int) w.n_out,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
        return;
    }
    const int64_t rows_per = std::max<int64_t>(1, (int64_t) (scratch_elems_ / w.n_in));
    for (int64_t r0 = 0; r0 < w.n_out; r0 += rows_per) {
        const int64_t nr = std::min(rows_per, w.n_out - r0);
        if (w.row_bytes == raw_rb) dequant_f32(w.type, (const char *) w.d + r0 * w.row_bytes, S.w_scratch, nr * w.n_in, S.s);
        else for (int64_t i = 0; i < nr; ++i)
            dequant_f32(w.type, (const char *) w.d + (r0 + i) * w.row_bytes, S.w_scratch + (size_t) i * w.n_in, w.n_in, S.s);
        CB(cublasSgemm(S.cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) nr, T, (int) w.n_in, &one, S.w_scratch, (int) w.n_in,
                       x, (int) w.n_in, &zero, y + r0, (int) w.n_out));
    }
}

// Two ranks: each copies its partial into the peer's peer buffer. The peer buffers alternate (ar_buf_), which makes
// the old "wait until the peer is done reading its peer buffer" handshake unnecessary: before rank r writes
// peer[b] of allreduce n, its stream has already waited for the peer's copy of allreduce n-1, which the peer's
// stream issued after its own reads of peer[b] (allreduce n-2). One cross-GPU wait per allreduce instead of two.
// P100E_AR_SAFE=1 restores the extra handshake.
// R > 2: the same with one copy per peer, each into the receiver's slot of this sender (peer_from); the next write of
// a buffer parity again follows the waits for every peer's copies of the allreduce in between.
void tp_model::allreduce_send(int T) {
    if (net_) { net_allreduce_send(T); return; }
    const int R = (int) rk_.size();
    if (R == 1) return;
    const size_t bytes = (size_t) T * E_ * 4;
    const int b = ar_buf_;
    ar_buf_ ^= 1;
    ar_cur_ = b;
    if (ar_safe_) {
        for (int r = 0; r < R; ++r) {
            CK(cudaSetDevice(rk_[r].dev));
            CK(cudaEventRecord(rk_[r].ev_part, rk_[r].s));
        }
    }
    for (int r = 0; r < R; ++r) {
        CK(cudaSetDevice(rk_[r].dev));
        for (int o = 0; o < R; ++o) {   // R = 2: the one peer o = 1 - r, slot 0 (as before)
            if (o == r) continue;
            if (ar_safe_) CK(cudaStreamWaitEvent(rk_[r].s, rk_[o].ev_part, 0));
            const size_t slot = (size_t) b * (R - 1) + (r < o ? r : r - 1);
            CK(cudaMemcpyPeerAsync(rk_[o].peer + slot * peer_stride_, rk_[o].dev, rk_[r].part, rk_[r].dev, bytes, rk_[r].s));
        }
        CK(cudaEventRecord(rk_[r].ev_copied, rk_[r].s));
    }
    for (int r = 0; r < R; ++r) {
        CK(cudaSetDevice(rk_[r].dev));
        for (int o = 0; o < R; ++o)
            if (o != r) CK(cudaStreamWaitEvent(rk_[r].s, rk_[o].ev_copied, 0));
    }
}

// every rank: x += sum_r part_r (rank 0's partial first on every rank, so the replicas stay bit-identical)
// (R = R_, every node's ranks; r0_ + r = the global rank of local rank r: one process, R_ = local count, r0_ = 0)
void tp_model::allreduce_add(int T) {
    const int R = R_, NL = (int) rk_.size();
    if (R == 1) {
        CK(cudaSetDevice(rk_[0].dev));
        k::add(rk_[0].x, rk_[0].part, (int64_t) T * E_, rk_[0].s);
        return;
    }
    allreduce_send(T);
    if (R > 2) {   // rank order on every rank
        for (int r = 0; r < NL; ++r) {
            CK(cudaSetDevice(rk_[r].dev));
            for (int q = 0; q < R; ++q) k::add(rk_[r].x, part_of(r, q), (int64_t) T * E_, rk_[r].s);
        }
        return;
    }
    for (int r = 0; r < NL; ++r) {
        CK(cudaSetDevice(rk_[r].dev));
        const bool g0 = r0_ + r == 0;   // global rank 0
        const float * first  = g0 ? rk_[r].part : peer_buf(r);
        const float * second = g0 ? peer_buf(r) : rk_[r].part;
        k::add(rk_[r].x, first, (int64_t) T * E_, rk_[r].s);
        k::add(rk_[r].x, second, (int64_t) T * E_, rk_[r].s);
    }
}

// norm_in (o = 0, y = S.cur, T rows) / norm_last (o = row T-1, one row into y) for R > 2: [x += every rank's partial of
// the last allreduce in rank order], y = rmsnorm(x) * w, S.aq = a16(y). One fast::addn_rmsnorm_q launch: P2P = wait
// for every peer's signal and read their partials in place; event path = the same kernel without the wait on the
// copies in S.peer (bitwise the same results). Shapes it refuses: k::add per partial + add2_rmsnorm_q.
void tp_model::norm_in_n(int r, const float * w, bool pending, int T, size_t o, float * y) {
    rank_state & S = rk_[r];
    const int R = R_;   // every node's partials (net mode: p2p_ar_ is off, so rk_[q] below is only read in one process)
    if (pending) {
        fast::p2p_ptrs pp{};
        for (int q = 0; q < R; ++q) pp.p[q] = (p2p_ar_ ? (ar_cur_ ? rk_[q].part2 : rk_[q].part) : part_of(r, q)) + o;
        if (p2p_ar_) {
            if (!fast::addn_rmsnorm_q_wait_ctr(S.x + o, pp, R, r, S.flag, S.flag + 2, w, y, S.aq, T, E_, eps_, S.s))
                throw std::runtime_error("P2P allreduce: unsupported shape");
            return;
        }
        if (fast::addn_rmsnorm_q(S.x + o, pp, R, w, y, S.aq, T, E_, eps_, S.s)) return;
        for (int q = 0; q < R; ++q) k::add(S.x + o, pp.p[q], (int64_t) T * E_, S.s);
    }
    if (!fast::add2_rmsnorm_q(S.x + o, nullptr, nullptr, w, y, S.aq, T, E_, eps_, S.s)) fast::matvec_quantize(y, E_, T, S.aq, S.s);
}

// fused decode (T rows): [x += partials of the last allreduce, rank 0 first], cur = rmsnorm(x) * w, aq = a16(cur)
void tp_model::norm_in(int r, const float * w, bool pending, int T) {
    rank_state & S = rk_[r];
    if (R_ > 2) { norm_in_n(r, w, pending, T, 0, S.cur); return; }
    const float * p0 = nullptr, * p1 = nullptr;
    if (pending && p2p_ar_) {
        // partials of the last allreduce sit in buffer ar_cur_ of each GPU; rank 0's first on both GPUs. The wait is
        // for the peer's signal of the same allreduce: flag[0] >= this GPU's own signal count flag[2] (device side)
        const float * mine = ar_cur_ ? S.part2 : S.part;
        const float * theirs = ar_cur_ ? rk_[1 - r].part2 : rk_[1 - r].part;
        p0 = r == 0 ? mine : theirs; p1 = r == 0 ? theirs : mine;
        if (sig_pend_[r]) {   // this GPU's signal of this allreduce inside the wait launch (P100E_AR_SIG_FUSE)
            sig_pend_[r] = false;
            if (!fast::add2_rmsnorm_q_sig_wait(S.x, p0, p1, S.flag, rk_[1 - r].flag, sig_par_[r], w, S.cur, S.aq, T, E_,
                                               eps_, S.s))
                throw std::runtime_error("P2P allreduce: unsupported shape");
            return;
        }
        if (!fast::add2_rmsnorm_q_wait_ctr(S.x, p0, p1, S.flag, S.flag + 2, w, S.cur, S.aq, T, E_, eps_, S.s))
            throw std::runtime_error("P2P allreduce: unsupported shape");
        return;
    }
    if (pending) {   // (global rank 0's partial first; net mode R_ = 2: one rank per node)
        const bool g0 = r0_ + r == 0;
        if (R_ == 1) p0 = S.part;
        else { p0 = g0 ? S.part : peer_buf(r); p1 = g0 ? peer_buf(r) : S.part; }
    }
    if (!fast::add2_rmsnorm_q(S.x, p0, p1, w, S.cur, S.aq, T, E_, eps_, S.s)) fast::matvec_quantize(S.cur, E_, T, S.aq, S.s);
}

void tp_model::norm_last(int r, const float * w, int T, float * y) {
    rank_state & S = rk_[r];
    const size_t o = (size_t) (T - 1) * E_;
    if (R_ > 2) { norm_in_n(r, w, true, 1, o, y); return; }
    if (p2p_ar_) {
        const float * mine = ar_cur_ ? S.part2 : S.part;
        const float * theirs = ar_cur_ ? rk_[1 - r].part2 : rk_[1 - r].part;
        const float * p0 = (r == 0 ? mine : theirs) + o, * p1 = (r == 0 ? theirs : mine) + o;
        if (sig_pend_[r]) {
            sig_pend_[r] = false;
            if (!fast::add2_rmsnorm_q_sig_wait(S.x + o, p0, p1, S.flag, rk_[1 - r].flag, sig_par_[r], w, y, S.aq, 1, E_,
                                               eps_, S.s))
                throw std::runtime_error("P2P allreduce: unsupported shape");
            return;
        }
        if (!fast::add2_rmsnorm_q_wait_ctr(S.x + o, p0, p1, S.flag, S.flag + 2, w, y, S.aq, 1, E_, eps_, S.s))
            throw std::runtime_error("P2P allreduce: unsupported shape");
        return;
    }
    const float * p0 = nullptr, * p1 = nullptr;
    const bool g0 = r0_ + r == 0;
    if (R_ == 1) p0 = S.part + o;
    else { p0 = (g0 ? S.part : peer_buf(r)) + o; p1 = (g0 ? peer_buf(r) : S.part) + o; }
    if (!fast::add2_rmsnorm_q(S.x + o, p0, p1, w, y, S.aq, 1, E_, eps_, S.s)) fast::matvec_quantize(y, E_, 1, S.aq, S.s);
}

// fused decode: every rank's partial for this allreduce is written (stream order) -> hand it to the peer
void tp_model::ar_post(int T) {
    if (!p2p_ar_) { allreduce_send(T); return; }
    if (rk_.size() > 2) {   // R > 2 (no fused-signal mode): every rank stores seq = ++its counter into each peer's flag
        const int R = (int) rk_.size();
        sig_need(0);
        for (int r = 0; r < R; ++r) {
            CK(cudaSetDevice(rk_[r].dev));
            fast::p2p_flags pf{};
            for (int q = 0; q < R; ++q) pf.f[q] = q == r ? nullptr : rk_[q].flag;
            fast::p2p_signal_n(pf, R, r, rk_[r].flag + 2, rk_[r].s);
        }
        sig_last_ = 0;
        ar_cur_ = ar_buf_;
        ar_buf_ ^= 1;
        return;
    }
    for (int r = 0; r < 2; ++r) if (sig_pend_[r]) { CK(cudaSetDevice(rk_[r].dev)); sig_flush(r); }   // never two owed
    // Fused only while capturing a graph: eagerly, rank 0's wait would spin while the host is still enqueueing rank 0's
    // next launches before rank 1's wait (and so its signal) - a hang until the timeout under WDDM launch batching
    // (seen on the GTX 1660 with --devices 0,0). The graph starts both ranks' chains at once. Eager steps (warm-ups,
    // no graphs) keep the separate p2p_signal_ctr launches.
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (sig_fuse_) CK(cudaStreamIsCapturing(rk_[0].s, &cs));
    if (sig_fuse_ && cs == cudaStreamCaptureStatusActive) {   // the signal goes into the next wait launch of each rank
        const int p = sig_last_ == 1 ? 0 : 1;   // write the word that does not hold the latest
        for (int r = 0; r < 2; ++r) { sig_pend_[r] = true; sig_par_[r] = p; }
        sig_last_ = p;
    } else {
        sig_need(0);
        for (int r = 0; r < 2; ++r) {   // seq = ++(own counter) on the device, stored into the peer's flag
            CK(cudaSetDevice(rk_[r].dev));
            fast::p2p_signal_ctr(rk_[1 - r].flag, rk_[r].flag + 2, rk_[r].s);
        }
        sig_last_ = 0;
    }
    ar_cur_ = ar_buf_;
    ar_buf_ ^= 1;
}

// Fused decode layers for T = 1..4 rows. Per layer and GPU: norm(+allreduce sum)+quantize, the input projections
// (same-type weights concatenated, one launch each; T > 1: each part's rows written to its own [T][n_out] per-op buffer
// by fast::matvec_prequantized_parts, P100E_CONCAT_T=0 / P100E_VERIFY_CAT=0: one launch per weight), the mixer
// (attention: attn_prep + attn_decode + sigmoid gate/quantize; delta net: gdn_prep + gdn_scan with the quantize in its
// epilogue), out projection, allreduce copy; then norm(+sum)+quantize, gate|up, silu*mul+quantize, down, allreduce copy.
// The last partials stay pending for the output norm. Same arithmetic as the per-op path, and every kernel's per-row
// result is independent of T (fast/matvec.cuh; attention fast/attn_dec2.cu at any depth): row t of a T-row call equals
// the T = 1 call for that token (MTP verify relies on this).
// keep_states (T > 1): the recurrent state after every token but the last is saved for rollback(): by default one
// T-token gdn_prep + gdn_scan writing the snapshots from inside the kernels (P100E_GDN_BATCH=0: the old per-token
// loop of T = 1 calls + D2D snapshot copies + a separate quantize; both bit-identical to T = 1 decode).
// one input-projection launch of the fused step: plain, or (T > 1, concatenated parts) per-part outputs
static void mv_run(const mv_launch & m, const rank_state & S, int T, cudaStream_t st = nullptr) {
    if (!st) st = S.s;
    if (m.outs.empty()) fast::matvec_prequantized(m.pk, S.cur, S.aq, m.y, T, st);
    else fast::matvec_prequantized_parts(m.pk, S.cur, S.aq, m.outs.data(), (int) m.outs.size(), T, st);
}

void tp_model::decode_layers(int T, bool keep_states, int nseq) {
    if (keep_states && T > 1 && T - nseq > n_snap_) throw std::runtime_error("keep_states: batch larger than enable_rollback()");
    for (int il = lb_; il < le_; ++il) fused_layer(il, T, keep_states, nseq);   // (pipe mode: this node's layers)
}

// delta-net descriptor of a fused step: rows / slots / snapshot slots per sequence from S.gseq (device), the layer's
// part of the slot pools and of the snapshot pool
fast::gdn_multi tp_model::gdn_desc(int r, int il, int T, int nseq, bool snap) const {
    const rank_state & S = rk_[r];
    const int g = gdn_of_[il];
    fast::gdn_multi m;
    m.seqs = S.gseq; m.nseq = nseq; m.Ttot = T;
    m.maxT = std::min(4, T - nseq + 1);   // the longest a sequence can be in this (T, nseq) shape: graphs stay valid
    m.conv_base = S.conv_pool + (size_t) g * S.csz; m.conv_slot = (size_t) n_gdn_ * S.csz;
    m.ssm_base = S.ssm_pool + (size_t) g * S.ssz; m.ssm_slot = (size_t) n_gdn_ * S.ssz;
    if (snap && n_snap_ > 0) { m.conv_snap = S.conv_snap[il]; m.ssm_snap = S.ssm_snap[il]; }
    return m;
}

void tp_model::fused_layer(int il, int T, bool keep_states, int nseq) {
    const int R = (int) rk_.size();
    const bool snap = keep_states && T > 1;
    {
        for (int r = 0; r < R; ++r) {
            rank_state & S = rk_[r];
            const layer_plan & P = S.lp[il];
            const int Hl = S.Hl, HVl = S.HVl, HKl = S.HKl, HKVl = S.HKVl, Cl = S.Cl;   // this rank's shares
            const size_t ssz = S.ssz, csz = S.csz;
            CK(cudaSetDevice(S.dev));
            norm_in(r, P.attn_norm, il > lb_, T);   // (the first layer of this process: S.x complete, nothing pending)
            // P100E_MERGE_INPROJ (step 6a): the two T = 1 in-projection launches of a mixed-type set (a16 weight + q8_0
            // weight, e.g. qkv|gate + beta|alpha) as one launch, same bits (fast::matvec_pair_t1)
            const auto & ins = T == 1 ? P.in_attn : P.in_attn_t;
            bool merged = false;
            if (T == 1 && merge_in_ && ins.size() == 2 && ins[0].outs.empty() && ins[1].outs.empty()) {
                merged = fast::matvec_pair_t1(ins[0].pk, S.aq, ins[0].y, ins[1].pk, S.cur, ins[1].y, S.s) ||
                         fast::matvec_pair_t1(ins[1].pk, S.aq, ins[1].y, ins[0].pk, S.cur, ins[0].y, S.s);
            }
            if (!merged && q8_fork_ && ins.size() == 2 && (ins[0].pk.layout == fast::LAYOUT_Q8R) != (ins[1].pk.layout == fast::LAYOUT_Q8R)) {
                // P100E_Q8_FORK: the small q8 weight on the side stream, overlapping the big GEMV; joined before the consumer
                const int jq = ins[0].pk.layout == fast::LAYOUT_Q8R ? 0 : 1;
                CK(cudaEventRecord(S.ev_q8f, S.s));
                CK(cudaStreamWaitEvent(S.s2, S.ev_q8f, 0));
                mv_run(ins[jq], S, T, S.s2);
                mv_run(ins[1 - jq], S, T);
                CK(cudaEventRecord(S.ev_q8j, S.s2));
                CK(cudaStreamWaitEvent(S.s, S.ev_q8j, 0));
            } else if (!merged) for (const auto & m : ins) mv_run(m, S, T);
            if (P.attn) {
                const float * q = T == 1 ? P.q : S.a, * kk = T == 1 ? P.k : S.e, * v = T == 1 ? P.v : S.g;
                // positions from S.pos on the device (cache rows pos[t]): no host n_past in the launches, so a captured
                // graph replays for every position. One sequence: its page table directly (the graph is keyed by its
                // slot), attention p0 = pos[0] (the pre-'sessions' launches); several sequences: per-row positions and
                // page tables (S.rtab), every row bitwise the T = 1 step of its sequence (fast/attn_dec2.cu;
                // tests/fast_sessions_test.cu)
                fast::kv_cache kv = nseq == 1 ? kv_seq(r, il) : kv_rows(r, il);
                if (!fast::attn_prep(q, kk, v, P.q_norm, P.k_norm, S.pos, S.c, S.d, kv, -1, T, Hl, HKVl, HD_,
                                     NROT_, rope_base_, eps_, S.s, 0, 0, 0, mrope(r)))
                    throw std::runtime_error("fast::attn_prep: unsupported shape");
                if (nseq == 1)
                {
                    if (!fast::attn_decode_dev(S.c, kv, S.b, T, S.pos, n_ctx_, Hl, 1.0f / sqrtf((float) HD_), S.s))
                        throw std::runtime_error("fast::attn_decode_dev: no kernel ran");
                } else if (!fast::attn_decode_rows(S.c, kv, S.b, T, S.pos, Hl, 1.0f / sqrtf((float) HD_), S.s))
                    throw std::runtime_error("fast::attn_decode_rows: unsupported shape");
                if (!fast::ew_quant(fast::EW_SIGMOID_MUL, S.b, S.d, S.b, S.aq, T, Hl * HD_, S.s)) {
                    k::sigmoid_mul(S.b, S.d, (int64_t) T * Hl * HD_, S.s);
                    fast::matvec_quantize(S.b, Hl * HD_, T, S.aq, S.s);
                }
                fast::matvec_prequantized(P.o, S.b, S.aq, part_out(r), T, S.s);
            } else {
                // T = 1: qkv / z / beta / alpha in place in the concatenated output; T > 1: dense [T][n] per-part buffers
                // qkv (a), gate z (b), beta (d), alpha (e). decay / beta land in place over alpha / beta.
                // One sequence: gdn_layer on the active sequence's state (S.conv_state / S.ssm_state = its slot; the graph
                // is keyed by the slot). keep_states: the kernels write the state after every token but the last into
                // the rollback snapshots (bit-identical to the per-token loop, tests/fast_gdn_verify_test.cu).
                // Several sequences: one launch pair for all (fast::gdn_layer_multi: rows / state slot / snapshot slots
                // per sequence from S.gseq; each bitwise its own gdn_layer call, tests/fast_sessions_test.cu)
                const float * xg = T == 1 ? P.qkv : S.a, * zg = T == 1 ? P.z : S.b;
                float * ag = T == 1 ? P.alpha : S.e, * bg = T == 1 ? P.beta : S.d;
                if (nseq == 1 && (!snap || gdn_batch_)) {
                    fast::gdn_layer(xg, ag, bg, zg, P.gw, S.conv_state[il], S.ssm_state[il], S.c, ag, bg, S.g, T, HKl, HVl,
                                    DK_, eps_, S.s, S.aq, snap ? S.conv_snap[il] : nullptr, snap ? S.ssm_snap[il] : nullptr,
                                    0, 0, 0, 0, gdn_fuse_ && T == 1 ? S.gdn_cnt : nullptr);
                } else if (snap && !gdn_batch_ && nseq == 1) {
                    for (int t = 0; t < T; ++t) {
                        fast::gdn_layer(S.a + (size_t) t * Cl, S.e + (size_t) t * HVl, S.d + (size_t) t * HVl,
                                        S.b + (size_t) t * HVl * DV_, P.gw, S.conv_state[il], S.ssm_state[il], S.c,
                                        S.e + (size_t) t * HVl, S.d + (size_t) t * HVl, S.g + (size_t) t * HVl * DV_,
                                        1, HKl, HVl, DK_, eps_, S.s);
                        if (t < T - 1) {
                            CK(cudaMemcpyAsync(S.ssm_snap[il] + t * ssz, S.ssm_state[il], ssz * 4, cudaMemcpyDeviceToDevice, S.s));
                            CK(cudaMemcpyAsync(S.conv_snap[il] + t * csz, S.conv_state[il], csz * 4, cudaMemcpyDeviceToDevice, S.s));
                        }
                    }
                    fast::matvec_quantize(S.g, (int64_t) HVl * DV_, T, S.aq, S.s);
                } else {
                    fast::gdn_layer_multi(xg, ag, bg, zg, P.gw, gdn_desc(r, il, T, nseq, snap), S.c, ag, bg, S.g, HKl, HVl,
                                          DK_, eps_, S.s, S.aq);
                }
                fast::matvec_prequantized(P.o, S.g, S.aq, part_out(r), T, S.s);
            }
        }
        ar_post(T);
        for (int r = 0; r < R; ++r) {
            rank_state & S = rk_[r];
            const layer_plan & P = S.lp[il];
            const int FFl = S.FFl;
            CK(cudaSetDevice(S.dev));
            norm_in(r, P.post_norm, true, T);
            for (const auto & m : T == 1 ? P.in_ffn : P.in_ffn_t) mv_run(m, S, T);
            float * fg = T == 1 ? P.fg : S.a, * fu = T == 1 ? P.fu : S.b;
            if (!fast::ew_quant(fast::EW_SILU_MUL, fg, fu, fg, S.aq, T, FFl, S.s)) {
                k::silu_mul(fg, fu, (int64_t) T * FFl, S.s);
                fast::matvec_quantize(fg, FFl, T, S.aq, S.s);
            }
            fast::matvec_prequantized(P.down, fg, S.aq, part_out(r), T, S.s);
        }
        ar_post(T);
    }
}

void tp_model::dbg_trace(const char * tag, int il, int T) {
    static const bool on = getenv("P100E_PF_DEBUG") != nullptr;
    if (!on) return;
    for (auto & S : rk_) { CK(cudaSetDevice(S.dev)); CK(cudaDeviceSynchronize()); }
    const int R = (int) rk_.size();
    size_t wide = 0;
    for (const auto & S : rk_) wide = std::max(wide, (size_t) T * std::max({S.FFl, 2 * S.Hl * HD_, S.Cl, E_}));
    auto h = [&](const float * d, size_t n) {
        std::vector<uint32_t> v(n);
        CK(cudaMemcpy(v.data(), d, n * 4, cudaMemcpyDeviceToHost));
        uint64_t x = 1469598103934665603ull;
        for (uint32_t u : v) x = (x ^ u) * 1099511628211ull;
        return (unsigned long long) x;
    };
    for (int r = 0; r < R; ++r) {
        rank_state & S = rk_[r];
        CK(cudaSetDevice(S.dev));
        printf("DBG L%02d %-4s r%d x %016llx a %016llx b %016llx g %016llx part %016llx\n", il, tag, r,
               h(S.x, (size_t) T * E_), h(S.a, std::min(wide, (size_t) T * 6144)), h(S.b, (size_t) T * 3072),
               h(S.g, (size_t) T * 512), h(S.part, (size_t) T * E_));
    }
}

// step descriptor of the active sequence -> every rank: positions p0..p0+T-1 (S.pos), its page table for every row
// (S.rtab), one delta-net sequence of T rows on its slot, snapshots from slot 0 (S.gseq). pinned: the copy is truly async
// (a pageable source is staged by the host first). Safe to rewrite: every forward / MTP run ends with a sync of every
// stream that read it.
// M-RoPE (step_mr_): the staged rows' rotary positions -> S.rpos ([3][max_batch]: t, h, w): image / staged rows their
// own (mm_, forward_mm), text rows of a sequence with images at row - its rope_delta.
void tp_model::set_pos(int p0, int T) {
    const int slot = sq().slot;
    if (slot < 0) throw std::runtime_error("set_pos: the active sequence is not resident");
    for (int i = 0; i < T; ++i) pos_h_[i] = p0 + i;
    for (int i = 0; i < std::min(T, (int) MAX_ROWS); ++i) desc_h_[DESC_RT + i] = slot * max_pages_;
    int32_t * g = desc_h_ + DESC_GS;
    g[0] = 0; g[1] = T; g[2] = slot; g[3] = 0;
    sq().snap0 = 0;
    const int B = max_batch_;
    bool mr = false;
    for (int i = 0; i < T; ++i) {
        if (mm_) {
            rpos_h_[i] = mm_[i].t; rpos_h_[B + i] = mm_[i].h; rpos_h_[2 * B + i] = mm_[i].w;
            mr = true;
        } else {
            mr = stage_rpos_text(i, sq(), p0 + i) || mr;
        }
    }
    step_mr_ = mr;
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        CK(cudaMemcpyAsync(S.desc, desc_h_, (size_t) (DESC_POS + T) * 4, cudaMemcpyHostToDevice, S.s));
    }
    if (mr) upload_rpos(T);
}

bool tp_model::stage_rpos_text(int i, const seq_t & q, int p) {
    const int B = max_batch_, rp = p - q.rope_delta;
    rpos_h_[i] = rpos_h_[B + i] = rpos_h_[2 * B + i] = rp;
    return q.rope_delta != 0;
}

void tp_model::upload_rpos(int T) {
    const int B = max_batch_;
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        for (int k = 0; k < 3; ++k)
            CK(cudaMemcpyAsync(S.rpos + (size_t) k * B, rpos_h_ + (size_t) k * B, (size_t) T * 4, cudaMemcpyHostToDevice, S.s));
    }
}

fast::mrope_pos tp_model::mrope(int r) const {
    fast::mrope_pos m;
    if (!mrope_now()) return m;
    m.p = rk_[r].rpos;
    m.stride = max_batch_;
    m.s0 = msec_[0]; m.s1 = msec_[1]; m.s2 = msec_[2]; m.sect = msect_;
    return m;
}

void tp_model::rope_rows(int r, float * x, int T, int H) {
    rank_state & S = rk_[r];
    if (mrope_now()) k::rope_mrope(x, mrope(r), T, H, HD_, NROT_, rope_base_, S.s);
    else k::rope_neox(x, S.pos, T, H, HD_, NROT_, rope_base_, S.s);
}

// Multimodal rows (see tp_model.h): the pieces become one row list (token or embedding row + M-RoPE position each),
// run in ubatch chunks (ubatch_chunks: the split a plain forward_n uses) through the prefill pipeline, or the classic
// path for <= 8 rows, with the chunk's rows staged in mm_ for set_pos / embed.
std::vector<float> tp_model::forward_mm(const std::vector<mm_in> & pieces,
                                        const std::function<void(int, const std::vector<int32_t> &)> & after_chunk) {
    static const bool pf_env = !getenv("P100E_PF") || atoi(getenv("P100E_PF")) != 0;
    static const int fast_maxt = getenv("P100E_FAST_MAXT") ? atoi(getenv("P100E_FAST_MAXT")) : 8;
    // debug (P100E_MROPE_SEQ=1): image rows at sequential 1-D positions like text (what a model without M-RoPE gets),
    // to show the 2-D positions matter
    static const bool seq_dbg = getenv("P100E_MROPE_SEQ") && atoi(getenv("P100E_MROPE_SEQ")) != 0;
    if (pnet_) throw std::runtime_error("forward_mm (images): not supported in the hybrid pipeline (--pipe-layers) yet");
    check_ok();
    std::vector<mm_row> rows;
    struct img_rows { int i0, i1, text_pos; };   // an image's rows [i0, i1) and the rotary position of the text after it
    std::vector<img_rows> imgs;
    // next_text[e]: the rotary position of the next text token if the context ended after the first e rows; -1 when
    // rows e - 1 and e belong to the same image (ending there would leave part of it)
    std::vector<int> next_text;
    int rn = sq().n_past - sq().rope_delta;   // rotary position of the next text token
    next_text.push_back(rn);
    for (const auto & p : pieces) {
        if (p.embd) {
            if (p.nx < 1 || p.ny < 1) throw std::runtime_error("forward_mm: empty image");
            if (p.n_embd != E_)
                throw std::runtime_error("forward_mm: image embedding rows are " + std::to_string(p.n_embd) +
                                         " wide, the model's are " + std::to_string(E_) + " (an mmproj of another model?)");
            const int i0 = (int) rows.size(), n = p.nx * p.ny;
            const int rn_after = rn + (seq_dbg ? n : std::max(p.nx, p.ny));
            for (int i = 0; i < n; ++i) {
                rows.push_back(seq_dbg ? mm_row{0, p.embd + (size_t) i * E_, rn + i, rn + i, rn + i}
                                       : mm_row{0, p.embd + (size_t) i * E_, rn, rn + i / p.nx, rn + i % p.nx});
                next_text.push_back(i + 1 < n ? -1 : rn_after);
            }
            rn = rn_after;
            imgs.push_back({i0, (int) rows.size(), rn});
        } else {
            for (int i = 0; i < p.n_tok; ++i) {
                rows.push_back(mm_row{p.tokens[i], nullptr, rn, rn, rn});
                ++rn;
                next_text.push_back(rn);
            }
        }
    }
    const int N = (int) rows.size();
    if (N < 1) throw std::runtime_error("forward_mm: empty prompt");
    if (!pool_ok_) throw std::runtime_error("forward: KV pool not allocated (alloc_pool)");
    if (sq().n_past + N > n_ctx_) throw std::runtime_error("context full");
    ensure_pages(active_, sq().n_past + N);
    rb_join();
    const int base = sq().n_past;
    // last_prefill_saturated() / last_prefill_wire_amax() cover this whole call, every chunk (as forward_n does): an
    // image prompt goes through the same f16 prefill wire
    const uint64_t sat0 = pf_sat_total_;
    pf_sat_last_ = false;
    float wamax = -1.0f;
    std::vector<float> logits;
    size_t ni = 0;   // the next image of imgs to record
    const std::vector<int> chunks = ubatch_chunks(N, 1);
    // the prefill ring stays ours across the chunks (and the after_chunk callbacks): no lend between them (idle_scratch)
    std::optional<lend_gate::busy_scope> ring_busy;
    if (pf_env && prefill_ok() && *std::max_element(chunks.begin(), chunks.end()) > fast_maxt) ring_busy.emplace(ring_gate_);
    int c0 = 0;
    for (int c : chunks) {
        std::vector<int32_t> toks(c);
        for (int i = 0; i < c; ++i) toks[i] = rows[c0 + i].tok;
        const int e = c0 + c;   // rows of this call in the context after the chunk
        mm_ = rows.data() + c0;
        bool fwd_done = false;
        try {
            const int p0 = sq().n_past;
            pf_wire_amax_ = -1.0f;
            logits = pf_env && c > fast_maxt && prefill_ok() ? forward_prefill(toks) : forward_classic(toks, 1, false);
            fwd_done = true;
            wamax = std::max(wamax, pf_wire_amax_);
            pf_sat_last_ = pf_sat_total_ != sat0;
            // the rows now in the context: completed images recorded, rope_delta_ at a boundary outside an image (inside
            // one it keeps its value: the next chunk's rows are staged with their own positions). Done before
            // after_chunk, so a throw there (or in a later chunk) leaves both matching the KV rows.
            for (; ni < imgs.size() && imgs[ni].i1 <= e; ++ni)
                sq().mmt.add(base + imgs[ni].i0, base + imgs[ni].i1, base + imgs[ni].i1 - imgs[ni].text_pos);
            if (next_text[e] >= 0) sq().rope_delta = sq().n_past - next_text[e];
            if (after_chunk) after_chunk(p0, toks);
        } catch (...) {
            mm_ = nullptr;
            pf_sat_last_ = pf_sat_total_ != sat0;
            pf_wire_amax_ = wamax;
            // a forward that failed part-way leaves its rows half written (KV rows, recurrent state); a chunk that ended
            // inside an image leaves part of it in the context: only reset() recovers from either
            if (!fwd_done || next_text[e] < 0) sq().mm_broken = true;
            throw;
        }
        mm_ = nullptr;
        c0 = e;
    }
    pf_sat_last_ = pf_sat_total_ != sat0;
    pf_wire_amax_ = wamax;
    return logits;
}

void tp_model::check_ok() const {
    if (active_ >= 0 && seqs_[active_].mm_broken)
        throw std::runtime_error("the context needs reset(): a multimodal forward failed part-way (rows half written or "
                                 "an image cut off)");
}

void tp_model::forward_image(const float * embd, int nx, int ny, int n_embd, const std::function<void(int, int)> & after_chunk) {
    mm_in p;
    p.embd = embd; p.nx = nx; p.ny = ny; p.n_embd = n_embd;
    if (!embd) throw std::runtime_error("forward_image: no embeddings");
    if (after_chunk) forward_mm({p}, [&](int p0, const std::vector<int32_t> & t) { after_chunk(p0, (int) t.size()); });
    else forward_mm({p});
}

void tp_model::embed(const std::vector<int32_t> & tokens) {
    const int T = (int) tokens.size();
    check_tokens(tokens, VOCAB_, "embed");   // (image rows carry token 0)
    const gguf_tensor * emb = g_.find("token_embd.weight");
    const size_t erow = emb->nbytes / emb->ne[1];
    if (mm_) {
        // multimodal rows: token rows gathered as below (image rows take token 0's row, overwritten next), then each run
        // of consecutive image rows copied as f32 straight into x on every GPU (the residual is replicated)
        bool any_tok = false;
        for (int i = 0; i < T; ++i) any_tok = any_tok || !mm_[i].embd;
        if (any_tok) {
            if (!emb_h_) CK(cudaMallocHost(&emb_h_, (size_t) max_batch_ * erow));
            for (int i = 0; i < T; ++i) memcpy(emb_h_ + (size_t) i * erow, emb->data + (size_t) tokens[i] * erow, erow);
            for (auto & S : rk_) {
                CK(cudaSetDevice(S.dev));
                CK(cudaMemcpyAsync(S.w_scratch, emb_h_, (size_t) T * erow, cudaMemcpyHostToDevice, S.s));
                dequant_f32(emb->type, S.w_scratch, S.x, (int64_t) T * E_, S.s);
            }
        }
        for (int i = 0; i < T;) {
            if (!mm_[i].embd) { ++i; continue; }
            int j = i + 1;
            while (j < T && mm_[j].embd == mm_[j - 1].embd + E_) ++j;
            for (auto & S : rk_) {
                CK(cudaSetDevice(S.dev));
                CK(cudaMemcpyAsync(S.x + (size_t) i * E_, mm_[i].embd, (size_t) (j - i) * E_ * 4, cudaMemcpyHostToDevice, S.s));
            }
            i = j;
        }
        return;
    }
    if (T > 8) {
        // long batches (per-op prompt rows, the MTP prompt catch-up): gathered on the host into pinned memory, one upload
        // per GPU (as forward_prefill, the same bytes). Safe to rewrite: every caller syncs its streams before the next
        if (!emb_h_) CK(cudaMallocHost(&emb_h_, (size_t) max_batch_ * erow));
        for (int i = 0; i < T; ++i) memcpy(emb_h_ + (size_t) i * erow, emb->data + (size_t) tokens[i] * erow, erow);
        for (auto & S : rk_) {
            CK(cudaSetDevice(S.dev));
            CK(cudaMemcpyAsync(S.w_scratch, emb_h_, (size_t) T * erow, cudaMemcpyHostToDevice, S.s));
            dequant_f32(emb->type, S.w_scratch, S.x, (int64_t) T * E_, S.s);
        }
        return;
    }
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        for (int i = 0; i < T; ++i) {
            CK(cudaMemcpyAsync((char *) S.w_scratch + i * erow, emb->data + (size_t) tokens[i] * erow, erow,
                               cudaMemcpyHostToDevice, S.s));
        }
        // dequant into x (scratch holds bytes at its start; dequant output goes to x)
        dequant_f32(emb->type, S.w_scratch, S.x, (int64_t) T * E_, S.s);
    }
}

// one decoder block on S.x through the per-op path (il == L_: the MTP / nextn block, full attention, own KV cache).
// keep_states (T > 1): the delta net also saves the state after every token but the last (rollback snapshots).
// kv_only (attention layers): only attn_norm + K / V (k_norm, rope) + the cache append, the same ops in the same order as
// the full block (identical cache bytes); no Q, attention, output projection or FFN, S.x unchanged (MTP prompt catch-up).
void tp_model::layer(int il, int p0, int T, bool keep_states, bool kv_only) {
    const int R = (int) rk_.size();
    const bool attn = il >= L_ || (il + 1) % full_attn_interval_ == 0;
    if (kv_only && !attn) throw std::runtime_error("layer: kv_only needs an attention layer");
    for (int r = 0; r < R; ++r) {
        rank_state & S = rk_[r];
        const int Hl = S.Hl, HKVl = S.HKVl, HKl = S.HKl, HVl = S.HVl, Cl = S.Cl;   // this rank's shares
        CK(cudaSetDevice(S.dev));
        norm(S.x, S.f[bn(il, "attn_norm.weight")], S.cur, T, E_, S.s);
        if (attn) {
            if (!kv_only) {
                matmul(r, bn(il, "attn_q.weight"), S.cur, S.a, T);
                k::split_q_gate(S.a, S.c, S.d, T, Hl, HD_, S.s);
                norm(S.c, S.f[bn(il, "attn_q_norm.weight")], S.c, (int64_t) T * Hl, HD_, S.s);
            }
            matmul(r, bn(il, "attn_k.weight"), S.cur, S.e, T);
            norm(S.e, S.f[bn(il, "attn_k_norm.weight")], S.e, (int64_t) T * HKVl, HD_, S.s);
            matmul(r, bn(il, "attn_v.weight"), S.cur, S.g, T);
            if (!kv_only) rope_rows(r, S.c, T, Hl);
            rope_rows(r, S.e, T, HKVl);
            if (kv_only) {
                if (use_fast_) { fast::kv_cache kv = kv_seq(r, il); fast::kv_append(kv, S.e, S.g, p0, T, S.s); }
                else k::kv_store(S.e, S.g, S.kc[il], S.vc[il], T, p0, HKVl * HD_, S.s);
                continue;
            }
            if (use_fast_) {
                fast::kv_cache kv = kv_seq(r, il);
                fast::kv_append(kv, S.e, S.g, p0, T, S.s);
                // one dec2 call for T = 5..8 (P100E_PF_TINY_ATTN, default on): the cache is read once instead of twice;
                // every row is bitwise the T = 1 computation at its position, so the grouping does not change the output
                if (!(pf_tiny_attn() && T > 4 && T <= fast::ATTN_MAX_ROWS &&
                      fast::attn_decode2(S.c, kv, S.b, T, p0, nullptr, Hl, 1.0f / sqrtf((float) HD_), S.s)))
                for (int t0 = 0; t0 < T; t0 += 4) {   // decode kernel takes up to 4 queries at a time
                    const int nt = std::min(4, T - t0);
                    if (!fast::attn_decode(S.c + (size_t) t0 * Hl * HD_, kv, S.b + (size_t) t0 * Hl * HD_, nt,
                                           p0 + t0, Hl, 1.0f / sqrtf((float) HD_), S.s))
                        throw std::runtime_error("fast::attn_decode: no kernel ran");
                }
            } else {
                k::kv_store(S.e, S.g, S.kc[il], S.vc[il], T, p0, HKVl * HD_, S.s);
                k::attention(S.c, S.kc[il], S.vc[il], S.b, T, p0, Hl, HKVl, HD_, 1.0f / sqrtf((float) HD_), S.s);
            }
            k::sigmoid_mul(S.b, S.d, (int64_t) T * Hl * HD_, S.s);
            matmul(r, bn(il, "attn_output.weight"), S.b, S.part, T);
        } else {
            matmul(r, bn(il, "attn_qkv.weight"), S.cur, S.a, T);
            matmul(r, bn(il, "attn_gate.weight"), S.cur, S.b, T);
            matmul(r, bn(il, "ssm_beta.weight"), S.cur, S.d, T);
            matmul(r, bn(il, "ssm_alpha.weight"), S.cur, S.e, T);
            if (use_fast_) {
                fast::GdnWeights gw{S.f[bn(il, "ssm_conv1d.weight")], S.f[bn(il, "ssm_dt.bias")], S.f[bn(il, "ssm_a")],
                                    S.f[bn(il, "ssm_norm.weight")]};
                if (keep_states && T > 1 && gdn_batch_) {
                    // T tokens in one call, the kernels saving the state after each token but the last (for rollback)
                    if (T - 1 > n_snap_) throw std::runtime_error("keep_states: batch larger than enable_rollback()");
                    fast::gdn_layer(S.a, S.e, S.d, S.b, gw, S.conv_state[il], S.ssm_state[il], S.c, S.e, S.d, S.g,
                                    T, HKl, HVl, DK_, eps_, S.s, nullptr, S.conv_snap[il], S.ssm_snap[il]);
                } else if (keep_states && T > 1) {
                    // one token at a time, saving the state after each token but the last (for rollback)
                    if (T - 1 > n_snap_) throw std::runtime_error("keep_states: batch larger than enable_rollback()");
                    const size_t ssz = (size_t) HVl * DV_ * DK_, csz = (size_t) Cl * (CONV_K_ - 1);
                    for (int t = 0; t < T; ++t) {
                        fast::gdn_layer(S.a + (size_t) t * Cl, S.e + (size_t) t * HVl, S.d + (size_t) t * HVl,
                                        S.b + (size_t) t * HVl * DV_, gw, S.conv_state[il], S.ssm_state[il], S.c,
                                        S.e + (size_t) t * HVl, S.d + (size_t) t * HVl, S.g + (size_t) t * HVl * DV_,
                                        1, HKl, HVl, DK_, eps_, S.s);
                        if (t < T - 1) {
                            CK(cudaMemcpyAsync(S.ssm_snap[il] + t * ssz, S.ssm_state[il], ssz * 4, cudaMemcpyDeviceToDevice, S.s));
                            CK(cudaMemcpyAsync(S.conv_snap[il] + t * csz, S.conv_state[il], csz * 4, cudaMemcpyDeviceToDevice, S.s));
                        }
                    }
                } else {
                    // x=qkv (a), alpha (e), beta (d), gate z (b); y scratch c; decay/beta alias e/d; out g
                    fast::gdn_layer(S.a, S.e, S.d, S.b, gw, S.conv_state[il], S.ssm_state[il], S.c, S.e, S.d, S.g,
                                    T, HKl, HVl, DK_, eps_, S.s);
                }
            } else {
                if (keep_states && T > 1) throw std::runtime_error("keep_states needs the fast path");
                k::gdn_gates(S.e, S.d, S.f[bn(il, "ssm_dt.bias")], S.f[bn(il, "ssm_a")], T, HVl, S.s);
                k::conv1d_silu(S.a, S.conv_state[il], S.f[bn(il, "ssm_conv1d.weight")], S.c, T, Cl, S.s);
                k::gdn_recurrent(S.c, S.e, S.d, S.ssm_state[il], S.g, T, HKl, HVl, DK_, DV_, eps_, S.s);
                k::rmsnorm_gated(S.g, S.f[bn(il, "ssm_norm.weight")], S.b, S.g, (int64_t) T * HVl, DV_, eps_, S.s);
            }
            matmul(r, bn(il, "ssm_out.weight"), S.g, S.part, T);
        }
    }
    if (kv_only) return;
    dbg_trace("mix", il, T);
    allreduce_add(T);
    dbg_trace("ar1", il, T);
    for (int r = 0; r < R; ++r) {
        rank_state & S = rk_[r];
        CK(cudaSetDevice(S.dev));
        norm(S.x, S.f[bn(il, "post_attention_norm.weight")], S.cur, T, E_, S.s);
        matmul(r, bn(il, "ffn_gate.weight"), S.cur, S.a, T);
        matmul(r, bn(il, "ffn_up.weight"), S.cur, S.b, T);
        k::silu_mul(S.a, S.b, (int64_t) T * S.FFl, S.s);
        matmul(r, bn(il, "ffn_down.weight"), S.a, S.part, T);
    }
    allreduce_add(T);
    dbg_trace("ar2", il, T);
}

std::vector<float> tp_model::forward(const std::vector<int32_t> & tokens) { return forward_n(tokens, 1, false); }

std::vector<int> tp_model::ubatch_chunks(int T, int n_logits) const {
    std::vector<int> ch;
    int i = 0;
    while (T - i > max_batch_) {
        const int c = std::min(max_batch_, T - i - n_logits);
        ch.push_back(c);
        i += c;
    }
    if (T - i > 0) ch.push_back(T - i);
    return ch;
}

std::vector<float> tp_model::forward_n(const std::vector<int32_t> & tokens, int n_logits, bool keep_states) {
    static const bool pf_env = !getenv("P100E_PF") || atoi(getenv("P100E_PF")) != 0;
    static const int fast_maxt = getenv("P100E_FAST_MAXT") ? atoi(getenv("P100E_FAST_MAXT")) : 8;
    check_ok();
    const int T = (int) tokens.size();
    if (!pool_ok_) throw std::runtime_error("forward: KV pool not allocated (alloc_pool)");
    // last_prefill_saturated() covers this whole call (every ubatch chunk): saturations counted since here
    const uint64_t sat0 = pf_sat_total_;
    pf_sat_last_ = false;
    pf_wire_amax_ = -1.0f;   // last_prefill_wire_amax(): the maximum over this call's prefills
    if (T > max_batch_ && !keep_states && n_logits >= 1 && n_logits <= 8 && sq().n_past + T <= n_ctx_) {
        // ubatch chunking: chunks of max_batch tokens in order; the last one keeps >= n_logits rows
        const std::vector<int> ch = ubatch_chunks(T, n_logits);
        // the prefill ring stays ours across the chunks: no lend between them (idle_scratch)
        std::optional<lend_gate::busy_scope> ring_busy;
        if (pf_env && prefill_ok()) ring_busy.emplace(ring_gate_);
        int i = 0;
        float amax = -1.0f;
        for (size_t j = 0; j + 1 < ch.size(); ++j) {
            forward_n(std::vector<int32_t>(tokens.begin() + i, tokens.begin() + i + ch[j]), 1, false);
            amax = std::max(amax, pf_wire_amax_);
            i += ch[j];
        }
        std::vector<float> l = forward_n(std::vector<int32_t>(tokens.begin() + i, tokens.end()), n_logits, false);
        pf_sat_last_ = pf_sat_total_ != sat0;
        pf_wire_amax_ = std::max(amax, pf_wire_amax_);
        return l;
    }
    if (T < 1 || T > max_batch_) throw std::runtime_error("batch size out of range");
    if (sq().n_past + T > n_ctx_) throw std::runtime_error("context full");
    if (n_logits < 1 || n_logits > std::min(8, T)) throw std::runtime_error("n_logits out of range");
    rb_join();
    ensure_pages(active_, sq().n_past + T);
    // the prefill pipeline returns the last row's logits and keeps no rollback snapshots
    std::vector<float> logits = pf_env && n_logits == 1 && !keep_states && T > fast_maxt && prefill_ok()
                                    ? forward_prefill(tokens) : forward_classic(tokens, n_logits, keep_states);
    pf_sat_last_ = pf_sat_total_ != sat0;
    static const bool dbg_hash = getenv("P100E_LOGITS_HASH") != nullptr;
    if (dbg_hash) {   // debug: exact fingerprint of the logits, to A/B paths that should be bit-identical
        for (int j = 0; j < n_logits; ++j) {
            uint64_t h = 1469598103934665603ull;
            double sum = 0;
            for (size_t i = 0; i < (size_t) VOCAB_; ++i) {
                const float v = logits[(size_t) j * VOCAB_ + i];
                uint32_t u; memcpy(&u, &v, 4); h = (h ^ u) * 1099511628211ull; sum += v;
            }
            if (n_logits == 1) printf("LOGITS T=%d n_past=%d hash %016llx sum %.6f\n", T, sq().n_past, (unsigned long long) h, sum);
            else printf("LOGITS T=%d n_past=%d row %d hash %016llx sum %.6f\n", T, sq().n_past, j, (unsigned long long) h, sum);
        }
    }
    return logits;
}

std::vector<int32_t> tp_model::forward_ids(const std::vector<int32_t> & tokens, int n_rows, bool keep_states) {
    check_ok();
    const int T = (int) tokens.size();
    if (!pool_ok_) throw std::runtime_error("forward: KV pool not allocated (alloc_pool)");
    if (T < 1 || T > max_batch_) throw std::runtime_error("batch size out of range");
    if (sq().n_past + T > n_ctx_) throw std::runtime_error("context full");
    if (n_rows < 1 || n_rows > std::min(8, T)) throw std::runtime_error("n_rows out of range");
    rb_join();
    ensure_pages(active_, sq().n_past + T);
    std::vector<int32_t> ids;
    forward_classic(tokens, n_rows, keep_states, &ids);
    return ids;
}

void tp_model::argmax_launch(int r, const float * logits, int n_rows) {
    rank_state & S = rk_[r];
    if (n_rows < 1 || n_rows > 8) throw std::runtime_error("argmax: 1..8 rows");
    argmax_rows_k<<<n_rows, 1024, 0, S.s>>>(logits, S.VL, (amax_pair *) S.amax);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync((amax_pair *) amax_h_ + (size_t) r * 8, S.amax, (size_t) n_rows * sizeof(amax_pair),
                       cudaMemcpyDeviceToHost, S.s));
}

std::vector<int32_t> tp_model::argmax_collect(int n_rows) const {
    if (net_) {   // every global rank's pairs (net_gather_amax), vocab offsets from the split table
        const amax_pair * p = (const amax_pair *) amax_net_.data();
        std::vector<int32_t> ids(n_rows);
        for (int j = 0; j < n_rows; ++j) {
            // the other nodes' pairs come off the link: each index must lie in its rank's vocab slice
            for (int q = 0; q < R_; ++q) {
                const int32_t i = p[(size_t) q * 8 + j].i;
                if (i < 0 || (int64_t) i >= (int64_t) split_.rk[q].voc.n)
                    throw std::runtime_error("net: argmax index " + std::to_string(i) + " from rank " + std::to_string(q) +
                                             " outside its vocab slice");
            }
            amax_pair best = p[j];
            best.i += (int32_t) split_.rk[0].voc.b;
            for (int q = 1; q < R_; ++q) {
                const amax_pair c = p[(size_t) q * 8 + j];
                if (c.v > best.v) best = amax_pair{c.v, c.i + (int32_t) split_.rk[q].voc.b};
            }
            ids[j] = best.i;
        }
        return ids;
    }
    const amax_pair * p = (const amax_pair *) amax_h_;
    std::vector<int32_t> ids(n_rows);
    for (int j = 0; j < n_rows; ++j) {
        amax_pair best = p[j];
        best.i += (int32_t) rk_[0].v0;
        for (int r = 1; r < (int) rk_.size(); ++r) {   // a later slice wins only with a strictly larger value
            const amax_pair q = p[(size_t) r * 8 + j];
            if (q.v > best.v) best = amax_pair{q.v, q.i + (int32_t) rk_[r].v0};
        }
        ids[j] = best.i;
    }
    return ids;
}

std::vector<float> tp_model::forward_classic(const std::vector<int32_t> & tokens, int n_logits, bool keep_states,
                                             std::vector<int32_t> * ids) {
    static const bool fuse_multi = !getenv("P100E_FUSE_VERIFY") || atoi(getenv("P100E_FUSE_VERIFY")) != 0;
    const int T = (int) tokens.size();
    const int R = (int) rk_.size();

    set_pos(sq().n_past, T);
    if (pnet_ && !has_embd()) pipe_recv_x(T);   // pipe node 1: node 0's residual rows after its layers (blocks for them)
    else embed(tokens);

    // fused: T = 1 decode, or an MTP verify batch (T <= 4, logits for every row)
    const bool fused = use_fast_ && fuse_ && plan_ok_ && (T == 1 || (fuse_multi && T <= 4 && n_logits == T));
    const bool keep = keep_states && T > 1;   // T = 1 keeps no snapshots either way
    bool capture = false, net_capt = false;
    if (fused && net_) {
        // net mode: every fused step starts at allreduce parity 0 (the streams are idle after the
        // previous forward's sync), so the segment graphs bake fixed peer slots; T = 1 replays them (P100E_NET_GRAPH)
        ar_buf_ = ar_cur_ = 0;
        const bool g = net_graph_ && !net_graph_failed_ && T == 1 && !keep;
        net_gset * NG = g ? net_find(T) : nullptr;
        if (NG) net_launch(*NG, T);
        else { fused_body(T, keep); net_capt = g; }
    } else if (fused) {
        step_graph * G = T <= graph_maxt_ ? find_graph(T, keep) : nullptr;
        if (G) {
            launch_graph(*G);
        } else {
            fused_body(T, keep);
            capture = T <= graph_maxt_ && !graph_failed_;   // this eager run was the warm-up: capture after the sync
        }
    } else {
        for (int il = lb_; il < le_; ++il) layer(il, sq().n_past, T, keep_states);   // (pipe mode: this node's layers)
    }

    if (!has_head()) {
        // pipe node 0: S.x holds the residual after its last layer (fused: folded by fused_body), every
        // rank the same replica; rank 0's rows go to node 1, which returns the greedy ids or the logit rows
        pipe_x_d2h(T);
        for (int r = 0; r < R; ++r) { CK(cudaSetDevice(rk_[r].dev)); CK(cudaStreamSynchronize(rk_[r].s)); }
        check_ar();
        pipe_send_x(T, 0);
        sq().n_past += T;   // (before the result handoff, as node 1 counts it there)
        sq().toks.insert(sq().toks.end(), tokens.begin(), tokens.end());
        last_T_ = T;
        pipe_result(n_logits, ids);
        if (capture) capture_graph(T, keep);
        if (ids) return {};
        return std::vector<float>(logits_h_, logits_h_ + (size_t) n_logits * VOCAB_);
    }

    // all ranks' lm_head first, then the copies (into pinned memory, so they are async and overlap): the old
    // per-rank norm + lm_head + pageable D2H blocked the host on each copy and serialized the two lm_heads (~1 ms/token)
    for (int r = 0; r < R && !fused; ++r) {
        rank_state & S = rk_[r];
        CK(cudaSetDevice(S.dev));
        // hn rows (all of them with the MTP head loaded, else the logit rows); logits for the last n_logits rows
        const int hr = n_nextn_ > 0 ? T : n_logits;
        norm(S.x + (int64_t) (T - hr) * E_, S.f["output_norm.weight"], S.hn + (int64_t) (T - hr) * E_, hr, E_, S.s);
        matmul(r, "output.weight", S.hn + (int64_t) (T - n_logits) * E_, S.logits, n_logits);
    }
    const int l0 = fused ? T - n_logits : 0;   // first logit row to return in S.logits
    for (int r = 0; r < R; ++r) {
        rank_state & S = rk_[r];
        const int VL = S.VL;
        CK(cudaSetDevice(S.dev));
        if (ids) { argmax_launch(r, S.logits + (size_t) l0 * VL, n_logits); continue; }
        for (int j = 0; j < n_logits; ++j)
            CK(cudaMemcpyAsync(logits_h_ + (size_t) j * VOCAB_ + (size_t) S.v0, S.logits + (size_t) (l0 + j) * VL,
                               (size_t) VL * 4, cudaMemcpyDeviceToHost, S.s));
    }
    for (int r = 0; r < R; ++r) { CK(cudaSetDevice(rk_[r].dev)); CK(cudaStreamSynchronize(rk_[r].s)); }
    check_ar();   // a P2P allreduce wait of this step (or of a draft run since the last step) timed out: abort
    if (net_) {   // the other nodes' vocab slices: argmax pairs (ids) or logit rows
        if (ids) net_gather_amax(n_logits);
        else net_gather_logits(n_logits);
    }
    sq().n_past += T;
    sq().toks.insert(sq().toks.end(), tokens.begin(), tokens.end());
    last_T_ = T;
    if (capture) capture_graph(T, keep);   // records only (nothing runs): the results above are untouched
    if (net_capt) net_capture(T);          // likewise (no network traffic either)
    if (ids) {
        *ids = argmax_collect(n_logits);
        if (pnet_) pipe_result(n_logits, ids);   // pipe node 1: the same ids to node 0
        return {};
    }
    if (pnet_) pipe_result(n_logits, nullptr);  // pipe node 1: the logit rows to node 0
    return std::vector<float>(logits_h_, logits_h_ + (size_t) n_logits * VOCAB_);
}

// fused step body on the staged inputs (S.x rows, S.pos): the layers, then output norm + lm_head for all T rows
// (S.logits [T][VL]) and hn = output_norm(x) of every row when the MTP head is loaded
void tp_model::fused_body(int T, bool keep_states, int nseq) {
    decode_layers(T, keep_states, nseq);
    if (!has_head()) {
        // pipe node 0: no LM head. Fold the last FFN partials into x with the add the next layer's
        // norm_in would do (rank 0's partial first, the P2P wait included), so S.x is the complete residual, bitwise what
        // a one-process run of this split holds entering layer le_. The norm it also writes (S.cur / S.aq) is unused.
        for (int r = 0; r < (int) rk_.size(); ++r) {
            CK(cudaSetDevice(rk_[r].dev));
            norm_in(r, rk_[r].lp[le_ - 1].post_norm, true, T);
        }
        return;
    }
    for (int r = 0; r < (int) rk_.size(); ++r) {
        rank_state & S = rk_[r];
        CK(cudaSetDevice(S.dev));
        norm_in(r, S.f["output_norm.weight"], true, T);
        fast::matvec_prequantized(S.w["output.weight"].pk, S.cur, S.aq, S.logits, T, S.s);
        // hn = output_norm(x) of every row: the MTP head's h input
        if (n_nextn_ > 0) CK(cudaMemcpyAsync(S.hn, S.cur, (size_t) T * E_ * 4, cudaMemcpyDeviceToDevice, S.s));
    }
}

tp_model::step_graph * tp_model::find_graph(int T, bool keep, int nseq) {
    const int slot = nseq == 1 ? sq().slot : -1;
    // the M-RoPE variant of attn_prep is baked into a captured graph: separate graphs with / without it
    for (auto & G : graphs_)
        if (G.T == T && G.keep == keep && G.nseq == nseq && G.slot == slot && G.mr == mrope_now() && G.sig == sig_fuse_ && G.q8f == q8_fork_)
            return &G;
    return nullptr;
}

void tp_model::drop_graphs(bool keep_only) {
    std::vector<step_graph> kept;
    for (auto & G : graphs_) {
        if (keep_only && !G.keep) { kept.push_back(G); continue; }
        if (G.ge) cudaGraphExecDestroy(G.ge);
        if (G.g) cudaGraphDestroy(G.g);
    }
    graphs_.swap(kept);
    for (auto & N : net_gsets_)   // net segment graphs: all of them
        for (auto & G : N.seg) { if (G.ge) cudaGraphExecDestroy(G.ge); if (G.g) cudaGraphDestroy(G.g); }
    net_gsets_.clear();
}

// Capture fused_body(T, keep) of every rank into one multi-device graph: rank 0's stream is the origin, the other
// ranks' streams join through ev_fork_ and rejoin through their ev_join, so the only cross-GPU edges are the fork
// and the join. The allreduces inside are the P2P flag protocol (kernels: signal + spin-wait on a device counter),
// which needs both GPUs' halves running at once - they are independent chains between fork and join. Capture
// records launches without running them; the host-side allreduce bookkeeping (ar_buf_ / ar_cur_) is restored after
// and the values a run of the body leaves are stored for launch_graph.
bool tp_model::capture_graph(int T, bool keep, int nseq) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    step_graph G;
    G.T = T; G.keep = keep; G.nseq = nseq; G.slot = nseq == 1 ? sq().slot : -1; G.mr = mrope_now();
    std::string err;
    if (!capture_body([&] { fused_body(T, keep, nseq); }, G, err)) {
        fprintf(stderr, "CUDA graph (T=%d): %s, eager launches from now on\n", T, err.c_str());
        graph_failed_ = true;
        return false;
    }
    graphs_.push_back(G);
    static const bool quiet = getenv("P100E_GRAPH_QUIET") != nullptr;
    if (!quiet)
        printf("CUDA graph T=%d%s%s: %zu nodes on %d GPU(s), captured + instantiated in %.1f ms\n", T,
               nseq > 1 ? (" (" + std::to_string(nseq) + " sequences)").c_str() : "", keep ? " (keep states)" : "",
               G.nodes, (int) rk_.size(), std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    return true;
}

bool tp_model::capture_body(const std::function<void()> & body, step_graph & G, std::string & err) {
    const int R = (int) rk_.size();
    auto ok = [&](cudaError_t e, const char * what) {
        if (e != cudaSuccess && err.empty()) err = std::string(what) + ": " + cudaGetErrorString(e);
        return e == cudaSuccess;
    };
    const int buf0 = ar_buf_, cur0 = ar_cur_;
    const int sig0 = sig_last_;
    if (sig_last_ < 0) sig_last_ = 1;   // the graph assumes this word holds the latest (launch_graph equalizes if not)
    G.q8f = q8_fork_; G.sig = sig_fuse_; G.sig_in = sig_last_;
    CK(cudaSetDevice(rk_[0].dev));
    bool began = ok(cudaStreamBeginCapture(rk_[0].s, cudaStreamCaptureModeRelaxed), "begin capture");
    if (began) {
        if (R > 1) ok(cudaEventRecord(ev_fork_, rk_[0].s), "fork record");
        for (int r = 1; r < R; ++r) { CK(cudaSetDevice(rk_[r].dev)); ok(cudaStreamWaitEvent(rk_[r].s, ev_fork_, 0), "fork wait"); }
        if (err.empty()) {
            try {
                body();
            } catch (const std::exception & e) {
                if (err.empty()) err = e.what();
            }
        }
        for (int r = 1; r < R; ++r) {
            CK(cudaSetDevice(rk_[r].dev));
            ok(cudaEventRecord(rk_[r].ev_join, rk_[r].s), "join record");
            CK(cudaSetDevice(rk_[0].dev));
            ok(cudaStreamWaitEvent(rk_[0].s, rk_[r].ev_join, 0), "join wait");
        }
    }
    G.buf_out = ar_buf_; G.cur_out = ar_cur_;
    ar_buf_ = buf0; ar_cur_ = cur0;
    G.sig_out = sig_last_;
    sig_last_ = sig0;
    if (sig_pend_[0] || sig_pend_[1]) {
        sig_pend_[0] = sig_pend_[1] = false;
        if (err.empty()) err = "captured body left an allreduce signal owed";
    }
    CK(cudaSetDevice(rk_[0].dev));
    if (began) ok(cudaStreamEndCapture(rk_[0].s, &G.g), "end capture");
    if (err.empty() && G.g) ok(cudaGraphInstantiateWithFlags(&G.ge, G.g, 0), "instantiate");
    if (err.empty() && G.ge) ok(cudaGraphUpload(G.ge, rk_[0].s), "upload");
    if (err.empty() && G.ge) ok(cudaStreamSynchronize(rk_[0].s), "upload sync");
    if (!err.empty() || !G.ge) {
        if (err.empty()) err = "no graph";
        if (G.ge) cudaGraphExecDestroy(G.ge);
        if (G.g) cudaGraphDestroy(G.g);
        G.ge = nullptr; G.g = nullptr;
        for (auto & S : rk_) { cudaSetDevice(S.dev); cudaGetLastError(); }
        return false;
    }
    cudaGraphGetNodes(G.g, nullptr, &G.nodes);
    return true;
}

// Replay a captured body. The eager inputs before it (set_pos / embed, and e.g. rollback() copies on the other ranks'
// streams) must precede every rank's part of the graph: rank r's stream joins rank 0's before the launch. After it the
// other ranks' streams wait for the graph, so the result copies queued on them follow it.
void tp_model::launch_graph(const step_graph & G) {
    const int R = (int) rk_.size();
    if (G.sig_in >= 0) sig_need(G.sig_in);   // on each rank's stream, before the join below
    for (int r = 1; r < R; ++r) {
        CK(cudaSetDevice(rk_[r].dev));
        CK(cudaEventRecord(rk_[r].ev_join, rk_[r].s));
        CK(cudaSetDevice(rk_[0].dev));
        CK(cudaStreamWaitEvent(rk_[0].s, rk_[r].ev_join, 0));
    }
    CK(cudaSetDevice(rk_[0].dev));
    CK(cudaGraphLaunch(G.ge, rk_[0].s));
    if (R > 1) {
        CK(cudaEventRecord(ev_fork_, rk_[0].s));
        for (int r = 1; r < R; ++r) { CK(cudaSetDevice(rk_[r].dev)); CK(cudaStreamWaitEvent(rk_[r].s, ev_fork_, 0)); }
    }
    if (G.sig_in >= 0) sig_last_ = G.sig_out;
    ar_buf_ = G.buf_out; ar_cur_ = G.cur_out;   // as if the body had been enqueued (parity only matters within a step)
}

// =====================================================================================================================
// Prefill pipeline (T > 8). Same arithmetic as forward_classic's prefill (gemm_packed = f16 unpack + prescale + fold
// GEMM; k::add x2 + fast::rmsnorm; k::silu_mul), so with P100E_PF_ATTN=0 the logits are bit-identical; the default
// attention is fast::attn_prefill (fp32, different summation order than the looped attn_decode).
//  - tokens are split into up to P100E_PF_CHUNKS (default 4) chunks of whole 128-token tiles, each on its own stream:
//    the row-parallel GEMM of a chunk, its peer copy, the add + norm and the next GEMMs of that chunk run while the
//    other chunks compute, so the PCIe exchange (~4.4 GB/s each way when both GPUs send) hides behind the GEMMs.
//    Only the mixer (GDN scan / attention) needs all tokens: one join + fork per layer.
//  - weights are unpacked to f16 on a side stream into a ring buffer ahead of use (P100E_PF_RING_MB, default 640);
//    the unpack kernels co-run beside the GEMM (the fold kernel leaves room for one small block per SM).
//  - activations are prescaled once per input (fused into the add + norm / SwiGLU) and every weight reading that
//    input runs in one segmented fold launch (q/k/v, qkv/gate/beta/alpha, gate/up): no 1-tile launches for the
//    24-row ssm_alpha/beta or the 512-row attn_k/v.
// Switches, read at every forward_prefill (so p100e-run --ab can interleave them in one process; ports of the fork's
// prefill exchange / fold GEMM):
//  - P100E_PF_WIRE16 (default on: quality-neutral vs BF16; =0 = exact f32 wire; fork GGML_PF_XCHG_F16 / GGML_CUDA_GEMM_FOLD=2): the row-parallel GEMMs (attn_output,
//    ssm_out, ffn_down) round their partials to f16 in the fold epilogue and write them as half, the exchange ships
//    half (exact: the wire carries what the producer made), the add widens them. Changes numerics once (the partials'
//    f16 rounding; both replicas add the same values). Saturation guard: a partial beyond the f16 range is clamped and
//    counted; the forward that saturated reports it at once (last_prefill_saturated(): its outputs / state carry the
//    clamped values, the caller redoes it and never caches it) and every later forward uses the f32 wire (warning).
//  - P100E_PF_WIRE_STATS=1: measure the largest |partial| of every exchange (an extra reduction per chunk, for
//    measurements only) and print one PFWIRE line per prefill: the maximum over both GPUs, where it was, and its
//    margin to 65504 (last_prefill_wire_amax()). With P100E_PF_WIRE16=0 the values are the exact f32 partials (what
//    the f16 wire would round); with the f16 wire, the rounded halves (65504 when clamped).
//  - P100E_PF_TEST_SAT=N (fault injection, tests): the next f16-wire prefill reports N extra saturated partials, so
//    last_prefill_saturated() / the switch to the f32 wire can be exercised without an overflowing prompt.
//  - P100E_PF_HOST_XCHG (P21, fork GGML_CUDA_AR_PASCAL copy-engine path): the exchange goes D2H into pinned host memory
//    on the sender and H2D on the receiver instead of a direct P2P copy.
//  - P100E_PF_RASTER (P3): fold block order, groups of N row tiles (0 = row tiles fastest). Same bits.
//  - P100E_PF_PRESCALE_V (P9): one-read prescale of the out-projection input. Same bits.
//  - P100E_UNPACK (P14): 1 = fork-style direct dequant-to-f16 kernels, 0 = the shared-memory unpack. Same bits.
//  - P100E_PF_SASS (P4, fork GGML_CUDA_GEMM_FOLD_SASS): the 128-token-tile fold launches with f32 outputs run the
//    register-bank-fixed sm_60 cubin of the same kernel (fast/gemm_sass_fold8.h): 1 (default) bank-fix variant 0,
//    2 the wide variant, 3 the unpatched cubin (debug), 0 the compiled kernel. Same bits (cc 6.0 only). The build
//    checks that the cubin comes from the current kernel (cmake/fast_gemm.cmake); a build that cannot run that check
//    (no Python) defines P100E_PF_SASS_DEFAULT=0.
//  - P100E_PF_FUSEDQ=1 (default off): q4kr weights are not unpacked to the f16 ring; the fold
//    GEMM reads them packed and dequantizes in its tile loads (fast::fold_opts::q4k). Same bits.
//  - P100E_GDN_CHUNKED (P25, fork GGML_CUDA_GDN_CHUNKED): 1 = the chunked delta net (fast/gdn_chunked.cu, fp32), 2 = its
//    fp64 mode, 0 = the sequential scan. Different rounding (see gdn_chunked.cuh); scratch = w_scratch.
// =====================================================================================================================

constexpr int PF_MAXCH = 8;
constexpr int PF_TILE = 128;
#ifndef P100E_PF_SASS_DEFAULT
#define P100E_PF_SASS_DEFAULT 1
#endif

struct pf_wentry {
    const dev_weight * w = nullptr;
    size_t off = 0, n = 0;                 // halves in the ring
    cudaEvent_t ready = nullptr;           // unpack done (weight stream)
    std::vector<cudaEvent_t> done;         // reads done (one per chunk stream)
    bool consumed = false;                 // all reads enqueued (done events recorded)
    bool fused = false;                    // P100E_PF_FUSEDQ: read packed by the GEMM, no ring space, no ready event
};

struct pf_rank {
    int dev = 0;
    cudaStream_t ws = nullptr;                     // weight unpack stream
    cudaStream_t cs[PF_MAXCH] = {};                // chunk streams
    cudaEvent_t ev_fork = nullptr, ev_wjoin = nullptr, ev_join[PF_MAXCH] = {};
    cudaEvent_t ev_sent[2][PF_MAXCH] = {}, ev_used[2][PF_MAXCH] = {};
    // chunk c's reads of the shared activation buffers are enqueued: after the out-projection prescale (ev_pre) and
    // after the SwiGLU prescale (ev_glu). See cross_wait in forward_prefill.
    cudaEvent_t ev_pre[PF_MAXCH] = {}, ev_glu[PF_MAXCH] = {};
    half * ring = nullptr;
    size_t ring_cap = 0, head = 0;                 // halves
    std::deque<pf_wentry> live;
    std::vector<cudaEvent_t> pool;                 // free events
    half * x16 = nullptr;                          // [max_batch][ldx] prescaled activations
    float * xcs = nullptr;                         // [max_batch] per-token scales
    int64_t ldx = 0;
    int * sat = nullptr;                           // f16 wire: count of saturated partials (device)
    float * wamax = nullptr;                       // P100E_PF_WIRE_STATS: max |partial| per exchange [2 * L] (device)
    void * host[2] = {};                           // P100E_PF_HOST_XCHG: pinned staging per parity ([max_batch][E] f32)
    int unpack_impl = 1;                           // fast::packed_to_f16 implementation for this forward
    bool fusedq = false;                           // P100E_PF_FUSEDQ: q4kr weights dequantized inside the fold GEMM

    cudaEvent_t ev() {
        if (pool.empty()) {
            cudaEvent_t e;
            CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            return e;
        }
        cudaEvent_t e = pool.back();
        pool.pop_back();
        return e;
    }
    void release(pf_wentry & e) {
        if (e.ready) pool.push_back(e.ready);
        for (cudaEvent_t d : e.done) pool.push_back(d);
        e.ready = nullptr;
        e.done.clear();
    }
    pf_wentry * find(const dev_weight * w) {
        for (auto & e : live) if (e.w == w) return &e;
        return nullptr;
    }
    // unpack w into the ring on ws. Only overwrites entries whose readers are all enqueued (ws first waits on their
    // done events). must=false: returns false instead of throwing when that is not possible yet.
    bool unpack(const dev_weight * w, bool must) {
        if (fusedq && w->pk.data && w->pk.layout == fast::LAYOUT_Q4KR && w->n_in % 256 == 0 &&
            (uintptr_t) w->pk.data % 16 == 0) {
            pf_wentry e;
            e.w = w;
            e.fused = true;
            live.push_back(e);
            return true;
        }
        const size_t n = (size_t) w->n_out * (size_t) w->n_in;
        if (n > ring_cap) throw std::runtime_error("prefill: weight larger than the f16 ring (P100E_PF_RING_MB)");
        const size_t o = head + n <= ring_cap ? head : 0;
        for (auto & e : live) {
            if (!e.fused && e.off < o + n && o < e.off + e.n && !e.consumed) {
                if (must) throw std::runtime_error("prefill: f16 ring too small for one GEMM group (P100E_PF_RING_MB)");
                return false;
            }
        }
        for (auto it = live.begin(); it != live.end();) {
            if (!it->fused && it->off < o + n && o < it->off + it->n) {
                for (cudaEvent_t d : it->done) CK(cudaStreamWaitEvent(ws, d, 0));
                release(*it);
                it = live.erase(it);
            } else {
                ++it;
            }
        }
        half * dst = ring + o;
        if (w->pk.data) {
            fast::packed_to_f16(w->pk, 0, w->n_out, dst, ws, unpack_impl);
        } else {
            to_fp16_cuda_t to16 = ggml_get_to_fp16_cuda((ggml_type) w->type);
            if (!to16) throw std::runtime_error("prefill: no f16 dequant for a weight type");
            to16(w->d, dst, (int64_t) n, ws);
        }
        pf_wentry e;
        e.w = w; e.off = o; e.n = n;
        e.ready = ev();
        CK(cudaEventRecord(e.ready, ws));
        live.push_back(e);
        head = o + n;
        return true;
    }
    void reset() {   // only after the streams are idle
        for (auto & e : live) release(e);
        live.clear();
        head = 0;
    }
};

void tp_model_pf_free(pf_rank * P) {
    if (!P) return;
    cudaSetDevice(P->dev);
    P->reset();
    for (cudaEvent_t e : P->pool) cudaEventDestroy(e);
    for (int c = 0; c < PF_MAXCH; ++c) {
        if (P->cs[c]) cudaStreamDestroy(P->cs[c]);
        if (P->ev_join[c]) cudaEventDestroy(P->ev_join[c]);
        for (int p = 0; p < 2; ++p) {
            if (P->ev_sent[p][c]) cudaEventDestroy(P->ev_sent[p][c]);
            if (P->ev_used[p][c]) cudaEventDestroy(P->ev_used[p][c]);
        }
        if (P->ev_pre[c]) cudaEventDestroy(P->ev_pre[c]);
        if (P->ev_glu[c]) cudaEventDestroy(P->ev_glu[c]);
    }
    if (P->ev_fork) cudaEventDestroy(P->ev_fork);
    if (P->ev_wjoin) cudaEventDestroy(P->ev_wjoin);
    if (P->ws) cudaStreamDestroy(P->ws);
    if (P->ring) cudaFree(P->ring);
    if (P->x16) cudaFree(P->x16);
    if (P->xcs) cudaFree(P->xcs);
    if (P->sat) cudaFree(P->sat);
    if (P->wamax) cudaFree(P->wamax);
    for (void * h : P->host) if (h) cudaFreeHost(h);
    delete P;
}

static size_t pf_ring_mb() {
    static const size_t mb = getenv("P100E_PF_RING_MB") ? (size_t) atoll(getenv("P100E_PF_RING_MB")) : 640;
    return mb;
}

// GEMM groups of the prefill pipeline in use order (layers [l0, l1): pipe mode runs a part): the weights of one group
// read the same prescaled input
static std::vector<std::vector<std::string>> pf_groups(int l0, int l1, int full_attn_interval) {
    std::vector<std::vector<std::string>> groups;
    for (int il = l0; il < l1; ++il) {
        if ((il + 1) % full_attn_interval == 0) {
            groups.push_back({bn(il, "attn_q.weight"), bn(il, "attn_k.weight"), bn(il, "attn_v.weight")});
            groups.push_back({bn(il, "attn_output.weight")});
        } else {
            groups.push_back({bn(il, "attn_qkv.weight"), bn(il, "attn_gate.weight"), bn(il, "ssm_beta.weight"),
                              bn(il, "ssm_alpha.weight")});
            groups.push_back({bn(il, "ssm_out.weight")});
        }
        groups.push_back({bn(il, "ffn_gate.weight"), bn(il, "ffn_up.weight")});
        groups.push_back({bn(il, "ffn_down.weight")});
    }
    return groups;
}

// per instance (verdict cached in pf_ok_): every case the pipeline would throw on falls back to forward_classic
bool tp_model::prefill_ok() const {
    if (!use_fast_ || rk_.size() > fast::P2P_MAX_RANKS || rk_.empty()) return false;
    if (pf_ok_ < 0) {
        const int R = (int) rk_.size();
        const char * why = nullptr;
        // row limits of the fused prescale kernels: glu_prescale K <= 9216, add_norm_prescale K <= 8192
        // (one GPU has FFl = 17408 on Qwen3.8-27B; an uneven split: every rank's share must fit)
        (void) R;
        for (const auto & S : rk_)
            if (S.FFl > 9216 || E_ > 8192) why = "FFN / embedding width beyond the fused prescale kernels";
        for (const auto & S : rk_)
            for (const auto & [name, w] : S.w) {
                if (name == "output.weight") continue;
                if (w.n_in % 64 || w.n_out % 4 || (!w.pk.data && w.packed)) why = "unsupported weight shape/layout";
                if (!w.pk.data && !ggml_get_to_fp16_cuda((ggml_type) w.type)) why = "weight type without an f16 unpack";
            }
        // the f16 ring must take any group wherever the previous one left its head: 2x the largest group suffices
        size_t big = 0;
        for (const auto & S : rk_)
            for (const auto & g : pf_groups(lb_, le_, full_attn_interval_)) {
                size_t n = 0;
                for (const auto & nm : g) {
                    auto it = S.w.find(nm);
                    if (it == S.w.end()) { why = "weight not loaded"; continue; }
                    n += (size_t) it->second.n_out * (size_t) it->second.n_in;
                }
                big = std::max(big, n);
            }
        if (2 * big > (pf_ring_mb() << 19)) why = "f16 ring (P100E_PF_RING_MB) smaller than 2x the largest GEMM group";
        pf_ok_ = why ? 0 : 1;
        if (why) printf("prefill pipeline: %s, using the classic prefill\n", why);
    }
    return pf_ok_ == 1;
}

// lazy setup of the prefill pipeline (S.pf is published only once every allocation succeeded); forward_prefill or,
// for memory planning before alloc_pool(-1), prefill_setup()
void tp_model::pf_setup() {
    const int R = (int) rk_.size();
    for (int r = 0; r < R; ++r) {
        rank_state & S = rk_[r];
        if (S.pf) continue;
        // prescaled-activation row stride: the widest input of this rank's GEMMs (its own shares)
        const int64_t LDX = ((std::max({(int64_t) E_, (int64_t) S.FFl, (int64_t) S.Hl * HD_, (int64_t) S.HVl * DV_}) + 63) / 64) * 64;
        CK(cudaSetDevice(S.dev));
        pf_rank * P = new pf_rank();
        P->dev = S.dev;
        try {
            // chunk streams at the highest priority, the unpack stream at the lowest: the block scheduler then hands
            // freed SMs to pending GEMM blocks first and the unpack fills what is left beside them
            // (P100E_PF_PRIO=0: all default priority)
            static const bool prio = !getenv("P100E_PF_PRIO") || atoi(getenv("P100E_PF_PRIO")) != 0;
            int lo = 0, hi = 0;
            CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
            if (!prio) lo = hi = 0;
            CK(cudaStreamCreateWithPriority(&P->ws, cudaStreamNonBlocking, lo));
            for (int c = 0; c < PF_MAXCH; ++c) {
                CK(cudaStreamCreateWithPriority(&P->cs[c], cudaStreamNonBlocking, hi));
                CK(cudaEventCreateWithFlags(&P->ev_join[c], cudaEventDisableTiming));
                for (int p = 0; p < 2; ++p) {
                    CK(cudaEventCreateWithFlags(&P->ev_sent[p][c], cudaEventDisableTiming));
                    CK(cudaEventCreateWithFlags(&P->ev_used[p][c], cudaEventDisableTiming));
                }
                CK(cudaEventCreateWithFlags(&P->ev_pre[c], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&P->ev_glu[c], cudaEventDisableTiming));
            }
            CK(cudaEventCreateWithFlags(&P->ev_fork, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&P->ev_wjoin, cudaEventDisableTiming));
            P->ring_cap = pf_ring_mb() << 19;   // MB -> halves
            CK(cudaMalloc(&P->ring, P->ring_cap * 2));
            P->ldx = LDX;
            CK(cudaMalloc(&P->x16, (size_t) max_batch_ * LDX * 2));
            CK(cudaMalloc(&P->xcs, (size_t) max_batch_ * 4));
            CK(cudaMalloc(&P->sat, sizeof(int)));
            CK(cudaMemset(P->sat, 0, sizeof(int)));
        } catch (...) {
            tp_model_pf_free(P);
            throw;
        }
        S.pf = P;
    }
}

// threading contract: tp_model.h (idle_scratch). Under the lease no prefill can run, so pf_setup / the ring are ours.
std::vector<tp_model::idle_mem> tp_model::idle_scratch() {
    if (!prefill_ok()) return {};
    std::shared_ptr<void> lease = ring_gate_.try_lend();
    if (!lease) return {};   // a prefill is running or waiting to start (or the ring is already lent)
    pf_setup();
    std::vector<idle_mem> v;
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        CK(cudaStreamSynchronize(S.pf->ws));   // forward_prefill ends with every stream idle and the ring reset
        v.push_back({S.dev, (void *) S.pf->ring, S.pf->ring_cap * 2, lease});
    }
    return v;
}

std::vector<float> tp_model::forward_prefill(const std::vector<int32_t> & tokens) {
    lend_gate::busy_scope ring_busy(ring_gate_);   // waits for a lent ring to come back (idle_scratch)
    const int T = (int) tokens.size();
    const int R = (int) rk_.size();
    static const int env_ch = getenv("P100E_PF_CHUNKS") ? atoi(getenv("P100E_PF_CHUNKS")) : 4;
    static const bool env_attn = !getenv("P100E_PF_ATTN") || atoi(getenv("P100E_PF_ATTN")) != 0;
    // P100E_PF_FOLD: 0 = fp32 attn_prefill only; 1 (default) = fold-GEMM attention (fast/attn_fold.cu: f16 products,
    // fp32 accumulation across 128-key tiles) once a chunk sees >= P100E_PF_FOLD_MIN keys (default 0) and has
    // T >= 64; 2 = fold for every T. Scratch = w_scratch (no extra VRAM; free while the mixer runs)
    // per-forward switches (see the block comment above); getenv per forward so they can be A/B'd in one process
    auto envi = [](const char * n, int d) { const char * e = getenv(n); return e && *e ? atoi(e) : d; };
    const int env_fold = envi("P100E_PF_FOLD", 1);
    const int env_fold_min = envi("P100E_PF_FOLD_MIN", 0);
    // Short chunks at depth (P100E_PF_FOLD_TMIN / _TMIN_DEPTH, default TMIN 64 = off): attn_prefill runs one block per 2
    // query tokens x kv head, so a T = 48 chunk at 100K keys has 24 x 2 blocks each scanning the whole cache serially
    // (37 ms per layer at 100K: 590 of 810 ms of the forward). The fold
    // path splits the keys over independent streams. Chunks of T >= TMIN take it once n_past >= TMIN_DEPTH.
    const int fold_tmin = envi("P100E_PF_FOLD_TMIN", 64), fold_tdepth = envi("P100E_PF_FOLD_TMIN_DEPTH", 0);
    const bool use_fold = env_attn && (env_fold == 2 || (env_fold == 1 && sq().n_past + T >= env_fold_min &&
                                                        (T >= 64 || (T >= fold_tmin && sq().n_past >= fold_tdepth))));
    // (R_: every node's ranks, so a node with one rank still writes the f16 wire its peers expect)
    const bool wire16 = R_ >= 2 && envi("P100E_PF_WIRE16", 1) != 0 && !pf_wire16_off_;
    const bool host_xchg = R == 2 && !net_ && envi("P100E_PF_HOST_XCHG", 0) != 0;
    const int raster = envi("P100E_PF_RASTER", 4);
    const bool prescale_v = envi("P100E_PF_PRESCALE_V", 1) != 0;
    const int unpack_impl = envi("P100E_UNPACK", fast::UNPACK_DIRECT);
    // P100E_PF_FUSEDQ: 0 off, 1 always, 2 (default) only for short prompts T<=P100E_PF_FUSEDQ_TMAX (32): the fused dequant
    // P100: wins T=9..32 (+29.7/+27.9/+13.2/+11.0%), loses T=48 -5.0%, T=64 -6.4%, so TMAX is 32.
    // It loses at pp2048 but removes the 24.5 GB/forward f16 unpack write+read that dominates T<=64
    const int fq = envi("P100E_PF_FUSEDQ", 2);
    const bool fusedq = fq == 1 || (fq == 2 && (int) tokens.size() <= envi("P100E_PF_FUSEDQ_TMAX", 32));
    const int gdn_chunked = envi("P100E_GDN_CHUNKED", 0);
    const bool wstats = envi("P100E_PF_WIRE_STATS", 0) != 0;
    const int test_sat = envi("P100E_PF_TEST_SAT", 0);
    fast::fold_opts fo;
    fo.raster = raster;
    fo.sass = envi("P100E_PF_SASS", P100E_PF_SASS_DEFAULT);
    pf_setup();
    for (int r = 0; r < R; ++r) {
        pf_rank & P = *rk_[r].pf;
        P.unpack_impl = unpack_impl;
        P.fusedq = fusedq;
        if (host_xchg && !P.host[0]) {   // pinned staging, allocated on first use
            CK(cudaSetDevice(P.dev));
            for (void *& h : P.host) CK(cudaMallocHost(&h, (size_t) max_batch_ * E_ * 4));
        }
        if (wstats) {   // max |partial| per exchange, zeroed on the main stream (the chunk streams fork from it)
            CK(cudaSetDevice(P.dev));
            if (!P.wamax) CK(cudaMalloc(&P.wamax, (size_t) 2 * L_ * sizeof(float)));
            CK(cudaMemsetAsync(P.wamax, 0, (size_t) 2 * L_ * sizeof(float), rk_[r].s));
        }
    }

    // ---- token chunks (whole 128-token tiles) ----
    const int ntile = (T + PF_TILE - 1) / PF_TILE;
    const int nch = std::max(1, std::min({env_ch, ntile, PF_MAXCH}));
    int cb[PF_MAXCH + 1];
    for (int c = 0; c <= nch; ++c) cb[c] = std::min(T, (int) ((int64_t) ntile * c / nch) * PF_TILE);

    // ---- weight plan: GEMM groups in use order ----
    const std::vector<std::vector<std::string>> groups = pf_groups(lb_, le_, full_attn_interval_);
    std::vector<std::vector<const dev_weight *>> flat(R);   // per rank, plan order
    std::vector<std::vector<int>> gfirst(R);
    for (int r = 0; r < R; ++r) {
        for (auto & g : groups) {
            gfirst[r].push_back((int) flat[r].size());
            for (auto & n : g) {
                auto it = rk_[r].w.find(n);
                if (it == rk_[r].w.end()) throw std::runtime_error("prefill: weight not loaded: " + n);
                flat[r].push_back(&it->second);
            }
        }
        gfirst[r].push_back((int) flat[r].size());
    }
    std::vector<size_t> next_unpack(R, 0);
    int gi = 0;   // next group

    auto dev = [&](int r) { CK(cudaSetDevice(rk_[r].dev)); };
    auto fork = [&]() {
        for (int r = 0; r < R; ++r) {
            dev(r);
            pf_rank & P = *rk_[r].pf;
            CK(cudaEventRecord(P.ev_fork, rk_[r].s));
            for (int c = 0; c < nch; ++c) CK(cudaStreamWaitEvent(P.cs[c], P.ev_fork, 0));
        }
    };
    auto join = [&]() {
        for (int r = 0; r < R; ++r) {
            dev(r);
            pf_rank & P = *rk_[r].pf;
            for (int c = 0; c < nch; ++c) {
                CK(cudaEventRecord(P.ev_join[c], P.cs[c]));
                CK(cudaStreamWaitEvent(rk_[r].s, P.ev_join[c], 0));
            }
        }
    };
    // The chunks share S.a / S.b / S.g, and successive ops use them with different row strides (FFl for gate/up vs
    // (each rank its own shares)
    // Hl*HD / HVl*DV for the mixer output vs the in-projection widths), so chunk c's rows of one op overlap other
    // chunks' rows of the neighbouring ops. Before a chunk writes such a buffer, every other chunk's reads of it must
    // have run: ffn gate/up (writes a, b) waits for every chunk's out-proj prescale (reads b or g, ev_pre); the next
    // layer's in-projections (write a, b, d, e, g) wait for every chunk's SwiGLU prescale (reads a, b; after its
    // ev_pre on the same stream, ev_glu). Only issue order and GPU scheduling protected this before.
    auto cross_wait = [&](bool glu) {
        if (nch < 2) return;
        for (int r = 0; r < R; ++r) {
            dev(r);
            pf_rank & P = *rk_[r].pf;
            const cudaEvent_t * ev = glu ? P.ev_glu : P.ev_pre;
            for (int c = 0; c < nch; ++c)
                for (int o = 0; o < nch; ++o)
                    if (o != c) CK(cudaStreamWaitEvent(P.cs[c], ev[o], 0));
        }
    };
    // the next group's weights in one segmented fold launch per chunk; ys(r)[i] = [T][n_out_i] output of weight i
    // (o16: the buffer holds [T][n_out_i] halves, fold_opts::out16)
    auto gemm_group = [&](auto ys, bool o16 = false) {
        const int g = gi++;
        for (int r = 0; r < R; ++r) {
            dev(r);
            pf_rank & P = *rk_[r].pf;
            const std::vector<float *> y = ys(r);
            const int k0 = gfirst[r][g], k1 = gfirst[r][g + 1];
            if ((int) y.size() != k1 - k0) throw std::runtime_error("prefill: group output count mismatch");
            while ((int) next_unpack[r] < k1) P.unpack(flat[r][next_unpack[r]++], true);
            std::vector<pf_wentry *> es;
            for (int k = k0; k < k1; ++k) {
                pf_wentry * e = P.find(flat[r][k]);
                if (!e) throw std::runtime_error("prefill: weight missing from the f16 ring");
                es.push_back(e);
            }
            const int K = (int) flat[r][k0]->n_in;
            fast::fold_opts o = fo;
            o.out16 = o16;
            o.sat = P.sat;
            for (int c = 0; c < nch; ++c) {
                // one launch for the f16 ring weights, one for the fused (packed q4kr) ones (P100E_PF_FUSEDQ); every
                // output is computed whole by one block either way, so the split changes no value
                for (int fz = 0; fz < 2; ++fz) {
                    fast::fold_seg segs[4];
                    int ns = 0;
                    for (size_t i = 0; i < es.size(); ++i) {
                        if (es[i]->fused != (fz == 1)) continue;
                        if (es[i]->w->n_in != K) throw std::runtime_error("prefill: group with different n_in");
                        if (!fz) CK(cudaStreamWaitEvent(P.cs[c], es[i]->ready, 0));
                        fast::fold_seg & sg = segs[ns++];
                        sg.w = fz ? (const half *) es[i]->w->pk.data : P.ring + es[i]->off;
                        sg.ldy = es[i]->w->n_out;
                        sg.rows = (int) es[i]->w->n_out;
                        if (o16) sg.y16 = (half *) y[i] + (int64_t) cb[c] * sg.ldy;
                        else sg.y = y[i] + (int64_t) cb[c] * sg.ldy;
                    }
                    if (!ns) continue;
                    o.q4k = fz == 1;
                    fast::fold_gemm(segs, ns, P.x16 + (int64_t) cb[c] * P.ldx, P.ldx, P.xcs + cb[c],
                                    cb[c + 1] - cb[c], K, P.cs[c], o);
                }
                for (auto * e : es) {
                    cudaEvent_t d = P.ev();
                    CK(cudaEventRecord(d, P.cs[c]));
                    e->done.push_back(d);
                }
            }
            for (auto * e : es) e->consumed = true;
            // prefetch the following weights as far as the ring allows
            while (next_unpack[r] < flat[r].size() && P.unpack(flat[r][next_unpack[r]], false)) ++next_unpack[r];
        }
    };
    // per chunk: send my partial to the peer, x += part_rank0 + part_rank1, then x16/xcs = prescale(rmsnorm(x) * w)
    // (norm_name empty: add only). wire16: the partials are halves (S.part as half [T][E], landing S.peer as half
    // [2][max_batch][E]); host_xchg: D2H into my pinned host[p], the peer H2D's it into its landing buffer.
    int parity = 0;
    const size_t esz = wire16 ? 2 : 4;   // bytes per exchanged value
    // rank r's landing buffer of parity p (bytes) for the partials of sender slot j (R > 2: q < r ? q : q - 1; R = 2: 0)
    auto landing = [&](int r, int p, int j = 0) {   // (R_ - 1 sender slots: net mode counts every node's ranks)
        return (char *) rk_[r].peer + ((size_t) p * std::max(1, R_ - 1) + j) * max_batch_ * E_ * esz;
    };
    // net mode, per chunk: every local rank's partial chunk D2H into its pinned slot (net_tx_), the local
    // peers get theirs by device copy as in one process; once the D2H copies are done, ONE message per node carries
    // all its ranks' chunks (net_link::exchange: send + receive at once); the received chunks go H2D into each local
    // rank's landing slots of the remote senders, on the chunk's stream; then the same add + norm + prescale kernels
    // over all R_ partials in global rank order. The host blocks on the D2H of chunk c while the GPUs keep running the
    // GEMMs of the later chunks (enqueued before the exchange), so chunk c's transfer overlaps their compute.
    // Staging reuse: chunk c sits at its own offset in every slot; the next exchange of chunk c first syncs the
    // chunk's stream (its D2H), which is ordered after this exchange's H2D from the same region.
    auto net_chunk = [&](int c, int p, size_t off, size_t bytes, int n, const float * const * wn, half * const * x16s) {
        const int NL = R;
        const size_t es_off = off * esz;
        // pre-add (P100E_NET_PREADD, f32 wire only): node 0 sends s = ((x + p0) + p1) .. of its ranks for this chunk, the
        // others set x = s and add -0.0f in place of those partials (the same add_norm_prescale calls and order)
        const bool pf_pre = net_preadd_ && !wire16;
        const bool tx_pre = pf_pre && net_->self() == 0, rx_pre = pf_pre && net_->self() != 0;
        const int pk = rx_pre ? node_nr_[0] : 0;
        for (int r = 0; r < NL; ++r) {
            dev(r);
            pf_rank & P = *rk_[r].pf;
            const char * src = (const char *) rk_[r].part + es_off;
            if (!tx_pre) CK(cudaMemcpyAsync(net_tx_ + (size_t) r * net_slot_ + es_off, src, bytes, cudaMemcpyDeviceToHost, P.cs[c]));
            for (int o = 0; o < NL; ++o) {   // local peers: device copy into their slot of my global rank
                if (o == r) continue;
                const int go = r0_ + o, gr = r0_ + r;
                CK(cudaStreamWaitEvent(P.cs[c], rk_[o].pf->ev_used[p][c], 0));
                CK(cudaMemcpyPeerAsync(landing(o, p, gr < go ? gr : gr - 1) + es_off, rk_[o].dev, src, rk_[r].dev, bytes,
                                       P.cs[c]));
            }
            CK(cudaEventRecord(P.ev_sent[p][c], P.cs[c]));
        }
        if (tx_pre) {   // on rank 0's chunk stream, before its own add of this chunk: s rows = x + p0 + p1 (+ p2 ..)
            dev(0);
            pf_rank & P = *rk_[0].pf;
            for (int o = 1; o < NL; ++o) CK(cudaStreamWaitEvent(P.cs[c], rk_[o].pf->ev_sent[p][c], 0));
            float * s = pre_buf_ + off;
            CK(cudaMemcpyAsync(s, rk_[0].x + off, bytes, cudaMemcpyDeviceToDevice, P.cs[c]));
            auto part_q = [&](int q) { return (const float *) (q == 0 ? (const char *) rk_[0].part + es_off : landing(0, p, q - 1) + es_off); };
            for (int q = 0; q < NL; q += 2)   // the receivers' add_norm_prescale pairs: (x + p0) + p1, then + p2 ..
                fast::add_norm_prescale(s, part_q(q), q + 1 < NL ? part_q(q + 1) : nullptr, nullptr, eps_, n, E_, nullptr,
                                        P.ldx, nullptr, P.cs[c]);
            CK(cudaMemcpyAsync(net_tx_ + es_off, s, bytes, cudaMemcpyDeviceToHost, P.cs[c]));
        }
        for (int r = 0; r < NL; ++r) { dev(r); CK(cudaStreamSynchronize(rk_[r].pf->cs[c])); }
        std::vector<net_link::seg> snd;
        for (int r = 0; r < (tx_pre ? 1 : NL); ++r) snd.push_back({net_tx_ + (size_t) r * net_slot_ + es_off, bytes});
        std::vector<std::vector<net_link::seg>> rcv(net_->n_nodes());
        for (int j = 0; j < net_->n_nodes(); ++j) {
            if (j == net_->self()) continue;
            if (pf_pre && j == 0) { rcv[j].push_back({net_rx_ + es_off, bytes}); continue; }
            for (int q = node_r0_[j]; q < node_r0_[j] + node_nr_[j]; ++q)
                rcv[j].push_back({net_rx_ + (size_t) q * net_slot_ + es_off, bytes});
        }
        net_->exchange(net_link::T_PREFILL, snd, rcv);
        for (int r = 0; r < NL; ++r) {
            rank_state & S = rk_[r];
            dev(r);
            pf_rank & P = *S.pf;
            const int gr = r0_ + r;
            const char * pq[fast::P2P_MAX_RANKS] = {};
            if (rx_pre) CK(cudaMemcpyAsync(S.x + off, net_rx_ + es_off, bytes, cudaMemcpyHostToDevice, P.cs[c]));   // x = s
            for (int q = 0; q < R_; ++q) {
                if (q < pk) { pq[q] = (const char *) (pad_[r] + off); continue; }   // folded into s: -0.0f
                if (q == gr) { pq[q] = (const char *) S.part + es_off; continue; }
                char * dst = landing(r, p, q < gr ? q : q - 1) + es_off;
                pq[q] = dst;
                if (q >= r0_ && q < r0_ + NL) CK(cudaStreamWaitEvent(P.cs[c], rk_[q - r0_].pf->ev_sent[p][c], 0));
                else CK(cudaMemcpyAsync(dst, net_rx_ + (size_t) q * net_slot_ + es_off, bytes, cudaMemcpyHostToDevice, P.cs[c]));
            }
            const float * w = wn[r];
            half * x16 = x16s[r];
            // the one-process kernels in the same order: R = 2 one call (rank 0's partial first), R > 2 two calls
            if (R_ == 2) {
                if (wire16) fast::add_norm_prescale16(S.x + off, (const half *) pq[0], (const half *) pq[1], w, eps_, n, E_, x16,
                                                      P.ldx, P.xcs + cb[c], P.cs[c]);
                else fast::add_norm_prescale(S.x + off, (const float *) pq[0], (const float *) pq[1], w, eps_, n, E_, x16,
                                             P.ldx, P.xcs + cb[c], P.cs[c]);
            } else if (wire16) {
                fast::add_norm_prescale16(S.x + off, (const half *) pq[0], (const half *) pq[1], nullptr, eps_, n, E_,
                                          nullptr, P.ldx, nullptr, P.cs[c]);
                fast::add_norm_prescale16(S.x + off, (const half *) pq[2], (const half *) pq[3], w, eps_, n, E_, x16,
                                          P.ldx, P.xcs + cb[c], P.cs[c]);
            } else {
                fast::add_norm_prescale(S.x + off, (const float *) pq[0], (const float *) pq[1], nullptr, eps_, n, E_,
                                        nullptr, P.ldx, nullptr, P.cs[c]);
                fast::add_norm_prescale(S.x + off, (const float *) pq[2], (const float *) pq[3], w, eps_, n, E_, x16,
                                        P.ldx, P.xcs + cb[c], P.cs[c]);
            }
            CK(cudaEventRecord(P.ev_used[p][c], P.cs[c]));
        }
    };
    int xi = 0;   // exchange index: 2 * layer (+1 for the FFN), the P100E_PF_WIRE_STATS slot
    auto exchange = [&](const std::string & norm_name) {
        const int p = parity;
        parity ^= 1;
        const int k = xi++;
        for (int c = 0; c < nch; ++c) {
            const int n = cb[c + 1] - cb[c];
            const size_t off = (size_t) cb[c] * E_;
            const size_t bytes = (size_t) n * E_ * esz;
            if (wstats)   // after this chunk's GEMM on its stream, before the partial is sent
                for (int r = 0; r < R; ++r) {
                    dev(r);
                    fast::abs_max((const char *) rk_[r].part + off * esz, wire16, (size_t) n * E_, rk_[r].pf->wamax + k,
                                  rk_[r].pf->cs[c]);
                }
            if (net_) {
                const float * wn[fast::P2P_MAX_RANKS] = {};
                half * x16s[fast::P2P_MAX_RANKS] = {};
                for (int r = 0; r < R; ++r) {
                    wn[r] = norm_name.empty() ? nullptr : rk_[r].f[norm_name];
                    x16s[r] = wn[r] ? rk_[r].pf->x16 + (int64_t) cb[c] * rk_[r].pf->ldx : nullptr;
                }
                net_chunk(c, p, off, bytes, n, wn, x16s);
                continue;
            }
            if (R == 2) {
                for (int r = 0; r < 2; ++r) {
                    const int o = 1 - r;
                    dev(r);
                    pf_rank & P = *rk_[r].pf;
                    // the peer's landing chunk is free (and, host_xchg, its H2D from my host[p] chunk has run)
                    CK(cudaStreamWaitEvent(P.cs[c], rk_[o].pf->ev_used[p][c], 0));
                    const char * src = (const char *) rk_[r].part + off * esz;
                    if (host_xchg) {
                        CK(cudaMemcpyAsync((char *) P.host[p] + off * esz, src, bytes, cudaMemcpyDeviceToHost, P.cs[c]));
                    } else {
                        CK(cudaMemcpyPeerAsync(landing(o, p) + off * esz, rk_[o].dev, src, rk_[r].dev, bytes, P.cs[c]));
                    }
                    CK(cudaEventRecord(P.ev_sent[p][c], P.cs[c]));
                }
            } else if (R > 2) {   // one copy per peer into its slot of this sender (host_xchg is R = 2 only)
                for (int r = 0; r < R; ++r) {
                    dev(r);
                    pf_rank & P = *rk_[r].pf;
                    const char * src = (const char *) rk_[r].part + off * esz;
                    for (int o = 0; o < R; ++o) {
                        if (o == r) continue;
                        CK(cudaStreamWaitEvent(P.cs[c], rk_[o].pf->ev_used[p][c], 0));
                        CK(cudaMemcpyPeerAsync(landing(o, p, r < o ? r : r - 1) + off * esz, rk_[o].dev, src, rk_[r].dev,
                                               bytes, P.cs[c]));
                    }
                    CK(cudaEventRecord(P.ev_sent[p][c], P.cs[c]));
                }
            }
            for (int r = 0; r < R; ++r) {
                rank_state & S = rk_[r];
                dev(r);
                pf_rank & P = *S.pf;
                const char * mine = (const char *) S.part + off * esz;
                const char * first = mine, * second = nullptr;
                if (R == 2) {
                    CK(cudaStreamWaitEvent(P.cs[c], rk_[1 - r].pf->ev_sent[p][c], 0));
                    char * other = landing(r, p) + off * esz;
                    if (host_xchg)
                        CK(cudaMemcpyAsync(other, (const char *) rk_[1 - r].pf->host[p] + off * esz, bytes,
                                           cudaMemcpyHostToDevice, P.cs[c]));
                    // rank 0's partial first on both ranks (bit-identical replicas)
                    first = r == 0 ? mine : other;
                    second = r == 0 ? other : mine;
                }
                const float * w = norm_name.empty() ? nullptr : S.f[norm_name];
                half * x16 = w ? P.x16 + (int64_t) cb[c] * P.ldx : nullptr;
                if (R > 2) {
                    // rank order on every rank: x += p0 + p1 (add only), then x += p2 (+ p3) with the norm + prescale:
                    // (((x + p0) + p1) + p2) + p3, the f32 adds of the R = 2 kernel extended
                    const char * pq[fast::P2P_MAX_RANKS] = {};
                    for (int q = 0; q < R; ++q) {
                        if (q != r) CK(cudaStreamWaitEvent(P.cs[c], rk_[q].pf->ev_sent[p][c], 0));
                        pq[q] = q == r ? mine : landing(r, p, q < r ? q : q - 1) + off * esz;
                    }
                    if (wire16) {
                        fast::add_norm_prescale16(S.x + off, (const half *) pq[0], (const half *) pq[1], nullptr, eps_, n, E_,
                                                  nullptr, P.ldx, nullptr, P.cs[c]);
                        fast::add_norm_prescale16(S.x + off, (const half *) pq[2], (const half *) pq[3], w, eps_, n, E_, x16,
                                                  P.ldx, P.xcs + cb[c], P.cs[c]);
                    } else {
                        fast::add_norm_prescale(S.x + off, (const float *) pq[0], (const float *) pq[1], nullptr, eps_, n, E_,
                                                nullptr, P.ldx, nullptr, P.cs[c]);
                        fast::add_norm_prescale(S.x + off, (const float *) pq[2], (const float *) pq[3], w, eps_, n, E_, x16,
                                                P.ldx, P.xcs + cb[c], P.cs[c]);
                    }
                } else if (wire16)
                    fast::add_norm_prescale16(S.x + off, (const half *) first, (const half *) second, w, eps_, n, E_, x16,
                                              P.ldx, P.xcs + cb[c], P.cs[c]);
                else
                    fast::add_norm_prescale(S.x + off, (const float *) first, (const float *) second, w, eps_, n, E_, x16,
                                            P.ldx, P.xcs + cb[c], P.cs[c]);
                CK(cudaEventRecord(P.ev_used[p][c], P.cs[c]));
            }
        }
    };
    auto prescale_chunks = [&](auto src, auto Kof) {   // src(r): [T][K] f32 -> x16 / xcs, K = Kof(r) (rank r's width)
        for (int c = 0; c < nch; ++c)
            for (int r = 0; r < R; ++r) {
                dev(r);
                pf_rank & P = *rk_[r].pf;
                const int K = Kof(r);
                fast::prescale_rows(src(r) + (int64_t) cb[c] * K, K, cb[c + 1] - cb[c], K,
                                    P.x16 + (int64_t) cb[c] * P.ldx, P.ldx, P.xcs + cb[c], P.cs[c], prescale_v);
                CK(cudaEventRecord(P.ev_pre[c], P.cs[c]));
            }
    };

    // ---- embeddings: gathered on the host into pinned memory (per instance, [max_batch] rows), one upload per GPU.
    // Safe to rewrite: the previous forward ended with a sync of every stream that read it.
    const gguf_tensor * emb = g_.find("token_embd.weight");
    const size_t erow = emb->nbytes / emb->ne[1];
    const int n_past = sq().n_past;
    set_pos(n_past, T);   // S.pos + descriptor (+ S.rpos with images in the context)
    if (pnet_ && !has_embd()) {
        pipe_recv_x(T);    // pipe node 1: node 0's residual rows after its layers, f32
    } else if (mm_) {
        embed(tokens);     // multimodal rows: token rows + the image rows' f32 embeddings
    } else {
        check_tokens(tokens, VOCAB_, "forward_prefill");
        if (!emb_h_) CK(cudaMallocHost(&emb_h_, (size_t) max_batch_ * erow));
        for (int i = 0; i < T; ++i) memcpy(emb_h_ + (size_t) i * erow, emb->data + (size_t) tokens[i] * erow, erow);
        for (int r = 0; r < R; ++r) {
            rank_state & S = rk_[r];
            dev(r);
            CK(cudaMemcpyAsync(S.w_scratch, emb_h_, (size_t) T * erow, cudaMemcpyHostToDevice, S.s));
            dequant_f32(emb->type, S.w_scratch, S.x, (int64_t) T * E_, S.s);
        }
    }
    fork();
    for (int c = 0; c < nch; ++c)
        for (int r = 0; r < R; ++r) {
            dev(r);
            pf_rank & P = *rk_[r].pf;
            fast::add_norm_prescale(rk_[r].x + (size_t) cb[c] * E_, nullptr, nullptr, rk_[r].f[bn(lb_, "attn_norm.weight")],
                                    eps_, cb[c + 1] - cb[c], E_, P.x16 + (int64_t) cb[c] * P.ldx, P.ldx, P.xcs + cb[c],
                                    P.cs[c]);
        }

    for (int il = lb_; il < le_; ++il) {   // (pipe mode: this node's layers)
        const bool attn = (il + 1) % full_attn_interval_ == 0;
        // in-projections, x16 = prescaled attn_norm(x)
        if (il > lb_) cross_wait(true);
        if (attn) gemm_group([&](int r) { return std::vector<float *>{rk_[r].a, rk_[r].e, rk_[r].g}; });
        else gemm_group([&](int r) { return std::vector<float *>{rk_[r].a, rk_[r].b, rk_[r].d, rk_[r].e}; });
        join();
        // mixer on the main stream (needs all tokens)
        for (int r = 0; r < R; ++r) {
            rank_state & S = rk_[r];
            const int Hl = S.Hl, HKVl = S.HKVl, HKl = S.HKl, HVl = S.HVl;   // this rank's shares
            dev(r);
            if (attn) {
                k::split_q_gate(S.a, S.c, S.d, T, Hl, HD_, S.s);
                norm(S.c, S.f[bn(il, "attn_q_norm.weight")], S.c, (int64_t) T * Hl, HD_, S.s);
                norm(S.e, S.f[bn(il, "attn_k_norm.weight")], S.e, (int64_t) T * HKVl, HD_, S.s);
                rope_rows(r, S.c, T, Hl);
                rope_rows(r, S.e, T, HKVl);
                fast::kv_cache kv = kv_seq(r, il);
                fast::kv_append(kv, S.e, S.g, n_past, T, S.s);
                const float scale = 1.0f / sqrtf((float) HD_);
                const bool folded = use_fold && fast::attn_fold(S.c, kv, S.b, T, n_past, Hl, scale, S.w_scratch,
                                                                scratch_elems_ * 4, S.s);
                if (!folded && !(env_attn && fast::attn_prefill(S.c, kv, S.b, T, n_past, Hl, scale, S.s)) &&
                    !(pf_tiny_attn() && T > 4 && T <= fast::ATTN_MAX_ROWS &&
                      fast::attn_decode2(S.c, kv, S.b, T, n_past, nullptr, Hl, scale, S.s))) {
                    for (int t0 = 0; t0 < T; t0 += 4) {
                        const int nt = std::min(4, T - t0);
                        if (!fast::attn_decode(S.c + (size_t) t0 * Hl * HD_, kv, S.b + (size_t) t0 * Hl * HD_, nt,
                                               n_past + t0, Hl, scale, S.s))
                            throw std::runtime_error("fast::attn_decode: no kernel ran");
                    }
                }
                k::sigmoid_mul(S.b, S.d, (int64_t) T * Hl * HD_, S.s);
            } else {
                fast::GdnWeights gw{S.f[bn(il, "ssm_conv1d.weight")], S.f[bn(il, "ssm_dt.bias")], S.f[bn(il, "ssm_a")],
                                    S.f[bn(il, "ssm_norm.weight")]};
                // gdn_layer = gdn_prep + gdn_scan; the chunked scan (P25) replaces the second step when it applies
                fast::gdn_prep(S.a, S.conv_state[il], gw, S.e, S.d, S.c, S.e, S.d, T, HKl, HVl, DK_, eps_, S.s);
                if (!(gdn_chunked > 0 &&
                      fast::gdn_scan_chunked(S.c, S.e, S.d, S.b, gw.norm_w, S.ssm_state[il], S.g, T, HKl, HVl, DK_, eps_,
                                             S.w_scratch, scratch_elems_ * 4, S.s, gdn_chunked)))
                    fast::gdn_scan(S.c, S.e, S.d, S.b, gw.norm_w, S.ssm_state[il], S.g, T, HKl, HVl, DK_, eps_, S.s);
            }
        }
        fork();
        // out-projection (row-parallel) -> part; exchange + add + post_attention_norm per chunk
        prescale_chunks([&](int r) { return attn ? rk_[r].b : rk_[r].g; },
                        [&](int r) { return attn ? rk_[r].Hl * HD_ : rk_[r].HVl * DV_; });
        gemm_group([&](int r) { return std::vector<float *>{rk_[r].part}; }, wire16);
        dbg_trace("mix", il, T);
        exchange(bn(il, "post_attention_norm.weight"));
        dbg_trace("ar1", il, T);
        // FFN per chunk: gate/up (one launch), SwiGLU fused into the prescale, down -> part; exchange + next attn_norm
        cross_wait(false);
        gemm_group([&](int r) { return std::vector<float *>{rk_[r].a, rk_[r].b}; });
        for (int c = 0; c < nch; ++c)
            for (int r = 0; r < R; ++r) {
                dev(r);
                pf_rank & P = *rk_[r].pf;
                const int FFl = rk_[r].FFl;
                fast::glu_prescale(rk_[r].a + (int64_t) cb[c] * FFl, rk_[r].b + (int64_t) cb[c] * FFl, cb[c + 1] - cb[c],
                                   FFl, P.x16 + (int64_t) cb[c] * P.ldx, P.ldx, P.xcs + cb[c], P.cs[c]);
                CK(cudaEventRecord(P.ev_glu[c], P.cs[c]));
            }
        gemm_group([&](int r) { return std::vector<float *>{rk_[r].part}; }, wire16);
        // (the last layer here: add only. Pipe node 0's S.x then holds the complete residual node 1 starts from)
        exchange(il + 1 < le_ ? bn(il + 1, "attn_norm.weight") : std::string());
        dbg_trace("ar2", il, T);
    }
    join();
    for (int r = 0; r < R; ++r) {   // the weight stream too, so the final sync covers it
        dev(r);
        pf_rank & P = *rk_[r].pf;
        CK(cudaEventRecord(P.ev_wjoin, P.ws));
        CK(cudaStreamWaitEvent(rk_[r].s, P.ev_wjoin, 0));
    }

    if (!has_head()) pipe_x_d2h(T);   // pipe node 0: rank 0's residual rows -> the handoff staging (sent after the sync)
    for (int r = 0; r < R && has_head(); ++r) {
        rank_state & S = rk_[r];
        dev(r);
        // hn = output_norm(x): every row with the MTP head loaded (its h input for the prompt), else the last row
        const int hr = n_nextn_ > 0 ? T : 1;
        norm(S.x + (int64_t) (T - hr) * E_, S.f["output_norm.weight"], S.hn + (int64_t) (T - hr) * E_, hr, E_, S.s);
        matmul(r, "output.weight", S.hn + (int64_t) (T - 1) * E_, S.logits, 1);
        CK(cudaMemcpyAsync(logits_h_ + (size_t) S.v0, S.logits, (size_t) S.VL * 4, cudaMemcpyDeviceToHost, S.s));
    }
    // full sync of every rank (main stream joined the chunk and weight streams above): the fused / graph decode step
    // that follows must never start while chunk streams can still touch part or peer
    for (int r = 0; r < R; ++r) {
        dev(r);
        CK(cudaStreamSynchronize(rk_[r].s));
        rk_[r].pf->reset();
    }
    if (net_) net_gather_logits(1);   // the other nodes' vocab slices of the last row
    int nsat = 0;
    if (wire16) {   // saturation guard of the f16 wire: this forward reports it (forward_n -> last_prefill_saturated())
        for (int r = 0; r < R; ++r) {
            int v = 0;
            dev(r);
            CK(cudaMemcpy(&v, rk_[r].pf->sat, sizeof(int), cudaMemcpyDeviceToHost));
            if (v) CK(cudaMemset(rk_[r].pf->sat, 0, sizeof(int)));
            nsat += v;
        }
        nsat += std::max(0, test_sat);   // P100E_PF_TEST_SAT fault injection
        if (net_) {   // every node's count: all switch to the f32 wire together (or none does)
            const int mine = nsat;
            nsat = 0;
            for (uint64_t v : net_->allgather((uint64_t) mine)) nsat += (int) v;
        }
        if (nsat) {
            fprintf(stderr, "prefill: %d f16 wire partial group(s) beyond the f16 range were saturated (T=%d at n_past %d):"
                            " this forward's outputs and state carry clamped values (last_prefill_saturated()); the "
                            "exchange uses the f32 wire from now on (P100E_PF_WIRE16)\n", nsat, T, sq().n_past);
            pf_wire16_off_ = true;
            pf_sat_total_ += (uint64_t) nsat;
        }
    }
    pf_wire_amax_ = -1.0f;
    if (wstats) {   // the largest |partial| of this prefill (streams synced above)
        std::vector<float> v((size_t) 2 * L_);
        int best_k = -1, best_r = 0;
        for (int r = 0; r < R; ++r) {
            dev(r);
            CK(cudaMemcpy(v.data(), rk_[r].pf->wamax, v.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (int k = 0; k < 2 * L_; ++k)   // NaN (if any) wins: !(a <= b)
                if (best_k < 0 || !(v[k] <= pf_wire_amax_)) { pf_wire_amax_ = v[k]; best_k = k; best_r = r; }
        }
        const int il = lb_ + best_k / 2;   // (exchange k of this process's layers)
        const char * what = best_k % 2 ? "ffn_down" : (il + 1) % full_attn_interval_ == 0 ? "attn_output" : "ssm_out";
        printf("PFWIRE T=%d n_past=%d wire=%s max|partial| %.6g at layer %d %s (GPU %d): f16 margin %.1fx, saturated %d\n",
               T, sq().n_past, wire16 ? "f16" : "f32", pf_wire_amax_, il, what, rk_[best_r].dev,
               pf_wire_amax_ > 0 ? 65504.0 / pf_wire_amax_ : 0.0, nsat);
        fflush(stdout);
    }
    if (pnet_ && !has_head()) pipe_send_x(T, nsat);   // node 0's rows (+ its saturation count) to node 1
    sq().n_past += T;
    sq().toks.insert(sq().toks.end(), tokens.begin(), tokens.end());
    last_T_ = T;
    // the last row's logits from node 1 (counted after n_past moved, as forward_classic does: a node may take either path)
    if (pnet_) pipe_result(1, nullptr);
    return std::vector<float>(logits_h_, logits_h_ + VOCAB_);
}

void tp_model::logits_rows(int row0, int n, float * host) {
    net_refuse("logits_rows (KL-divergence scoring)");
    if (row0 < 0 || n < 0 || row0 + n > last_T_) throw std::runtime_error("logits_rows: rows beyond the last forward");
    if (n == 0) return;
    rb_join();
    static const bool gemm = getenv("P100E_KLD_HEAD") && !strcmp(getenv("P100E_KLD_HEAD"), "gemm");
    const int G = gemm ? 64 : 8;   // rows per LM-head call (gemm: > P100E_FAST_MAXT = 8 selects matmul's GEMM path)
    const int R = (int) rk_.size();
    for (int r = 0; r < R; ++r) {
        rank_state & S = rk_[r];
        CK(cudaSetDevice(S.dev));
        if (!S.lrows) CK(cudaMalloc(&S.lrows, (size_t) 64 * S.VL * 4));
        // the same rmsnorm call the forwards use for their logit rows, into S.cur (scratch between forwards)
        norm(S.x + (int64_t) row0 * E_, S.f["output_norm.weight"], S.cur + (int64_t) row0 * E_, n, E_, S.s);
    }
    for (int g0 = 0; g0 < n; g0 += G) {
        const int nb = std::min(G, n - g0);
        for (int r = 0; r < R; ++r) {   // both GPUs' head groups interleaved: they run concurrently
            rank_state & S = rk_[r];
            CK(cudaSetDevice(S.dev));
            matmul(r, "output.weight", S.cur + (int64_t) (row0 + g0) * E_, S.lrows, nb);
            CK(cudaMemcpy2DAsync(host + (size_t) g0 * VOCAB_ + (size_t) S.v0, (size_t) VOCAB_ * 4, S.lrows,
                                 (size_t) S.VL * 4, (size_t) S.VL * 4, nb, cudaMemcpyDeviceToHost, S.s));
        }
    }
    for (int r = 0; r < R; ++r) { CK(cudaSetDevice(rk_[r].dev)); CK(cudaStreamSynchronize(rk_[r].s)); }
}

// rollback copies of up to MAX_ROWS sequences in one launch: for sequence z (blockIdx.z), every delta-net layer g and
// both states (blockIdx.y = 2 g + kind): state slot slot[z] <- snapshot slot snap[z] (float4 units; the pools are
// [slot][gdn layer][...], the snapshots [gdn layer][snapshot slot][...])
struct rb_args {
    const float4 * snp[2]; float4 * pool[2]; int n4[2];
    int n_gdn, n_snap;
    int slot[8], snap[8];
};
static __global__ void __launch_bounds__(256) rollback_k(const rb_args a) {
    const int z = blockIdx.z, g = blockIdx.y >> 1, kind = blockIdx.y & 1;
    const int n4 = a.n4[kind];
    const float4 * __restrict__ src = a.snp[kind] + ((size_t) g * a.n_snap + a.snap[z]) * n4;
    float4 * __restrict__ dst = a.pool[kind] + ((size_t) a.slot[z] * a.n_gdn + g) * n4;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) dst[i] = src[i];
}

void tp_model::rb_join() {
    if (!rb_pending_) return;
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        CK(cudaStreamWaitEvent(S.s, S.ev_rb, 0));
    }
    rb_pending_ = false;
}

void tp_model::enable_rollback(int max_tokens) {
    const int n = std::max(0, max_tokens - 1);
    if (n <= n_snap_) return;
    rb_join();
    static const bool env_k = !getenv("P100E_RB_KERNEL") || atoi(getenv("P100E_RB_KERNEL")) != 0;
    static const bool env_a = !getenv("P100E_RB_ASYNC") || atoi(getenv("P100E_RB_ASYNC")) != 0;
    rb_kernel_ = env_k;
    rb_async_ = env_k && env_a;
    drop_graphs(true);   // captured keep_states graphs hold the old snapshot pointers
    // the kernel copies float4s: every state / snapshot slot must be a whole number of 16-byte units (every rank's)
    for (const auto & S : rk_) if (S.ssz % 4 || S.csz % 4) rb_kernel_ = rb_async_ = false;
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        CK(cudaStreamSynchronize(S.s));
        if (S.rb_s) CK(cudaStreamSynchronize(S.rb_s));
        // snapshot pools [gdn layer][snapshot slot][...], shared by every sequence of a step (each takes its rows' range)
        if (S.conv_snp) cudaFree(S.conv_snp);
        if (S.ssm_snp) cudaFree(S.ssm_snp);
        S.conv_snp = S.ssm_snp = nullptr;
        CK(cudaMalloc(&S.conv_snp, (size_t) n_gdn_ * n * S.csz * 4));
        CK(cudaMalloc(&S.ssm_snp, (size_t) n_gdn_ * n * S.ssz * 4));
        S.conv_snap.assign(L_, nullptr); S.ssm_snap.assign(L_, nullptr);
        for (int il = 0; il < L_; ++il) {
            if (gdn_of_[il] < 0) continue;
            S.conv_snap[il] = S.conv_snp + (size_t) gdn_of_[il] * n * S.csz;
            S.ssm_snap[il] = S.ssm_snp + (size_t) gdn_of_[il] * n * S.ssz;
        }
        if (!S.rb_s) {
            CK(cudaStreamCreateWithFlags(&S.rb_s, cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&S.ev_rb_in, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&S.ev_rb, cudaEventDisableTiming));
        }
    }
    n_snap_ = n;
}

// state slot <- snapshot slot for every (slot, snapshot) pair, all delta-net layers. Kernel path: one launch per GPU,
// async on rb_s after the verify that wrote the snapshots, overlapping whatever the caller enqueues next on S.s that
// does not touch the recurrent state (the MTP draft head); the next forward / reset joins it (rb_join).
// P100E_RB_KERNEL=0: cudaMemcpyAsync per state on S.s.
void tp_model::rollback_slots(const std::vector<std::pair<int, int>> & sl) {
    if (sl.empty()) return;
    if (sl.size() > 8) throw std::runtime_error("rollback: more than 8 sequences");
    rb_join();
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        if (rb_kernel_) {
            cudaStream_t st = rb_async_ ? S.rb_s : S.s;
            if (rb_async_) {
                CK(cudaEventRecord(S.ev_rb_in, S.s));
                CK(cudaStreamWaitEvent(S.rb_s, S.ev_rb_in, 0));
            }
            rb_args a{};
            a.snp[0] = (const float4 *) S.ssm_snp; a.snp[1] = (const float4 *) S.conv_snp;
            a.pool[0] = (float4 *) S.ssm_pool; a.pool[1] = (float4 *) S.conv_pool;
            a.n4[0] = (int) (S.ssz / 4); a.n4[1] = (int) (S.csz / 4);
            a.n_gdn = n_gdn_; a.n_snap = n_snap_;
            for (size_t i = 0; i < sl.size(); ++i) { a.slot[i] = sl[i].first; a.snap[i] = sl[i].second; }
            rollback_k<<<dim3(48, 2 * n_gdn_, (unsigned) sl.size()), 256, 0, st>>>(a);
            CK(cudaGetLastError());
            if (rb_async_) CK(cudaEventRecord(S.ev_rb, S.rb_s));
            continue;
        }
        for (const auto & [slot, sn] : sl)
            for (int g = 0; g < n_gdn_; ++g) {
                CK(cudaMemcpyAsync(S.ssm_pool + ((size_t) slot * n_gdn_ + g) * S.ssz, S.ssm_snp + ((size_t) g * n_snap_ + sn) * S.ssz,
                                   S.ssz * 4, cudaMemcpyDeviceToDevice, S.s));
                CK(cudaMemcpyAsync(S.conv_pool + ((size_t) slot * n_gdn_ + g) * S.csz, S.conv_snp + ((size_t) g * n_snap_ + sn) * S.csz,
                                   S.csz * 4, cudaMemcpyDeviceToDevice, S.s));
            }
    }
    rb_pending_ = rb_kernel_ && rb_async_;
}

void tp_model::rollback(int T, int n_keep) {
    check_ok();
    if (n_keep < 1 || n_keep > T || T > sq().n_past) throw std::runtime_error("rollback out of range");
    if (n_keep == T) return;
    if (T - 1 > n_snap_) throw std::runtime_error("rollback: batch larger than enable_rollback() allows");
    const int n_new = sq().n_past - (T - n_keep);
    sq().mmt.check_cut(n_new);   // a cut inside an image: refused before anything changes
    rollback_slots({{sq().slot, sq().snap0 + n_keep - 1}});
    sq().n_past = n_new;   // attention KV: later positions are simply overwritten
    sq().toks.resize(sq().n_past);
    sq().rope_delta = sq().mmt.cut(n_new);   // images dropped by the cut no longer shift the text after them
}

void tp_model::rollback_batch(const std::vector<rb_req> & list) {
    std::vector<std::pair<int, int>> sl;
    for (const rb_req & q : list) {
        seq_t & s = seqs_.at(q.seq);
        if (q.n_keep < 1 || q.n_keep > q.T || q.T > s.n_past || s.slot < 0) throw std::runtime_error("rollback_batch out of range");
        if (q.n_keep == q.T) continue;
        if (s.snap0 + q.T - 1 > n_snap_) throw std::runtime_error("rollback_batch: snapshot range");
        s.mmt.check_cut(s.n_past - (q.T - q.n_keep));
    }
    for (const rb_req & q : list) {
        seq_t & s = seqs_.at(q.seq);
        if (q.n_keep == q.T) continue;
        sl.push_back({s.slot, s.snap0 + q.n_keep - 1});
        s.n_past -= q.T - q.n_keep;
        s.toks.resize(s.n_past);
        s.rope_delta = s.mmt.cut(s.n_past);
    }
    rollback_slots(sl);
}

void tp_model::bench_fill_kv(int n) {
    if (n > n_ctx_) throw std::runtime_error("bench_fill_kv: beyond the context");
    if (n <= sq().n_past) return;
    if (!use_fast_) throw std::runtime_error("bench_fill_kv needs the fast path");
    ensure_pages(active_, n);
    for (int r = 0; r < (int) rk_.size(); ++r) {
        rank_state & S = rk_[r];
        CK(cudaSetDevice(S.dev));
        CK(cudaStreamSynchronize(S.s));   // the page-table upload (kv_fill_random reads the table back)
        for (size_t il = 0; il < S.kv.size(); ++il)
            if (kvl_of_[il] >= 0 || (!paged_ && S.kv[il].k)) {
                fast::kv_cache kv = kv_seq(r, (int) il);
                fast::kv_fill_random(kv, sq().n_past, n, 0x9e3779b9u * (uint32_t) (il + 1), S.s);
            }
        CK(cudaStreamSynchronize(S.s));
    }
    sq().n_past = n;
    sq().toks.resize(n, -1);
}

void tp_model::sig_need(int word) {
    if (sig_last_ < 0 || sig_last_ == word || !p2p_ar_) return;
    for (auto & S : rk_) { CK(cudaSetDevice(S.dev)); fast::p2p_sig_equalize(S.flag, S.s); }
    sig_last_ = -1;
}

void tp_model::sig_flush(int r) {
    sig_pend_[r] = false;
    fast::p2p_sig_flush(rk_[1 - r].flag, rk_[r].flag, sig_par_[r], rk_[r].s);
}

void tp_model::reload_env() {
    // kernel choices baked into the captured graphs: P100E_MVWRP_Q* (the decode GEMV kernels) and the decode launch
    // merges (step 6: P100E_MERGE_INPROJ, P100E_GDN_FUSE). A change drops this model's step graphs and bumps
    // kern_epoch_: tp_mtp's draft graphs bake the same fused_layer launches (nextn block), and tp_mtp::check_epoch drops
    // them before its next run (they would replay the old kernels: same outputs, wrong timings).
    bool kern = fast::wrx_reload_env();
    kern = fast::attn_dec2_reload_env() || kern;   // P100E_DEC2_LEAN (decode attention K staging)
    g_pf_tiny_attn = -1;   // P100E_PF_TINY_ATTN (eager classic chunks, no graph)
    {
        auto flag = [](const char * n) { const char * e = getenv(n); return e && *e && atoi(e) != 0; };
        const bool m = flag("P100E_MERGE_INPROJ"), g = flag("P100E_GDN_FUSE");
        if (m != merge_in_ || g != gdn_fuse_) {
            merge_in_ = m; gdn_fuse_ = g;   // host flags, read when launches are enqueued
            kern = true;
        }
    }
    if (kern) {
        for (auto & S : rk_) { CK(cudaSetDevice(S.dev)); CK(cudaStreamSynchronize(S.s)); }
        drop_graphs();
        ++kern_epoch_;
    }
    if (!ar_status_h_) return;
    {   // graphs are keyed by the mode (find_graph); the counter words are tracked (sig_last_), so a switch is safe
        const char * e = getenv("P100E_AR_SIG_FUSE");
        sig_fuse_ = e && *e && atoi(e) != 0 && rk_.size() <= 2;   // R > 2: separate signals (ar_post)
        const char * q8 = getenv("P100E_Q8_FORK");
        q8_fork_ = q8 && *q8 && atoi(q8) != 0;
    }
    const char * e = getenv("P100E_AR_FENCE");
    const int fence = e && *e ? atoi(e) : 0;
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        CK(cudaStreamSynchronize(S.s));
        CK(cudaMemcpy(S.flag + fast::P2P_FLAG_FENCE, &fence, sizeof fence, cudaMemcpyHostToDevice));
    }
}

bool tp_model::ar_failed() const {
    if (!ar_status_h_) return false;
    for (int r = 0; r < (int) rk_.size(); ++r)
        if (((const volatile int *) ar_status_h_)[(size_t) r * 16]) return true;
    return false;
}

void tp_model::check_ar() const {
    if (gl_) {   // P100E_NET_GPULINK: a recv_copy wait timed out, or the spin thread failed (it also sets the count)
        const std::string e = gl_->error();
        if (!e.empty() || gl_->timeouts())
            throw std::runtime_error("net gpu link: " + (e.empty() ? std::to_string(gl_->timeouts()) +
                                     " GPU wait(s) for a peer's partial timed out" : "the pdl spin thread failed: " + e) +
                                     "; the results are invalid, aborting (restart every node)");
    }
    if (!ar_status_h_) return;
    for (int r = 0; r < (int) rk_.size(); ++r) {
        const volatile int * st = ar_status_h_ + (size_t) r * 16;
        if (st[0])
            throw std::runtime_error("P2P allreduce: GPU " + std::to_string(rk_[r].dev) + " timed out after " +
                                     std::to_string(st[3]) + " ms waiting for the peer's signal " + std::to_string(st[1]) +
                                     " (last seen " + std::to_string(st[2]) + "); the results are invalid, aborting "
                                     "(fatal for this instance: restart the process)");
    }
}

void tp_model::bench_rewind(int n) {
    if (n < 0 || n > sq().n_past) throw std::runtime_error("bench_rewind: n out of range");
    rb_join();
    for (auto & S : rk_) { CK(cudaSetDevice(S.dev)); CK(cudaStreamSynchronize(S.s)); }
    sq().n_past = n;
    sq().toks.resize(n, -1);
}

void tp_model::print_mem(const char * tag) const {
    for (const auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        printf("MEM %s: GPU %d free %.0f MiB of %.0f MiB\n", tag, S.dev, fr / 1048576.0, tot / 1048576.0);
    }
}

// the active sequence: empty (zero delta-net state, n_past 0, its pages back to the pool)
void tp_model::reset() {
    sync_all();
    seq_t & q = sq();
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        CK(cudaMemsetAsync(S.ssm_pool + (size_t) q.slot * n_gdn_ * S.ssz, 0, (size_t) n_gdn_ * S.ssz * 4, S.s));
        CK(cudaMemsetAsync(S.conv_pool + (size_t) q.slot * n_gdn_ * S.csz, 0, (size_t) n_gdn_ * S.csz * 4, S.s));
        CK(cudaStreamSynchronize(S.s));
    }
    for (auto it = q.pages.rbegin(); it != q.pages.rend(); ++it) free_pages_.push_back(*it);
    q.pages.clear();
    q.n_past = 0;
    q.toks.clear();
    q.rope_delta = 0;
    q.mmt.clear();
    q.mm_broken = false;
    last_T_ = 0;
}

// =====================================================================================================================
// Sequences
// =====================================================================================================================

fast::kv_cache tp_model::kv_seq(int r, int il) const {
    fast::kv_cache c = rk_[r].kv[il];
    if (paged_) { c.pt = rk_[r].ptab + (size_t) sq().slot * max_pages_; c.rtab = nullptr; }
    return c;
}

fast::kv_cache tp_model::kv_rows(int r, int il) const {
    fast::kv_cache c = rk_[r].kv[il];
    if (paged_) { c.pt = rk_[r].ptab; c.rtab = rk_[r].rtab; }
    return c;
}

void tp_model::sync_all() {
    rb_join();
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        CK(cudaStreamSynchronize(S.s));
        if (S.rb_s) CK(cudaStreamSynchronize(S.rb_s));
    }
}

void tp_model::bind_states() {
    const int slot = active_ >= 0 && seqs_[active_].slot >= 0 ? seqs_[active_].slot : 0;
    for (auto & S : rk_)
        for (int il = 0; il < L_; ++il) {
            if (gdn_of_[il] < 0) { S.conv_state[il] = S.ssm_state[il] = nullptr; continue; }
            S.conv_state[il] = S.conv_pool + ((size_t) slot * n_gdn_ + gdn_of_[il]) * S.csz;
            S.ssm_state[il] = S.ssm_pool + ((size_t) slot * n_gdn_ + gdn_of_[il]) * S.ssz;
        }
}

void tp_model::ensure_pages(int id, int n) {
    seq_t & q = seqs_.at(id);
    if (!q.used || q.slot < 0) throw std::runtime_error("sequence " + std::to_string(id) + " is not resident");
    if (!paged_) return;
    if (!pool_ok_) throw std::runtime_error("KV pool not allocated (alloc_pool)");
    if (n > n_ctx_) throw std::runtime_error("context full");
    const int need = (n + (1 << psh_) - 1) >> psh_;
    const int have = (int) q.pages.size();
    if (need <= have) return;
    if (need - have > (int) free_pages_.size())
        throw std::runtime_error("KV pool full (" + std::to_string(free_pages_.size()) + " free pages, need " +
                                 std::to_string(need - have) + ")");
    int32_t * row = ptab_h_ + (size_t) q.slot * max_pages_;
    for (int i = have; i < need; ++i) {
        const int pg = free_pages_.back();
        free_pages_.pop_back();
        q.pages.push_back(pg);
        row[i] = pg;
    }
    // stream-ordered upload of the new entries (pinned source: entries [have, need) are not written again while it is
    // in flight - later growth writes later entries, slot reuse goes through sync_all)
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        CK(cudaMemcpyAsync(S.ptab + (size_t) q.slot * max_pages_ + have, row + have, (size_t) (need - have) * 4,
                           cudaMemcpyHostToDevice, S.s));
    }
}

int tp_model::seq_create() {
    if (free_slots_.empty()) throw std::runtime_error("no free sequence slot (max_seqs " + std::to_string(max_slots_) + ")");
    sync_all();
    int id = -1;
    for (int i = 0; i < (int) seqs_.size() && id < 0; ++i) if (!seqs_[i].used) id = i;
    if (id < 0) { id = (int) seqs_.size(); seqs_.emplace_back(); }
    seq_t & q = seqs_[id];
    q = seq_t{};
    q.used = true;
    q.slot = free_slots_.back();
    free_slots_.pop_back();
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        CK(cudaMemsetAsync(S.ssm_pool + (size_t) q.slot * n_gdn_ * S.ssz, 0, (size_t) n_gdn_ * S.ssz * 4, S.s));
        CK(cudaMemsetAsync(S.conv_pool + (size_t) q.slot * n_gdn_ * S.csz, 0, (size_t) n_gdn_ * S.csz * 4, S.s));
    }
    return id;
}

void tp_model::seq_free(int id) {
    seq_t & q = seqs_.at(id);
    if (!q.used) return;
    sync_all();
    for (auto it = q.pages.rbegin(); it != q.pages.rend(); ++it) free_pages_.push_back(*it);
    if (q.slot >= 0) free_slots_.push_back(q.slot);
    for (char * h : q.host) if (h) cudaFreeHost(h);
    host_bytes_ -= q.host_bytes;
    q = seq_t{};
    if (active_ == id) active_ = -1;
}

void tp_model::seq_select(int id) {
    const seq_t & q = seqs_.at(id);
    if (!q.used || q.slot < 0) throw std::runtime_error("seq_select: sequence " + std::to_string(id) + " is not resident");
    active_ = id;
    bind_states();
}

int tp_model::seq_len(int id) const { return seqs_.at(id).n_past; }
bool tp_model::seq_resident(int id) const { return seqs_.at(id).used && seqs_.at(id).slot >= 0; }
int tp_model::seq_pages(int id) const { const seq_t & q = seqs_.at(id); return q.slot >= 0 ? (int) q.pages.size() : q.host_pages; }
size_t tp_model::seq_host_bytes(int id) const { return seqs_.at(id).host_bytes; }

// Offload: every page the sequence holds (all KV layers incl. the MTP block, K and V: two 2-D blocks per page and GPU)
// and its delta-net slot (conv + ssm, contiguous per slot) -> pinned host memory, both GPUs at once. Rank r's copy is
// [pages][its page_bytes] | conv | ssm in its own sizes (an uneven split: the ranks' copies differ in size).
void tp_model::seq_offload(int id) {
    net_refuse("seq_offload (sequences to host RAM)");
    seq_t & q = seqs_.at(id);
    if (!q.used || q.slot < 0) throw std::runtime_error("seq_offload: sequence not resident");
    if (!paged_) throw std::runtime_error("seq_offload needs the paged KV pool");
    std::vector<size_t> bytes(rk_.size());
    size_t total = 0;
    for (size_t r = 0; r < rk_.size(); ++r) {
        bytes[r] = q.pages.size() * rk_[r].page_bytes + state_bytes((int) r);
        total += bytes[r];
    }
    if (host_used() + total > host_budget_)
        throw std::runtime_error("seq_offload: host budget exceeded (" + std::to_string((host_used() + total) >> 20) +
                                 " MiB > " + std::to_string(host_budget_ >> 20) + " MiB)");
    sync_all();
    q.host.assign(rk_.size(), nullptr);
    for (size_t r = 0; r < rk_.size(); ++r) {
        CK(cudaSetDevice(rk_[r].dev));
        if (cudaHostAlloc((void **) &q.host[r], bytes[r], cudaHostAllocPortable) != cudaSuccess) {
            cudaGetLastError();
            for (char *& h : q.host) if (h) { cudaFreeHost(h); h = nullptr; }
            throw std::runtime_error("seq_offload: pinned host allocation of " + std::to_string(bytes[r] >> 20) + " MiB failed");
        }
    }
    for (size_t r = 0; r < rk_.size(); ++r) {
        rank_state & S = rk_[r];
        const size_t pblk = S.pblk, pb = S.page_bytes, csz = S.csz, ssz = S.ssz;
        CK(cudaSetDevice(S.dev));
        char * h = q.host[r];
        const size_t region = (size_t) n_pages_ * pblk;   // one layer's K (or V) pages
        for (size_t i = 0; i < q.pages.size(); ++i)
            for (int kv = 0; kv < 2; ++kv)   // [kv][layer] blocks of this page -> host [page][kv][layer][pblk]
                CK(cudaMemcpy2DAsync(h + i * pb + (size_t) kv * n_kvl_ * pblk, pblk,
                                     S.kv_pool + (size_t) kv * n_kvl_ * region + (size_t) q.pages[i] * pblk, region, pblk,
                                     n_kvl_, cudaMemcpyDeviceToHost, S.s));
        h += q.pages.size() * pb;
        CK(cudaMemcpyAsync(h, S.conv_pool + (size_t) q.slot * n_gdn_ * csz, (size_t) n_gdn_ * csz * 4, cudaMemcpyDeviceToHost, S.s));
        CK(cudaMemcpyAsync(h + (size_t) n_gdn_ * csz * 4, S.ssm_pool + (size_t) q.slot * n_gdn_ * ssz, (size_t) n_gdn_ * ssz * 4,
                           cudaMemcpyDeviceToHost, S.s));
    }
    sync_all();
    q.host_pages = (int) q.pages.size();
    q.host_bytes = total;
    host_bytes_ += q.host_bytes;
    for (auto it = q.pages.rbegin(); it != q.pages.rend(); ++it) free_pages_.push_back(*it);
    q.pages.clear();
    free_slots_.push_back(q.slot);
    q.slot = -1;
    if (active_ == id) active_ = -1;
}

void tp_model::seq_restore(int id) {
    const seq_t & q = seqs_.at(id);
    if (!q.used || q.slot >= 0) throw std::runtime_error("seq_restore: sequence not offloaded");
    restore_impl(id, q.host_pages, true);
}

// offloaded -> resident with the first np saved pages; state: also the delta-net slot (the host copy's, i.e. of n_past)
void tp_model::restore_impl(int id, int np, bool state) {
    seq_t & q = seqs_.at(id);
    if (!q.used || q.slot >= 0) throw std::runtime_error("seq_restore: sequence not offloaded");
    if (np < 0 || np > q.host_pages) throw std::runtime_error("seq_restore: page count out of range");
    if (free_slots_.empty()) throw std::runtime_error("seq_restore: no free sequence slot");
    if ((int) free_pages_.size() < np) throw std::runtime_error("seq_restore: KV pool full");
    sync_all();
    q.slot = free_slots_.back();
    free_slots_.pop_back();
    q.pages.clear();
    if (np > 0) ensure_pages(id, std::min(np << psh_, n_ctx_));   // np pages (+ the table upload)
    const int hp = q.host_pages;
    for (size_t r = 0; r < rk_.size(); ++r) {
        rank_state & S = rk_[r];
        const size_t pblk = S.pblk, pb = S.page_bytes, csz = S.csz, ssz = S.ssz;
        CK(cudaSetDevice(S.dev));
        const char * h = q.host[r];
        const size_t region = (size_t) n_pages_ * pblk;
        for (int i = 0; i < np; ++i)
            for (int kv = 0; kv < 2; ++kv)
                CK(cudaMemcpy2DAsync(S.kv_pool + (size_t) kv * n_kvl_ * region + (size_t) q.pages[i] * pblk, region,
                                     h + (size_t) i * pb + (size_t) kv * n_kvl_ * pblk, pblk, pblk, n_kvl_,
                                     cudaMemcpyHostToDevice, S.s));
        h += (size_t) hp * pb;
        if (state) {
            CK(cudaMemcpyAsync(S.conv_pool + (size_t) q.slot * n_gdn_ * csz, h, (size_t) n_gdn_ * csz * 4, cudaMemcpyHostToDevice, S.s));
            CK(cudaMemcpyAsync(S.ssm_pool + (size_t) q.slot * n_gdn_ * ssz, h + (size_t) n_gdn_ * csz * 4, (size_t) n_gdn_ * ssz * 4,
                               cudaMemcpyHostToDevice, S.s));
        }
    }
    sync_all();
    for (char * h : q.host) if (h) cudaFreeHost(h);
    q.host.clear();
    host_bytes_ -= q.host_bytes;
    q.host_bytes = 0;
    q.host_pages = 0;
}

// ---- prompt cache primitives ----

void tp_model::seq_truncate(int id, int n) {
    seq_t & q = seqs_.at(id);
    if (!q.used || q.slot < 0) throw std::runtime_error("seq_truncate: sequence not resident");
    if (n < 0 || n > q.n_past) throw std::runtime_error("seq_truncate: position out of range");
    if (n == q.n_past) return;
    q.mmt.check_cut(n);   // a cut inside an image: refused before anything changes
    sync_all();   // nothing in flight may still read the pages given back, or the page-table entries rewritten later
    if (paged_) {
        const int keep = (n + (1 << psh_) - 1) >> psh_;
        for (int i = (int) q.pages.size() - 1; i >= keep; --i) free_pages_.push_back(q.pages[i]);
        if ((int) q.pages.size() > keep) q.pages.resize(keep);
    }
    q.n_past = n;
    q.toks.resize(n);
    q.rope_delta = q.mmt.cut(n);
}

void tp_model::seq_restore_prefix(int id, int n) {
    seq_t & q = seqs_.at(id);
    if (!q.used || q.slot >= 0) throw std::runtime_error("seq_restore_prefix: sequence not offloaded");
    if (n < 0 || n > q.n_past) throw std::runtime_error("seq_restore_prefix: position out of range");
    q.mmt.check_cut(n);
    restore_impl(id, std::min(q.host_pages, (n + (1 << psh_) - 1) >> psh_), n == q.n_past);
    q.n_past = n;
    q.toks.resize(n);
    q.rope_delta = q.mmt.cut(n);
}

int tp_model::seq_fork(int src, int n) {
    if (!paged_) throw std::runtime_error("seq_fork needs the paged KV pool");
    {
        const seq_t & q = seqs_.at(src);
        if (!q.used || q.slot < 0) throw std::runtime_error("seq_fork: source not resident");
        if (n < 0 || n > q.n_past) throw std::runtime_error("seq_fork: position out of range");
        q.mmt.check_cut(n);
        const int np = (n + (1 << psh_) - 1) >> psh_;
        if (free_slots_.empty()) throw std::runtime_error("seq_fork: no free sequence slot");
        if ((int) free_pages_.size() < np) throw std::runtime_error("seq_fork: KV pool full");
    }
    const int id = seq_create();   // may grow seqs_: references taken after
    seq_t & q = seqs_.at(src);
    seq_t & d = seqs_.at(id);
    const int np = (n + (1 << psh_) - 1) >> psh_;
    if (np > 0) ensure_pages(id, std::min(np << psh_, n_ctx_));
    for (auto & S : rk_) {
        const size_t region = (size_t) n_pages_ * S.pblk;
        CK(cudaSetDevice(S.dev));
        for (int i = 0; i < np; ++i)   // every KV layer's K and V of page i: one 2-D copy (rows = 2 x n_kvl, pitch region)
            CK(cudaMemcpy2DAsync(S.kv_pool + (size_t) d.pages[i] * S.pblk, region, S.kv_pool + (size_t) q.pages[i] * S.pblk,
                                 region, S.pblk, (size_t) 2 * n_kvl_, cudaMemcpyDeviceToDevice, S.s));
    }
    sync_all();
    d.n_past = n;
    d.toks.assign(q.toks.begin(), q.toks.begin() + n);
    d.mmt = q.mmt;
    d.rope_delta = d.mmt.cut(n);
    return id;
}

void tp_model::state_save(int id, const std::vector<char *> & dst) {
    const seq_t & q = seqs_.at(id);
    if (!q.used || q.slot < 0) throw std::runtime_error("state_save: sequence not resident");
    if (dst.size() != rk_.size()) throw std::runtime_error("state_save: one buffer per GPU");
    rb_join();
    for (size_t r = 0; r < rk_.size(); ++r) {
        rank_state & S = rk_[r];
        CK(cudaSetDevice(S.dev));
        CK(cudaMemcpyAsync(dst[r], S.conv_pool + (size_t) q.slot * n_gdn_ * S.csz, (size_t) n_gdn_ * S.csz * 4,
                           cudaMemcpyDeviceToHost, S.s));
        CK(cudaMemcpyAsync(dst[r] + (size_t) n_gdn_ * S.csz * 4, S.ssm_pool + (size_t) q.slot * n_gdn_ * S.ssz,
                           (size_t) n_gdn_ * S.ssz * 4, cudaMemcpyDeviceToHost, S.s));
        // an uneven split: the tail of a smaller rank's state_bytes() buffer is zeroed (fixed buffer contents)
        if (state_bytes((int) r) < state_bytes()) memset(dst[r] + state_bytes((int) r), 0, state_bytes() - state_bytes((int) r));
    }
    sync_all();
}

void tp_model::state_save_async(int id, const std::vector<char *> & dst) {
    const seq_t & q = seqs_.at(id);
    if (!q.used || q.slot < 0) throw std::runtime_error("state_save_async: sequence not resident");
    if (dst.size() != rk_.size()) throw std::runtime_error("state_save_async: one buffer per GPU");
    rb_join();
    for (size_t r = 0; r < rk_.size(); ++r) {
        rank_state & S = rk_[r];
        const size_t cb = (size_t) n_gdn_ * S.csz * 4, sb = (size_t) n_gdn_ * S.ssz * 4;
        // an uneven split: the tail of a smaller rank's state_bytes() buffer is zeroed (host side; the D2H below writes
        // only the first cb + sb bytes)
        if (cb + sb < state_bytes()) memset(dst[r] + cb + sb, 0, state_bytes() - cb - sb);
        CK(cudaSetDevice(S.dev));
        if (!S.ck_s) {
            CK(cudaMalloc(&S.ck_stage, cb + sb));
            CK(cudaStreamCreateWithFlags(&S.ck_s, cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&S.ev_ck_in, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&S.ev_ck, cudaEventDisableTiming));
        } else if (ck_pending_) {
            CK(cudaStreamWaitEvent(S.s, S.ev_ck, 0));   // the previous D2H must have left the staging buffer
        }
        // D2D on S.s: ordered after the forward that produced the state and before the next one that overwrites it
        CK(cudaMemcpyAsync(S.ck_stage, S.conv_pool + (size_t) q.slot * n_gdn_ * S.csz, cb, cudaMemcpyDeviceToDevice, S.s));
        CK(cudaMemcpyAsync(S.ck_stage + cb, S.ssm_pool + (size_t) q.slot * n_gdn_ * S.ssz, sb, cudaMemcpyDeviceToDevice, S.s));
        CK(cudaEventRecord(S.ev_ck_in, S.s));
        CK(cudaStreamWaitEvent(S.ck_s, S.ev_ck_in, 0));
        CK(cudaMemcpyAsync(dst[r], S.ck_stage, cb + sb, cudaMemcpyDeviceToHost, S.ck_s));
        CK(cudaEventRecord(S.ev_ck, S.ck_s));
    }
    ck_pending_ = true;
}

void tp_model::state_save_join() {
    if (!ck_pending_) return;
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        CK(cudaEventSynchronize(S.ev_ck));
    }
    ck_pending_ = false;
}

void tp_model::state_load(int id, const std::vector<const char *> & src) {
    const seq_t & q = seqs_.at(id);
    if (!q.used || q.slot < 0) throw std::runtime_error("state_load: sequence not resident");
    if (src.size() != rk_.size()) throw std::runtime_error("state_load: one buffer per GPU");
    rb_join();
    for (size_t r = 0; r < rk_.size(); ++r) {
        rank_state & S = rk_[r];
        CK(cudaSetDevice(S.dev));
        CK(cudaMemcpyAsync(S.conv_pool + (size_t) q.slot * n_gdn_ * S.csz, src[r], (size_t) n_gdn_ * S.csz * 4,
                           cudaMemcpyHostToDevice, S.s));
        CK(cudaMemcpyAsync(S.ssm_pool + (size_t) q.slot * n_gdn_ * S.ssz, src[r] + (size_t) n_gdn_ * S.csz * 4,
                           (size_t) n_gdn_ * S.ssz * 4, cudaMemcpyHostToDevice, S.s));
    }
    sync_all();
}

void tp_model::state_zero(int id) {
    const seq_t & q = seqs_.at(id);
    if (!q.used || q.slot < 0) throw std::runtime_error("state_zero: sequence not resident");
    rb_join();
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        CK(cudaMemsetAsync(S.ssm_pool + (size_t) q.slot * n_gdn_ * S.ssz, 0, (size_t) n_gdn_ * S.ssz * 4, S.s));
        CK(cudaMemsetAsync(S.conv_pool + (size_t) q.slot * n_gdn_ * S.csz, 0, (size_t) n_gdn_ * S.csz * 4, S.s));
    }
    sync_all();
}

// rank r's resident pages p0..p0+n-1 of sequence id -> dst ([page][K|V][kv layer][page rows], the offload layout), async
void tp_model::pages_out(int id, int r, int p0, int n, char * dst) {
    const seq_t & q = seqs_.at(id);
    rank_state & S = rk_[r];
    CK(cudaSetDevice(S.dev));
    const size_t pblk = S.pblk, region = (size_t) n_pages_ * pblk;
    for (int i = 0; i < n; ++i)
        for (int kv = 0; kv < 2; ++kv)
            CK(cudaMemcpy2DAsync(dst + (size_t) i * S.page_bytes + (size_t) kv * n_kvl_ * pblk, pblk,
                                 S.kv_pool + (size_t) kv * n_kvl_ * region + (size_t) q.pages[p0 + i] * pblk, region,
                                 pblk, n_kvl_, cudaMemcpyDeviceToHost, S.s));
}

// sequence file: "P100ESQ2" | fingerprint (15 x u64) | n_past, n_tok, pages (i32) | tokens | per GPU: pages x page_bytes
// ([page][K|V][kv layer][page rows]: the offload layout, independent of the pool size) + state [conv | ssm] (an uneven
// split: each GPU's in its own sizes; the split is in the "GPU count" field: R, | its range hash << 8 when uneven)
static const char kSeqMagic[8] = {'P', '1', '0', '0', 'E', 'S', 'Q', '2'};
static const char * kFpNames[15] = {"model file size", "tensor count", "embedding", "layers", "vocab", "KV type",
                                    "page shift", "GPU count / tensor-parallel split (--p100e-split, P100E_SPLIT)",
                                    "page bytes", "state bytes", "KV layers", "MTP block",
                                    "GGUF metadata hash", "tensor table hash", "tensor content sample hash"};

// P100E_* switches that never change the numbers a forward writes (output / debug only, memory budgets, cache and draft
// policy, stream priorities, CUDA graphs = replays of the same kernels, rollback copy method, allreduce handshake);
// every other P100E_* variable counts in the numerics hash (an unknown new switch is assumed to change numerics)
static const std::vector<std::string> kNumericsNeutralEnv = {
    "P100E_GRAPH_QUIET", "P100E_HOST_BUDGET_MB", "P100E_CKPT_BUDGET_MB", "P100E_CACHE", "P100E_CACHE_EXACT",
    "P100E_DECODE_SHARE", "P100E_DEBUG_NAN", "P100E_PF_DEBUG", "P100E_LOGITS_HASH", "P100E_MTP_PROF", "P100E_MTP_CALIB",
    "P100E_MTP_CHECK", "P100E_PRINT_LOGITS", "P100E_TOPK", "P100E_SESS_ORACLE", "P100E_DRAFT_BATCH", "P100E_FAST_CHECK",
    "P100E_PF_PRIO", "P100E_RB_ASYNC", "P100E_RB_KERNEL", "P100E_WPAD", "P100E_GRAPH",
    "P100E_DRAFT_GRAPH", "P100E_AR_SAFE",
    "P100E_SPLIT",   // the split itself is hashed (identity_init: an uneven split's range table), not its spelling
    "P100E_MERGE_INPROJ", "P100E_GDN_FUSE"};   // launch merges, bit-identical (record 'STAT p100e decode merges ...')

void tp_model::identity_init() {
    // an even split (dim / R everywhere) keeps the pre-'tp-uneven' identity: R alone (old session files stay loadable,
    // the numerics hash unchanged); an uneven one adds its range table (another split = other weights per GPU)
    const uint64_t gpus = (uint64_t) R_ | (split_.even ? 0 : split_.hash() << 8);   // (R_: every node's ranks)
    fp_ = {g_.file_size, (uint64_t) g_.tensors.size(), (uint64_t) E_, (uint64_t) L_, (uint64_t) VOCAB_,
           (uint64_t) kv_type_, (uint64_t) psh_, gpus, (uint64_t) page_bytes_max_, (uint64_t) state_bytes(),
           (uint64_t) n_kvl_, (uint64_t) n_nextn_, g_.meta_hash(), g_.tensor_table_hash(), g_.tensor_sample_hash()};
    uint64_t h = kHashSeed;
    auto add = [&](uint64_t v) { h = hash_bytes(h, &v, sizeof v); };
    add(NUMERICS_VERSION);
    add((uint64_t) max_batch_);
    add((uint64_t) kv_type_);
    add((uint64_t) R_);
    if (!split_.even) add(split_.hash());
    for (const auto & S : rk_) {
        cudaDeviceProp pr;
        CK(cudaGetDeviceProperties(&pr, S.dev));
        add((uint64_t) (pr.major * 10 + pr.minor));
    }
    int rt = 0;
    cudaRuntimeGetVersion(&rt);
    add((uint64_t) rt);
    int cbv = 0;
    if (!rk_.empty() && rk_[0].cb) cublasGetVersion(rk_[0].cb, &cbv);
    add((uint64_t) cbv);
    add(env_hash("P100E_", kNumericsNeutralEnv, &numerics_env_));
    numerics_ = h;
}

void tp_model::seq_write(int id, FILE * f) {
    net_refuse("seq_write (session files)");
    seq_t & q = seqs_.at(id);
    if (!q.used) throw std::runtime_error("seq_write: no such sequence");
    if (!q.mmt.spans.empty()) throw std::runtime_error("seq_write: sequences with images are not saved (M-RoPE state)");
    if (!paged_) throw std::runtime_error("seq_write needs the paged KV pool");
    const std::vector<uint64_t> fp = fingerprint();
    const bool res = q.slot >= 0;
    const int np = res ? (int) q.pages.size() : q.host_pages;
    const int32_t hdr[3] = {q.n_past, (int32_t) q.toks.size(), np};
    auto wr = [&](const void * p, size_t n) {
        if (n && fwrite(p, 1, n, f) != n) throw std::runtime_error("seq_write: write failed (disk full?)");
    };
    wr(kSeqMagic, 8);
    wr(fp.data(), fp.size() * 8);
    wr(hdr, sizeof hdr);
    wr(q.toks.data(), q.toks.size() * 4);
    if (!res) {
        for (size_t r = 0; r < rk_.size(); ++r) wr(q.host[r], (size_t) np * rk_[r].page_bytes + state_bytes((int) r));
        return;
    }
    // resident: through a pinned staging buffer, up to 32 pages at a time
    sync_all();
    const int chunk = std::max(1, std::min(np, 32));
    char * st = nullptr;
    if (cudaHostAlloc((void **) &st, std::max((size_t) chunk * page_bytes_max_, state_bytes()), cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        throw std::runtime_error("seq_write: pinned staging allocation failed");
    }
    try {
        for (size_t r = 0; r < rk_.size(); ++r) {
            rank_state & S = rk_[r];
            for (int p0 = 0; p0 < np; p0 += chunk) {
                const int n = std::min(chunk, np - p0);
                pages_out(id, (int) r, p0, n, st);
                CK(cudaStreamSynchronize(S.s));
                wr(st, (size_t) n * S.page_bytes);
            }
            CK(cudaSetDevice(S.dev));
            CK(cudaMemcpyAsync(st, S.conv_pool + (size_t) q.slot * n_gdn_ * S.csz, (size_t) n_gdn_ * S.csz * 4,
                               cudaMemcpyDeviceToHost, S.s));
            CK(cudaMemcpyAsync(st + (size_t) n_gdn_ * S.csz * 4, S.ssm_pool + (size_t) q.slot * n_gdn_ * S.ssz,
                               (size_t) n_gdn_ * S.ssz * 4, cudaMemcpyDeviceToHost, S.s));
            CK(cudaStreamSynchronize(S.s));
            wr(st, state_bytes((int) r));
        }
    } catch (...) {
        cudaFreeHost(st);
        throw;
    }
    cudaFreeHost(st);
}

int tp_model::seq_read(FILE * f) {
    net_refuse("seq_read (session files)");
    if (!paged_ || !pool_ok_) throw std::runtime_error("seq_read needs the paged KV pool");
    auto rd = [&](void * p, size_t n) {
        if (n && fread(p, 1, n, f) != n) throw std::runtime_error("seq_read: short file");
    };
    char mg[8];
    rd(mg, 8);
    if (!memcmp(mg, "P100ESQ1", 8))
        throw std::runtime_error("seq_read: a version 1 sequence file (no content / numerics fingerprint): not loadable");
    if (memcmp(mg, kSeqMagic, 8)) throw std::runtime_error("seq_read: not a p100e sequence file");
    const std::vector<uint64_t> want = fingerprint();
    std::vector<uint64_t> fp(want.size());
    rd(fp.data(), fp.size() * 8);
    for (size_t i = 0; i < want.size(); ++i)
        if (fp[i] != want[i]) {
            if (i == 7)   // GPU count | split hash << 8: name both parts
                throw std::runtime_error("seq_read: saved with another tensor-parallel layout (" +
                                         std::to_string(fp[i] & 0xff) + " GPUs, " + (fp[i] >> 8 ? "uneven" : "even") +
                                         " split vs " + std::to_string(want[i] & 0xff) + " GPUs, " +
                                         (want[i] >> 8 ? "uneven split " + split_.str() : std::string("even split")) +
                                         "; the weights per GPU must match: --p100e-split / P100E_SPLIT)");
            throw std::runtime_error(std::string("seq_read: saved with another model / KV layout (") + kFpNames[i] + " " +
                                     std::to_string(fp[i]) + " vs " + std::to_string(want[i]) + ")");
        }
    int32_t hdr[3];
    rd(hdr, sizeof hdr);
    const int n_past = hdr[0], ntok = hdr[1], np = hdr[2];
    if (n_past < 0 || n_past > n_ctx_ || ntok != n_past || np < 0 || np > max_pages_ || ((int64_t) np << psh_) < n_past)
        throw std::runtime_error("seq_read: bad header (or the sequence exceeds this engine's context)");
    std::vector<int32_t> toks(ntok);
    rd(toks.data(), (size_t) ntok * 4);
    for (int32_t t : toks)
        if (t < -1 || t >= VOCAB_) throw std::runtime_error("seq_read: token id " + std::to_string(t) + " out of range");
    std::vector<size_t> bytes(rk_.size());   // rank r: [np][its page_bytes] | its state (the offload layout)
    size_t total = 0;
    for (size_t r = 0; r < rk_.size(); ++r) {
        bytes[r] = (size_t) np * rk_[r].page_bytes + state_bytes((int) r);
        total += bytes[r];
    }
    if (host_used() + total > host_budget_)
        throw std::runtime_error("seq_read: host budget exceeded (" + std::to_string((host_used() + total) >> 20) +
                                 " MiB > " + std::to_string(host_budget_ >> 20) + " MiB)");
    std::vector<char *> host(rk_.size(), nullptr);
    try {
        for (size_t r = 0; r < rk_.size(); ++r) {
            CK(cudaSetDevice(rk_[r].dev));
            if (cudaHostAlloc((void **) &host[r], bytes[r], cudaHostAllocPortable) != cudaSuccess) {
                cudaGetLastError();
                throw std::runtime_error("seq_read: pinned host allocation of " + std::to_string(bytes[r] >> 20) + " MiB failed");
            }
            rd(host[r], bytes[r]);
        }
    } catch (...) {
        for (char * h : host) if (h) cudaFreeHost(h);
        throw;
    }
    int id = -1;
    for (int i = 0; i < (int) seqs_.size() && id < 0; ++i) if (!seqs_[i].used) id = i;
    if (id < 0) { id = (int) seqs_.size(); seqs_.emplace_back(); }
    seq_t & q = seqs_[id];
    q = seq_t{};
    q.used = true;
    q.slot = -1;
    q.n_past = n_past;
    q.toks = std::move(toks);
    q.host = host;
    q.host_pages = np;
    q.host_bytes = total;
    host_bytes_ += q.host_bytes;
    return id;
}

void tp_model::prefill_setup() {
    if (prefill_ok()) pf_setup();   // the prefill pipeline's streams, f16 weight ring and activation buffers
    // the decode attention's split scratch (sized for MAX_ROWS rows, allocated on first use)
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        (void) fast::attn_dec2_nsplit(S.HKVl);
    }
    const gguf_tensor * emb = g_.find("token_embd.weight");
    if (!emb_h_) CK(cudaMallocHost(&emb_h_, (size_t) max_batch_ * (emb->nbytes / emb->ne[1])));
}

// KV pool: one allocation per GPU, [K | V][kv layer][page][2^psh][row]: each layer's K (and V) pages are one contiguous
// region, so a sequence on consecutive pages (the free list hands them out in order) reads exactly the contiguous layout
// of an unpaged cache (TLB / DRAM locality as before paging); a page across layers is a 2-D block (offload: two
// cudaMemcpy2DAsync per page and GPU). Every sequence's page table row maps its positions into it. Sized by tokens, or
// (tokens < 0) by the free memory of the smaller GPU.
void tp_model::alloc_pool(int64_t tokens, const std::vector<size_t> & reserve_mb, size_t margin_mb) {
    if (!paged_) { pool_ok_ = true; return; }
    if (pool_ok_) throw std::runtime_error("alloc_pool: the pool exists already");
    const int R = (int) rk_.size();
    const int ps = 1 << psh_;
    page_bytes_max_ = 0;
    for (auto & S : rk_) {   // each rank's pages hold its own KV heads (an uneven split: different page sizes)
        const size_t rowb = fast::kv_row_bytes(kv_type_, S.HKVl, HD_);
        S.pblk = (size_t) ps * rowb;            // one page of one layer's K (or V)
        S.page_bytes = (size_t) n_kvl_ * 2 * S.pblk;
        page_bytes_max_ = std::max(page_bytes_max_, S.page_bytes);
    }
    sync_all();
    drop_graphs();
    int64_t np = 0;
    // default reserve: 1024 MiB on the last GPU (the vision encoder, ~950 MiB at a 2048-token image cap)
    std::vector<size_t> res = reserve_mb;
    if (res.empty()) { res.assign(R, 0); res.back() = 1024; }
    if (tokens >= 0) {
        np = (tokens + ps - 1) >> psh_;
    } else {
        np = INT64_MAX;
        for (int r = 0; r < R; ++r) {
            CK(cudaSetDevice(rk_[r].dev));
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            const size_t keep = ((r < (int) res.size() ? res[r] : 0) + margin_mb) << 20;
            // ranks sharing this GPU (local tests: --devices 0,0,0,0) split its free memory: one page of every rank on
            // it per pool page (equal pages: (free / page) / ranks, the same floor as before)
            size_t share = 0;
            for (const auto & o : rk_) if (o.dev == rk_[r].dev) share += o.page_bytes;
            const int64_t p = fr > keep ? (int64_t) ((fr - keep) / std::max<size_t>(1, share)) : 0;
            np = std::min(np, p);
        }
        np = std::min<int64_t>(np, (int64_t) max_slots_ * max_pages_);   // more than every slot at n_ctx is useless
    }
    if (np < 1) throw std::runtime_error("alloc_pool: no room for a single KV page");
    // the decode attention addresses a layer's K (or V) region with 32-bit offsets
    for (const auto & S : rk_)
        if ((size_t) np * S.pblk >= ((size_t) 1 << 32)) np = (int64_t) ((((size_t) 1 << 32) - 1) / S.pblk);
    // the same pool (page count) on every node: identical page allocation, lockstep (pipe mode too: a node whose pool
    // ran out while the other's did not would leave the common control flow)
    if (net_ || pnet_) np = net_min(np);
    if (np > INT32_MAX / 2) throw std::runtime_error("alloc_pool: too many pages");
    for (int r = 0; r < R; ++r) {
        rank_state & S = rk_[r];
        CK(cudaSetDevice(S.dev));
        if (cudaMalloc(&S.kv_pool, (size_t) np * S.page_bytes) != cudaSuccess) {
            cudaGetLastError();
            throw std::runtime_error("alloc_pool: cudaMalloc of " + std::to_string(((size_t) np * S.page_bytes) >> 20) +
                                     " MiB failed on GPU " + std::to_string(S.dev));
        }
        // finite contents: the fast kernels may read (and weight by zero) rows past a sequence's end inside a page
        CK(cudaMemset(S.kv_pool, 0, (size_t) np * S.page_bytes));
        for (int il = 0; il < (int) S.kv.size(); ++il) {
            const int li = kvl_of_[il];
            if (li < 0) continue;
            fast::kv_cache & c = S.kv[il];
            c.k = S.kv_pool + (size_t) li * np * S.pblk;
            c.v = S.kv_pool + (size_t) (n_kvl_ + li) * np * S.pblk;
            c.pt = S.ptab; c.rtab = nullptr; c.psh = psh_; c.pstride = S.pblk;
            c.n_ctx = n_ctx_;
        }
    }
    n_pages_ = (int) np;
    free_pages_.clear();
    for (int p = n_pages_ - 1; p >= 0; --p) free_pages_.push_back(p);
    pool_ok_ = true;
    CK(cudaSetDevice(rk_[0].dev));
    for (auto & S : rk_) { CK(cudaSetDevice(S.dev)); CK(cudaDeviceSynchronize()); }
    std::string fr_s;
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        fr_s += " GPU " + std::to_string(S.dev) + " " + std::to_string(fr >> 20) + " MiB";
    }
    std::string how = tokens >= 0 ? std::string("requested") : "free VRAM minus reserve";
    if (tokens < 0) for (size_t r = 0; r < res.size(); ++r) how += (r ? "/" : " ") + std::to_string(res[r]);
    if (tokens < 0) how += " MiB + margin " + std::to_string(margin_mb) + " MiB";
    printf("KV pool: %d pages x %d positions = %lld tokens (%s; %s, %d KV layers, %.2f MiB per page and GPU, %.0f MiB per "
           "GPU%s), %d sequence slot(s), per-sequence cap %d; free after:%s\n",
           n_pages_, ps, (long long) pool_tokens(), how.c_str(),
           kv_type_ == fast::KV_F16 ? "f16" : kv_type_ == fast::KV_Q8_0 ? "q8_0" : "q4_0", n_kvl_, page_bytes_max_ / 1048576.0,
           (double) n_pages_ * page_bytes_max_ / 1048576.0, split_.even ? "" : " on the GPU with the most KV heads",
           max_slots_, n_ctx_, fr_s.c_str());
}

// One fused step over rows of several sequences (graph per (T, nseq, keep)). Descriptor: row t of sequence i = position
// n_past(i) + t, its page-table row; delta-net sequence i = (first row, rows, state slot, first snapshot slot).
std::vector<int32_t> tp_model::forward_batch_ids(const std::vector<seq_rows> & b, bool keep_states,
                                                  const std::vector<int> * logit_rows, std::vector<float> * logits) {
    net_refuse("forward_batch_ids (batched sequences, the server)");
    const int nseq = (int) b.size();
    if (!pool_ok_) throw std::runtime_error("forward_batch_ids: KV pool not allocated");
    if (!(use_fast_ && fuse_ && plan_ok_)) throw std::runtime_error("forward_batch_ids needs the fused decode path");
    if (nseq < 1 || nseq > MAX_ROWS) throw std::runtime_error("forward_batch_ids: 1..8 sequences");
    int T = 0, nsnap = 0;
    for (int i = 0; i < nseq; ++i) {
        const int Ti = (int) b[i].tokens.size();
        if (Ti < 1 || Ti > 4) throw std::runtime_error("forward_batch_ids: 1..4 rows per sequence");
        for (int j = 0; j < i; ++j) if (b[j].seq == b[i].seq) throw std::runtime_error("forward_batch_ids: duplicate sequence");
        const seq_t & q = seqs_.at(b[i].seq);
        if (!q.used || q.slot < 0) throw std::runtime_error("forward_batch_ids: sequence not resident");
        if (q.n_past + Ti > n_ctx_) throw std::runtime_error("context full");
        T += Ti;
        nsnap += Ti - 1;
    }
    if (T > MAX_ROWS) throw std::runtime_error("forward_batch_ids: more than 8 rows");
    // logit rows checked before the step changes any state (logits_h_ holds MAX_ROWS rows)
    if (logit_rows) {
        if (logit_rows->size() > (size_t) MAX_ROWS) throw std::runtime_error("forward_batch_ids: more than 8 logit rows");
        for (const int row : *logit_rows)
            if (row < 0 || row >= T) throw std::runtime_error("forward_batch_ids: logit row out of range");
    }
    const bool keep = keep_states && T > nseq;
    if (keep && nsnap > n_snap_) throw std::runtime_error("forward_batch_ids: snapshots beyond enable_rollback()");
    rb_join();
    if (nseq == 1) seq_select(b[0].seq);   // the host-pointer fallbacks (P100E_GDN_BATCH=0) use the active sequence
    for (const auto & s : b) ensure_pages(s.seq, seqs_[s.seq].n_past + (int) s.tokens.size());
    // descriptor
    std::vector<int32_t> toks;
    int row = 0, sn = 0;
    bool mr = false;   // a sequence with images: every row gets its rotary position (S.rpos)
    for (int i = 0; i < nseq; ++i) {
        seq_t & q = seqs_[b[i].seq];
        const int Ti = (int) b[i].tokens.size();
        for (int t = 0; t < Ti; ++t) {
            pos_h_[row + t] = q.n_past + t;
            desc_h_[DESC_RT + row + t] = q.slot * max_pages_;
            mr = stage_rpos_text(row + t, q, q.n_past + t) || mr;
        }
        int32_t * g = desc_h_ + DESC_GS + 4 * i;
        g[0] = row; g[1] = Ti; g[2] = q.slot; g[3] = sn;
        q.snap0 = sn;
        if (keep) sn += Ti - 1;
        row += Ti;
        toks.insert(toks.end(), b[i].tokens.begin(), b[i].tokens.end());
    }
    for (auto & S : rk_) {
        CK(cudaSetDevice(S.dev));
        CK(cudaMemcpyAsync(S.desc, desc_h_, (size_t) (DESC_POS + T) * 4, cudaMemcpyHostToDevice, S.s));
    }
    step_mr_ = mr;
    if (mr) upload_rpos(T);
    embed(toks);
    bool capture = false;
    step_graph * G = T <= graph_maxt_ ? find_graph(T, keep, nseq) : nullptr;
    if (G) {
        launch_graph(*G);
    } else {
        fused_body(T, keep, nseq);
        capture = T <= graph_maxt_ && !graph_failed_;
    }
    bool need_argmax = true;
    if (logit_rows && (int) logit_rows->size() == T) {
        need_argmax = false;
        for (int j = 0; j < T; ++j) {
            if ((*logit_rows)[j] != j) {
                need_argmax = true;
                break;
            }
        }
    }
    for (int r = 0; r < (int) rk_.size(); ++r) {
        rank_state & S = rk_[r];
        CK(cudaSetDevice(S.dev));
        if (need_argmax) argmax_launch(r, S.logits, T);
        if (logit_rows) {
            const int VL = S.VL;
            for (size_t j = 0; j < logit_rows->size(); ++j) {
                const int row = (*logit_rows)[j];   // (range checked above)
                CK(cudaMemcpyAsync(logits_h_ + j * VOCAB_ + (size_t) S.v0, S.logits + (size_t) row * VL, (size_t) VL * 4,
                                   cudaMemcpyDeviceToHost, S.s));
            }
        }
    }
    for (auto & S : rk_) { CK(cudaSetDevice(S.dev)); CK(cudaStreamSynchronize(S.s)); }
    if (logit_rows && logits) logits->assign(logits_h_, logits_h_ + logit_rows->size() * (size_t) VOCAB_);
    for (const auto & s : b) {
        seq_t & q = seqs_[s.seq];
        q.n_past += (int) s.tokens.size();
        q.toks.insert(q.toks.end(), s.tokens.begin(), s.tokens.end());
    }
    if (capture) capture_graph(T, keep, nseq);
    return need_argmax ? argmax_collect(T) : std::vector<int32_t>(T, 0);
}

// =====================================================================================================================
// Network tensor parallel: ranks on several machines, SPMD lockstep (tp_model.h, src/net_link.h)
// =====================================================================================================================

// P100E_* switches that change which paths / collectives a node runs (chunking, wire width, fused vs per-op, prefill
// pipeline vs classic, the split). They must be equal on every node or the nodes would leave the common control flow;
// everything else (kernel variants of one rank, debug output, priorities) may differ per node.
static const char * const kNetControlEnv[] = {
    "P100E_FAST", "P100E_FUSE", "P100E_FUSE_VERIFY", "P100E_PF", "P100E_FAST_MAXT", "P100E_GEMM", "P100E_PF_CHUNKS",
    "P100E_PF_WIRE16", "P100E_PF_TEST_SAT", "P100E_PAGED", "P100E_PAGE_SHIFT", "P100E_SPLIT", "P100E_CONCAT",
    "P100E_NET"};   // the transport
// (P100E_NET_PREADD sets message sizes: its effective value is agreed in init; P100E_NET_GRAPH / _SPIN are node-local)

void tp_model::net_refuse(const char * what) const {
    if (net_) throw std::runtime_error(std::string(what) + ": not supported over the network (--net-nodes) yet");
    if (pnet_) throw std::runtime_error(std::string(what) + ": not supported in the hybrid pipeline (--pipe-layers) yet");
}

void tp_model::net_init(const std::vector<int> & devices, const opts & o) {
    auto envi = [](const char * n, int d) { const char * e = getenv(n); return e && *e ? atoi(e) : d; };
    const int connect_s = envi("P100E_NET_CONNECT_S", o.net_connect_s), io_s = envi("P100E_NET_TIMEOUT_S", o.net_timeout_s);
    std::string list;
    for (size_t i = 0; i < o.net_nodes.size(); ++i) list += (i ? "," : "") + o.net_nodes[i];
    printf("net: node %d of %s, %zu local GPU(s); waiting up to %d s for the other nodes\n", o.net_node, list.c_str(),
           devices.size(), connect_s);
    fflush(stdout);
    net_ = net_transport_open(o.net_nodes, o.net_node, connect_s * 1000, io_s * 1000);   // P100E_NET=tcp|pdl
    printf("net: transport %s\n", net_->kind());
    // first collective: the mode (pipe_init sends 1 here), so a node started with --pipe-layers next to one without is
    // refused with this name instead of at a later collective whose sizes no longer match
    net_->agree("the multi-node mode (0 = tensor parallel over the nodes, 1 = --pipe-layers hybrid pipeline)", 0);
    // global ranks: node order, each node's local GPUs in --devices order
    const std::vector<uint64_t> cnt = net_->allgather((uint64_t) devices.size());
    node_r0_.assign(cnt.size(), 0);
    node_nr_.assign(cnt.size(), 0);
    R_ = 0;
    for (size_t j = 0; j < cnt.size(); ++j) { node_r0_[j] = R_; node_nr_[j] = (int) cnt[j]; R_ += (int) cnt[j]; }
    r0_ = node_r0_[net_->self()];
    if (R_ > fast::P2P_MAX_RANKS)
        throw std::runtime_error("net: " + std::to_string(R_) + " ranks over all nodes: at most " +
                                 std::to_string(fast::P2P_MAX_RANKS) + " supported");
    // refused on any mismatch (every node throws the same message: agree() compares all values on every node)
    net_->agree("the model file size", g_.file_size);
    net_->agree("the model metadata (GGUF keys)", g_.meta_hash());
    net_->agree("the model tensor table", g_.tensor_table_hash());
    net_->agree("the model tensor data (sample)", g_.tensor_sample_hash());
    net_->agree("n_ctx (--ctx)", (uint64_t) n_ctx_);
    net_->agree("max_batch (ubatch)", (uint64_t) max_batch_);
    net_->agree("the KV cache type (--kv)", (uint64_t) kv_type_);
    net_->agree("load_mtp", (uint64_t) o.load_mtp);
    net_->agree("max_seqs", (uint64_t) max_slots_);
    net_->agree("pool_tokens", (uint64_t) o.pool_tokens);
    net_->agree("the run arguments", o.net_args_hash);
    for (const char * n : kNetControlEnv) {
        const char * e = getenv(n);
        const std::string v = e ? e : "<unset>";
        net_->agree(std::string("the switch ") + n + " (" + v + " here)", hash_bytes(kHashSeed, v.data(), v.size()));
    }
    if (o.load_mtp) throw std::runtime_error("MTP over the network (--net-nodes) is not supported yet: run without --mtp / --load-mtp");
}

void tp_model::net_check(const opts & o, const std::vector<double> & w) {
    (void) o;
    uint64_t h = kHashSeed;
    for (double x : w) h = hash_bytes(h, &x, sizeof x);
    net_->agree("the split weights (--split / P100E_SPLIT)", h);
    net_->agree("the split table", split_.hash());
    std::string mine;
    for (int r = 0; r < node_nr_[net_->self()]; ++r) mine += (r ? "," : "") + std::to_string(r0_ + r);
    printf("net: %d ranks over %d nodes; this node holds global rank(s) %s\n", R_, net_->n_nodes(), mine.c_str());
}

int64_t tp_model::net_min(int64_t v) {   // (net or pipe link)
    net_transport & l = net_ ? *net_ : *pnet_;
    int64_t m = v;
    for (uint64_t x : l.allgather((uint64_t) v)) m = std::min(m, (int64_t) x);
    return m;
}

// allreduce_send over the network: the event path's peer slots, filled by device copies between this node's ranks
// and by D2H -> one TCP message per node -> H2D for the other nodes' ranks. The adds that follow (allreduce_add,
// norm_in / norm_in_n) read all R_ partials in global rank order, as in one process. Three phases, so that a CUDA
// graph segment (P100E_NET_GRAPH) can end after pre and the next one start with post; eager: pre, wire, post.
void tp_model::net_allreduce_send(int T) {
    net_ar_pre(T);
    if (gl_on(T)) {   // the GPU link: nothing on the host between pre and post (a graph segment just continues)
    } else if (net_cap_) {   // capturing: this segment ends here, the next one starts with the H2D of what will arrive
        if (!net_seg_end(*net_cap_out_)) throw std::runtime_error(net_cap_err_);
        net_seg_begin();
    } else {
        net_ar_wire(T);
    }
    net_ar_post(T);
}

void tp_model::net_ar_pre(int T) {
    const int NL = (int) rk_.size();
    const size_t bytes = (size_t) T * E_ * 4;
    const int b = ar_buf_;
    ar_buf_ ^= 1;
    ar_cur_ = b;
    const bool pre_tx = net_preadd_ && net_->self() == 0;
    const bool gl = gl_on(T);
    if (gl) {   // traffic as enqueued (net_counters); a captured step adds its share per replay (net_launch)
        const uint64_t peers = (uint64_t) net_->n_nodes() - 1;
        gl_out_ += (uint64_t) bytes * (pre_tx ? 1 : NL) * peers;
        for (int q = 0; q < R_; ++q)
            if ((q < r0_ || q >= r0_ + NL) && !(q > 0 && q < pre_k_)) gl_in_ += bytes;
        ++gl_calls_;
    }
    for (int r = 0; r < NL; ++r) {
        rank_state & S = rk_[r];
        CK(cudaSetDevice(S.dev));
        // GPU link: the partial goes on the wire first (publish kernel -> mapped ring -> spin thread), then the local
        // peer copies; with pre-add only rank 0 sends (s, below) and the other ranks just step their counter
        if (gl && !pre_tx) gl_->publish(r, S.part, (size_t) T * E_, S.s);
        else if (gl && r > 0) gl_->tick(r, S.s);
        if (!pre_tx && !gl) CK(cudaMemcpyAsync(net_tx_ + (size_t) r * net_slot_, S.part, bytes, cudaMemcpyDeviceToHost, S.s));
        for (int o = 0; o < NL; ++o) {   // into o's slot of my global rank (parity b)
            if (o == r) continue;
            const int g = r0_ + o, q = r0_ + r;
            float * dst = rk_[o].peer + ((size_t) b * (R_ - 1) + (q < g ? q : q - 1)) * peer_stride_;
            // (cudaMemcpyPeerAsync cannot be captured; a graph segment takes the UVA copy, same bytes)
            if (net_cap_) CK(cudaMemcpyAsync(dst, S.part, bytes, cudaMemcpyDefault, S.s));
            else CK(cudaMemcpyPeerAsync(dst, rk_[o].dev, S.part, S.dev, bytes, S.s));
        }
        CK(cudaEventRecord(S.ev_copied, S.s));
    }
    if (pre_tx) {   // s = ((x + p0) + p1) + .. on rank 0: the receivers' fold up to node 0's last rank (P100E_NET_PREADD)
        rank_state & S = rk_[0];
        CK(cudaSetDevice(S.dev));
        for (int o = 1; o < NL; ++o) CK(cudaStreamWaitEvent(S.s, rk_[o].ev_copied, 0));
        CK(cudaMemcpyAsync(pre_buf_, S.x, bytes, cudaMemcpyDeviceToDevice, S.s));
        for (int q = 0; q < NL; ++q) k::add(pre_buf_, part_of(0, q), (int64_t) T * E_, S.s);
        if (gl) gl_->publish(0, pre_buf_, (size_t) T * E_, S.s);
        else CK(cudaMemcpyAsync(net_tx_, pre_buf_, bytes, cudaMemcpyDeviceToHost, S.s));
    }
}

void tp_model::net_ar_wire(int T) {
    if (gl_on(T)) return;   // the GPU link moved this all-reduce (no host step)
    const int NL = (int) rk_.size();
    const size_t bytes = (size_t) T * E_ * 4;
    for (int r = 0; r < NL; ++r) { CK(cudaSetDevice(rk_[r].dev)); CK(cudaStreamSynchronize(rk_[r].s)); }
    const bool pre_tx = net_preadd_ && net_->self() == 0;
    std::vector<net_link::seg> snd;
    for (int r = 0; r < (pre_tx ? 1 : NL); ++r) snd.push_back({net_tx_ + (size_t) r * net_slot_, bytes});
    std::vector<std::vector<net_link::seg>> rcv(net_->n_nodes());
    for (int j = 0; j < net_->n_nodes(); ++j) {
        if (j == net_->self()) continue;
        if (net_preadd_ && j == 0) { rcv[j].push_back({net_rx_, bytes}); continue; }   // s, in slot 0
        for (int q = node_r0_[j]; q < node_r0_[j] + node_nr_[j]; ++q) rcv[j].push_back({net_rx_ + (size_t) q * net_slot_, bytes});
    }
    net_->exchange(net_link::T_ALLREDUCE, snd, rcv);
}

void tp_model::net_ar_post(int T) {
    const int NL = (int) rk_.size(), R = R_, b = ar_cur_;
    const size_t bytes = (size_t) T * E_ * 4;
    if (gl_on(T)) {   // the GPU link: wait for + copy every remote partial into the places the H2D below fills
        for (int r = 0; r < NL; ++r) {
            rank_state & S = rk_[r];
            CK(cudaSetDevice(S.dev));
            const int g = r0_ + r;
            float * dst[pdl::GPU_MAX_RANKS] = {};
            if (pre_k_ > 0) dst[0] = S.x;   // x = s (part_of: pads for ranks 1..pre_k_-1, which send nothing)
            for (int q = pre_k_; q < R; ++q) {
                if (q >= r0_ && q < r0_ + NL) continue;
                dst[q] = S.peer + ((size_t) b * (R - 1) + (q < g ? q : q - 1)) * peer_stride_;
            }
            gl_->recv_copy(r, dst, (size_t) T * E_, S.s);
            // the local copies of pre: always waited here (no host sync between graph segments covers them now; the
            // events were recorded in the same capture, so a graph may wait on them)
            for (int o = 0; o < NL; ++o) if (o != r) CK(cudaStreamWaitEvent(S.s, rk_[o].ev_copied, 0));
        }
        return;
    }
    for (int r = 0; r < NL; ++r) {
        rank_state & S = rk_[r];
        CK(cudaSetDevice(S.dev));
        const int g = r0_ + r;
        if (pre_k_ > 0) CK(cudaMemcpyAsync(S.x, net_rx_, bytes, cudaMemcpyHostToDevice, S.s));   // x = s (part_of: pads)
        for (int q = pre_k_; q < R; ++q) {
            if (q >= r0_ && q < r0_ + NL) continue;
            float * dst = S.peer + ((size_t) b * (R - 1) + (q < g ? q : q - 1)) * peer_stride_;
            CK(cudaMemcpyAsync(dst, net_rx_ + (size_t) q * net_slot_, bytes, cudaMemcpyHostToDevice, S.s));
        }
        // the local copies of pre (a graph segment: the host sync between the segments covers them, and an event of
        // another capture cannot be waited on)
        if (!net_cap_) for (int o = 0; o < NL; ++o) if (o != r) CK(cudaStreamWaitEvent(S.s, rk_[o].ev_copied, 0));
    }
}

// ---- segment graphs (P100E_NET_GRAPH): rank 0's stream is the capture origin, the other local ranks fork from it and
// join back (as capture_body). Errors are collected in net_cap_err_ (no throw while a capture is open).
void tp_model::net_seg_begin() {
    const int NL = (int) rk_.size();
    auto ok = [&](cudaError_t e, const char * w) { if (e != cudaSuccess && net_cap_err_.empty()) net_cap_err_ = std::string(w) + ": " + cudaGetErrorString(e); };
    CK(cudaSetDevice(rk_[0].dev));
    ok(cudaStreamBeginCapture(rk_[0].s, cudaStreamCaptureModeRelaxed), "begin capture");
    if (NL > 1) ok(cudaEventRecord(ev_fork_, rk_[0].s), "fork record");
    for (int r = 1; r < NL; ++r) { CK(cudaSetDevice(rk_[r].dev)); ok(cudaStreamWaitEvent(rk_[r].s, ev_fork_, 0), "fork wait"); }
}

bool tp_model::net_seg_end(std::vector<step_graph> & out) {
    const int NL = (int) rk_.size();
    auto ok = [&](cudaError_t e, const char * w) { if (e != cudaSuccess && net_cap_err_.empty()) net_cap_err_ = std::string(w) + ": " + cudaGetErrorString(e); };
    for (int r = 1; r < NL; ++r) {
        CK(cudaSetDevice(rk_[r].dev));
        ok(cudaEventRecord(rk_[r].ev_join, rk_[r].s), "join record");
        CK(cudaSetDevice(rk_[0].dev));
        ok(cudaStreamWaitEvent(rk_[0].s, rk_[r].ev_join, 0), "join wait");
    }
    CK(cudaSetDevice(rk_[0].dev));
    step_graph G;
    ok(cudaStreamEndCapture(rk_[0].s, &G.g), "end capture");
    if (net_cap_err_.empty() && G.g) ok(cudaGraphInstantiateWithFlags(&G.ge, G.g, 0), "instantiate");
    if (net_cap_err_.empty() && G.ge) ok(cudaGraphUpload(G.ge, rk_[0].s), "upload");
    if (G.g) cudaGraphGetNodes(G.g, nullptr, &G.nodes);
    G.buf_out = ar_buf_; G.cur_out = ar_cur_;   // the parity a run of the segment leaves (launch_graph restores it)
    G.sig_in = -1;
    out.push_back(G);   // also on failure: freed with the set
    return net_cap_err_.empty() && G.ge;
}

tp_model::net_gset * tp_model::net_find(int T) {
    for (auto & G : net_gsets_)
        if (G.T == T && G.slot == sq().slot && G.mr == mrope_now() && G.q8f == q8_fork_) return &G;
    return nullptr;
}

// record fused_body(T) (nothing runs; the streams are idle): segment k = the work between allreduce k-1's exchange and
// allreduce k's, the last one ends with the LM head. The parity starts at 0, as every net fused step does.
bool tp_model::net_capture(int T) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    net_gset G;
    G.T = T; G.slot = sq().slot; G.mr = mrope_now(); G.q8f = q8_fork_;
    const int b0 = ar_buf_, c0 = ar_cur_;
    ar_buf_ = ar_cur_ = 0;
    net_cap_ = true;
    net_cap_out_ = &G.seg;
    net_cap_err_.clear();
    const uint64_t go0 = gl_out_, gi0 = gl_in_, gc0 = gl_calls_;   // nothing moves while capturing: kept per replay
    try {
        net_seg_begin();
        fused_body(T, false, 1);
        net_seg_end(G.seg);
    } catch (const std::exception & e) {
        if (net_cap_err_.empty()) net_cap_err_ = e.what();
    }
    net_cap_ = false;
    net_cap_out_ = nullptr;
    G.gl_out = gl_out_ - go0; G.gl_in = gl_in_ - gi0; G.gl_calls = gl_calls_ - gc0;
    gl_out_ = go0; gl_in_ = gi0; gl_calls_ = gc0;
    ar_buf_ = b0; ar_cur_ = c0;
    CK(cudaSetDevice(rk_[0].dev));
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(rk_[0].s, &cs);
    if (cs != cudaStreamCaptureStatusNone) {   // an exception left a capture open: close it (its graph is discarded)
        cudaGraph_t g = nullptr;
        cudaStreamEndCapture(rk_[0].s, &g);
        if (g) cudaGraphDestroy(g);
    }
    if (!net_cap_err_.empty()) {
        for (auto & S : G.seg) { if (S.ge) cudaGraphExecDestroy(S.ge); if (S.g) cudaGraphDestroy(S.g); }
        for (auto & S : rk_) { cudaSetDevice(S.dev); cudaGetLastError(); }
        fprintf(stderr, "net CUDA graphs (T=%d): %s, eager decode steps from now on\n", T, net_cap_err_.c_str());
        net_graph_failed_ = true;
        return false;
    }
    size_t nodes = 0;
    for (auto & S : G.seg) nodes += S.nodes;
    if (!getenv("P100E_GRAPH_QUIET"))
        printf("net CUDA graphs T=%d: %zu segments, %zu nodes on %d local GPU(s), captured in %.1f ms\n", T, G.seg.size(), nodes,
               (int) rk_.size(), std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    net_gsets_.push_back(std::move(G));
    return true;
}

// run a captured step: segment, (host sync + exchange), segment, ... ; the exchanges are the same collectives in the
// same order as an eager step's, so the other nodes cannot tell (they may run eager or graphs)
void tp_model::net_launch(const net_gset & G, int T) {
    for (size_t k = 0; k < G.seg.size(); ++k) {
        launch_graph(G.seg[k]);
        if (k + 1 < G.seg.size()) net_ar_wire(T);
    }
    if (gl_on(T)) {   // one graph per step: flush the WDDM launch batch (the peers wait for its publishes)
        CK(cudaSetDevice(rk_[0].dev));
        (void) cudaStreamQuery(rk_[0].s);
        gl_out_ += G.gl_out; gl_in_ += G.gl_in; gl_calls_ += G.gl_calls;
    }
}

net_stats tp_model::net_counters() const {
    const net_transport * l = link();   // net: tensor parallel over the nodes; pipe: the layer handoff
    if (!l) return net_stats{};
    net_stats n = l->stats();
    n.bytes_out += gl_out_;
    n.bytes_in += gl_in_;
    n.calls += gl_calls_;
    return n;
}

// P100E_NET_GPULINK=1 (P100E_NET=pdl only): the zero-copy GPU path of pdl_gpu.cuh for the
// all-reduces of up to gl_maxt_ rows (decode, MTP-size verify; bigger ones and prefill keep the host path, which
// shares the link: pdl_link::exchange posts and waits while the spin thread polls). Every node must agree (it changes
// which pdl streams carry the partials), so the effective value is agreed like the pre-add.
void tp_model::gl_init() {
    auto envi = [](const char * n, int d) { const char * e = getenv(n); return e && *e ? atoi(e) : d; };
    pdl_link * pl = dynamic_cast<pdl_link *>(net_.get());
    const bool asked = envi("P100E_NET_GPULINK", 0) != 0, want = asked && pl != nullptr;
    if (asked && !pl) printf("net: P100E_NET_GPULINK needs P100E_NET=pdl: gpu link off\n");
    net_->agree("the gpu link (P100E_NET_GPULINK with P100E_NET=pdl, effective)", (uint64_t) want);
    if (!want) return;
    gl_maxt_ = std::max(1, std::min(envi("P100E_NET_GPULINK_MAXT", 8), max_batch_));
    net_->agree("P100E_NET_GPULINK_MAXT (effective)", (uint64_t) gl_maxt_);
    pdl::gpu_link_opts go;
    // ring depth = the engine's send ring (proto_opts::send_ring, 8): a NACKed message is resent from its tx slot,
    // which must not have been reused yet
    go.depth = 8;
    go.slot_bytes = (size_t) gl_maxt_ * E_ * 4;
    for (size_t j = 0; j < node_nr_.size(); ++j) for (int k = 0; k < node_nr_[j]; ++k) go.node_of_rank.push_back((int) j);
    go.sends.assign((size_t) R_, 1);
    if (net_preadd_) for (int q = 1; q < node_nr_[0]; ++q) go.sends[(size_t) q] = 0;   // folded into rank 0's s
    for (auto & S : rk_) go.devices.push_back(S.dev);
    // GPU spin bound: a display GPU's watchdog kills a kernel after ~2 s (as the P2P all-reduce: at most 750 ms there)
    int ms = envi("P100E_NET_GPULINK_TIMEOUT_MS", 10000);
    for (auto & S : rk_) {
        int wd = 0;
        CK(cudaDeviceGetAttribute(&wd, cudaDevAttrKernelExecTimeout, S.dev));
        if (wd) ms = std::min(ms, 750);
    }
    go.timeout_ms = std::max(10, ms);
    CK(cudaSetDevice(rk_[0].dev));
    gl_ = new pdl::gpu_link(*pl, go);
    net_->agree("gpu link rings allocated", 1);   // no node's wait kernel spins before every node can answer
    gl_->start();
    const char * pc = getenv("P100E_PDL_CPU");
    printf("net: gpu link: ring depth %d, slot %zu bytes (T <= %d), GPU wait bound %d ms, spin thread %s%s\n", go.depth,
           go.slot_bytes, gl_maxt_, go.timeout_ms, pc && *pc ? "pinned to core " : "unpinned", pc && *pc ? pc : "");
}
// logits_h_ rows 0..n_rows-1 hold this node's vocab slices (its ranks' ranges are adjacent: one block per row);
// the other nodes' blocks are filled in place
void tp_model::net_gather_logits(int n_rows) {
    auto block = [&](int j, int64_t & b, int64_t & e) {
        b = split_.rk[node_r0_[j]].voc.b;
        e = split_.rk[node_r0_[j] + node_nr_[j] - 1].voc.e();
    };
    int64_t vb, ve;
    block(net_->self(), vb, ve);
    std::vector<net_link::seg> snd;
    for (int i = 0; i < n_rows; ++i) snd.push_back({logits_h_ + (size_t) i * VOCAB_ + vb, (size_t) (ve - vb) * 4});
    std::vector<std::vector<net_link::seg>> rcv(net_->n_nodes());
    for (int j = 0; j < net_->n_nodes(); ++j) {
        if (j == net_->self()) continue;
        int64_t b, e;
        block(j, b, e);
        for (int i = 0; i < n_rows; ++i) rcv[j].push_back({logits_h_ + (size_t) i * VOCAB_ + b, (size_t) (e - b) * 4});
    }
    net_->exchange(net_link::T_LOGITS, snd, rcv);
}

// amax_h_ [local rank][8] -> amax_net_ [global rank][8], the other nodes' pairs received in place
void tp_model::net_gather_amax(int n_rows) {
    const int NL = (int) rk_.size();
    const size_t ps = sizeof(amax_pair), n = (size_t) n_rows * ps;
    char * g = amax_net_.data();
    std::vector<net_link::seg> snd;
    for (int r = 0; r < NL; ++r) {
        memcpy(g + (size_t) (r0_ + r) * 8 * ps, (const char *) amax_h_ + (size_t) r * 8 * ps, n);
        snd.push_back({(char *) amax_h_ + (size_t) r * 8 * ps, n});
    }
    std::vector<std::vector<net_link::seg>> rcv(net_->n_nodes());
    for (int j = 0; j < net_->n_nodes(); ++j) {
        if (j == net_->self()) continue;
        for (int q = node_r0_[j]; q < node_r0_[j] + node_nr_[j]; ++q) rcv[j].push_back({g + (size_t) q * 8 * ps, n});
    }
    net_->exchange(net_link::T_ARGMAX, snd, rcv);
}

// =====================================================================================================================
// Hybrid pipeline: the tensor split inside each node, whole layers between the nodes (tp_model.h top).
// Per forward of T rows two collectives on the link (both nodes take part in each: SPMD lockstep):
//   1. rows:   node 0 sends [pipe_hdr][T][E] f32 (rank 0's S.x after its last layer), node 1 sends a pipe_hdr
//   2. result: node 1 sends [pipe_hdr][n_rows] int32 greedy ids or [n_rows][VOCAB] f32 logits, node 0 a pipe_hdr
// Every header carries (rows, n_past, step): both sides compare them, so a node that left the common control flow is
// caught at the next handoff ("lockstep lost") instead of computing on another position's rows. Node 1 blocks in (1)
// while node 0 computes, node 0 in (2) while node 1 does: one token = node 0's layers + one way + node 1's layers +
// head + the way back (a decode step moves E x 4 = 20 KB at E = 5120, plus 4 bytes of id).
// =====================================================================================================================

namespace {
constexpr uint32_t PIPE_MAGIC = 0x50495031u;   // "1PIP"
struct pipe_hdr { uint32_t magic, rows; int32_t n_past, aux; uint64_t step; };   // aux: rows = node 0's nsat, result = kind
static_assert(sizeof(pipe_hdr) == 24, "pipe_hdr layout");
}

// switches that change the pipe protocol or the pool page arithmetic: equal on both nodes. Everything else (kernel
// choices, the prefill pipeline vs classic, the f16 prefill wire, the split) is node local: each node's control flow
// inside a forward is its own, only the handoffs are common.
static const char * const kPipeControlEnv[] = {"P100E_PAGED", "P100E_PAGE_SHIFT", "P100E_NET"};

void tp_model::pipe_init(const std::vector<int> & devices, const opts & o) {
    auto envi = [](const char * n, int d) { const char * e = getenv(n); return e && *e ? atoi(e) : d; };
    const int connect_s = envi("P100E_NET_CONNECT_S", o.net_connect_s), io_s = envi("P100E_NET_TIMEOUT_S", o.net_timeout_s);
    if (o.net_nodes.size() != 2)
        throw std::runtime_error("the hybrid pipeline (--pipe-layers) runs on exactly 2 nodes (--net-nodes a,b), got " +
                                 std::to_string(o.net_nodes.size()));
    if (o.load_mtp) throw std::runtime_error("MTP in the hybrid pipeline (--pipe-layers) is not supported yet: run without --mtp / --load-mtp");
    std::string list;
    for (size_t i = 0; i < o.net_nodes.size(); ++i) list += (i ? "," : "") + o.net_nodes[i];
    printf("pipe: node %d of %s, %zu local GPU(s); waiting up to %d s for the other node\n", o.net_node, list.c_str(),
           devices.size(), connect_s);
    fflush(stdout);
    pnet_ = net_transport_open(o.net_nodes, o.net_node, connect_s * 1000, io_s * 1000);   // P100E_NET=tcp|pdl
    printf("pipe: transport %s\n", pnet_->kind());
    // refused on any mismatch (both nodes throw the same message); the mode first, as net_init
    pnet_->agree("the multi-node mode (0 = tensor parallel over the nodes, 1 = --pipe-layers hybrid pipeline)", 1);
    pnet_->agree("the model file size", g_.file_size);
    pnet_->agree("the model metadata (GGUF keys)", g_.meta_hash());
    pnet_->agree("the model tensor table", g_.tensor_table_hash());
    pnet_->agree("the model tensor data (sample)", g_.tensor_sample_hash());
    pnet_->agree("n_ctx (--ctx)", (uint64_t) n_ctx_);
    pnet_->agree("max_batch (ubatch)", (uint64_t) max_batch_);
    pnet_->agree("the KV cache type (--kv)", (uint64_t) kv_type_);
    pnet_->agree("max_seqs", (uint64_t) max_slots_);
    pnet_->agree("pool_tokens", (uint64_t) o.pool_tokens);
    pnet_->agree("the run arguments", o.net_args_hash);
    pnet_->agree("the pipe layers (--pipe-layers / --pipe-split)", (uint64_t) (int64_t) o.pipe_layers);
    pnet_->agree("the pipe total (--pipe-split a,b: a + b)", (uint64_t) o.pipe_total);
    for (const char * n : kPipeControlEnv) {
        const char * e = getenv(n);
        const std::string v = e ? e : "<unset>";
        pnet_->agree(std::string("the switch ") + n + " (" + v + " here)", hash_bytes(kHashSeed, v.data(), v.size()));
    }
    if (o.pipe_total && o.pipe_total != L_)
        throw std::runtime_error("--pipe-split: the parts add up to " + std::to_string(o.pipe_total) + " layers, the model has " +
                                 std::to_string(L_));

    // ---- byte plan: what each node needs per GPU for a layer range, against its free VRAM ----
    // Free VRAM of this node: per distinct GPU (ranks sharing a GPU - local tests, --devices 0,0 - count once), the
    // smallest. Exchanged with the GPU counts, so both nodes compute the same plan and the same verdict.
    std::vector<int> udev(devices);
    std::sort(udev.begin(), udev.end());
    udev.erase(std::unique(udev.begin(), udev.end()), udev.end());
    size_t fr_min = SIZE_MAX;
    for (int d : udev) {
        CK(cudaSetDevice(d));
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        fr_min = std::min(fr_min, fr);
    }
    const std::vector<uint64_t> fr_all = pnet_->allgather((uint64_t) fr_min);
    const std::vector<uint64_t> nr_all = pnet_->allgather((uint64_t) devices.size());
    const std::vector<uint64_t> nd_all = pnet_->allgather((uint64_t) udev.size());
    const int64_t pool_tok = o.pool_tokens > 0 ? o.pool_tokens : (int64_t) n_ctx_ * max_slots_;   // (deferred: n_ctx)
    auto layer_bytes = [&](int il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        uint64_t b = 0;
        for (const auto & t : g_.tensors) if (t.name.compare(0, p.size(), p) == 0) b += t.nbytes;
        return b;
    };
    struct need { double w, head, kv, st, buf, per_gpu; };
    // node j running layers [l0, l1): weights (quantized bytes as in the file; every rank holds 1/R of them), the head,
    // the KV pool (pool_tok positions of every own attention layer, K + V), the delta-net states (one slot per
    // sequence), and per rank the fixed buffers: f32 dequant scratch 256 MiB, the activation buffers at max_batch rows,
    // the logit rows, the prefill pipeline's f16 ring (640 MiB, when its kernels take the shares) and 256 MiB of
    // margin (cuBLAS workspace, matvec scratch, the loader's f32 temporaries, the CUDA runtime's own)
    auto plan = [&](int j, int l0, int l1) {
        need n{};
        const int R = (int) nr_all[j], nd = std::max<int>(1, (int) nd_all[j]);
        int n_att = 0, n_gdn = 0;
        for (int il = l0; il < l1; ++il) {
            n.w += (double) layer_bytes(il);
            if ((il + 1) % full_attn_interval_ == 0) ++n_att; else ++n_gdn;
        }
        if (l1 == L_) {
            if (const gguf_tensor * t = g_.find("output.weight")) n.head += (double) t->nbytes;
            if (const gguf_tensor * t = g_.find("output_norm.weight")) n.head += (double) t->nelements() * 4;
        }
        n.kv = (double) pool_tok * n_att * 2.0 * (double) fast::kv_row_bytes(kv_type_, HKV_, HD_);
        const double C = 2.0 * HK_ * DK_ + (double) HV_ * DV_;
        n.st = (double) max_slots_ * n_gdn * (C * (CONV_K_ - 1) + (double) HV_ * DV_ * DK_) * 4;
        const double FFl = (double) FF_ / R, mb = max_batch_;
        const double wide = std::max({FFl, 2.0 * H_ / R * HD_, C / R, (double) E_});
        double per_rank = 256.0 * 1048576 + (4.0 + 2.0 * std::max(1, R - 1)) * mb * E_ * 4 + 6.0 * mb * wide * 4 + 256.0 * 1048576;
        if (l1 == L_) per_rank += 8.0 * VOCAB_ / R * 4;
        if (FFl <= 9216 && E_ <= 8192) per_rank += 640.0 * 1048576 + mb * std::max(FFl, (double) E_) * 2;
        n.buf = per_rank * R / nd;
        n.per_gpu = (n.w + n.head + n.kv + n.st) / nd + n.buf;
        return n;
    };
    auto fits = [&](int j, const need & n) { return n.per_gpu <= (double) fr_all[j]; };
    int k = o.pipe_layers;
    if (k < 0) {   // auto: the most layers on node 1 (smallest k) that both nodes hold
        for (k = 1; k < L_; ++k)
            if (fits(1, plan(1, k, L_)) && fits(0, plan(0, 0, k))) break;
        if (k >= L_) {
            const need a = plan(0, 0, L_ - 1), b = plan(1, L_ - 1, L_);
            throw std::runtime_error("pipe: no layer split fits both nodes (node 0 with " + std::to_string(L_ - 1) +
                                     " layers needs " + std::to_string((int) (a.per_gpu / 1048576)) + " MiB per GPU of " +
                                     std::to_string((int) (fr_all[0] >> 20)) + " MiB free; node 1 with 1 layer + head " +
                                     std::to_string((int) (b.per_gpu / 1048576)) + " MiB of " +
                                     std::to_string((int) (fr_all[1] >> 20)) + " MiB): use a shorter --ctx or more GPUs");
        }
    }
    if (k < 1 || k >= L_)
        throw std::runtime_error("--pipe-layers " + std::to_string(k) + ": node 0 runs layers [0, k), node 1 [k, " +
                                 std::to_string(L_) + "): k must be 1.." + std::to_string(L_ - 1));
    const need p0 = plan(0, 0, k), p1 = plan(1, k, L_);
    const char * kvs = kv_type_ == fast::KV_F16 ? "f16" : kv_type_ == fast::KV_Q8_0 ? "q8_0" : "q4_0";
    auto gib = [](double b) { return b / 1073741824.0; };
    printf("pipe: byte plan (%s, KV %s for %lld positions, ubatch %d): node 0 layers [0, %d) + embedding, node 1 layers "
           "[%d, %d) + output norm + LM head\n", o.pipe_layers < 0 ? "auto split" : "--pipe-layers", kvs,
           (long long) pool_tok, max_batch_, k, k, L_);
    const need * pl[2] = {&p0, &p1};
    for (int j = 0; j < 2; ++j) {
        const need & n = *pl[j];
        printf("pipe:   node %d: %d GPU(s) (%d rank(s)): weights %.2f GiB + head %.2f GiB + KV %.2f GiB + state %.2f GiB "
               "+ buffers %.2f GiB per GPU = %.2f GiB per GPU, free %.2f GiB per GPU: %s\n", j, (int) nd_all[j],
               (int) nr_all[j], gib(n.w), gib(n.head), gib(n.kv), gib(n.st), gib(n.buf), gib(n.per_gpu), gib((double) fr_all[j]),
               fits(j, n) ? "fits" : "DOES NOT FIT");
    }
    fflush(stdout);
    for (int j = 0; j < 2; ++j)
        if (!fits(j, *pl[j]))
            throw std::runtime_error("pipe: node " + std::to_string(j) + " does not fit: " +
                                     std::to_string((int) (pl[j]->per_gpu / 1048576)) + " MiB per GPU needed, " +
                                     std::to_string((int) (fr_all[j] >> 20)) + " MiB free (" +
                                     (j == 1 ? "give node 1 fewer layers: a larger --pipe-layers" :
                                               "give node 0 fewer layers: a smaller --pipe-layers") +
                                     ", a shorter --ctx, or --pipe-layers auto)");
    lb_ = pnet_->self() == 0 ? 0 : k;
    le_ = pnet_->self() == 0 ? k : L_;
}

void tp_model::pipe_x_d2h(int T) {
    rank_state & S = rk_[0];
    CK(cudaSetDevice(S.dev));
    CK(cudaMemcpyAsync(pipe_h_, S.x, (size_t) T * E_ * 4, cudaMemcpyDeviceToHost, S.s));
}

static void pipe_check(const pipe_hdr & a, const pipe_hdr & b, const char * what) {
    if (b.magic != PIPE_MAGIC || a.rows != b.rows || a.n_past != b.n_past || a.step != b.step ||
        (std::string(what) == "result" && a.aux != b.aux))
        throw std::runtime_error(std::string("pipe: lockstep lost at the ") + what + " handoff: this node is at step " +
                                 std::to_string(a.step) + " (" + std::to_string(a.rows) + " rows at n_past " +
                                 std::to_string(a.n_past) + "), the other at step " + std::to_string(b.step) + " (" +
                                 std::to_string(b.rows) + " rows at n_past " + std::to_string(b.n_past) + ")");
}

void tp_model::pipe_send_x(int T, int nsat) {
    pipe_hdr h{PIPE_MAGIC, (uint32_t) T, sq().n_past, nsat, ++pipe_step_}, peer{};
    std::vector<std::vector<net_link::seg>> rcv(2);
    rcv[1] = {{&peer, sizeof peer}};
    pnet_->exchange(net_link::T_PIPE, {{&h, sizeof h}, {pipe_h_, (size_t) T * E_ * 4}}, rcv);
    pipe_check(h, peer, "rows");
}

void tp_model::pipe_recv_x(int T) {
    pipe_hdr mine{PIPE_MAGIC, (uint32_t) T, sq().n_past, 0, ++pipe_step_}, h{};
    std::vector<std::vector<net_link::seg>> rcv(2);
    rcv[0] = {{&h, sizeof h}, {pipe_h_, (size_t) T * E_ * 4}};
    pnet_->exchange(net_link::T_PIPE, {{&mine, sizeof mine}}, rcv);
    pipe_check(mine, h, "rows");
    for (auto & S : rk_) {   // every local rank's replica of the residual stream (the staging is rewritten only after
        CK(cudaSetDevice(S.dev));   // this forward's final sync)
        CK(cudaMemcpyAsync(S.x, pipe_h_, (size_t) T * E_ * 4, cudaMemcpyHostToDevice, S.s));
    }
    if (h.aux > 0) {   // node 0's f16 prefill wire saturated: this forward's rows carry clamped values there
        pf_sat_total_ += (uint64_t) h.aux;
        fprintf(stderr, "prefill: node 0 saturated %d f16 wire partial group(s) in its layers (T=%d at n_past %d): this "
                        "forward's outputs carry clamped values (last_prefill_saturated())\n", h.aux, T, sq().n_past);
    }
}

void tp_model::pipe_result(int n_rows, std::vector<int32_t> * ids) {
    pipe_hdr mine{PIPE_MAGIC, (uint32_t) n_rows, sq().n_past, ids ? 1 : 2, ++pipe_step_}, peer{};
    if (ids) ids->resize((size_t) n_rows);
    void * pay = ids ? (void *) ids->data() : (void *) logits_h_;
    const size_t bytes = ids ? (size_t) n_rows * 4 : (size_t) n_rows * VOCAB_ * 4;
    std::vector<std::vector<net_link::seg>> rcv(2);
    if (has_head()) {   // node 1: the result out
        rcv[0] = {{&peer, sizeof peer}};
        pnet_->exchange(net_link::T_PIPE, {{&mine, sizeof mine}, {pay, bytes}}, rcv);
    } else {            // node 0: the result in (ids / logits_h_ rows)
        rcv[1] = {{&peer, sizeof peer}, {pay, bytes}};
        pnet_->exchange(net_link::T_PIPE, {{&mine, sizeof mine}}, rcv);
    }
    pipe_check(mine, peer, "result");
    if (ids && !has_head()) check_tokens(*ids, VOCAB_, "pipe: node 1's greedy ids");   // node 0 feeds them back
}

} // namespace p100e
