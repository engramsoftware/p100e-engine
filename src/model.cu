#include "model.h"
#include "kernels.cuh"
#include "../third_party/ggml/convert.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace p100e {

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error( \
    std::string("CUDA ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)
#define CB(x) do { cublasStatus_t e_ = (x); if (e_ != CUBLAS_STATUS_SUCCESS) throw std::runtime_error( \
    std::string("cuBLAS error ") + std::to_string((int) e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

static constexpr size_t k_chunk_elems = 64ull << 20;   // f32 weight scratch: 64M floats = 256 MiB

static std::string bn(int il, const char * suffix) { return "blk." + std::to_string(il) + "." + suffix; }

model::model(const std::string & path, int n_ctx, int max_batch) : g_(path), n_ctx_(n_ctx), max_batch_(max_batch) {
    const std::string a = g_.get_str("general.architecture", "");
    if (a != "qwen35") throw std::runtime_error("unsupported architecture '" + a + "' (want qwen35)");
    hp_.n_embd      = (int) g_.get_int(a + ".embedding_length", 0);
    hp_.n_layer_all = (int) g_.get_int(a + ".block_count", 0);
    hp_.n_layer     = hp_.n_layer_all - (int) g_.get_int(a + ".nextn_predict_layers", 0);
    hp_.n_ff        = (int) g_.get_int(a + ".feed_forward_length", 0);
    hp_.n_head      = (int) g_.get_int(a + ".attention.head_count", 0);
    hp_.n_head_kv   = (int) g_.get_int(a + ".attention.head_count_kv", 0);
    hp_.head_dim    = (int) g_.get_int(a + ".attention.key_length", 0);
    hp_.n_rot       = (int) g_.get_int(a + ".rope.dimension_count", 0);
    hp_.rms_eps     = (float) g_.get_float(a + ".attention.layer_norm_rms_epsilon", 1e-6);
    hp_.rope_base   = (float) g_.get_float(a + ".rope.freq_base", 1e7);
    hp_.full_attn_interval = (int) g_.get_int(a + ".full_attention_interval", 4);
    hp_.ssm_conv    = (int) g_.get_int(a + ".ssm.conv_kernel", 4);
    hp_.ssm_inner   = (int) g_.get_int(a + ".ssm.inner_size", 0);
    hp_.ssm_state   = (int) g_.get_int(a + ".ssm.state_size", 0);
    hp_.ssm_groups  = (int) g_.get_int(a + ".ssm.group_count", 0);
    hp_.ssm_dt_rank = (int) g_.get_int(a + ".ssm.time_step_rank", 0);
    const gguf_tensor * emb = g_.find("token_embd.weight");
    if (!emb) throw std::runtime_error("missing token_embd.weight");
    hp_.n_vocab = (int) emb->ne[1];
    if (hp_.ssm_conv != 4) throw std::runtime_error("only conv kernel 4 supported");
    // divisors and loop bounds below (is_attn's modulus, ssm_inner / ssm_dt_rank): a missing or zero key is a bad
    // model, not an integer divide by zero
    if (hp_.n_layer <= 0 || hp_.n_embd <= 0 || hp_.head_dim <= 0 || hp_.full_attn_interval <= 0 || hp_.ssm_dt_rank <= 0 ||
        hp_.ssm_inner % hp_.ssm_dt_rank != 0)
        throw std::runtime_error("bad qwen35 hparams (block_count, embedding_length, attention.key_length, "
                                 "full_attention_interval, ssm.time_step_rank / ssm.inner_size)");

    CK(cudaStreamCreate(&s_));
    CB(cublasCreate(&cb_));
    CB(cublasSetStream(cb_, s_));
    CB(cublasSetMathMode(cb_, CUBLAS_PEDANTIC_MATH));   // true f32 reference

    // scratch for the largest weight chunk (bytes) and its f32 dequant
    size_t max_row_bytes = 0;
    for (const auto & t : g_.tensors) {
        if (t.ne[1] > 1) max_row_bytes = std::max(max_row_bytes, (size_t)(t.nbytes / (t.ne[1] * t.ne[2] * t.ne[3])));
    }
    w_elems_ = k_chunk_elems;
    q_bytes_ = std::max(max_row_bytes * (size_t) max_batch_, (size_t)(k_chunk_elems * 4));   // >= any chunk's bytes
    CK(cudaMalloc(&d_q_, q_bytes_));
    CK(cudaMalloc(&d_w_, w_elems_ * sizeof(float)));

    const int conv_dim = 2 * hp_.ssm_groups * hp_.ssm_state + hp_.ssm_inner;
    const size_t wide = (size_t) max_batch_ * std::max({hp_.n_ff, 2 * hp_.n_head * hp_.head_dim, conv_dim, hp_.n_embd});
    const size_t emb_sz = (size_t) max_batch_ * hp_.n_embd;
    CK(cudaMalloc(&x_, emb_sz * 4));
    CK(cudaMalloc(&cur_, emb_sz * 4));
    CK(cudaMalloc(&res_, emb_sz * 4));
    for (float ** p : {&a_, &b_, &c_, &d_, &e_, &f_}) CK(cudaMalloc(p, wide * 4));
    CK(cudaMalloc(&d_pos_, max_batch_ * sizeof(int32_t)));

    const size_t kv_row = (size_t) hp_.n_head_kv * hp_.head_dim;
    kc_.assign(hp_.n_layer, nullptr); vc_.assign(hp_.n_layer, nullptr);
    conv_state_.assign(hp_.n_layer, nullptr); ssm_state_.assign(hp_.n_layer, nullptr);
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (hp_.is_attn(il)) {
            CK(cudaMalloc(&kc_[il], (size_t) n_ctx_ * kv_row * 4));
            CK(cudaMalloc(&vc_[il], (size_t) n_ctx_ * kv_row * 4));
        } else {
            const size_t cs = (size_t) conv_dim * (hp_.ssm_conv - 1);
            const size_t ss = (size_t) hp_.ssm_dt_rank * (hp_.ssm_inner / hp_.ssm_dt_rank) * hp_.ssm_state;
            CK(cudaMalloc(&conv_state_[il], cs * 4));
            CK(cudaMalloc(&ssm_state_[il], ss * 4));
            CK(cudaMemset(conv_state_[il], 0, cs * 4));
            CK(cudaMemset(ssm_state_[il], 0, ss * 4));
        }
    }
}

model::~model() {
    for (auto & [n, p] : small_) cudaFree(p);
    for (auto * v : {&kc_, &vc_, &conv_state_, &ssm_state_}) for (float * p : *v) if (p) cudaFree(p);
    for (float * p : {x_, cur_, res_, a_, b_, c_, d_, e_, f_}) cudaFree(p);
    cudaFree(d_pos_); cudaFree(d_q_); cudaFree(d_w_);
    if (cb_) cublasDestroy(cb_);
    if (s_) cudaStreamDestroy(s_);
}

// dequantize n elements of type `type` from device bytes `src` to f32 `dst`
static void dequant_f32(uint32_t type, const void * src, float * dst, int64_t n, cudaStream_t s) {
    if (type == GGML_TYPE_F32) {
        CK(cudaMemcpyAsync(dst, src, n * 4, cudaMemcpyDeviceToDevice, s));
        return;
    }
    to_fp32_cuda_t f = ggml_get_to_fp32_cuda((ggml_type) type);
    if (!f) throw std::runtime_error(std::string("no dequant for type ") + get_type_info(type).name);
    f(src, dst, n, s);
}

const float * model::small(const std::string & name) {
    auto it = small_.find(name);
    if (it != small_.end()) return it->second;
    const gguf_tensor * t = g_.find(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    float * d = nullptr;
    CK(cudaMalloc(&d, t->nelements() * 4));
    CK(cudaMemcpyAsync(d_q_, t->data, t->nbytes, cudaMemcpyHostToDevice, s_));
    dequant_f32(t->type, d_q_, d, t->nelements(), s_);
    CK(cudaStreamSynchronize(s_));
    small_[name] = d;
    return d;
}

void model::matmul(const std::string & name, const float * x, float * y, int T) {
    const gguf_tensor * t = g_.find(name);
    if (!t) throw std::runtime_error("missing tensor " + name);
    const int64_t n_in = t->ne[0], n_out = t->ne[1];
    const size_t row_bytes = t->nbytes / n_out;
    const int64_t rows_per = std::max<int64_t>(1, (int64_t)(w_elems_ / n_in));
    const float one = 1.0f, zero = 0.0f;
    for (int64_t r0 = 0; r0 < n_out; r0 += rows_per) {
        const int64_t nr = std::min(rows_per, n_out - r0);
        CK(cudaMemcpyAsync(d_q_, t->data + r0 * row_bytes, nr * row_bytes, cudaMemcpyHostToDevice, s_));
        dequant_f32(t->type, d_q_, d_w_, nr * n_in, s_);
        // y (col-major n_out x T) rows r0..r0+nr = W[r0..](nr x n_in) * x (n_in x T)
        CB(cublasSgemm(cb_, CUBLAS_OP_T, CUBLAS_OP_N, (int) nr, T, (int) n_in, &one, d_w_, (int) n_in,
                       x, (int) n_in, &zero, y + r0, (int) n_out));
    }
}

void model::embed(const std::vector<int32_t> & tokens, float * x) {
    const gguf_tensor * t = g_.find("token_embd.weight");
    const size_t row_bytes = t->nbytes / t->ne[1];
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] < 0 || tokens[i] >= hp_.n_vocab) throw std::runtime_error("token id out of range");
        CK(cudaMemcpyAsync((char *) d_q_ + i * row_bytes, t->data + (size_t) tokens[i] * row_bytes, row_bytes,
                           cudaMemcpyHostToDevice, s_));
    }
    dequant_f32(t->type, d_q_, x, (int64_t) tokens.size() * hp_.n_embd, s_);
}

void model::dbg(const char * name, int il, const float * x, int64_t n) {
    if (!debug_) return;
    std::vector<float> h(n);
    CK(cudaMemcpyAsync(h.data(), x, n * 4, cudaMemcpyDeviceToHost, s_));
    CK(cudaStreamSynchronize(s_));
    double sum = 0;
    for (float v : h) sum += v;
    if (il >= 0) printf("DBG %s-%d sum = %.6f  first = %.4f %.4f %.4f\n", name, il, sum, h[0], h[1], h[2]);
    else         printf("DBG %s sum = %.6f  first = %.4f %.4f %.4f\n", name, sum, h[0], h[1], h[2]);
}

std::vector<float> model::forward(const std::vector<int32_t> & tokens) {
    const int T = (int) tokens.size();
    if (T < 1 || T > max_batch_) throw std::runtime_error("batch size out of range");
    if (n_past_ + T > n_ctx_) throw std::runtime_error("context full");
    const hparams & h = hp_;
    const int E = h.n_embd, HD = h.head_dim;
    const int Hk = h.ssm_groups, Hv = h.ssm_dt_rank, Dk = h.ssm_state, Dv = h.ssm_inner / h.ssm_dt_rank;
    const int conv_dim = 2 * Hk * Dk + h.ssm_inner;

    std::vector<int32_t> pos(T);
    for (int i = 0; i < T; ++i) pos[i] = n_past_ + i;
    CK(cudaMemcpyAsync(d_pos_, pos.data(), T * 4, cudaMemcpyHostToDevice, s_));

    embed(tokens, x_);
    dbg("model.input_embed", -1, x_, (int64_t) T * E);

    for (int il = 0; il < h.n_layer; ++il) {
        k::rmsnorm(x_, small(bn(il, "attn_norm.weight")), cur_, T, E, h.rms_eps, s_);
        dbg("attn_norm", il, cur_, (int64_t) T * E);

        if (h.is_attn(il)) {
            const int H = h.n_head, Hkv = h.n_head_kv;
            matmul(bn(il, "attn_q.weight"), cur_, a_, T);                    // [T][H][2*HD]
            k::split_q_gate(a_, c_, d_, T, H, HD, s_);                       // q -> c, gate -> d
            k::rmsnorm(c_, small(bn(il, "attn_q_norm.weight")), c_, (int64_t) T * H, HD, h.rms_eps, s_);
            matmul(bn(il, "attn_k.weight"), cur_, e_, T);                    // [T][Hkv][HD]
            k::rmsnorm(e_, small(bn(il, "attn_k_norm.weight")), e_, (int64_t) T * Hkv, HD, h.rms_eps, s_);
            matmul(bn(il, "attn_v.weight"), cur_, f_, T);
            k::rope_neox(c_, d_pos_, T, H, HD, h.n_rot, h.rope_base, s_);
            k::rope_neox(e_, d_pos_, T, Hkv, HD, h.n_rot, h.rope_base, s_);
            k::kv_store(e_, f_, kc_[il], vc_[il], T, n_past_, Hkv * HD, s_);
            k::attention(c_, kc_[il], vc_[il], b_, T, n_past_, H, Hkv, HD, 1.0f / sqrtf((float) HD), s_);
            dbg("attn_pregate", il, b_, (int64_t) T * H * HD);
            k::sigmoid_mul(b_, d_, (int64_t) T * H * HD, s_);
            matmul(bn(il, "attn_output.weight"), b_, res_, T);
        } else {
            matmul(bn(il, "attn_qkv.weight"), cur_, a_, T);                  // [T][conv_dim]
            matmul(bn(il, "attn_gate.weight"), cur_, b_, T);                 // z [T][Hv*Dv]
            matmul(bn(il, "ssm_beta.weight"), cur_, d_, T);                  // [T][Hv]
            matmul(bn(il, "ssm_alpha.weight"), cur_, e_, T);                 // [T][Hv]
            k::gdn_gates(e_, d_, small(bn(il, "ssm_dt.bias")), small(bn(il, "ssm_a")), T, Hv, s_);
            k::conv1d_silu(a_, conv_state_[il], small(bn(il, "ssm_conv1d.weight")), c_, T, conv_dim, s_);
            dbg("conv_output_silu", il, c_, (int64_t) T * conv_dim);
            k::gdn_recurrent(c_, e_, d_, ssm_state_[il], f_, T, Hk, Hv, Dk, Dv, h.rms_eps, s_);
            dbg("attn_output", il, f_, (int64_t) T * Hv * Dv);
            k::rmsnorm_gated(f_, small(bn(il, "ssm_norm.weight")), b_, f_, (int64_t) T * Hv, Dv, h.rms_eps, s_);
            matmul(bn(il, "ssm_out.weight"), f_, res_, T);
        }
        k::add(x_, res_, (int64_t) T * E, s_);                                // attn residual
        dbg("attn_residual", il, x_, (int64_t) T * E);

        k::rmsnorm(x_, small(bn(il, "post_attention_norm.weight")), cur_, T, E, h.rms_eps, s_);
        matmul(bn(il, "ffn_gate.weight"), cur_, a_, T);
        matmul(bn(il, "ffn_up.weight"), cur_, b_, T);
        k::silu_mul(a_, b_, (int64_t) T * h.n_ff, s_);
        matmul(bn(il, "ffn_down.weight"), a_, res_, T);
        k::add(x_, res_, (int64_t) T * E, s_);
        dbg("l_out", il, x_, (int64_t) T * E);
    }

    // final norm + LM head on the last token only
    const float * last = x_ + (int64_t)(T - 1) * E;
    k::rmsnorm(last, small("output_norm.weight"), cur_, 1, E, h.rms_eps, s_);
    float * d_logits = nullptr;
    CK(cudaMalloc(&d_logits, (size_t) h.n_vocab * 4));
    matmul(g_.find("output.weight") ? "output.weight" : "token_embd.weight", cur_, d_logits, 1);
    std::vector<float> logits(h.n_vocab);
    CK(cudaMemcpyAsync(logits.data(), d_logits, (size_t) h.n_vocab * 4, cudaMemcpyDeviceToHost, s_));
    CK(cudaStreamSynchronize(s_));
    cudaFree(d_logits);
    n_past_ += T;
    return logits;
}

} // namespace p100e
