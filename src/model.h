// Qwen3.8 (GGUF arch "qwen35": hybrid gated-delta-net + gated full attention, dense FFN) model.
// v0 = correctness reference: quantized weights stay in host memory (mmap) and are streamed to the
// GPU per matmul, dequantized to f32 and multiplied with cuBLAS. One GPU, f32 activations.
#pragma once

#include "gguf.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace p100e {

struct hparams {
    int n_embd = 0, n_layer = 0, n_layer_all = 0, n_ff = 0, n_vocab = 0;
    int n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0, full_attn_interval = 4;
    float rms_eps = 1e-6f, rope_base = 1e7f;
    int ssm_conv = 4, ssm_inner = 0, ssm_state = 0, ssm_groups = 0, ssm_dt_rank = 0;
    bool is_attn(int il) const { return (il + 1) % full_attn_interval == 0; }
};

class model {
public:
    model(const std::string & path, int n_ctx, int max_batch);
    ~model();

    const hparams & hp() const { return hp_; }
    // run tokens at positions n_past..n_past+T-1, return logits of the last token
    std::vector<float> forward(const std::vector<int32_t> & tokens);
    int n_past() const { return n_past_; }
    void set_debug(bool d) { debug_ = d; }

private:
    void matmul(const std::string & w, const float * x, float * y, int T);   // y[T][out] = x[T][in] * W^T
    const float * small(const std::string & name);                           // resident f32 copy
    void embed(const std::vector<int32_t> & tokens, float * x);
    void dbg(const char * name, int il, const float * x, int64_t n);

    gguf_file g_;
    hparams hp_;
    int n_ctx_, max_batch_, n_past_ = 0;
    bool debug_ = false;
    cudaStream_t s_ = nullptr;
    cublasHandle_t cb_ = nullptr;

    // weight streaming scratch
    void  * d_q_ = nullptr;   size_t q_bytes_ = 0;
    float * d_w_ = nullptr;   size_t w_elems_ = 0;
    std::map<std::string, float *> small_;

    // activations
    float *x_, *cur_, *res_, *a_, *b_, *c_, *d_, *e_, *f_;
    int32_t * d_pos_;
    // state: KV caches for attention layers, conv + delta states for linear layers
    std::vector<float *> kc_, vc_, conv_state_, ssm_state_;
};

} // namespace p100e
