#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Description of a folded-in SSM state GET_ROWS: read row rows[seq] straight out of base, and write
// the same values to gather_dst so the folded GET_ROWS node is preserved.
struct ggml_cuda_gated_delta_net_fused_gather {
    const float *   base       = nullptr; // the GET_ROWS input (the state cache)
    const int32_t * rows       = nullptr; // row indices (n_seqs of them)
    int64_t         row_stride = 0;       // floats per cache row
    float *         gather_dst = nullptr; // dst of the folded GET_ROWS node
};

// Variant with beta = sigmoid(x) folded into the prologue.  cache behaves as before (nullptr
// disables it).  With fuse_beta_sigmoid the kernel reads dst->src[4]->src[0], applies sigmoid,
// and also writes the result to dst->src[4]->data, so the folded node is preserved.
void ggml_cuda_op_gated_delta_net_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                        const ggml_cuda_gated_delta_net_fused_cache *  cache,
                                        bool                                           fuse_beta_sigmoid,
                                        const ggml_cuda_gated_delta_net_fused_gather * gather = nullptr);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// Ported from Kmic-68/llama.cpp 77e05601d (MIT): gdn-chunked.cu, the chunked prefill form and an
// fp64 reference, both opt-in by env (GGML_CUDA_GDN_CHUNKED, GGML_CUDA_GDN_REF; default off here).
// pf-gdn: beta_pre/beta_sig/gather_dst carry our beta-sigmoid and state-gather fusions
// (nullptr when not fused); s_idx/s_row are the gather rows, curr_state is then the cache base.
struct ggml_cuda_gdn_alt_args {
    const float * q; const float * k; const float * v; const float * g; const float * beta;
    const float * curr_state; float * dst; float * state;
    int64_t H, n_tokens, n_seqs, neqk1, rq3;
    int64_t sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3;
    float scale; int64_t state_slot_stride; int K; const int32_t * s_idx; int64_t s_row;
    const float * beta_pre; float * beta_sig; float * gather_dst;
};
// returns false when the shipped recurrence should run (always, unless one of the envs is set)
bool ggml_cuda_gdn_alt(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_alt_args & a);
