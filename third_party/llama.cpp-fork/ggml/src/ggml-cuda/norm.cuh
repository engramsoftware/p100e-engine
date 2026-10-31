#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

// Fuse ADD -> RMS_NORM -> MUL into one launch (residual add folded into rms_norm's prologue)
void ggml_cuda_op_rms_norm_fused_pre_add(ggml_backend_cuda_context & ctx,
                                         ggml_tensor *               add_tensor,
                                         ggml_tensor *               dst,
                                         ggml_tensor *               mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Merge a run of identically shaped L2_NORM nodes into one launch (implementation and
// preconditions in norm.cu)
#define CUDA_L2_NORM_FUSE_MAX 4

bool ggml_cuda_l2_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor ** nodes, int n);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
