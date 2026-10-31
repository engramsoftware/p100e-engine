#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

// Merge a run of identically shaped CPY nodes into one launch (implementation and preconditions
// in cpy.cu)
#define CUDA_CPY_FUSE_MAX 8

bool ggml_cuda_cpy_fused(ggml_backend_cuda_context & ctx, ggml_tensor ** nodes, int n);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
