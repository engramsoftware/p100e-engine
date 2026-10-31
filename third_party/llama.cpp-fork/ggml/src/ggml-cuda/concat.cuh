#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

// Description of a GET_ROWS folded into a CONCAT: read row rows[0] straight out of base.
struct ggml_cuda_concat_gather {
    const char *    base       = nullptr;   // start of the GET_ROWS src0 (the cache)
    const int32_t * rows       = nullptr;   // the saved row index (a single element)
    int64_t         row_stride = 0;         // bytes per cache row
    char *          gdst       = nullptr;   // dst of the folded GET_ROWS (written, node-preserving)
};

// Can this get_rows node be folded into this concat node?  GET_ROWS execution is only deferred
// when it can be: deferring without the concat taking the per-row path corrupts the output.
bool ggml_cuda_concat_can_fuse_gather(const ggml_tensor * concat, const ggml_tensor * gr);

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
