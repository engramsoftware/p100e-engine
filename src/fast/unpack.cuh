// p100e::fast: dequantize PACKED weights (fast::pack_weight layouts) to f16 for the prefill GEMM.
#pragma once
#include "fast/matvec.cuh"
#include <cuda_fp16.h>

namespace p100e::fast {

// implementations (bit-identical outputs)
enum {
    UNPACK_SMEM   = 0,   // one block per row: stage + un-permute in shared memory, then ggml's dequant functions
    UNPACK_DIRECT = 1,   // fork-style direct kernels (q4r-sm60.cu q*_to_f16): flat grid, a thread (warp) per group,
                         // 32-bit loads, 8/16-byte stores, no shared memory
};

// P100E_UNPACK (0 = smem, 1 = direct; default direct), read once
int unpack_impl_default();

// dst[nrows][n_in] (f16, device) = rows [row0, row0 + nrows) of w, bit-identical to ggml's to-f16 dequant of the
// original raw GGUF rows. Every packed layout is a per-row permutation (row stride = fast::row_stride(w): the raw row
// bytes, or a padded stride for Q6_K, whose pad bytes are never read), so row0 may be any row. LAYOUT_RAW / LAYOUT_KQRAW call the ggml converter on the row range. The direct kernels need a 4-byte
// aligned row stride (8 for q8r) and a 16-byte aligned dst; otherwise the smem kernel runs.
// Throws std::runtime_error for an unsupported type/layout or (smem kernel) a row too large for shared memory
// (2*row_bytes > 48 KiB).
void packed_to_f16(const packed_weight & w, int64_t row0, int64_t nrows, half * dst, cudaStream_t s);
void packed_to_f16(const packed_weight & w, int64_t row0, int64_t nrows, half * dst, cudaStream_t s, int impl);

} // namespace p100e::fast
