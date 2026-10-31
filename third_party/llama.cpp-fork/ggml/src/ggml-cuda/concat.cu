#include "concat.cuh"

#include <stdint.h>

// contiguous kernels
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE) concat_cont(const T * x,
                                                                             const T * y,
                                                                             T *       dst,
                                                                             int64_t   ne00,
                                                                             int64_t   ne01,
                                                                             int64_t   ne02,
                                                                             int64_t   ne0,
                                                                             int64_t   ne1,
                                                                             int64_t   ne2) {
    static_assert(dim >= 0 && dim <= 2, "dim must be in [0, 2]");

    const int64_t n = ne0 * ne1 * ne2;

    ggml_cuda_pdl_sync();
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        if constexpr (dim == 0) {
            const int64_t row = i / ne0;
            const int64_t i0  = i - row * ne0;

            if (i0 < ne00) {
                dst[i] = x[row * ne00 + i0];
            } else {
                dst[i] = y[row * (ne0 - ne00) + (i0 - ne00)];
            }
        } else if constexpr (dim == 1) {
            const int64_t dst_plane  = ne0 * ne1;
            const int64_t src0_plane = ne0 * ne01;
            const int64_t src1_plane = dst_plane - src0_plane;
            const int64_t i2         = i / dst_plane;
            const int64_t i01        = i - i2 * dst_plane;

            if (i01 < src0_plane) {
                dst[i] = x[i2 * src0_plane + i01];
            } else {
                dst[i] = y[i2 * src1_plane + (i01 - src0_plane)];
            }
        } else {
            const int64_t src0_size = ne0 * ne1 * ne02;

            if (i < src0_size) {
                dst[i] = x[i];
            } else {
                dst[i] = y[i - src0_size];
            }
        }
    }
}

template <typename T>
static void concat_cont_cuda(const T * x,
                             const T * y,
                             T *       dst,
                             int64_t   ne00,
                             int64_t   ne01,
                             int64_t   ne02,
                             int64_t   ne0,
                             int64_t   ne1,
                             int64_t   ne2,
                             int       dim,
                             cudaStream_t stream) {
    const int64_t n          = ne0 * ne1 * ne2;
    const int     num_blocks = (n + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;

    if (dim == 0) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream);
        ggml_cuda_kernel_launch(concat_cont<T, 0>, launch_params, x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
        return;
    }
    if (dim == 1) {
        concat_cont<T, 1><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
        return;
    }
    concat_cont<T, 2><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
}

// non-contiguous kernel (slow)
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3) {
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const T * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const T *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10);
            } else if constexpr (dim == 1) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + (i1 - ne01)*nb11 + i0*nb10);
            } else if constexpr (dim == 2) {
                x = (const T *)(src1 + i3*nb13 + (i2 - ne02)*nb12 + i1*nb11 + i0*nb10);
            } else if constexpr (dim == 3) {
                x = (const T *)(src1 + (i3 - ne03)*nb13 + i2*nb12 + i1*nb11 + i0*nb10);
            }
        }

        T * y = (T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}

// Flattened variant of the non-contiguous kernel.
//
// The kernel above maps one CUDA block to each (i1, i2, i3) and loops i0 over blockDim.x, so only
// min(ne0, blockDim.x) of the CUDA_CONCAT_BLOCK_SIZE threads in a block ever execute.  For the
// delta-net conv-state concat of Qwen3-Next / Qwen3.5 (src/models/delta-net-base.cpp,
// build_conv_state(): concat(conv_states, ggml_transpose(qkv_mixed), /*dim=*/0) -- the transpose is
// what forces the non-contiguous path) the shape is
//   ne0 = (conv_kernel_size - 1) + n_tokens = 4..7,  ne1 = conv_channels = 8192,
// so 2.1M threads are launched to move ~32k elements: a 128 KiB copy costs 33 us, which is 4 GB/s
// on a card with 732 GB/s of HBM2 (the ne0 = 7 end of the range moves 224 KiB).  Mapping one
// thread per output element instead measures, at the same shape,
// 18.0 -> 4.7 us on a Tesla P100 (sm_60) with bit-identical output.
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont_flat(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
             uint3   ne0_fdv,
             uint3   ne1_fdv,
             uint3   ne2_fdv,
          uint32_t   nelem,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3) {
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    for (uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x; idx < nelem; idx += blockDim.x*gridDim.x) {
        const uint2 dm0 = fast_div_modulo(idx,   ne0_fdv);
        const uint2 dm1 = fast_div_modulo(dm0.x, ne1_fdv);
        const uint2 dm2 = fast_div_modulo(dm1.x, ne2_fdv);

        const int64_t i0 = dm0.y;
        const int64_t i1 = dm1.y;
        const int64_t i2 = dm2.y;
        const int64_t i3 = dm2.x;

        const T * x;

        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const T *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10);
            } else if constexpr (dim == 1) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + (i1 - ne01)*nb11 + i0*nb10);
            } else if constexpr (dim == 2) {
                x = (const T *)(src1 + i3*nb13 + (i2 - ne02)*nb12 + i1*nb11 + i0*nb10);
            } else if constexpr (dim == 3) {
                x = (const T *)(src1 + (i3 - ne03)*nb13 + i2*nb12 + i1*nb11 + i0*nb10);
            }
        }

        T * y = (T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}

// Ported from Kmic-68/llama.cpp dd24ec3b8 (MIT): tiled dim-0 concat for a transposed src1.
// GGML_CUDA_CONCAT_TILED=1 enables it (default OFF in this fork; Kmic's default is on).
// Dim-0 concat whose src1 is a transposed view (src1 contiguous along dim 1), i.e. the delta net's conv
// input during prefill: concat(conv_states [3, C, S], transpose(qkv_mixed) [T, C, S], 0) with C = 5120
// per GPU under -sm tensor. The block-per-row kernel (ne0 >= 256) and the flat kernel (ne0 < 256) both
// read src1 with a stride of a whole qkv row (20 KiB) between neighbouring threads; a 32x32 shared tile
// reads src1 along dim 1 and writes dst along dim 0, both coalesced. Same copy, same bytes.
template <typename T>
static __global__ void concat_dim0_tiled(
        const char * src0, const char * src1, char * dst, const int64_t ne00, const int64_t ne0, const int64_t ne1,
        const uint64_t nb00, const uint64_t nb01, const uint64_t nb02, const uint64_t nb10, const uint64_t nb12,
        const uint64_t nb1, const uint64_t nb2) {
    __shared__ T tile[32][33];
    const int64_t i00 = (int64_t) blockIdx.x*32;
    const int64_t i10 = (int64_t) blockIdx.y*32;
    const int64_t i2  = blockIdx.z;
    // read: threadIdx.x runs along dim 1 (contiguous in src1)
    for (int r = threadIdx.y; r < 32; r += blockDim.y) {
        const int64_t i0 = i00 + r;
        const int64_t i1 = i10 + threadIdx.x;
        if (i0 < ne0 && i1 < ne1) {
            tile[r][threadIdx.x] = i0 < ne00 ? *(const T *) (src0 + i2*nb02 + i1*nb01 + i0*nb00)
                                             : *(const T *) (src1 + i2*nb12 + i1*sizeof(T) + (i0 - ne00)*nb10);
        }
    }
    __syncthreads();
    // write: threadIdx.x runs along dim 0 (contiguous in dst)
    for (int r = threadIdx.y; r < 32; r += blockDim.y) {
        const int64_t i0 = i00 + threadIdx.x;
        const int64_t i1 = i10 + r;
        if (i0 < ne0 && i1 < ne1) {
            *(T *) (dst + i2*nb2 + i1*nb1 + i0*sizeof(T)) = tile[threadIdx.x][r];
        }
    }
}

// Launches concat_dim0_tiled when enabled and the shapes match; false otherwise (caller falls through).
template <typename T>
static bool concat_dim0_tiled_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int dim,
                                   cudaStream_t stream) {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_CONCAT_TILED");
        return e != nullptr && std::atoi(e) != 0;
    }();
    if (!on || dim != 0 || dst->ne[0] <= 64 || ggml_is_quantized(src0->type) ||
        (size_t) src1->nb[1] != sizeof(T) || src1->nb[0] <= src1->nb[1] || (size_t) dst->nb[0] != sizeof(T) ||
        src0->ne[3] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1 ||
        (dst->ne[1] + 31)/32 > 65535 || dst->ne[2] > 65535) {
        return false;
    }
    const dim3 grid((dst->ne[0] + 31)/32, (dst->ne[1] + 31)/32, dst->ne[2]);
    concat_dim0_tiled<T><<<grid, dim3(32, 8, 1), 0, stream>>>(
        (const char *) src0->data, (const char *) src1->data, (char *) dst->data, src0->ne[0], dst->ne[0], dst->ne[1],
        src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[0], src1->nb[2], dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// CONCAT along dim 0 recovers the indices per element, which becomes the bottleneck for shapes
// with short rows and many of them -- delta-net's conv state is ne0 = 4..8 with ne1 = 8192, and
// measures 51 GB/s on a P100, 8% of the 606 GB/s ceiling.
// One thread per row reduces the indexing to a single multiply per row, and when the dst row is
// exactly 16 B it can be written with one vector store.  Element order and the number of reads
// and writes are unchanged, so the output is bit-identical.
#define CUDA_CONCAT_ROWS_MAX_NE0 8

// Where output element c is read from: src0's row when c < ne00, otherwise src1's row.
// The address is selected first and read once, so the side not selected is never dereferenced.
#define CUDA_CONCAT_ROWS_SRC(c) ((c) < ne00 ? (x + (int64_t) (c)*nb00) : (y + (int64_t) ((c) - ne00)*nb10))

template <typename T, int NE0, bool VEC16>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_rows(const char * __restrict__ src0,
                const char * __restrict__ src1,
                      char * __restrict__ dst,
                const int     ne00,
                const int     nrows,
                const int64_t nb00, const int64_t nb01,
                const int64_t nb10, const int64_t nb11,
                const int64_t nb0,  const int64_t nb1) {
    ggml_cuda_pdl_lc();

    const int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }

    const char * x = src0 + (int64_t) row*nb01;
    const char * y = src1 + (int64_t) row*nb11;
    char *       d = dst  + (int64_t) row*nb1;

    ggml_cuda_pdl_sync();

    if constexpr (VEC16) {
        // Only taken for NE0 == 4 with a 4-byte type: the dst row is exactly 16 B, one store
        int4 v;
        v.x = *(const int *) CUDA_CONCAT_ROWS_SRC(0);
        v.y = *(const int *) CUDA_CONCAT_ROWS_SRC(1);
        v.z = *(const int *) CUDA_CONCAT_ROWS_SRC(2);
        v.w = *(const int *) CUDA_CONCAT_ROWS_SRC(3);
        *(int4 *) d = v;
    } else {
        // Write as we read; staging into a temporary array risks spilling it to local memory
        // (the destinations do not overlap, so the order does not affect the result)
#pragma unroll
        for (int c = 0; c < NE0; ++c) {
            *(T *) (d + (int64_t) c*nb0) = *(const T *) CUDA_CONCAT_ROWS_SRC(c);
        }
    }
}

// The kernel above with src0 replaced by "row rows[0] of the cache", which folds away the
// preceding GET_ROWS.  Node-preserving, so the folded GET_ROWS dst is written as well.
template <typename T, int NE0, bool VEC16>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_rows_gather(const char    * __restrict__ cache,
                       const int32_t * __restrict__ rows,
                       const char    * __restrict__ src1,
                             char    * __restrict__ dst,
                             char    * __restrict__ gdst,
                       const int     ne00,
                       const int     nrows,
                       const int64_t crow,
                       const int64_t nb00, const int64_t nb01,
                       const int64_t nb10, const int64_t nb11,
                       const int64_t nb0,  const int64_t nb1) {
    ggml_cuda_pdl_lc();

    const int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }

    // The folded GET_ROWS output is a contiguous copy from the start of the cache row, so offsets
    // within the row carry over unchanged (shifted by the row index)
    const char * x = cache + (int64_t) rows[0]*crow + (int64_t) row*nb01;
    const char * y = src1  + (int64_t) row*nb11;
    char *       d = dst   + (int64_t) row*nb1;
    char *       g = gdst  + (int64_t) row*nb01;

    ggml_cuda_pdl_sync();

    if constexpr (VEC16) {
        int4 v;
        v.x = *(const int *) CUDA_CONCAT_ROWS_SRC(0);
        v.y = *(const int *) CUDA_CONCAT_ROWS_SRC(1);
        v.z = *(const int *) CUDA_CONCAT_ROWS_SRC(2);
        v.w = *(const int *) CUDA_CONCAT_ROWS_SRC(3);
        *(int4 *) d = v;
        if (0 < ne00) { *(int *) (g              ) = v.x; }
        if (1 < ne00) { *(int *) (g +   nb00) = v.y; }
        if (2 < ne00) { *(int *) (g + 2*nb00) = v.z; }
        if (3 < ne00) { *(int *) (g + 3*nb00) = v.w; }
    } else {
#pragma unroll
        for (int c = 0; c < NE0; ++c) {
            const T val = *(const T *) CUDA_CONCAT_ROWS_SRC(c);
            *(T *) (d + (int64_t) c*nb0) = val;
            if (c < ne00) {
                *(T *) (g + (int64_t) c*nb00) = val;
            }
        }
    }
}

// Decide whether the per-row path applies and, if so, launch it and return true.
// Is the dst row exactly 16 B, i.e. can one thread write it with a single vector store?
static bool concat_rows_vec16(const ggml_tensor * dst) {
    const size_t ts = ggml_type_size(dst->type);
    return dst->ne[0]*ts == 16 && (size_t) dst->nb[0] == ts &&
           dst->nb[1] % 16 == 0 && ((uintptr_t) dst->data) % 16 == 0;
}

// The part of the test that does not depend on the template parameter T, shared with
// ggml_cuda_concat_can_fuse_gather.
// for_gather means "a preceding GET_ROWS is being folded in", i.e. a whole launch disappears.
static bool concat_rows_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst,
                                 int dim, bool for_gather) {
    static const bool disable = getenv("GGML_CUDA_DISABLE_CONCAT_ROWS") != nullptr &&
                                std::atoi(getenv("GGML_CUDA_DISABLE_CONCAT_ROWS"));
    if (disable || dim != 0 || ggml_is_quantized(src0->type) || ggml_type_size(dst->type) != 4) {
        return false;
    }

    const int64_t ne0  = dst->ne[0];
    const int64_t ne00 = src0->ne[0];
    if (ne0 < 2 || ne0 > CUDA_CONCAT_ROWS_MAX_NE0 || ne00 < 1 || ne00 >= ne0) {
        return false;
    }

    // One thread per row only wins on its own when the dst row is exactly 16 B, i.e. a single
    // vector store.  For longer rows neighbouring threads write 20-32 B apart, coalescing breaks
    // down, and it ties or slightly loses against the one-thread-per-element path whose dst is
    // fully contiguous.  When a GET_ROWS is folded in, removing an entire launch dominates that,
    // so the restriction is relaxed there.
    if (!for_gather && !concat_rows_vec16(dst)) {
        return false;
    }

    // (i1, i2, i3) must collapse to a row index in equal steps of nb1.  Length-1 dimensions are
    // exempt from the stride test: ggml_transpose does not swap nb[2] / nb[3], so testing them
    // strictly gives the wrong answer.
    auto rowform = [](const ggml_tensor * t) {
        return (t->ne[2] == 1 || (uint64_t) t->nb[2] == (uint64_t) t->ne[1]*t->nb[1]) &&
               (t->ne[3] == 1 || (uint64_t) t->nb[3] == (uint64_t) t->ne[2]*t->nb[2]);
    };
    if (!rowform(src0) || !rowform(src1) || !rowform(dst)) {
        return false;
    }
    if (src0->ne[1] != dst->ne[1] || src0->ne[2] != dst->ne[2] || src0->ne[3] != dst->ne[3] ||
        src1->ne[1] != dst->ne[1] || src1->ne[2] != dst->ne[2] || src1->ne[3] != dst->ne[3]) {
        return false;
    }

    const int64_t nrows = dst->ne[1]*dst->ne[2]*dst->ne[3];
    return nrows > 0 && nrows <= INT_MAX;
}

template <typename T>
static bool concat_rows_cuda(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst,
                             int dim, cudaStream_t stream, const ggml_cuda_concat_gather * gather) {
    if constexpr (sizeof(T) != 4) {
        // 4-byte types (f32 / i32) only; other widths do not occur here and are not worth the code
        GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst); GGML_UNUSED(dim); GGML_UNUSED(stream);
        GGML_ASSERT(gather == nullptr);
        return false;
    } else {
        if (!concat_rows_eligible(src0, src1, dst, dim, gather != nullptr)) {
            return false;
        }

        const int64_t ne0   = dst->ne[0];
        const int64_t ne00  = src0->ne[0];
        const int64_t nrows = dst->ne[1]*dst->ne[2]*dst->ne[3];

        const bool vec16 = concat_rows_vec16(dst);
        // Microbenchmark optima: block 256 for the vector store, block 64 for per-element writes
        // (the gather variant always uses 64)
        // (the latter gives each thread little work, so smaller blocks spread better over the SMs)
        const int block  = (vec16 && gather == nullptr) ? CUDA_CONCAT_BLOCK_SIZE : 64;
        const int nblk   = (int) ((nrows + block - 1) / block);
        const int ne00_i = (int) ne00;
        const int nrow_i = (int) nrows;

        const char * s0 = (const char *) src0->data;
        const char * s1 = (const char *) src1->data;
        char *       d  = (char *)       dst->data;

#define CUDA_CONCAT_ROWS_LAUNCH(NE0, VEC)                                                        \
        do {                                                                                     \
            const ggml_cuda_kernel_launch_params lp =                                            \
                ggml_cuda_kernel_launch_params(nblk, block, 0, stream);                          \
            if (gather != nullptr) {                                                             \
                ggml_cuda_kernel_launch(concat_rows_gather<T, NE0, VEC>, lp,                     \
                                    gather->base, gather->rows, s1, d, gather->gdst,             \
                                    ne00_i, nrow_i, gather->row_stride,                          \
                                    (int64_t) src0->nb[0], (int64_t) src0->nb[1],                \
                                    (int64_t) src1->nb[0], (int64_t) src1->nb[1],                \
                                    (int64_t) dst->nb[0],  (int64_t) dst->nb[1]);                \
            } else {                                                                             \
                ggml_cuda_kernel_launch(concat_rows<T, NE0, VEC>, lp, s0, s1, d, ne00_i, nrow_i,  \
                                    (int64_t) src0->nb[0], (int64_t) src0->nb[1],                \
                                    (int64_t) src1->nb[0], (int64_t) src1->nb[1],                \
                                    (int64_t) dst->nb[0],  (int64_t) dst->nb[1]);                \
            }                                                                                    \
        } while (0)

        switch (ne0) {
            case 2: CUDA_CONCAT_ROWS_LAUNCH(2, false); break;
            case 3: CUDA_CONCAT_ROWS_LAUNCH(3, false); break;
            case 4:
                if (vec16) { CUDA_CONCAT_ROWS_LAUNCH(4, true); }
                else       { CUDA_CONCAT_ROWS_LAUNCH(4, false); }
                break;
            case 5: CUDA_CONCAT_ROWS_LAUNCH(5, false); break;
            case 6: CUDA_CONCAT_ROWS_LAUNCH(6, false); break;
            case 7: CUDA_CONCAT_ROWS_LAUNCH(7, false); break;
            case 8: CUDA_CONCAT_ROWS_LAUNCH(8, false); break;
            default: return false;
        }
#undef CUDA_CONCAT_ROWS_LAUNCH
        return true;
    }
}

bool ggml_cuda_concat_can_fuse_gather(const ggml_tensor * concat, const ggml_tensor * gr) {
    static const bool disable = getenv("GGML_CUDA_DISABLE_FUSE_CONCAT_GATHER") != nullptr &&
                                std::atoi(getenv("GGML_CUDA_DISABLE_FUSE_CONCAT_GATHER"));
    if (disable || concat->op != GGML_OP_CONCAT || gr->op != GGML_OP_GET_ROWS) {
        return false;
    }
    if (!concat_rows_eligible(concat->src[0], concat->src[1], concat,
                              ((const int32_t *) concat->op_params)[0], /*for_gather =*/ true)) {
        return false;
    }
    // The fold candidate: a contiguous f32 gather of exactly one row, from a contiguous cache row.
    // With a single row, the position in the cache follows from the row index with no division.
    return gr->type == GGML_TYPE_F32 && gr->src[0] != nullptr && gr->src[1] != nullptr &&
           gr->src[0]->type == GGML_TYPE_F32 && gr->src[1]->type == GGML_TYPE_I32 &&
           !(gr->flags & GGML_TENSOR_FLAG_OUTPUT) &&
           ggml_is_contiguous(gr) && gr->ne[2] == 1 && gr->ne[3] == 1 &&
           ggml_nelements(gr->src[1]) == 1 &&
           ggml_is_contiguous_rows(gr->src[0]) && gr->src[0]->nb[0] == sizeof(float) &&
           (size_t) concat->src[0]->nb[0] == sizeof(float) && gr->data != nullptr;
}

template <typename T>
static void concat_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int dim, cudaStream_t stream,
                        const ggml_cuda_concat_gather * gather = nullptr) {
    if (concat_rows_cuda<T>(src0, src1, dst, dim, stream, gather)) {
        return;
    }
    // Having committed to folding, failing to take the per-row path would leave the GET_ROWS
    // unexecuted and corrupt the result.  ggml_cuda_concat_can_fuse_gather already decided this,
    // so reaching here is a bug.
    GGML_ASSERT(gather == nullptr);

    if (dim != 3 && ggml_is_contiguous_to_3(src0) && ggml_is_contiguous_to_3(src1)) {
        const T * src0_d = (const T *) src0->data;
        const T * src1_d = (const T *) src1->data;
        T *       dst_d  = (T *) dst->data;

        for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            concat_cont_cuda(
                    src0_d + i3*(src0->nb[3] / sizeof(T)),
                    src1_d + i3*(src1->nb[3] / sizeof(T)),
                    dst_d  + i3*( dst->nb[3] / sizeof(T)),
                    ggml_row_size(src0->type, src0->ne[0])/sizeof(T), src0->ne[1], src0->ne[2],
                    ggml_row_size(dst->type, dst->ne[0])/sizeof(T),  dst->ne[1],  dst->ne[2], dim, stream);
        }
    } else if (dim == 3 && ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const size_t size0 = ggml_nbytes(src0);
        const size_t size1 = ggml_nbytes(src1);

        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data,         src0->data, size0, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data + size0, src1->data, size1, cudaMemcpyDeviceToDevice, stream));
    } else {
        GGML_ASSERT(!ggml_is_quantized(src0->type));

        // Ported from Kmic-68/llama.cpp dd24ec3b8 (MIT); GGML_CUDA_CONCAT_TILED=1
        if (concat_dim0_tiled_cuda<T>(src0, src1, dst, dim, stream)) {
            return;
        }

        // The block-per-row kernel below leaves blockDim.x - ne0 threads per block idle, which
        // dominates whenever ne0 is small and ne1 is large (delta-net conv state: ne0 = 4..7 with
        // ne1 = 8192).  Use the flattened one-thread-per-element mapping in that regime.  Gated on
        // uint32 because fast_div_modulo() operates on 32-bit indices.
        const int64_t nelem = ggml_nelements(dst);

        if (dst->ne[0] < CUDA_CONCAT_BLOCK_SIZE && nelem <= (int64_t) std::numeric_limits<uint32_t>::max()) {
            const int num_blocks = (int) ((nelem + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE);

            const uint3 ne0_fdv = init_fastdiv_values(dst->ne[0]);
            const uint3 ne1_fdv = init_fastdiv_values(dst->ne[1]);
            const uint3 ne2_fdv = init_fastdiv_values(dst->ne[2]);

            auto launch_flat = [&](auto dim) {
                concat_non_cont_flat<T, dim><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                    (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                    ne0_fdv, ne1_fdv, ne2_fdv, (uint32_t) nelem,
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            };
            switch (dim) {
                case 0:
                    launch_flat(std::integral_constant<int, 0>{});
                    break;
                case 1:
                    launch_flat(std::integral_constant<int, 1>{});
                    break;
                case 2:
                    launch_flat(std::integral_constant<int, 2>{});
                    break;
                case 3:
                    launch_flat(std::integral_constant<int, 3>{});
                    break;
                default:
                    GGML_ABORT("Invalid dim: %d", dim);
                    break;
            }
            return;
        }

        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        auto launch_kernel = [&](auto dim) {
            concat_non_cont<T, dim><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        };
        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
                break;
        }
    }
}

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    // Collect the GET_ROWS that was deferred for this CONCAT, if any
    ggml_cuda_concat_gather gather;
    bool has_gather = false;
    if (ctx.gdn_gather_owner == dst && ctx.gdn_gather_node != nullptr) {
        const ggml_tensor * gr = ctx.gdn_gather_node;
        gather.base       = (const char *) gr->src[0]->data;
        gather.rows       = ctx.gdn_rows_scratch;
        gather.row_stride = (int64_t) gr->src[0]->nb[1];
        gather.gdst       = (char *) gr->data;
        ctx.gdn_gather_clear();
        has_gather = gather.base != nullptr && gather.rows != nullptr &&
                     gather.gdst != nullptr && gather.row_stride > 0;
        GGML_ASSERT(has_gather);
    }

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(dst->type  == src0->type);

    if (ggml_is_quantized(src0->type)) {
        if (dim == 3) {
            GGML_ASSERT(ggml_is_contiguous(src0));
            GGML_ASSERT(ggml_is_contiguous(src1));
        } else {
            GGML_ASSERT(ggml_is_contiguous_to_3(src0));
            GGML_ASSERT(ggml_is_contiguous_to_3(src1));
        }
        GGML_ASSERT(src0->ne[0] % ggml_blck_size(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_size(src1->type) == 0);

        // if first 3 dimensions are contiguous and ne[0] is multiple of the block size we can concat both tensors as byte tensors
        GGML_ASSERT(!has_gather);
        concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
    } else {
        GGML_ASSERT(ggml_blck_size(src0->type) == 1);

        switch (ggml_type_size(src0->type)) {
            case 1:
                concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
                break;
            case 2:
                concat_cuda<uint16_t>(src0, src1, dst, dim, stream);
                break;
            case 4:
                concat_cuda<uint32_t>(src0, src1, dst, dim, stream, has_gather ? &gather : nullptr);
                break;
            case 8:
                concat_cuda<uint64_t>(src0, src1, dst, dim, stream);
                break;
            default:
                GGML_ABORT("Unsupported type size: %zu", ggml_type_size(src0->type));
                break;
        }
    }
}
