#include "norm.cuh"
#include <cstdint>

template <int block_size>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float2 mean_var = make_float2(0.0f, 0.0f);

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = (x[col] - mean) * inv_std;
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    const int start =     blockIdx.x*group_size + threadIdx.x;
    const int end   = min(blockIdx.x*group_size + group_size,  ne_elements);

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum + 32);

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

template <int block_size, bool do_multiply = false, bool do_add = false, bool do_pre_add = false>
static __global__ void rms_norm_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    // Fold the residual add into the prologue: compute x = pre_a + pre_b, store it to
                                    // pre_dst (the ADD's destination, which is where rms_norm reads x from) and feed it
                                    // straight into the sum of squares.  rms_norm maps one row to one block, so the row
                                    // is contained in the block and no cross-block synchronization is needed.
                                    const float * pre_a                = nullptr,
                                    const float * pre_b                = nullptr,
                                    float *       pre_dst              = nullptr) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_pre_add) {
        // The predicate guarantees the ADD's three tensors are contiguous and identically
        // shaped, so the same offset as x is correct
        const int64_t pre_off = sample*stride_sample + channel*stride_channel + row*stride_row;
        pre_a   += pre_off;
        pre_b   += pre_off;
        pre_dst += pre_off;
    }

    if constexpr (do_add) {
        const int add_row     = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    // x is read twice (once for the sum of squares, once for the scaling).
    // For a single row the grid is 1 block, so this kernel runs on one SM and is limited by the
    // bytes it moves through it; on a Tesla P100 the 4096-wide norms of a decode step cost
    // 7.82 us each, 7075 times per 256 tokens = 1.9% of the decode wall.  When the row fits in
    // max_cache values per thread, keep it in registers instead and drop the second read.
    // The <1024> instantiation is the only one launched with a row wide enough to need more than
    // one value per thread (rms_norm_f32_cuda picks <256> when ncols < 1024), and adding the
    // branch to the narrow one COST ~6% on the 128/256-wide norms, so gate it at compile time.
    constexpr int  max_cache = 4;
    constexpr bool do_cache  = block_size >= 1024;
    const bool cached = do_cache && ncols > block_size && ncols <= max_cache*block_size;
    float xv[max_cache];

    ggml_cuda_pdl_sync();
    if (cached) {
#pragma unroll
        for (int k = 0; k < max_cache; ++k) {
            const int col = tid + k*block_size;
            float     xi  = 0.0f;
            if (col < ncols) {
                if constexpr (do_pre_add) {
                    xi           = pre_a[col] + pre_b[col];
                    pre_dst[col] = xi; // always written: the next layer reads it as its residual
                } else {
                    xi = x[col];
                }
            }
            xv[k] = xi;
            tmp  += xi * xi;
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            float xi;
            if constexpr (do_pre_add) {
                xi           = pre_a[col] + pre_b[col];
                pre_dst[col] = xi;
            } else {
                xi = x[col];
            }
            tmp += xi * xi;
        }
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    auto store = [&](const int col, const float xi) {
        if constexpr (do_multiply && do_add) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const int add_col = fastmodulo(col, add_ncols_packed);
            dst[col]          = scale * xi * mul[mul_col] + add[add_col];
        } else if constexpr (do_multiply) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col]          = scale * xi * mul[mul_col];
        } else {
            dst[col] = scale * xi;
        }
    };

    if (cached) {
#pragma unroll
        for (int k = 0; k < max_cache; ++k) {
            const int col = tid + k*block_size;
            if (col < ncols) {
                store(col, xv[k]);
            }
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            store(col, x[col]);
        }
    }
}

// Ported from Kmic-68/llama.cpp c5d1110a2 (MIT): warp-per-row RMS norm for short rows.
// GGML_CUDA_NORM_WARP=1 enables it (default OFF in this fork; Kmic's default is on).
// rms_norm_f32<256, ...> for rows of 32..256 floats, one warp per row (RPB rows per block). The 256-thread
// block left most of its threads idle on such rows and ran at ~50 GB/s. Bit-identical to rms_norm_f32<256>:
// lane l holds x[32w + l]^2 for each "virtual warp" w, sums each with the same butterfly the block's warp w
// ran (warp_reduce_sum), and combines the NW sums with the same 32-lane butterfly block_reduce runs over its
// 8 warp sums (lanes >= NW contribute exact zeros, as the block's empty warps and its sentinel lanes do);
// every output uses the same expression as the block kernel's store().
// Adapted: only the plain / mul / mul+add variants (this fork has no fused scale, pair-scale or gate norm);
// the do_pre_add variant (residual fold, patch "fuse pre-add") keeps the block kernel.
template <int NW, bool do_multiply, bool do_add>
static __global__ void __launch_bounds__(256) rms_norm_f32_warp(
        const float * x, float * dst, const int nrows, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps,
        const float * mul, const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed, const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const float * add, const int64_t add_stride_row, const int64_t add_stride_channel, const int64_t add_stride_sample,
        const uint3 add_ncols_packed, const uint3 add_nrows_packed, const uint3 add_nchannels_packed, const uint3 add_nsamples_packed) {
    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");
    constexpr int ncols = 32*NW;
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row >= nrows) {
        return; // whole warp: the row index is uniform across a warp
    }
    const int nchannels = gridDim.y;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int lane      = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }
    if constexpr (do_add) {
        const int add_row     = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float xv[NW];
    float ws[NW];
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        xv[w] = x[32*w + lane];
        float t = 0.0f;
        t += xv[w] * xv[w];
        ws[w] = warp_reduce_sum(t);
    }
    float v2 = 0.0f;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        v2 = lane == w ? ws[w] : v2;
    }
    const float tmp   = warp_reduce_sum(v2);
    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        const int col = 32*w + lane;
        if constexpr (do_multiply && do_add) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const int add_col = fastmodulo(col, add_ncols_packed);
            dst[col]          = scale * xv[w] * mul[mul_col] + add[add_col];
        } else if constexpr (do_multiply) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col]          = scale * xv[w] * mul[mul_col];
        } else {
            dst[col] = scale * xv[w];
        }
    }
}

static bool rms_norm_warp_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_NORM_WARP");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

// Launches rms_norm_f32_warp in place of rms_norm_f32<256, ...> when enabled and the row fits
// (32..256 floats, a multiple of 32); returns false otherwise (caller launches the block kernel).
template <bool do_multiply, bool do_add>
static bool rms_norm_f32_try_warp(const dim3 blocks_num, const int ncols, cudaStream_t stream,
        const float * x, float * dst, const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps,
        const float * mul, const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed, const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const float * add, const int64_t add_stride_row, const int64_t add_stride_channel, const int64_t add_stride_sample,
        const uint3 add_ncols_packed, const uint3 add_nrows_packed, const uint3 add_nchannels_packed, const uint3 add_nsamples_packed) {
    if (!rms_norm_warp_enabled() || ncols <= 0 || ncols % 32 != 0 || ncols > 256) {
        return false;
    }
    constexpr int RPB = 8;
    const int nrows = (int) blocks_num.x;
    const dim3 grid((nrows + RPB - 1)/RPB, blocks_num.y, blocks_num.z);
    const dim3 block(32, RPB, 1);
#define RMS_WARP_CASE(NW) case NW: rms_norm_f32_warp<NW, do_multiply, do_add><<<grid, block, 0, stream>>>( \
        x, dst, nrows, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, \
        mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add, add_stride_row, add_stride_channel, \
        add_stride_sample, add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed); break;
    switch (ncols/32) {
        RMS_WARP_CASE(1) RMS_WARP_CASE(2) RMS_WARP_CASE(3) RMS_WARP_CASE(4)
        RMS_WARP_CASE(5) RMS_WARP_CASE(6) RMS_WARP_CASE(7) RMS_WARP_CASE(8)
        default: return false;
    }
#undef RMS_WARP_CASE
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template <int block_size>
static __global__ void rms_norm_back_f32(
        const float * grad, const float * xf, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row)*ncols;
    xf   += int64_t(row)*ncols;
    dst  += int64_t(row)*ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps  = sum_xx + ncols*eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x    = -scale_grad * sum_xg/sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad*grad[col] + scale_x*xf[col];
    }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

template <int block_size>
static __global__ void l2_norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    // Same double read as rms_norm_f32 above; cache the row in registers
    // when it fits so the scaling pass does not go back to memory.
    constexpr int max_cache = 4;
    const bool cached = ncols > block_size && ncols <= max_cache*block_size;
    float xv[max_cache];

    ggml_cuda_pdl_sync();
    if (cached) {
#pragma unroll
        for (int k = 0; k < max_cache; ++k) {
            const int col  = tid + k*block_size;
            const float xi = col < ncols ? x[col] : 0.0f;
            xv[k] = xi;
            tmp  += xi * xi;
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            const float xi = x[col];
            tmp += xi * xi;
        }
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    ggml_cuda_pdl_lc();

    // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
    const float scale = rsqrtf(fmaxf(tmp, eps * eps));

    if (cached) {
#pragma unroll
        for (int k = 0; k < max_cache; ++k) {
            const int col = tid + k*block_size;
            if (col < ncols) {
                dst[col] = scale * xv[k];
            }
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            dst[col] = scale * x[col];
        }
    }
}

// Merge a run of identically shaped l2_norm calls into a single launch.
// Qwen3.5's linear-attention layers call l2_norm back to back on two views of the same tensor
// (q_conv / k_conv).  Measured on a P100 each call takes 2.26 us while doing only 16 blocks x 32
// threads of work, so it is almost entirely launch overhead.  Extending grid.x to nrows*n merges
// them.  The in-block reduction order is unchanged, so the output is bit-identical.
struct l2_norm_fused_ptrs {
    const float * x  [CUDA_L2_NORM_FUSE_MAX];
    float *       dst[CUDA_L2_NORM_FUSE_MAX];
};

template <int block_size>
static __global__ void l2_norm_f32_fused(
        const l2_norm_fused_ptrs p, const int nrows, const int ncols, const int64_t stride_row,
        const int64_t stride_channel, const int64_t stride_sample, const float eps) {
    const int nchannels = gridDim.y;

    const int k         = blockIdx.x / nrows;
    const int row       = blockIdx.x - k*nrows;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    // Constant indices, for the same reason as in cpy.cu: a runtime index spills the parameter
    // struct to local memory.
    const float * xb = p.x[0];
    float *       db = p.dst[0];
#pragma unroll
    for (int j = 1; j < CUDA_L2_NORM_FUSE_MAX; ++j) {
        if (k == j) { xb = p.x[j]; db = p.dst[j]; }
    }

    const float * x = xb + sample*stride_sample + channel*stride_channel + row*stride_row;
    float *     dst = db + ((sample*nchannels + channel)*nrows + row)*(int64_t) ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    ggml_cuda_pdl_lc();

    const float scale = rsqrtf(fmaxf(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col];
    }
}

// Execute nodes[0..n) as a single L2_NORM launch and return true, if that is possible.
// Independence has already been checked on the graph side; this only decides whether one kernel
// can express all of them.
bool ggml_cuda_l2_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor ** nodes, int n) {
    if (n < 2 || n > CUDA_L2_NORM_FUSE_MAX) {
        return false;
    }

    const ggml_tensor * src0 = nodes[0]->src[0];

    if (src0->type != GGML_TYPE_F32 || nodes[0]->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(nodes[0])) {
        return false;
    }

    float eps;
    memcpy(&eps, nodes[0]->op_params, sizeof(float));
    if (!(eps >= 0.0f)) {
        return false;
    }

    const size_t ts0 = ggml_type_size(src0->type);
    if (src0->nb[0] != ts0) {
        return false;
    }

    // Shape, strides and eps must match exactly -- only the data pointers may differ
    for (int k = 1; k < n; ++k) {
        const ggml_tensor * s = nodes[k]->src[0];
        if (!s || s->type != GGML_TYPE_F32 || nodes[k]->type != GGML_TYPE_F32) {
            return false;
        }
        if (!ggml_is_contiguous(nodes[k])) {
            return false;
        }
        float eps_k;
        memcpy(&eps_k, nodes[k]->op_params, sizeof(float));
        if (memcmp(&eps_k, &eps, sizeof(float)) != 0) {
            return false;
        }
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (s->ne[i] != src0->ne[i] || s->nb[i] != src0->nb[i] ||
                nodes[k]->ne[i] != nodes[0]->ne[i] || nodes[k]->nb[i] != nodes[0]->nb[i]) {
                return false;
            }
        }
    }

    const int64_t ncols     = src0->ne[0];
    const int64_t nrows     = src0->ne[1];
    const int64_t nchannels = src0->ne[2];
    const int64_t nsamples  = src0->ne[3];

    if (nrows*n > INT_MAX) {
        return false;
    }

    l2_norm_fused_ptrs p = {};
    for (int k = 0; k < n; ++k) {
        p.x[k]   = (const float *) nodes[k]->src[0]->data;
        p.dst[k] = (float *)       nodes[k]->data;
    }

    const int64_t s01 = src0->nb[1] / ts0;
    const int64_t s02 = src0->nb[2] / ts0;
    const int64_t s03 = src0->nb[3] / ts0;

    const dim3 blocks_num((unsigned int) (nrows*n), (unsigned int) nchannels, (unsigned int) nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params =
            ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, ctx.stream()};
        ggml_cuda_kernel_launch(l2_norm_f32_fused<WARP_SIZE>, launch_params,
            p, (int) nrows, (int) ncols, s01, s02, s03, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params =
            ggml_cuda_kernel_launch_params{blocks_num, block_dims, 32*sizeof(float), ctx.stream()};
        ggml_cuda_kernel_launch(l2_norm_f32_fused<1024>, launch_params,
            p, (int) nrows, (int) ncols, s01, s02, s03, eps);
    }

    return true;
}

static void norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

static void group_norm_f32_cuda(
        const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, block_dims.x > WARP_SIZE ? 2 * 32 * sizeof(float): 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

static void rms_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        // Ported from Kmic-68/llama.cpp c5d1110a2 (MIT); GGML_CUDA_NORM_WARP=1
        if (rms_norm_f32_try_warp<false, false>(blocks_num, ncols, stream, x, dst, stride_row, stride_channel, stride_sample, eps,
                nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
                nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0))) {
            return;
        }
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<256, false>, launch_params,
            x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, nullptr, nullptr);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<1024, false>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, nullptr, nullptr);
    }
}

static void rms_norm_mul_f32_cuda(const float *  x,
                                  const float *  mul,
                                  const float *  add,
                                  float *        dst,
                                  const int      ncols,
                                  const int      nrows,
                                  const int      nchannels,
                                  const int      nsamples,
                                  const int64_t  stride_row,
                                  const int64_t  stride_channel,
                                  const int64_t  stride_sample,
                                  const int64_t  mul_stride_row,
                                  const int64_t  mul_stride_channel,
                                  const int64_t  mul_stride_sample,
                                  const uint32_t mul_ncols,
                                  const uint32_t mul_nrows,
                                  const uint32_t mul_nchannels,
                                  const uint32_t mul_nsamples,
                                  const int64_t  add_stride_row,
                                  const int64_t  add_stride_channel,
                                  const int64_t  add_stride_sample,
                                  const uint32_t add_ncols,
                                  const uint32_t add_nrows,
                                  const uint32_t add_nchannels,
                                  const uint32_t add_nsamples,
                                  const float    eps,
                                  cudaStream_t   stream,
                                  // non-null folds the residual add into the prologue
                                  const float *  pre_a   = nullptr,
                                  const float *  pre_b   = nullptr,
                                  float *        pre_dst = nullptr) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (mul != nullptr && add == nullptr && pre_a != nullptr) {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
        const dim3 block_dims(ncols < 1024 ? 256 : 1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params =
            ggml_cuda_kernel_launch_params{blocks_num, block_dims, 32 * sizeof(float), stream};
        if (ncols < 1024) {
            ggml_cuda_kernel_launch(rms_norm_f32<256, true, false, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row,
                mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed,
                mul_nchannels_packed, mul_nsamples_packed,
                nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
                pre_a, pre_b, pre_dst);
        } else {
            ggml_cuda_kernel_launch(rms_norm_f32<1024, true, false, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row,
                mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed,
                mul_nchannels_packed, mul_nsamples_packed,
                nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
                pre_a, pre_b, pre_dst);
        }
        return;
    }
    if (mul == nullptr) {
        rms_norm_f32_cuda(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    if (add == nullptr) {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
        if (ncols < 1024) {
            // Ported from Kmic-68/llama.cpp c5d1110a2 (MIT); GGML_CUDA_NORM_WARP=1
            if (rms_norm_f32_try_warp<true, false>(blocks_num, ncols, stream, x, dst, stride_row, stride_channel, stride_sample, eps,
                    mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed,
                    mul_nchannels_packed, mul_nsamples_packed,
                    nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0))) {
                return;
            }
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<256, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, nullptr, nullptr);
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<1024, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, nullptr, nullptr);
        }
    } else {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        if (ncols < 1024) {
            // Ported from Kmic-68/llama.cpp c5d1110a2 (MIT); GGML_CUDA_NORM_WARP=1
            if (rms_norm_f32_try_warp<true, true>(blocks_num, ncols, stream, x, dst, stride_row, stride_channel, stride_sample, eps,
                    mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed,
                    mul_nchannels_packed, mul_nsamples_packed, add, add_stride_row, add_stride_channel, add_stride_sample,
                    add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed)) {
                return;
            }
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims,block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<256, true, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
                nullptr, nullptr, nullptr);
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<1024, true, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
                nullptr, nullptr, nullptr);
        }
    }
}

static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    }
}

static void l2_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<WARP_SIZE>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<1024>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float * src0_d = (const float *) rms_norm_src->data;
    const float * mul_d = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_d = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = (float *) mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_cuda(src0_d, mul_d, nullptr, dst_d,
                          ne00, ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ 0, 0, 0,
                          0, 0, 0, 0,
                          eps, stream);
}

// Fuse ADD -> RMS_NORM -> MUL into one launch.  The ADD's output (the next layer's residual) is
// still written, so no node is eliminated; the only requirement is that rms_norm's input is
// exactly the preceding ADD.
void ggml_cuda_op_rms_norm_fused_pre_add(ggml_backend_cuda_context & ctx,
                                         ggml_tensor *               add_tensor,
                                         ggml_tensor *               dst,
                                         ggml_tensor *               mul_tensor) {
    const ggml_tensor * rms_norm_src = (const ggml_tensor *) dst->src[0];
    GGML_ASSERT(rms_norm_src == add_tensor);

    float eps = 0.0f;
    memcpy(&eps, dst->op_params, sizeof(float));

    const float * mul_d   = nullptr;
    const ggml_tensor * mul_src = nullptr;
    if (mul_tensor->src[0] == dst) {
        mul_d   = (const float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_d   = (const float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    rms_norm_mul_f32_cuda((const float *) rms_norm_src->data, mul_d, nullptr, (float *) mul_tensor->data,
                          ne00, ne01, ne02, ne03,
                          s01, s02, s03,
                          mul_s01, mul_s02, mul_s03,
                          (int) mul_src->ne[0], (int) mul_src->ne[1], (int) mul_src->ne[2], (int) mul_src->ne[3],
                          0, 0, 0,
                          0, 0, 0, 0,
                          eps, ctx.stream(),
                          (const float *) add_tensor->src[0]->data,
                          (const float *) add_tensor->src[1]->data,
                          (float *)       add_tensor->data);
}

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float               eps          = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float *       src0_d  = (const float *) rms_norm_src->data;
    const float *       mul_d   = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d   = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_d   = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const float *       add_d   = nullptr;
    const ggml_tensor * add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_d   = (float *) add_tensor->src[1]->data;
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_d   = (float *) add_tensor->src[0]->data;
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float *      dst_d  = (float *) add_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    rms_norm_mul_f32_cuda(src0_d, mul_d,add_d,dst_d,
                          ne00,ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ add_s01, add_s02, add_s03,
                          add_ncols, add_nrows, add_nchannels, add_nsamples,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    l2_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}
