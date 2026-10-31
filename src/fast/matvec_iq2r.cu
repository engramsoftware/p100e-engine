// IQ2_XXS / IQ2_XS / IQ2_S fast layouts (iq2xxsr, iq2xsr, iq2sr): repack, a16 HFMA2 GEMV (T = 1..4), packed -> f16.
//
// All three formats code a 32-value group as four 8-value chunks; chunk c is a grid index (256 / 512 / 1024 entries of
// eight bytes in {8, 25, 43}) plus an 8-bit sign pattern, and the group carries 4-bit scales (one for the group in
// IQ2_XXS, one per 16 values in IQ2_XS / IQ2_S).  value = d * (0.5 + scale) * 0.25 * grid * sign.
//
// Repacked rows (the same bytes as the GGUF row, rb = 66 / 74 / 82 * nb), group t = 8 * sb + g:
//   iq2xxsr [group plane: 8 B per group = the block's qs bytes as is (4 grid bytes, aux32 = 4 x 7-bit sign idx | scale << 28)]
//           [d plane: 2 B per sb]
//   iq2xsr  [group plane: 4 x u16 (grid idx 9 bits | sign idx << 9)][scales plane: 8 B per sb][d plane]
//   iq2sr   [group plane: 4 grid-low bytes, 4 sign bytes][qh plane 8 B per sb][scales plane 8 B per sb][d plane]
// i.e. d (which sits in front of the odd-sized block) moves to a plane so the 8-byte group loads are aligned, and for
// IQ2_S the grid-low and sign halves of qs are interleaved per group.  A lane owns one group of a row, KQ_ROWS rows per
// block; consecutive lanes read consecutive 8-byte chunks (256 contiguous bytes per warp load).
//
// Decode (per chunk): the grid (8 bytes) and the sign pattern (four half2 sign masks) come from shared-memory tables
// (filled once per block; blocks are persistent over row tiles); PRMT puts bytes (j, j + 2) as the low mantissa bytes of
// 1024 + b (magic 0x64), HSUB2 gives the exact b, XOR with the sign mask applies the sign: the half2 weight of the a16
// slot pair (v, v + 2).  Two fp16 partial sums per row and column (values 0-15, 16-31: the two scales), 8 products each
// (|a| <= 1, b <= 43: no overflow), then f32.   y = yd * sum_g [ dA * SA + dB * SB ].
// Numerics differ from the generic f32 GEMV only by the a16 activation rounding and the fp16 partial sums.
#include "matvec.cuh"
#include "matvec_internal.cuh"
#include "dequantize.cuh"
#include "convert.cuh"
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace p100e::fast {

enum { IQ2_XXS_T = 0, IQ2_XS_T = 1, IQ2_S_T = 2 };

static int iq2_type_of(int layout) { return layout - LAYOUT_IQ2XXSR; }
static __host__ __device__ __forceinline__ int iq2_block_bytes(int ty) { return ty == IQ2_XXS_T ? 66 : ty == IQ2_XS_T ? 74 : 82; }

// ------------------------------------------------------------------------------------------------------------- repack
// One CUDA block per row; the row is staged in shared memory (rb <= 48 KiB, checked by pack_weight), then rewritten.
template <int TY>
static __global__ void iq2r_convert(char * data, const int nb, const size_t row_bytes) {
    extern __shared__ uint8_t iq2row[];
    uint8_t * row = (uint8_t *) (data + blockIdx.x*row_bytes);
    const int n32 = (int) (row_bytes/4);
    for (int i = threadIdx.x; i < n32; i += blockDim.x) ((uint32_t *) iq2row)[i] = ((const uint32_t *) row)[i];
    __syncthreads();
    const int bb = iq2_block_bytes(TY);
    for (int o = threadIdx.x; o < (int) row_bytes; o += blockDim.x) {
        uint8_t v;
        if constexpr (TY == IQ2_XXS_T) {
            if (o < 64*nb) { const int sb = o >> 6; v = iq2row[bb*sb + 2 + (o & 63)]; }
            else           { const int k = o - 64*nb, sb = k >> 1; v = iq2row[bb*sb + (k & 1)]; }
        } else if constexpr (TY == IQ2_XS_T) {
            if (o < 64*nb)      { const int sb = o >> 6; v = iq2row[bb*sb + 2 + (o & 63)]; }
            else if (o < 72*nb) { const int k = o - 64*nb, sb = k >> 3; v = iq2row[bb*sb + 66 + (k & 7)]; }
            else                { const int k = o - 72*nb, sb = k >> 1; v = iq2row[bb*sb + (k & 1)]; }
        } else {
            if (o < 64*nb) {
                const int sb = o >> 6, g = (o >> 3) & 7, k = o & 7;   // k < 4: qs grid low byte 4g + k, else sign byte 32 + 4g + k - 4
                v = iq2row[bb*sb + 2 + (k < 4 ? 4*g + k : 32 + 4*g + (k - 4))];
            } else if (o < 72*nb) { const int k = o - 64*nb, sb = k >> 3; v = iq2row[bb*sb + 66 + (k & 7)]; }       // qh
            else if (o < 80*nb)   { const int k = o - 72*nb, sb = k >> 3; v = iq2row[bb*sb + 74 + (k & 7)]; }       // scales
            else                  { const int k = o - 80*nb, sb = k >> 1; v = iq2row[bb*sb + (k & 1)]; }
        }
        row[o] = v;
    }
}

bool iq2r_repack(int layout, void * dp, int64_t nrows, int64_t n_in, size_t row_bytes, cudaStream_t s) {
    const int nb = (int) (n_in/256);
    const unsigned g = (unsigned) nrows;
    char * d = (char *) dp;
    switch (iq2_type_of(layout)) {
        case IQ2_XXS_T: iq2r_convert<IQ2_XXS_T><<<g, 256, row_bytes, s>>>(d, nb, row_bytes); break;
        case IQ2_XS_T:  iq2r_convert<IQ2_XS_T ><<<g, 256, row_bytes, s>>>(d, nb, row_bytes); break;
        case IQ2_S_T:   iq2r_convert<IQ2_S_T  ><<<g, 256, row_bytes, s>>>(d, nb, row_bytes); break;
        default: return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

// ----------------------------------------------------------------------------------------------- shared decode pieces
template <int TY> struct iq2_traits;
template <> struct iq2_traits<IQ2_XXS_T> { static constexpr int ngrid = 256,  nsign = 128; };
template <> struct iq2_traits<IQ2_XS_T>  { static constexpr int ngrid = 512,  nsign = 128; };
template <> struct iq2_traits<IQ2_S_T>   { static constexpr int ngrid = 1024, nsign = 256; };

static __device__ __forceinline__ uint64_t iq2_grid_at(const int ty, const int i) {
    return ty == IQ2_XXS_T ? iq2xxs_grid[i] : ty == IQ2_XS_T ? iq2xs_grid[i] : iq2s_grid[i];
}

// sign pattern s (bit j = value j negative) -> four half2 sign masks for the slot pairs (0,2) (1,3) (4,6) (5,7)
static __device__ __forceinline__ uint4 iq2_sign_masks(const uint32_t s) {
    uint32_t m[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int u = 4*(p >> 1) + (p & 1);
        m[p] = (((s >> u) & 1u) << 15) | (((s >> (u + 2)) & 1u) << 31);
    }
    return make_uint4(m[0], m[1], m[2], m[3]);
}

// the per-group fields shared by GEMV and unpack: chunk c -> grid index, sign table index, and the scale of its half
template <int TY>
struct iq2_group {
    uint2    gw;
    uint32_t extra;   // XS: scale byte; S: qh | scale << 8; XXS: unused
    __device__ __forceinline__ int grid_idx(const int c) const {
        if constexpr (TY == IQ2_XXS_T) return (gw.x >> (8*c)) & 0xFF;
        else if constexpr (TY == IQ2_XS_T) return (((c < 2 ? gw.x : gw.y) >> (16*(c & 1))) & 0xFFFF) & 511;
        else return ((gw.x >> (8*c)) & 0xFF) | ((((extra & 0xFF) << (8 - 2*c))) & 0x300);
    }
    __device__ __forceinline__ int sign_idx(const int c) const {
        if constexpr (TY == IQ2_XXS_T) return (gw.y >> (7*c)) & 127;
        else if constexpr (TY == IQ2_XS_T) return (((c < 2 ? gw.x : gw.y) >> (16*(c & 1))) & 0xFFFF) >> 9;
        else return (gw.y >> (8*c)) & 0xFF;
    }
    // (0.5 + scale) * 0.25 factor operand: 4-bit scale of half h (0: values 0-15, 1: 16-31)
    __device__ __forceinline__ int scale4(const int h) const {
        if constexpr (TY == IQ2_XXS_T) return gw.y >> 28;
        else if constexpr (TY == IQ2_XS_T) return (extra >> (4*h)) & 0xF;
        else return ((extra >> 8) >> (4*h)) & 0xF;
    }
};

// ------------------------------------------------------------------------------------------------------------- GEMV
#define IQ2_ROWS 4

template <int TY, int ncols_dst, int nwarps>
static __device__ __forceinline__ void iq2r_body(
        const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
        float * __restrict__ dst, const int ngroups, const int row_bytes, const int ntiles,
        const int stride_col_aq, const int stride_col_ads, const int stride_col_dst) {
#if defined(FP16_AVAILABLE)
    constexpr int NG = iq2_traits<TY>::ngrid, NS = iq2_traits<TY>::nsign;
    __shared__ uint2 sgrid[NG];
    __shared__ uint4 ssign[NS];
    __shared__ float tmp[nwarps][ncols_dst][IQ2_ROWS];

    const int nb       = ngroups/8;
    const int tid      = threadIdx.x + threadIdx.y*WARP_SIZE;
    const int nthreads = WARP_SIZE*nwarps;
    for (int i = tid; i < NG; i += nthreads) {
        const uint64_t g = iq2_grid_at(TY, i);
        sgrid[i] = make_uint2((uint32_t) g, (uint32_t) (g >> 32));
    }
    for (int i = tid; i < NS; i += nthreads) {
        ssign[i] = iq2_sign_masks(TY == IQ2_S_T ? (uint32_t) i : (uint32_t) ksigns_iq2xs[i]);
    }
    __syncthreads();

    const half2 sub = __float2half2_rn(1024.0f);
    const bool wide = (row_bytes & 7) == 0;
    // plane offsets inside a row (bytes)
    const int off_x = TY == IQ2_XXS_T ? 0 : TY == IQ2_XS_T ? 64*nb : 72*nb;   // XS: scales plane; S: scales plane (qh at 64 nb)
    const int off_d = TY == IQ2_XXS_T ? 64*nb : TY == IQ2_XS_T ? 72*nb : 80*nb;

    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const int row0 = tile*IQ2_ROWS;
        const char * const xbase = vx + (size_t) row0*row_bytes;
        float sumf[ncols_dst][IQ2_ROWS] = {{0.0f}};

        for (int t = tid; t < ngroups; t += nthreads) {
            const int sb = t >> 3, g = t & 7;

            iq2_group<TY> gr[IQ2_ROWS];
            uint32_t dh[IQ2_ROWS];
#pragma unroll
            for (int i = 0; i < IQ2_ROWS; ++i) {
                const char * xr = xbase + (size_t) i*row_bytes;
                if (wide) {
                    gr[i].gw = *((const uint2 *) (xr + 8*t));
                } else {
                    const uint32_t * p = (const uint32_t *) (xr + 8*t);
                    gr[i].gw = make_uint2(p[0], p[1]);
                }
                gr[i].extra = 0;
                if constexpr (TY == IQ2_XS_T) {
                    gr[i].extra = *((const uint8_t *) (xr + off_x + 8*sb + g));
                } else if constexpr (TY == IQ2_S_T) {
                    gr[i].extra = *((const uint8_t *) (xr + 64*nb + 8*sb + g)) | ((uint32_t) *((const uint8_t *) (xr + off_x + 8*sb + g)) << 8);
                }
                dh[i] = *((const uint16_t *) (xr + off_d + 2*sb));
            }

            half2 a[ncols_dst][16];
            float yd[ncols_dst];
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const int4 * ap = (const int4 *) (aq + (size_t) j*stride_col_aq + t*16);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int4 av = ap[c];
                    a[j][4*c + 0] = *((const half2 *) &av.x);
                    a[j][4*c + 1] = *((const half2 *) &av.y);
                    a[j][4*c + 2] = *((const half2 *) &av.z);
                    a[j][4*c + 3] = *((const half2 *) &av.w);
                }
                yd[j] = __low2float(ads[j*stride_col_ads + t]);
            }

#pragma unroll
            for (int i = 0; i < IQ2_ROWS; ++i) {
                const float dd = __half2float(__ushort_as_half((unsigned short) dh[i]));
                const float dA = dd*(0.5f + (float) gr[i].scale4(0))*0.25f;
                const float dB = dd*(0.5f + (float) gr[i].scale4(1))*0.25f;
                half2 sA[ncols_dst], sB[ncols_dst];
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    sA[j] = make_half2(0.0f, 0.0f);
                    sB[j] = make_half2(0.0f, 0.0f);
                }
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint2 gv = sgrid[gr[i].grid_idx(c)];
                    const uint4 sm = ssign[gr[i].sign_idx(c)];
#pragma unroll
                    for (int kk = 0; kk < 2; ++kk) {
                        const uint32_t G = kk ? gv.y : gv.x;
#pragma unroll
                        for (int jj = 0; jj < 2; ++jj) {
                            const uint32_t x = __byte_perm(G, 0x64646464u, jj ? 0x4341 : 0x4240);
                            const uint32_t mk = jj ? (kk ? sm.w : sm.y) : (kk ? sm.z : sm.x);
                            half2 w = __hsub2(*((const half2 *) &x), sub);
                            const uint32_t wu = (*((const uint32_t *) &w)) ^ mk;
                            w = *((const half2 *) &wu);
                            const int slot = 4*(2*(c & 1) + kk) + 2*jj + (c >> 1);
#pragma unroll
                            for (int j = 0; j < ncols_dst; ++j) {
                                if (c < 2) sA[j] = __hfma2(w, a[j][slot], sA[j]);
                                else       sB[j] = __hfma2(w, a[j][slot], sB[j]);
                            }
                        }
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 fa = __half22float2(sA[j]);
                    const float2 fb = __half22float2(sB[j]);
                    sumf[j][i] += yd[j]*(dA*(fa.x + fa.y) + dB*(fb.x + fb.y));
                }
            }
        }

        if constexpr (nwarps == 1) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < IQ2_ROWS; ++i) {
                    const float v = warp_reduce_sum(sumf[j][i]);
                    if (threadIdx.x == i) {
                        dst[j*stride_col_dst + row0 + i] = v;
                    }
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < IQ2_ROWS; ++i) {
                    const float v = warp_reduce_sum(sumf[j][i]);
                    if (threadIdx.x == 0) {
                        tmp[threadIdx.y][j][i] = v;
                    }
                }
            }
            __syncthreads();
            if (threadIdx.y == 0) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    if (threadIdx.x < IQ2_ROWS) {
                        float v = 0.0f;
#pragma unroll
                        for (int w = 0; w < nwarps; ++w) {
                            v += tmp[w][j][threadIdx.x];
                        }
                        dst[j*stride_col_dst + row0 + threadIdx.x] = v;
                    }
                }
            }
            __syncthreads();   // tmp is reused by this block's next tile
        }
    }
#else
    GGML_UNUSED_VARS(vx, aq, ads, dst, ngroups, row_bytes, ntiles, stride_col_aq, stride_col_ads, stride_col_dst);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

template <int TY, int ncols_dst, int nwarps>
static __global__ void __launch_bounds__(WARP_SIZE*nwarps)
mul_mat_vec_iq2r(
        const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
        float * __restrict__ dst, const int ngroups, const int row_bytes, const int ntiles,
        const int stride_col_aq, const int stride_col_ads, const int stride_col_dst) {
    iq2r_body<TY, ncols_dst, nwarps>(vx, aq, ads, dst, ngroups, row_bytes, ntiles, stride_col_aq, stride_col_ads, stride_col_dst);
}

// grouped: blockIdx.y = group g of tab: the full GEMV of weight tab[g].w on a16 column tab[g].col into dst column
// tab[g].out; the same body as the ncols_dst = 1 kernel, so a column's bits equal the single-column launch
template <int TY, int NC, int nwarps>
static __global__ void __launch_bounds__(WARP_SIZE*nwarps)
mul_mat_vec_iq2r_grouped(
        const mv_group * __restrict__ tab, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
        float * __restrict__ dst, const int ngroups, const int row_bytes, const int ntiles,
        const int stride_col_aq, const int stride_col_ads, const int stride_col_dst) {
    const mv_group gr = tab[blockIdx.y];
    iq2r_body<TY, NC, nwarps>((const char *) gr.w, aq + (size_t) gr.col*stride_col_aq, ads + (size_t) gr.col*stride_col_ads,
                              dst + (size_t) gr.out*stride_col_dst, ngroups, row_bytes, ntiles, stride_col_aq, stride_col_ads,
                              stride_col_dst);
}

// warps per block: the largest of {5, 4, 2, 1} that keeps >= 90% of the lanes busy over the ngroups group loop
// (n_in 5120 -> 5, 10240 -> 5, 17408 -> 2, 8704 -> 1); independent of the column count, so a column's sum never
// depends on T
static int iq2_pick_nw(const int ngroups) {
    int best = 1; double beff = 0;
    for (int nw : {5, 4, 2, 1}) {
        const int per = 32*nw;
        const double eff = (double) ngroups/(((ngroups + per - 1)/per)*per);
        if (eff >= 0.9) return nw;
        if (eff > beff) { beff = eff; best = nw; }
    }
    return best;
}

template <int TY, int NC, int NW>
static void iq2r_launch_nw(const char * vx, const a16_args & a, cudaStream_t stream) {
    static int bps[16] = {0}, nsm[16] = {0};
    int dev = 0;
    cudaGetDevice(&dev);
    dev &= 15;
    if (!bps[dev]) {
        int n = 0, sms = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, mul_mat_vec_iq2r<TY, NC, NW>, WARP_SIZE*NW, 0);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        nsm[dev] = std::max(sms, 1);
        bps[dev] = std::max(n, 1);
    }
    const int ntiles = a.nrows/IQ2_ROWS;
    const int blocks = std::min(ntiles, nsm[dev]*bps[dev]);
    mul_mat_vec_iq2r<TY, NC, NW><<<blocks, dim3(WARP_SIZE, NW, 1), 0, stream>>>(
        vx, a.aq, a.ads, a.dst, a.ngroups, a.row_bytes, ntiles, a.ngroups*16, a.ngroups, a.nrows);
}

template <int TY, int NC>
static void iq2r_launch_n(const char * vx, const a16_args & a, cudaStream_t s) {
    switch (iq2_pick_nw(a.ngroups)) {
        case 5: iq2r_launch_nw<TY, NC, 5>(vx, a, s); break;
        case 4: iq2r_launch_nw<TY, NC, 4>(vx, a, s); break;
        case 2: iq2r_launch_nw<TY, NC, 2>(vx, a, s); break;
        default: iq2r_launch_nw<TY, NC, 1>(vx, a, s); break;
    }
}

template <int TY>
static void iq2r_launch(const char * vx, const a16_args & a, int ncols, cudaStream_t s) {
    switch (ncols) {
        case 1: iq2r_launch_n<TY, 1>(vx, a, s); break;
        case 2: iq2r_launch_n<TY, 2>(vx, a, s); break;
        case 3: iq2r_launch_n<TY, 3>(vx, a, s); break;
        case 4: iq2r_launch_n<TY, 4>(vx, a, s); break;
        default: break;
    }
}
template <int TY, int NC, int NW>
static void iq2r_grouped_nw(const mv_group * tab, int G, const a16_args & a, cudaStream_t stream) {
    static int bps[16] = {0}, nsm[16] = {0};
    int dev = 0;
    cudaGetDevice(&dev);
    dev &= 15;
    if (!bps[dev]) {
        int n = 0, sms = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, mul_mat_vec_iq2r_grouped<TY, NC, NW>, WARP_SIZE*NW, 0);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        nsm[dev] = std::max(sms, 1);
        bps[dev] = std::max(n, 1);
    }
    const int ntiles = a.nrows/IQ2_ROWS;
    const int bx = std::max(1, std::min(ntiles, std::max(1, nsm[dev]*bps[dev]/std::max(1, G))));
    mul_mat_vec_iq2r_grouped<TY, NC, NW><<<dim3(bx, G), dim3(WARP_SIZE, NW, 1), 0, stream>>>(
        tab, a.aq, a.ads, a.dst, a.ngroups, a.row_bytes, ntiles, a.ngroups*16, a.ngroups, a.nrows);
}
// nwarps from the group count only (iq2_pick_nw), never from NC: a column's bits are the same at every NC
template <int TY, int NC>
static void iq2r_grouped_n(const mv_group * tab, int G, const a16_args & a, cudaStream_t s) {
    switch (iq2_pick_nw(a.ngroups)) {
        case 5: iq2r_grouped_nw<TY, NC, 5>(tab, G, a, s); break;
        case 4: iq2r_grouped_nw<TY, NC, 4>(tab, G, a, s); break;
        case 2: iq2r_grouped_nw<TY, NC, 2>(tab, G, a, s); break;
        default: iq2r_grouped_nw<TY, NC, 1>(tab, G, a, s); break;
    }
}
template <int TY>
static void iq2r_grouped(const mv_group * tab, int G, const a16_args & a, int nc, cudaStream_t s) {
    switch (nc) {
        case 1: iq2r_grouped_n<TY, 1>(tab, G, a, s); break;
        case 2: iq2r_grouped_n<TY, 2>(tab, G, a, s); break;
        case 3: iq2r_grouped_n<TY, 3>(tab, G, a, s); break;
        default: iq2r_grouped_n<TY, 4>(tab, G, a, s); break;
    }
}
void launch_iq2sr_grouped(const mv_group * tab, int G, const a16_args & a, cudaStream_t s) { iq2r_grouped<IQ2_S_T>(tab, G, a, 1, s); }
void launch_iq2sr_grouped_nc(const mv_group * tab, int G, const a16_args & a, int nc, cudaStream_t s) { iq2r_grouped<IQ2_S_T>(tab, G, a, nc, s); }

void launch_iq2xxsr(const char * vx, const a16_args & a, int ncols, cudaStream_t s) { iq2r_launch<IQ2_XXS_T>(vx, a, ncols, s); }
void launch_iq2xsr (const char * vx, const a16_args & a, int ncols, cudaStream_t s) { iq2r_launch<IQ2_XS_T >(vx, a, ncols, s); }
void launch_iq2sr  (const char * vx, const a16_args & a, int ncols, cudaStream_t s) { iq2r_launch<IQ2_S_T  >(vx, a, ncols, s); }

// ------------------------------------------------------------------------------------------------------ packed -> f16
// One thread per 32-value group: ggml's dequantize_iq2_* arithmetic (d * (0.5 + scale) * 0.25 as f32, times the grid
// byte, times the sign, rounded to half), 32 halves out as four 16-byte stores (or scalar stores when dst is not 16-byte
// aligned), so the f16 bits equal ggml_get_to_fp16_cuda on the raw rows.
template <int TY, bool ALIGNED>
static __global__ void __launch_bounds__(256) iq2r_to_f16_k(const char * __restrict__ w, half * __restrict__ y,
                                                           const int nb, const int row_bytes, const int64_t ngroups) {
    const int64_t gt = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (gt >= ngroups) return;
    const int64_t row = gt/(8*nb);
    const int t = (int) (gt - row*8*nb);
    const int sb = t >> 3, g = t & 7;
    const char * xr = w + row*row_bytes;
    iq2_group<TY> gr;
    const uint32_t * p = (const uint32_t *) (xr + 8*t);
    gr.gw = make_uint2(p[0], p[1]);
    gr.extra = 0;
    if constexpr (TY == IQ2_XS_T) {
        gr.extra = *((const uint8_t *) (xr + 64*nb + 8*sb + g));
    } else if constexpr (TY == IQ2_S_T) {
        gr.extra = *((const uint8_t *) (xr + 64*nb + 8*sb + g)) | ((uint32_t) *((const uint8_t *) (xr + 72*nb + 8*sb + g)) << 8);
    }
    const int off_d = TY == IQ2_XXS_T ? 64*nb : TY == IQ2_XS_T ? 72*nb : 80*nb;
    const half hd = __ushort_as_half(*((const uint16_t *) (xr + off_d + 2*sb)));
    __align__(16) half out[32];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const uint64_t gv = iq2_grid_at(TY, gr.grid_idx(c));
        const uint8_t * grid = (const uint8_t *) &gv;
        const uint8_t signs = TY == IQ2_S_T ? (uint8_t) gr.sign_idx(c) : ksigns_iq2xs[gr.sign_idx(c)];
        const float d = (float) hd*(0.5f + (float) gr.scale4(c >> 1))*0.25f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            out[8*c + j] = ggml_cuda_cast<half>(d*grid[j]*(signs & kmask_iq2xs[j] ? -1.f : 1.f));
        }
    }
    half * yo = y + (size_t) gt*32;
    if constexpr (ALIGNED) {
#pragma unroll
        for (int c = 0; c < 4; ++c) ((uint4 *) yo)[c] = ((const uint4 *) out)[c];
    } else {
#pragma unroll
        for (int i = 0; i < 32; ++i) yo[i] = out[i];
    }
}

bool iq2r_to_f16(int layout, const char * base, int64_t nrows, int64_t n_in, int64_t rb, half * dst, cudaStream_t s) {
    if (((uintptr_t) base) % 4 || rb % 4 || n_in % 256) return false;
    const int nb = (int) (n_in/256);
    const int64_t ng = nrows*nb*8;
    const unsigned blocks = (unsigned) ((ng + 255)/256);
    const bool al = ((uintptr_t) dst) % 16 == 0;
#define IQ2U(TY) if (al) iq2r_to_f16_k<TY, true ><<<blocks, 256, 0, s>>>(base, dst, nb, (int) rb, ng); \
                 else    iq2r_to_f16_k<TY, false><<<blocks, 256, 0, s>>>(base, dst, nb, (int) rb, ng);
    switch (iq2_type_of(layout)) {
        case IQ2_XXS_T: IQ2U(IQ2_XXS_T) break;
        case IQ2_XS_T:  IQ2U(IQ2_XS_T)  break;
        case IQ2_S_T:   IQ2U(IQ2_S_T)   break;
        default: return false;
    }
#undef IQ2U
    return true;
}

} // namespace p100e::fast
