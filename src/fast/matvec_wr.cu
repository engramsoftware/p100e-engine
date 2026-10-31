// p100e fast: decode GEMV kernels tuned for the P100 tensor-parallel shard shapes (T = 1).
//
// "wr" (warp rows): every warp owns RW whole output rows and walks all of K with its 32 lanes (lane l takes the
// 32-value groups l, l + 32, ...).  The block first stages the a16 activation of the whole row (aq + ads) in shared
// memory, so the activation is read from L2 once per NW*RW rows instead of once per 4 rows (the K-split kernels
// re-read it per block, which roughly doubled the L2->SM traffic of the weights), with conflict-free LDS.128.
// Weight loads are 16-byte and coalesced (lane l reads group l: 512 contiguous bytes per warp instruction).
// The reduction is a warp shuffle per row (no shared memory, no second sync).
//
// "ks" (K split): for tiny n_out (the 24-row ssm_alpha/ssm_beta shards) every block owns RB rows and all of its
// threads split K, so 24 rows still spread over 24 blocks of 160+ threads (one round trip each).
//
// The per-(row, group) dot products are the decoders below: the same dequantized half2 values, the same fp16
// partial sums in the same order and the same fp32 expression as the K-split kernels in matvec_q4kr.cu /
// matvec_q3r.cu / matvec_kq.cu, so every group contribution is bit-identical to them; only the fp32 order in
// which the group contributions are summed differs (lane-strided then warp tree, instead of thread-strided,
// warp tree and cross-warp sum).
#include "matvec.cuh"
#include "matvec_internal.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "matvec_dec.cuh"

namespace p100e::fast {

// ------------------------------------------------------------------------------------------------- wr kernel
// Shared memory: aq as 4 chunk planes of ngp uint4 (chunk c of group g at [c*ngp + g]; ngp = ngroups rounded up to
// 2 mod 8 so that both the staging stores and the lane reads are bank-conflict free), then ads as float2.
static inline int wr_ngp(const int ngroups) { return ngroups + ((10 - ngroups % 8) % 8); }
static inline size_t wr_smem(const int ngroups) { return (size_t) 4*wr_ngp(ngroups)*16 + (size_t) ngroups*8; }

template <class D, int NW, int RW, int MINB, int U>
static __global__ void __launch_bounds__(WARP_SIZE*NW, MINB)
mv_wr(const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
      float * __restrict__ dst, const int ngroups, const int nrows, const int row_bytes, const int ngp) {
    extern __shared__ uint4 wr_sm[];
    const int lane = threadIdx.x;
    const int tid  = threadIdx.x + WARP_SIZE*threadIdx.y;
    {
        const uint4 * ag = (const uint4 *) aq;
        for (int i = tid; i < 4*ngroups; i += WARP_SIZE*NW) {
            wr_sm[(i & 3)*ngp + (i >> 2)] = ag[i];
        }
        float2 * sds = (float2 *) (wr_sm + 4*ngp);
        for (int i = tid; i < ngroups; i += WARP_SIZE*NW) {
            sds[i] = __half22float2(ads[i]);
        }
    }
    __syncthreads();
    const float2 * sds = (const float2 *) (wr_sm + 4*ngp);

    const int row0 = (blockIdx.x*NW + threadIdx.y)*RW;
    if (row0 >= nrows) {
        return;
    }
    const D dec(lane);
    const int nb = ngroups/8;
    const char * xr[RW];
#pragma unroll
    for (int i = 0; i < RW; ++i) {
        xr[i] = vx + (size_t) min(row0 + i, nrows - 1)*row_bytes;
    }
    float acc[RW];
#pragma unroll
    for (int i = 0; i < RW; ++i) acc[i] = 0.0f;

    // U groups per lane per iteration: all U*RW raw loads are issued before the first dot
    for (int t0 = lane; t0 < ngroups; t0 += WARP_SIZE*U) {
        typename D::raw r[U][RW];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int t = t0 + WARP_SIZE*u;
            if (U == 1 || t < ngroups) {
#pragma unroll
                for (int i = 0; i < RW; ++i) dec.load(r[u][i], xr[i], t, nb);
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int t = t0 + WARP_SIZE*u;
            if (U == 1 || t < ngroups) {
                half2 a[16];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint4 v = wr_sm[c*ngp + t];
                    a[4*c + 0] = as_h2(v.x); a[4*c + 1] = as_h2(v.y); a[4*c + 2] = as_h2(v.z); a[4*c + 3] = as_h2(v.w);
                }
                const float2 ds = sds[t];
#pragma unroll
                for (int i = 0; i < RW; ++i) acc[i] += dec.dot(r[u][i], a, ds.x, ds.y);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < RW; ++i) {
        const float v = warp_reduce_sum(acc[i]);
        if (lane == i && row0 + i < nrows) dst[row0 + i] = v;
    }
}

// ------------------------------------------------------------------------------------------------- ks kernel
// RB rows per block, all NW*32 threads split K (thread tid takes groups tid, tid + NW*32, ...), activation from
// global (only n_out/RB blocks read it).
// NC columns: activation column j at aq + j*ngroups*16 / ads + j*ngroups, output column j at dst + j*nrows.
template <class D, int NW, int RB, int NC>
static __global__ void __launch_bounds__(WARP_SIZE*NW)
mv_ks(const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
      float * __restrict__ dst, const int ngroups, const int nrows, const int row_bytes) {
    const int lane = threadIdx.x;
    const int tid  = threadIdx.x + WARP_SIZE*threadIdx.y;
    const int row0 = blockIdx.x*RB;
    const D dec(tid);
    const int nb = ngroups/8;
    float acc[NC][RB];
#pragma unroll
    for (int j = 0; j < NC; ++j)
#pragma unroll
        for (int i = 0; i < RB; ++i) acc[j][i] = 0.0f;
    for (int t = tid; t < ngroups; t += WARP_SIZE*NW) {
        typename D::raw r[RB];
#pragma unroll
        for (int i = 0; i < RB; ++i) dec.load(r[i], vx + (size_t) min(row0 + i, nrows - 1)*row_bytes, t, nb);
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            half2 a[16];
            const uint4 * ap = (const uint4 *) (aq + (size_t) j*ngroups*16 + (size_t) t*16);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint4 v = ap[c];
                a[4*c + 0] = as_h2(v.x); a[4*c + 1] = as_h2(v.y); a[4*c + 2] = as_h2(v.z); a[4*c + 3] = as_h2(v.w);
            }
            const float2 ds = __half22float2(ads[j*ngroups + t]);
#pragma unroll
            for (int i = 0; i < RB; ++i) acc[j][i] += dec.dot(r[i], a, ds.x, ds.y);
        }
    }
    __shared__ float red[NW][NC][RB];
#pragma unroll
    for (int j = 0; j < NC; ++j)
#pragma unroll
        for (int i = 0; i < RB; ++i) {
            const float v = warp_reduce_sum(acc[j][i]);
            if (lane == 0) red[threadIdx.y][j][i] = v;
        }
    __syncthreads();
    if (tid < NC*RB) {
        const int j = tid / RB, i = tid % RB;
        if (row0 + i < nrows) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < NW; ++w) v += red[w][j][i];
            dst[(size_t) j*nrows + row0 + i] = v;
        }
    }
}

// "wrp": persistent warp-rows with the activation quantize fused in.  Every block quantizes the whole f32 x into
// shared memory once (one thread per group, same planes as mv_wr), then each warp walks row tiles of RW rows,
// tile = global warp id + k * (total warps), with no further synchronisation.  The first tile's weights are
// prefetched to L2 before the quantize so the DRAM pipe is busy from the start.
// PQ ("wrq", variants 501..): the activation is already quantized (aq/ads, the a16 layout written by
// fast::matvec_quantize or a fused producer); the block stages it into the same planes instead of quantizing x.
// The planes hold the same bits either way, so wrq rows equal wrp / wr rows bit for bit.
template <class D, int NW, int RW, int MINB, bool PF, bool PQ>
static __global__ void __launch_bounds__(WARP_SIZE*NW, MINB)
mv_wrp(const char * __restrict__ vx, const float * __restrict__ x, const half2 * __restrict__ aq,
       const half2 * __restrict__ ads, float * __restrict__ dst,
       const int ngroups, const int nrows, const int row_bytes, const int ngp) {
    extern __shared__ uint4 wr_sm[];
    const int lane = threadIdx.x;
    const int tid  = threadIdx.x + WARP_SIZE*threadIdx.y;
    const int ntiles = (nrows + RW - 1)/RW;
    const int wstep  = gridDim.x*NW;
    int tile = blockIdx.x*NW + threadIdx.y;
    if constexpr (PF) {
        if (tile < ntiles) {
            const int nb = ngroups/8;
#pragma unroll
            for (int i = 0; i < RW; ++i) {
                const char * xr = vx + (size_t) min(tile*RW + i, nrows - 1)*row_bytes;
                for (int t = lane; t < ngroups; t += WARP_SIZE) D::prefetch(xr, t, nb);
            }
        }
    }
    float2 * sds = (float2 *) (wr_sm + 4*ngp);
    if constexpr (PQ) {
        (void) x;
        const uint4 * ag = (const uint4 *) aq;
        for (int i = tid; i < 4*ngroups; i += WARP_SIZE*NW) wr_sm[(i & 3)*ngp + (i >> 2)] = ag[i];
        for (int i = tid; i < ngroups; i += WARP_SIZE*NW) sds[i] = __half22float2(ads[i]);
    } else {
        (void) aq; (void) ads;
        for (int g = tid; g < ngroups; g += WARP_SIZE*NW) {
            uint4 q[4];
            float2 ds;
            a16_group(x + 32*g, q, ds);
#pragma unroll
            for (int c = 0; c < 4; ++c) wr_sm[c*ngp + g] = q[c];
            sds[g] = ds;
        }
    }
    __syncthreads();

    const D dec(lane);
    const int nb = ngroups/8;
    for (; tile < ntiles; tile += wstep) {
        const int row0 = tile*RW;
        const char * xr[RW];
#pragma unroll
        for (int i = 0; i < RW; ++i) xr[i] = vx + (size_t) min(row0 + i, nrows - 1)*row_bytes;
        float acc[RW];
#pragma unroll
        for (int i = 0; i < RW; ++i) acc[i] = 0.0f;
        for (int t = lane; t < ngroups; t += WARP_SIZE) {
            typename D::raw r[RW];
#pragma unroll
            for (int i = 0; i < RW; ++i) dec.load(r[i], xr[i], t, nb);
            half2 a[16];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint4 v = wr_sm[c*ngp + t];
                a[4*c + 0] = as_h2(v.x); a[4*c + 1] = as_h2(v.y); a[4*c + 2] = as_h2(v.z); a[4*c + 3] = as_h2(v.w);
            }
            const float2 ds = sds[t];
#pragma unroll
            for (int i = 0; i < RW; ++i) acc[i] += dec.dot(r[i], a, ds.x, ds.y);
        }
#pragma unroll
        for (int i = 0; i < RW; ++i) {
            const float v = warp_reduce_sum(acc[i]);
            if (lane == i && row0 + i < nrows) dst[row0 + i] = v;
        }
    }
}

// "ksf": K split (RB rows per block, all threads split K) with the quantize fused: each thread quantizes the
// groups it visits itself (no shared memory).  For tiny n_out.
template <class D, int NW, int RB>
static __global__ void __launch_bounds__(WARP_SIZE*NW)
mv_ksf(const char * __restrict__ vx, const float * __restrict__ x, float * __restrict__ dst,
       const int ngroups, const int nrows, const int row_bytes) {
    const int lane = threadIdx.x;
    const int tid  = threadIdx.x + WARP_SIZE*threadIdx.y;
    const int row0 = blockIdx.x*RB;
    const D dec(tid);
    const int nb = ngroups/8;
    float acc[RB];
#pragma unroll
    for (int i = 0; i < RB; ++i) acc[i] = 0.0f;
    for (int t = tid; t < ngroups; t += WARP_SIZE*NW) {
        typename D::raw r[RB];
#pragma unroll
        for (int i = 0; i < RB; ++i) dec.load(r[i], vx + (size_t) min(row0 + i, nrows - 1)*row_bytes, t, nb);
        uint4 q[4];
        float2 ds;
        a16_group(x + 32*t, q, ds);
        half2 a[16];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            a[4*c + 0] = as_h2(q[c].x); a[4*c + 1] = as_h2(q[c].y); a[4*c + 2] = as_h2(q[c].z); a[4*c + 3] = as_h2(q[c].w);
        }
#pragma unroll
        for (int i = 0; i < RB; ++i) acc[i] += dec.dot(r[i], a, ds.x, ds.y);
    }
    __shared__ float red[NW][RB];
#pragma unroll
    for (int i = 0; i < RB; ++i) {
        const float v = warp_reduce_sum(acc[i]);
        if (lane == 0) red[threadIdx.y][i] = v;
    }
    __syncthreads();
    if (tid < RB && row0 + tid < nrows) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) v += red[w][tid];
        dst[row0 + tid] = v;
    }
}

// ------------------------------------------------------------------------------------------------- launchers
template <class D, int NW, int RW, int MINB, int U>
static bool wr_launch(const char * vx, const a16_args & a, cudaStream_t s) {
    const size_t smem = wr_smem(a.ngroups);
    if (smem > 48*1024) return false;
    const int rows_per_block = NW*RW;
    const dim3 grid((a.nrows + rows_per_block - 1)/rows_per_block);
    mv_wr<D, NW, RW, MINB, U><<<grid, dim3(WARP_SIZE, NW), smem, s>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.nrows, a.row_bytes,
                                                                   wr_ngp(a.ngroups));
    return true;
}

template <class D, int NW, int RB, int NC = 1>
static bool ks_launch(const char * vx, const a16_args & a, cudaStream_t s) {
    const dim3 grid((a.nrows + RB - 1)/RB);
    mv_ks<D, NW, RB, NC><<<grid, dim3(WARP_SIZE, NW), 0, s>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.nrows, a.row_bytes);
    return true;
}

// variant table: id -> (kind, NW, RW, MINB, load policy).  Variant 0 = default selection by shape.
#define WR_VARIANTS(X) \
    X( 1, wr, 4, 1, 1, 0, 1) X( 2, wr, 4, 2, 1, 0, 1) X( 3, wr, 4, 4, 1, 0, 1) \
    X( 4, wr, 8, 1, 1, 0, 1) X( 5, wr, 8, 2, 1, 0, 1) X( 6, wr, 8, 4, 1, 0, 1) \
    X( 7, wr, 2, 2, 1, 0, 1) X( 8, wr, 2, 4, 1, 0, 1) X( 9, wr,16, 1, 1, 0, 1) X(10, wr,16, 2, 1, 0, 1) \
    X(11, wr, 8, 2, 1, 1, 1) X(12, wr, 8, 2, 1, 2, 1) X(13, wr, 4, 4, 1, 1, 1) X(14, wr, 4, 2, 1, 1, 1) \
    X(15, wr, 8, 2, 4, 0, 1) X(16, wr, 4, 4, 4, 0, 1) \
    X(17, wr, 8, 1, 1, 0, 2) X(18, wr, 8, 2, 1, 0, 2) X(19, wr, 4, 2, 1, 0, 2) X(20, wr, 4, 4, 1, 0, 2) \
    X(21, wr, 4, 1, 1, 0, 2) X(22, wr, 8, 1, 1, 2, 2) \
    X(23, wr, 4, 8, 1, 0, 1) X(24, wr, 2, 8, 1, 0, 1) X(25, wr, 8, 4, 2, 0, 1) X(26, wr, 4, 4, 5, 0, 1) \
    X(27, wr, 4, 4, 6, 0, 1) X(28, wr, 8, 2, 5, 0, 1)

int wr_variant_count() { return 28; }

const char * wr_variant_name(int v) {
    static char buf[64];
    switch (v) {
#define X(ID, K, NW, RW, MINB, LP, U) case ID: snprintf(buf, sizeof buf, #K " nw%d rw%d mb%d lp%d u%d", NW, RW, MINB, LP, U); return buf;
        WR_VARIANTS(X)
#undef X
        default: return "default";
    }
}

template <template <int> class DT>
static bool wr_dispatch_t(const char * vx, const a16_args & a, int variant, cudaStream_t s) {
    switch (variant) {
#define X(ID, K, NW, RW, MINB, LP, U) case ID: return wr_launch<DT<LP>, NW, RW, MINB, U>(vx, a, s);
        WR_VARIANTS(X)
#undef X
        default: return false;
    }
}

template <int LP> using dec_q6r2_16 = dec_q6r2<LP, 16>;
template <int LP> using dec_q6r2_8  = dec_q6r2<LP, 8>;
template <int LP> using dec_q6r3_16 = dec_q6r3<LP, 16>;
template <int LP> using dec_q6r3_8  = dec_q6r3<LP, 8>;

// ks variants for tiny n_out: 101 = nw5 rb1, 102 = nw4 rb1, 103 = nw8 rb1, 104 = nw4 rb2, 105 = nw2 rb4
template <template <int> class DT>
static bool ks_dispatch_t(const char * vx, const a16_args & a, int variant, cudaStream_t s) {
    switch (variant) {
        case 101: return ks_launch<DT<0>, 5, 1>(vx, a, s);
        case 102: return ks_launch<DT<0>, 4, 1>(vx, a, s);
        case 103: return ks_launch<DT<0>, 8, 1>(vx, a, s);
        case 104: return ks_launch<DT<0>, 4, 2>(vx, a, s);
        case 105: return ks_launch<DT<0>, 2, 4>(vx, a, s);
        default: return false;
    }
}

const char * ks_variant_name(int v) {
    switch (v) {
        case 101: return "ks nw5 rb1"; case 102: return "ks nw4 rb1"; case 103: return "ks nw8 rb1";
        case 104: return "ks nw4 rb2"; case 105: return "ks nw2 rb4"; default: return "?";
    }
}

// layout: LAYOUT_Q4KR / LAYOUT_Q3R / LAYOUT_Q6R2 (VA from the row stride).  Returns false if not handled.
bool launch_wr(int layout, const char * vx, const a16_args & a, int variant, cudaStream_t s) {
    const bool ks = variant >= 101;
    switch (layout) {
        case LAYOUT_Q4KR: return ks ? ks_dispatch_t<dec_q4kr>(vx, a, variant, s) : wr_dispatch_t<dec_q4kr>(vx, a, variant, s);
        case LAYOUT_Q3R:  return ks ? ks_dispatch_t<dec_q3r >(vx, a, variant, s) : wr_dispatch_t<dec_q3r >(vx, a, variant, s);
        case LAYOUT_Q6R2:
            if (a.row_bytes % 16 == 0) return ks ? ks_dispatch_t<dec_q6r2_16>(vx, a, variant, s) : wr_dispatch_t<dec_q6r2_16>(vx, a, variant, s);
            if (a.row_bytes % 8 == 0)  return ks ? ks_dispatch_t<dec_q6r2_8 >(vx, a, variant, s) : wr_dispatch_t<dec_q6r2_8 >(vx, a, variant, s);
            return false;
        case LAYOUT_Q6R3:
            if (a.row_bytes % 16 == 0) return ks ? ks_dispatch_t<dec_q6r3_16>(vx, a, variant, s) : wr_dispatch_t<dec_q6r3_16>(vx, a, variant, s);
            if (a.row_bytes % 8 == 0)  return ks ? ks_dispatch_t<dec_q6r3_8 >(vx, a, variant, s) : wr_dispatch_t<dec_q6r3_8 >(vx, a, variant, s);
            return false;
        case LAYOUT_Q5KR: return ks ? ks_dispatch_t<dec_q5kr>(vx, a, variant, s) : wr_dispatch_t<dec_q5kr>(vx, a, variant, s);
        default: return false;
    }
}

// T = 1..4 on Q6R3 / Q5KR with the legacy geometry (2 warps, 4 rows per block, K split)
template <class D>
static bool ks_cols_t(const char * vx, const a16_args & a, int ncols, cudaStream_t s) {
    switch (ncols) {
        case 1: return ks_launch<D, 2, 4, 1>(vx, a, s);
        case 2: return ks_launch<D, 2, 4, 2>(vx, a, s);
        case 3: return ks_launch<D, 2, 4, 3>(vx, a, s);
        case 4: return ks_launch<D, 2, 4, 4>(vx, a, s);
        default: return false;
    }
}

// ---- persistent warp rows, T = 1: variants 301.. (wrp, quantize fused in), 501.. (wrq, prequantized input);
// 401.. (ksf, K split with the quantize fused in)
template <class D, int NW, int RW, int MINB, bool PF, bool PQ>
static bool wrp_launch(const char * vx, const float * x, const half2 * aq, const half2 * ads, float * y, int ngroups,
                       int nrows, int row_bytes, cudaStream_t s) {
    const size_t smem = wr_smem(ngroups);
    if (smem > 48*1024) return false;
    // persistent grid: as many blocks as fit at once, but no more than there are tiles.  Cached per (device,
    // smem size) for this instantiation; the occupancy query only runs the first time a pair is seen.
    struct entry { int dev; size_t smem; int nblk; };
    static entry cache[16];
    static int ncache = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    int nblk = 0;
    for (int i = 0; i < ncache; ++i) if (cache[i].dev == dev && cache[i].smem == smem) { nblk = cache[i].nblk; break; }
    if (nblk == 0) {
        int per_sm = 0, nsm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, mv_wrp<D, NW, RW, MINB, PF, PQ>, WARP_SIZE*NW, smem);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        nblk = std::max(1, per_sm)*nsm;
        if (ncache < 16) cache[ncache++] = entry{dev, smem, nblk};
    }
    const int ntiles = (nrows + RW - 1)/RW;
    const int grid = std::max(1, std::min(nblk, (ntiles + NW - 1)/NW));
    mv_wrp<D, NW, RW, MINB, PF, PQ><<<grid, dim3(WARP_SIZE, NW), smem, s>>>(vx, x, aq, ads, y, ngroups, nrows, row_bytes,
                                                                          wr_ngp(ngroups));
    return true;
}

template <class D, int NW, int RB>
static bool ksf_launch(const char * vx, const float * x, float * y, int ngroups, int nrows, int row_bytes, cudaStream_t s) {
    mv_ksf<D, NW, RB><<<(nrows + RB - 1)/RB, dim3(WARP_SIZE, NW), 0, s>>>(vx, x, y, ngroups, nrows, row_bytes);
    return true;
}

// id 3xx = wrp (f32 x, quantize fused in); id 3xx + 200 = wrq, the same geometry on a prequantized activation
#define WRP_VARIANTS(X) \
    X(301, 4, 4, 1, false) X(302, 4, 4, 1, true) X(303, 8, 2, 1, false) X(304, 8, 2, 1, true) \
    X(305, 4, 2, 1, false) X(306, 8, 4, 1, false) X(307, 4, 8, 1, false) X(308, 4, 4, 4, false) \
    X(309, 8, 2, 4, false) X(310, 4, 4, 5, false) X(311, 4, 4, 6, false) X(312, 2, 4, 1, false) \
    X(313, 2, 8, 1, false) X(314, 16, 2, 1, false) X(315, 8, 1, 1, false) X(316, 4, 4, 4, true) \
    X(317, 16, 2, 2, false) X(318, 24, 2, 1, false) X(319, 16, 4, 1, false) X(320, 12, 4, 1, false) \
    X(321, 8, 8, 1, false) X(322, 32, 1, 1, false) X(323, 12, 2, 1, false) X(324, 32, 2, 1, false)

const char * fused_variant_name(int v) {
    static char buf[64];
    switch (v) {
#define X(ID, NW, RW, MINB, PF) \
        case ID:       snprintf(buf, sizeof buf, "wrp nw%d rw%d mb%d pf%d", NW, RW, MINB, (int) PF); return buf; \
        case ID + 200: snprintf(buf, sizeof buf, "wrq nw%d rw%d mb%d pf%d", NW, RW, MINB, (int) PF); return buf;
        WRP_VARIANTS(X)
#undef X
        case 401: return "ksf nw5 rb1"; case 402: return "ksf nw4 rb1"; case 403: return "ksf nw4 rb2"; case 404: return "ksf nw2 rb4";
        default: return "?";
    }
}

template <class D>
static bool fused_t(const char * vx, const float * x, const half2 * aq, const half2 * ads, float * y, int ngroups, int nrows,
                    int row_bytes, int variant, cudaStream_t s) {
    switch (variant) {
#define X(ID, NW, RW, MINB, PF) \
        case ID:       return x  ? wrp_launch<D, NW, RW, MINB, PF, false>(vx, x, nullptr, nullptr, y, ngroups, nrows, row_bytes, s) : false; \
        case ID + 200: return aq ? wrp_launch<D, NW, RW, MINB, PF, true >(vx, nullptr, aq, ads, y, ngroups, nrows, row_bytes, s) : false;
        WRP_VARIANTS(X)
#undef X
        case 401: return x ? ksf_launch<D, 5, 1>(vx, x, y, ngroups, nrows, row_bytes, s) : false;
        case 402: return x ? ksf_launch<D, 4, 1>(vx, x, y, ngroups, nrows, row_bytes, s) : false;
        case 403: return x ? ksf_launch<D, 4, 2>(vx, x, y, ngroups, nrows, row_bytes, s) : false;
        case 404: return x ? ksf_launch<D, 2, 4>(vx, x, y, ngroups, nrows, row_bytes, s) : false;
        default: return false;
    }
}

static bool fused_layout(int layout, const char * vx, const float * x, const half2 * aq, const half2 * ads, float * y,
                         int ngroups, int nrows, int row_bytes, int variant, cudaStream_t s) {
    // P100E_MVWRP_Q3/_Q4/_Q5/_Q6: the inline-PTX twin of mv_wrp (matvec_wrx.cu) for the default geometries, same bits
    if (variant < 400 || variant >= 500) {
        if (launch_wrpx(layout, vx, x, aq, ads, y, ngroups, nrows, row_bytes, variant, s)) return true;
    }
    switch (layout) {
        case LAYOUT_Q4KR: return fused_t<dec_q4kr<0>>(vx, x, aq, ads, y, ngroups, nrows, row_bytes, variant, s);
        case LAYOUT_Q3R:  return fused_t<dec_q3r<0> >(vx, x, aq, ads, y, ngroups, nrows, row_bytes, variant, s);
        case LAYOUT_Q5KR: return fused_t<dec_q5kr<0>>(vx, x, aq, ads, y, ngroups, nrows, row_bytes, variant, s);
        case LAYOUT_Q6R2:
            if (row_bytes % 16 == 0) return fused_t<dec_q6r2<0, 16>>(vx, x, aq, ads, y, ngroups, nrows, row_bytes, variant, s);
            if (row_bytes % 8 == 0)  return fused_t<dec_q6r2<0, 8> >(vx, x, aq, ads, y, ngroups, nrows, row_bytes, variant, s);
            return false;
        case LAYOUT_Q6R3:
            if (row_bytes % 16 == 0) return fused_t<dec_q6r3<0, 16>>(vx, x, aq, ads, y, ngroups, nrows, row_bytes, variant, s);
            if (row_bytes % 8 == 0)  return fused_t<dec_q6r3<0, 8> >(vx, x, aq, ads, y, ngroups, nrows, row_bytes, variant, s);
            return false;
        default: return false;
    }
}

bool launch_fused_t1(int layout, const char * vx, const float * x, float * y, int ngroups, int nrows, int row_bytes,
                     int variant, cudaStream_t s) {
    if (((uintptr_t) x) % 16 != 0 || variant < 300 || variant >= 500) return false;
    return fused_layout(layout, vx, x, nullptr, nullptr, y, ngroups, nrows, row_bytes, variant, s);
}

bool launch_wrq(int layout, const char * vx, const a16_args & a, int variant, cudaStream_t s) {
    if (variant < 501 || variant >= 600) return false;
    return fused_layout(layout, vx, nullptr, a.aq, a.ads, a.dst, a.ngroups, a.nrows, a.row_bytes, variant, s);
}

// T = 1..4 on Q6R3 / Q5KR with the legacy geometry (2 warps, 4 rows per block, K split): the P100E_MV_VARIANT=-1 path
// of the layouts that have no legacy kernel.  Q6R3 needs an 8-byte aligned row stride (nb % 4 == 0, see pack_weight).
bool launch_ks_cols(int layout, const char * vx, const a16_args & a, int ncols, cudaStream_t s) {
    switch (layout) {
        case LAYOUT_Q6R3:
            if (a.row_bytes % 16 == 0) return ks_cols_t<dec_q6r3<0, 16>>(vx, a, ncols, s);
            if (a.row_bytes % 8 == 0)  return ks_cols_t<dec_q6r3<0, 8>>(vx, a, ncols, s);
            return false;
        case LAYOUT_Q5KR: return ks_cols_t<dec_q5kr<0>>(vx, a, ncols, s);
        default: return false;
    }
}

// ------------------------------------------------------------------------------------------------- wrc (T = 1..4)
// Warp rows for NC activation columns in one pass over the weights: every warp owns RW rows, lane l takes the groups
// l, l + 32, ... in order and accumulates each column separately, then the same xor warp tree per (column, row).
// That is exactly the per-row order of mv_wr / wrp / wrq, so column j of a T-column call equals the T = 1 result for
// that column, bit for bit, whatever T (the per-op verify / prefill chunks stay T-invariant).  The activations are
// read through the read-only cache (no staging, no shared memory limit on n_in).
template <class D, int NW, int RW, int NC>
static __global__ void __launch_bounds__(WARP_SIZE*NW)
mv_wrc(const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
       float * __restrict__ dst, const int ngroups, const int nrows, const int row_bytes) {
    const int lane = threadIdx.x;
    const int row0 = (blockIdx.x*NW + threadIdx.y)*RW;
    if (row0 >= nrows) {
        return;
    }
    const D dec(lane);
    const int nb = ngroups/8;
    const char * xr[RW];
#pragma unroll
    for (int i = 0; i < RW; ++i) xr[i] = vx + (size_t) min(row0 + i, nrows - 1)*row_bytes;
    float acc[NC][RW];
#pragma unroll
    for (int j = 0; j < NC; ++j)
#pragma unroll
        for (int i = 0; i < RW; ++i) acc[j][i] = 0.0f;
    for (int t = lane; t < ngroups; t += WARP_SIZE) {
        typename D::raw r[RW];
#pragma unroll
        for (int i = 0; i < RW; ++i) dec.load(r[i], xr[i], t, nb);
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const uint4 * ap = (const uint4 *) (aq + ((size_t) j*ngroups + t)*16);
            half2 a[16];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint4 v = __ldg(ap + c);
                a[4*c + 0] = as_h2(v.x); a[4*c + 1] = as_h2(v.y); a[4*c + 2] = as_h2(v.z); a[4*c + 3] = as_h2(v.w);
            }
            const float2 ds = __half22float2(__ldg(ads + (size_t) j*ngroups + t));
#pragma unroll
            for (int i = 0; i < RW; ++i) acc[j][i] += dec.dot(r[i], a, ds.x, ds.y);
        }
    }
#pragma unroll
    for (int j = 0; j < NC; ++j)
#pragma unroll
        for (int i = 0; i < RW; ++i) {
            const float v = warp_reduce_sum(acc[j][i]);
            if (lane == ((j*RW + i) & (WARP_SIZE - 1)) && row0 + i < nrows) dst[(size_t) j*nrows + row0 + i] = v;
        }
}

template <class D, int NW, int RW>
static bool wrc_launch(const char * vx, const a16_args & a, int ncols, cudaStream_t s) {
    const dim3 grid((a.nrows + NW*RW - 1)/(NW*RW)), blk(WARP_SIZE, NW);
    switch (ncols) {
        case 1: mv_wrc<D, NW, RW, 1><<<grid, blk, 0, s>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.nrows, a.row_bytes); return true;
        case 2: mv_wrc<D, NW, RW, 2><<<grid, blk, 0, s>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.nrows, a.row_bytes); return true;
        case 3: mv_wrc<D, NW, RW, 3><<<grid, blk, 0, s>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.nrows, a.row_bytes); return true;
        case 4: mv_wrc<D, NW, RW, 4><<<grid, blk, 0, s>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.nrows, a.row_bytes); return true;
        default: return false;
    }
}

// "wrcp": the persistent form for NC columns (prequantized), like wrq: every block stages the NC activation columns in
// shared memory once (column j: 4 chunk planes of ngp uint4, then ngroups float2 (amax, sum)), then its warps walk row
// tiles of RW rows with no further synchronisation. Same per-row order as mv_wrc / wr / wrq (bit-identical).
static inline int wrcp_cs(const int ngroups) { return 4*wr_ngp(ngroups) + (ngroups + 1)/2; }   // column stride, uint4

template <class D, int NW, int RW, int NC>
static __global__ void __launch_bounds__(WARP_SIZE*NW)
mv_wrcp(const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
        float * __restrict__ dst, const int ngroups, const int nrows, const int row_bytes, const int ngp, const int cs) {
    extern __shared__ uint4 wr_sm[];
    const int lane = threadIdx.x;
    const int tid  = threadIdx.x + WARP_SIZE*threadIdx.y;
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        const uint4 * ag = (const uint4 *) (aq + (size_t) j*ngroups*16);
        uint4 * pl = wr_sm + j*cs;
        for (int i = tid; i < 4*ngroups; i += WARP_SIZE*NW) pl[(i & 3)*ngp + (i >> 2)] = ag[i];
        float2 * sd = (float2 *) (pl + 4*ngp);
        for (int i = tid; i < ngroups; i += WARP_SIZE*NW) sd[i] = __half22float2(ads[(size_t) j*ngroups + i]);
    }
    __syncthreads();

    const int ntiles = (nrows + RW - 1)/RW;
    const int wstep  = gridDim.x*NW;
    const D dec(lane);
    const int nb = ngroups/8;
    for (int tile = blockIdx.x*NW + threadIdx.y; tile < ntiles; tile += wstep) {
        const int row0 = tile*RW;
        const char * xr[RW];
#pragma unroll
        for (int i = 0; i < RW; ++i) xr[i] = vx + (size_t) min(row0 + i, nrows - 1)*row_bytes;
        float acc[NC][RW];
#pragma unroll
        for (int j = 0; j < NC; ++j)
#pragma unroll
            for (int i = 0; i < RW; ++i) acc[j][i] = 0.0f;
        for (int t = lane; t < ngroups; t += WARP_SIZE) {
            typename D::raw r[RW];
#pragma unroll
            for (int i = 0; i < RW; ++i) dec.load(r[i], xr[i], t, nb);
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                const uint4 * pl = wr_sm + j*cs;
                half2 a[16];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint4 v = pl[c*ngp + t];
                    a[4*c + 0] = as_h2(v.x); a[4*c + 1] = as_h2(v.y); a[4*c + 2] = as_h2(v.z); a[4*c + 3] = as_h2(v.w);
                }
                const float2 ds = ((const float2 *) (pl + 4*ngp))[t];
#pragma unroll
                for (int i = 0; i < RW; ++i) acc[j][i] += dec.dot(r[i], a, ds.x, ds.y);
            }
        }
#pragma unroll
        for (int j = 0; j < NC; ++j)
#pragma unroll
            for (int i = 0; i < RW; ++i) {
                const float v = warp_reduce_sum(acc[j][i]);
                if (lane == ((j*RW + i) & (WARP_SIZE - 1)) && row0 + i < nrows) dst[(size_t) j*nrows + row0 + i] = v;
            }
    }
}

template <class D, int NW, int RW, int NC>
static bool wrcp_launch1(const char * vx, const a16_args & a, cudaStream_t s) {
    const int cs = wrcp_cs(a.ngroups);
    const size_t smem = (size_t) NC*cs*16;
    if (smem > 48*1024) return false;
    struct entry { int dev; size_t smem; int nblk; };
    static entry cache[16];
    static int ncache = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    int nblk = 0;
    for (int i = 0; i < ncache; ++i) if (cache[i].dev == dev && cache[i].smem == smem) { nblk = cache[i].nblk; break; }
    if (nblk == 0) {
        int per_sm = 0, nsm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, mv_wrcp<D, NW, RW, NC>, WARP_SIZE*NW, smem);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        nblk = std::max(1, per_sm)*nsm;
        if (ncache < 16) cache[ncache++] = entry{dev, smem, nblk};
    }
    const int ntiles = (a.nrows + RW - 1)/RW;
    const int grid = std::max(1, std::min(nblk, (ntiles + NW - 1)/NW));
    mv_wrcp<D, NW, RW, NC><<<grid, dim3(WARP_SIZE, NW), smem, s>>>(vx, a.aq, a.ads, a.dst, a.ngroups, a.nrows, a.row_bytes,
                                                                 wr_ngp(a.ngroups), cs);
    return true;
}

template <class D, int NW, int RW>
static bool wrcp_launch(const char * vx, const a16_args & a, int ncols, cudaStream_t s) {
    switch (ncols) {
        case 1: return wrcp_launch1<D, NW, RW, 1>(vx, a, s);
        case 2: return wrcp_launch1<D, NW, RW, 2>(vx, a, s);
        case 3: return wrcp_launch1<D, NW, RW, 3>(vx, a, s);
        case 4: return wrcp_launch1<D, NW, RW, 4>(vx, a, s);
        default: return false;
    }
}

// T columns (1..8): chunks of as many columns as fit staged in shared memory (at most 4; n_in 5120: 4, n_in 8704: 2),
// persistent wrcp; when not even one column fits (n_in > ~21k), the unstaged mv_wrc. P100E_MV_WRC = "NW,RW" picks
// the wrcp geometry (16,2 default; 8,4 / 8,2 / 4,4 / 4,2) for tuning; every geometry gives the same bits.
// P100 rank-0 matvec totals per step incl. the activation quantize, T = 2 / 3 / 4:
// 16,2: 24.80 / 32.26 / 38.02 ms; 8,4: 24.76 / 33.72 / 40.49; 8,2: 26.59 / 37.35 / 41.88; 4,4: 30.05 / 45.29 / 58.33;
// legacy K-split kernels (not T-invariant): 26.07 / 34.22 / 48.16.
template <class D>
static bool wrc_t(const char * vx, const a16_args & a, int ncols, cudaStream_t s) {
    static const int geo = [] {
        const char * e = getenv("P100E_MV_WRC");
        int nw = 0, rw = 0;
        if (!e || sscanf(e, "%d,%d", &nw, &rw) != 2) return 162;
        return nw*10 + rw;
    }();
    const int fit = (int) std::min<size_t>(4, (size_t) 48*1024/((size_t) wrcp_cs(a.ngroups)*16));
    const int step = fit > 0 ? fit : 4;
    for (int c0 = 0; c0 < ncols; c0 += step) {
        const int nc = std::min(step, ncols - c0);
        a16_args b = a;
        b.aq = a.aq + (size_t) c0*a.ngroups*16; b.ads = a.ads + (size_t) c0*a.ngroups; b.dst = a.dst + (size_t) c0*a.nrows;
        bool ok;
        if (fit == 0) ok = wrc_launch<D, 4, 2>(vx, b, nc, s);
        else switch (geo) {
            case 84:  ok = wrcp_launch<D, 8, 4>(vx, b, nc, s); break;
            case 82:  ok = wrcp_launch<D, 8, 2>(vx, b, nc, s); break;
            case 44:  ok = wrcp_launch<D, 4, 4>(vx, b, nc, s); break;
            case 42:  ok = wrcp_launch<D, 4, 2>(vx, b, nc, s); break;
            default:  ok = wrcp_launch<D, 16, 2>(vx, b, nc, s); break;
        }
        if (!ok) return false;
    }
    return true;
}

bool launch_wrc(int layout, const char * vx, const a16_args & a, int ncols, cudaStream_t s) {
    switch (layout) {
        case LAYOUT_Q4KR: return wrc_t<dec_q4kr<0>>(vx, a, ncols, s);
        case LAYOUT_Q3R:  return wrc_t<dec_q3r<0> >(vx, a, ncols, s);
        case LAYOUT_Q5KR: return wrc_t<dec_q5kr<0>>(vx, a, ncols, s);
        case LAYOUT_Q6R2:
            if (a.row_bytes % 16 == 0) return wrc_t<dec_q6r2<0, 16>>(vx, a, ncols, s);
            if (a.row_bytes % 8 == 0)  return wrc_t<dec_q6r2<0, 8> >(vx, a, ncols, s);
            return false;
        case LAYOUT_Q6R3:
            if (a.row_bytes % 16 == 0) return wrc_t<dec_q6r3<0, 16>>(vx, a, ncols, s);
            if (a.row_bytes % 8 == 0)  return wrc_t<dec_q6r3<0, 8> >(vx, a, ncols, s);
            return false;
        default: return false;
    }
}

// ------------------------------------------------------------------------------------------------- q8_0 (f32)
// Q8_0 repacked row: [qs plane 32*nb][d plane nb halves] (matvec.cu).  Chunk c = 16 weights = one uint4 at 16c, its
// scale is d[c >> 1].  Per chunk the same fp32 FMA chain as q8r_gemv (matvec_generic.cu), accumulated as
// acc = fma(d, s, acc); only the order of the chunk sums differs.
static __device__ __forceinline__ float q8_chunk(const uint4 q, const float4 (&xv)[4]) {
    const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        s = fmaf((float) (int8_t) (qw[c]      ), xv[c].x, s);
        s = fmaf((float) (int8_t) (qw[c] >>  8), xv[c].y, s);
        s = fmaf((float) (int8_t) (qw[c] >> 16), xv[c].z, s);
        s = fmaf((float) (int8_t) (qw[c] >> 24), xv[c].w, s);
    }
    return s;
}

// K split over the block, RB rows per block (tiny n_out)
template <int NW, int RB>
static __global__ void __launch_bounds__(WARP_SIZE*NW)
q8_ks(const char * __restrict__ w, const float * __restrict__ x, float * __restrict__ y, const int nb, const int n_out,
      const int row_bytes) {
    const int lane = threadIdx.x;
    const int tid  = threadIdx.x + WARP_SIZE*threadIdx.y;
    const int row0 = blockIdx.x*RB;
    float acc[RB];
#pragma unroll
    for (int i = 0; i < RB; ++i) acc[i] = 0.0f;
    for (int c = tid; c < 2*nb; c += WARP_SIZE*NW) {
        float4 xv[4];
        const float4 * xp = (const float4 *) (x + 16*c);
#pragma unroll
        for (int k = 0; k < 4; ++k) xv[k] = __ldg(xp + k);
#pragma unroll
        for (int i = 0; i < RB; ++i) {
            const char * r = w + (size_t) min(row0 + i, n_out - 1)*row_bytes;
            const uint4 q = __ldg((const uint4 *) (r + 16*c));
            const unsigned short db = __ldg((const unsigned short *) (r + 32*nb + 2*(c >> 1)));
            const float d = __half2float(*((const half *) &db));
            acc[i] = fmaf(d, q8_chunk(q, xv), acc[i]);
        }
    }
    __shared__ float red[NW][RB];
#pragma unroll
    for (int i = 0; i < RB; ++i) {
        const float v = warp_reduce_sum(acc[i]);
        if (lane == 0) red[threadIdx.y][i] = v;
    }
    __syncthreads();
    if (tid < RB && row0 + tid < n_out) {
        float v = 0.0f;
#pragma unroll
        for (int k = 0; k < NW; ++k) v += red[k][tid];
        y[row0 + tid] = v;
    }
}

// warp rows: each warp RW rows, lanes split K.  STAGE: x staged in shared memory as 4 float4 planes (chunk-major,
// conflict free) instead of read through L1.
template <int NW, int RW, bool STAGE>
static __global__ void __launch_bounds__(WARP_SIZE*NW)
q8_wr(const char * __restrict__ w, const float * __restrict__ x, float * __restrict__ y, const int nb, const int n_out,
      const int row_bytes, const int ncp) {
    extern __shared__ float4 q8_sm[];
    const int lane = threadIdx.x;
    const int nch  = 2*nb;
    if constexpr (STAGE) {
        const int tid = threadIdx.x + WARP_SIZE*threadIdx.y;
        const float4 * xg = (const float4 *) x;
        for (int i = tid; i < 4*nch; i += WARP_SIZE*NW) q8_sm[(i & 3)*ncp + (i >> 2)] = xg[i];
        __syncthreads();
    }
    const int row0 = (blockIdx.x*NW + threadIdx.y)*RW;
    if (row0 >= n_out) return;
    const char * xr[RW];
#pragma unroll
    for (int i = 0; i < RW; ++i) xr[i] = w + (size_t) min(row0 + i, n_out - 1)*row_bytes;
    float acc[RW];
#pragma unroll
    for (int i = 0; i < RW; ++i) acc[i] = 0.0f;
    for (int c = lane; c < nch; c += WARP_SIZE) {
        uint4 q[RW];
        unsigned short db[RW];
#pragma unroll
        for (int i = 0; i < RW; ++i) {
            q[i]  = __ldg((const uint4 *) (xr[i] + 16*c));
            db[i] = __ldg((const unsigned short *) (xr[i] + 32*nb + 2*(c >> 1)));
        }
        float4 xv[4];
        if constexpr (STAGE) {
#pragma unroll
            for (int k = 0; k < 4; ++k) xv[k] = q8_sm[k*ncp + c];
        } else {
            const float4 * xp = (const float4 *) (x + 16*c);
#pragma unroll
            for (int k = 0; k < 4; ++k) xv[k] = __ldg(xp + k);
        }
#pragma unroll
        for (int i = 0; i < RW; ++i) {
            const float d = __half2float(*((const half *) &db[i]));
            acc[i] = fmaf(d, q8_chunk(q[i], xv), acc[i]);
        }
    }
#pragma unroll
    for (int i = 0; i < RW; ++i) {
        const float v = warp_reduce_sum(acc[i]);
        if (lane == i && row0 + i < n_out) y[row0 + i] = v;
    }
}

const char * q8_variant_name(int v) {
    switch (v) {
        case 201: return "q8ks nw4 rb1"; case 202: return "q8ks nw8 rb1"; case 203: return "q8ks nw4 rb2"; case 204: return "q8ks nw2 rb4";
        case 211: return "q8wr nw4 rw1"; case 212: return "q8wr nw4 rw2"; case 213: return "q8wr nw8 rw1 st";
        case 214: return "q8wr nw8 rw2 st"; case 215: return "q8wr nw4 rw2 st"; case 216: return "q8wr nw4 rw4 st";
        case 217: return "q8wr nw2 rw1"; case 218: return "q8wr nw1 rw1";
        default: return "?";
    }
}

template <int NW, int RW, bool STAGE>
static bool q8wr_launch(const char * w, const float * x, float * y, int nb, int n_out, int rb, cudaStream_t s) {
    const int nch = 2*nb;
    const int ncp = nch + ((10 - nch % 8) % 8);
    const size_t smem = STAGE ? (size_t) 4*ncp*16 : 0;
    if (smem > 48*1024) return false;
    const dim3 grid((n_out + NW*RW - 1)/(NW*RW));
    q8_wr<NW, RW, STAGE><<<grid, dim3(WARP_SIZE, NW), smem, s>>>(w, x, y, nb, n_out, rb, ncp);
    return true;
}
template <int NW, int RB>
static bool q8ks_launch(const char * w, const float * x, float * y, int nb, int n_out, int rb, cudaStream_t s) {
    q8_ks<NW, RB><<<(n_out + RB - 1)/RB, dim3(WARP_SIZE, NW), 0, s>>>(w, x, y, nb, n_out, rb);
    return true;
}

bool launch_q8_t1(const char * w, const float * x, float * y, int64_t n_in, int64_t n_out, size_t row_bytes, int variant,
                  cudaStream_t s) {
    const int nb = (int) (n_in/32), no = (int) n_out, rb = (int) row_bytes;
    if (n_in % 256 != 0 || row_bytes % 16 != 0 || ((uintptr_t) x) % 16 != 0) return false;
    switch (variant) {
        case 201: return q8ks_launch<4, 1>(w, x, y, nb, no, rb, s);
        case 202: return q8ks_launch<8, 1>(w, x, y, nb, no, rb, s);
        case 203: return q8ks_launch<4, 2>(w, x, y, nb, no, rb, s);
        case 204: return q8ks_launch<2, 4>(w, x, y, nb, no, rb, s);
        case 211: return q8wr_launch<4, 1, false>(w, x, y, nb, no, rb, s);
        case 212: return q8wr_launch<4, 2, false>(w, x, y, nb, no, rb, s);
        case 213: return q8wr_launch<8, 1, true >(w, x, y, nb, no, rb, s);
        case 214: return q8wr_launch<8, 2, true >(w, x, y, nb, no, rb, s);
        case 215: return q8wr_launch<4, 2, true >(w, x, y, nb, no, rb, s);
        case 216: return q8wr_launch<4, 4, true >(w, x, y, nb, no, rb, s);
        case 217: return q8wr_launch<2, 1, false>(w, x, y, nb, no, rb, s);
        case 218: return q8wr_launch<1, 1, false>(w, x, y, nb, no, rb, s);
        default: return false;
    }
}

// T columns at once (MTP verify, T = 2..4): q8_wr without staging, each row summed exactly as q8_wr / q8r_gemv do for
// one column (lane-strided chunks c = lane, lane + 32, ..., acc = fma(d, chunk FMA chain, acc), xor tree), so column j
// equals the T = 1 call on column j. q8r_gemv<T> runs one warp per row at 4 rows per block, which leaves a 24-row
// ssm_alpha / ssm_beta shard on 6 blocks (18.6 us at T = 3 on the P100 vs 6.6 us for the T = 1 beta|alpha launch).
template <int NW, int RW, int T>
static __global__ void __launch_bounds__(WARP_SIZE*NW)
q8_wrt(const char * __restrict__ w, const float * __restrict__ x, float * __restrict__ y, const int nb, const int n_in,
       const int n_out, const int row_bytes) {
    const int lane = threadIdx.x;
    const int nch  = 2*nb;
    const int row0 = (blockIdx.x*NW + threadIdx.y)*RW;
    if (row0 >= n_out) return;
    const char * xr[RW];
#pragma unroll
    for (int i = 0; i < RW; ++i) xr[i] = w + (size_t) min(row0 + i, n_out - 1)*row_bytes;
    float acc[RW][T];
#pragma unroll
    for (int i = 0; i < RW; ++i)
#pragma unroll
        for (int j = 0; j < T; ++j) acc[i][j] = 0.0f;
    for (int c = lane; c < nch; c += WARP_SIZE) {
        uint4 q[RW];
        unsigned short db[RW];
#pragma unroll
        for (int i = 0; i < RW; ++i) {
            q[i]  = __ldg((const uint4 *) (xr[i] + 16*c));
            db[i] = __ldg((const unsigned short *) (xr[i] + 32*nb + 2*(c >> 1)));
        }
#pragma unroll
        for (int j = 0; j < T; ++j) {
            float4 xv[4];
            const float4 * xp = (const float4 *) (x + (size_t) j*n_in + 16*c);
#pragma unroll
            for (int k = 0; k < 4; ++k) xv[k] = __ldg(xp + k);
#pragma unroll
            for (int i = 0; i < RW; ++i) {
                const float d = __half2float(*((const half *) &db[i]));
                acc[i][j] = fmaf(d, q8_chunk(q[i], xv), acc[i][j]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < RW; ++i)
#pragma unroll
        for (int j = 0; j < T; ++j) {
            const float v = warp_reduce_sum(acc[i][j]);
            if (lane == i && row0 + i < n_out) y[(size_t) j*n_out + row0 + i] = v;
        }
}

template <int NW, int RW>
static bool q8wrt_launch(const char * w, const float * x, float * y, int T, int nb, int n_in, int n_out, int rb,
                         cudaStream_t s) {
    const dim3 grid((n_out + NW*RW - 1)/(NW*RW)), blk(WARP_SIZE, NW);
    switch (T) {
        case 2: q8_wrt<NW, RW, 2><<<grid, blk, 0, s>>>(w, x, y, nb, n_in, n_out, rb); return true;
        case 3: q8_wrt<NW, RW, 3><<<grid, blk, 0, s>>>(w, x, y, nb, n_in, n_out, rb); return true;
        case 4: q8_wrt<NW, RW, 4><<<grid, blk, 0, s>>>(w, x, y, nb, n_in, n_out, rb); return true;
        default: return false;
    }
}

bool launch_q8_tn(const char * w, const float * x, float * y, int T, int64_t n_in, int64_t n_out, size_t row_bytes,
                  cudaStream_t s) {
    const int nb = (int) (n_in/32), no = (int) n_out, rb = (int) row_bytes;
    if (T < 2 || T > 4 || n_out > 2048 || n_in % 256 != 0 || row_bytes % 16 != 0 || ((uintptr_t) x) % 16 != 0) return false;
    // tiny shards (ssm_alpha / ssm_beta, 24 rows): one row per 1-warp block, as the T = 1 choice (218); else 4 warps x 2 rows
    if (n_out <= 64) return q8wrt_launch<1, 1>(w, x, y, T, nb, (int) n_in, no, rb, s);
    return q8wrt_launch<4, 2>(w, x, y, T, nb, (int) n_in, no, rb, s);
}

// ------------------------------------------------------------------------------------------------- wrq + q8 pair
// P100E_MERGE_INPROJ (decode step 6a): a layer's two T = 1 input-projection launches (a warp-rows weight on the a16
// activation, a Q8R weight on the f32 activation: the test model qkv|gate + beta|alpha, q + k|v) as one launch. Blocks
// [0, gq) run q8_wr's row loop (non-staged: the staged copy holds the same x bits), blocks [gq, gq + ga) mv_wrp's PQ
// body with the tile stride ga*NW. Both families sum a row in one warp (lane-strided groups / chunks, xor tree), so a
// row's value does not depend on the geometry: the outputs are bit-identical to the two launches.
template <class D, int NW, int RW, int MINB, int RWQ>
static __global__ void __launch_bounds__(WARP_SIZE*NW, MINB)
mv_wrq_q8(const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
          float * __restrict__ dst, const int ngroups, const int nrows, const int row_bytes, const int ngp, const int ga,
          const char * __restrict__ w8, const float * __restrict__ x8, float * __restrict__ y8, const int nb8,
          const int n8, const int rb8, const int gq) {
    const int lane = threadIdx.x;
    if ((int) blockIdx.x < gq) {
        const int nch  = 2*nb8;
        const int row0 = (blockIdx.x*NW + threadIdx.y)*RWQ;
        if (row0 >= n8) return;
        const char * xr[RWQ];
#pragma unroll
        for (int i = 0; i < RWQ; ++i) xr[i] = w8 + (size_t) min(row0 + i, n8 - 1)*rb8;
        float acc[RWQ];
#pragma unroll
        for (int i = 0; i < RWQ; ++i) acc[i] = 0.0f;
        for (int c = lane; c < nch; c += WARP_SIZE) {
            uint4 q[RWQ];
            unsigned short db[RWQ];
#pragma unroll
            for (int i = 0; i < RWQ; ++i) {
                q[i]  = __ldg((const uint4 *) (xr[i] + 16*c));
                db[i] = __ldg((const unsigned short *) (xr[i] + 32*nb8 + 2*(c >> 1)));
            }
            float4 xv[4];
            const float4 * xp = (const float4 *) (x8 + 16*c);
#pragma unroll
            for (int k = 0; k < 4; ++k) xv[k] = __ldg(xp + k);
#pragma unroll
            for (int i = 0; i < RWQ; ++i) {
                const float d = __half2float(*((const half *) &db[i]));
                acc[i] = fmaf(d, q8_chunk(q[i], xv), acc[i]);
            }
        }
#pragma unroll
        for (int i = 0; i < RWQ; ++i) {
            const float v = warp_reduce_sum(acc[i]);
            if (lane == i && row0 + i < n8) y8[row0 + i] = v;
        }
        return;
    }
    extern __shared__ uint4 wr_sm[];
    const int tid    = threadIdx.x + WARP_SIZE*threadIdx.y;
    const int ntiles = (nrows + RW - 1)/RW;
    const int wstep  = ga*NW;
    int tile = ((int) blockIdx.x - gq)*NW + threadIdx.y;
    float2 * sds = (float2 *) (wr_sm + 4*ngp);
    const uint4 * ag = (const uint4 *) aq;
    for (int i = tid; i < 4*ngroups; i += WARP_SIZE*NW) wr_sm[(i & 3)*ngp + (i >> 2)] = ag[i];
    for (int i = tid; i < ngroups; i += WARP_SIZE*NW) sds[i] = __half22float2(ads[i]);
    __syncthreads();
    const D dec(lane);
    const int nb = ngroups/8;
    for (; tile < ntiles; tile += wstep) {
        const int row0 = tile*RW;
        const char * xr[RW];
#pragma unroll
        for (int i = 0; i < RW; ++i) xr[i] = vx + (size_t) min(row0 + i, nrows - 1)*row_bytes;
        float acc[RW];
#pragma unroll
        for (int i = 0; i < RW; ++i) acc[i] = 0.0f;
        for (int t = lane; t < ngroups; t += WARP_SIZE) {
            typename D::raw r[RW];
#pragma unroll
            for (int i = 0; i < RW; ++i) dec.load(r[i], xr[i], t, nb);
            half2 a[16];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint4 v = wr_sm[c*ngp + t];
                a[4*c + 0] = as_h2(v.x); a[4*c + 1] = as_h2(v.y); a[4*c + 2] = as_h2(v.z); a[4*c + 3] = as_h2(v.w);
            }
            const float2 ds = sds[t];
#pragma unroll
            for (int i = 0; i < RW; ++i) acc[i] += dec.dot(r[i], a, ds.x, ds.y);
        }
#pragma unroll
        for (int i = 0; i < RW; ++i) {
            const float v = warp_reduce_sum(acc[i]);
            if (lane == i && row0 + i < nrows) dst[row0 + i] = v;
        }
    }
}

template <class D, int NW, int RW, int MINB, int RWQ>
static bool wrq_q8_launch(const char * vx, const a16_args & a, const char * w8, const float * x8, float * y8, int nb8,
                          int n8, int rb8, cudaStream_t s) {
    const size_t smem = wr_smem(a.ngroups);
    if (smem > 48*1024) return false;
    // the wrp part's persistent grid sized as wrp_launch does
    struct entry { int dev; size_t smem; int nblk; };
    static entry cache[16];
    static int ncache = 0;
    int dev = 0;
    cudaGetDevice(&dev);
    int nblk = 0;
    for (int i = 0; i < ncache; ++i) if (cache[i].dev == dev && cache[i].smem == smem) { nblk = cache[i].nblk; break; }
    if (nblk == 0) {
        int per_sm = 0, nsm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, mv_wrq_q8<D, NW, RW, MINB, RWQ>, WARP_SIZE*NW, smem);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        nblk = std::max(1, per_sm)*nsm;
        if (ncache < 16) cache[ncache++] = entry{dev, smem, nblk};
    }
    const int ntiles = (a.nrows + RW - 1)/RW;
    const int ga = std::max(1, std::min(nblk, (ntiles + NW - 1)/NW));
    const int gq = (n8 + NW*RWQ - 1)/(NW*RWQ);
    mv_wrq_q8<D, NW, RW, MINB, RWQ><<<ga + gq, dim3(WARP_SIZE, NW), smem, s>>>(
        vx, a.aq, a.ads, a.dst, a.ngroups, a.nrows, a.row_bytes, wr_ngp(a.ngroups), ga, w8, x8, y8, nb8, n8, rb8, gq);
    return true;
}

template <class D>
static bool wrq_q8_t(const char * vx, const a16_args & a, int variant, const char * w8, const float * x8, float * y8,
                     int nb8, int n8, int rb8, cudaStream_t s) {
    const bool tiny = n8 <= 64;   // beta|alpha (48 rows): one row per warp spreads them over more warps
    // the default_t1_variant geometries only (303 / 306 / 314 / 315 / 320; the WRP_VARIANTS rows): other ids keep two launches
    switch (variant) {
#define X(ID, NW, RW, MINB) \
        case ID + 200: \
            return tiny ? wrq_q8_launch<D, NW, RW, MINB, 1>(vx, a, w8, x8, y8, nb8, n8, rb8, s) \
                        : wrq_q8_launch<D, NW, RW, MINB, 2>(vx, a, w8, x8, y8, nb8, n8, rb8, s);
        X(303, 8, 2, 1) X(306, 8, 4, 1) X(314, 16, 2, 1) X(315, 8, 1, 1) X(320, 12, 4, 1)
#undef X
        default: return false;
    }
}

bool launch_wrq_q8(int layout, const char * vx, const a16_args & a, int variant, const char * w8, const float * x8,
                   float * y8, int64_t n_in8, int64_t n8, size_t rb8, cudaStream_t s) {
    if (variant < 501 || variant >= 600) return false;
    if (n_in8 % 256 != 0 || rb8 % 16 != 0 || ((uintptr_t) x8) % 16 != 0) return false;
    // a layout whose inline-PTX twin is switched on keeps its own launch (that kernel is not merged here)
    const int xi = layout == LAYOUT_Q3R ? 0 : layout == LAYOUT_Q4KR ? 1 : layout == LAYOUT_Q5KR ? 2 : layout == LAYOUT_Q6R3 ? 3 : -1;
    if (xi >= 0 && wrx_level(xi) > 0) return false;
    const int nb8 = (int) (n_in8/32), n8i = (int) n8, rb8i = (int) rb8;
    switch (layout) {
        case LAYOUT_Q4KR: return wrq_q8_t<dec_q4kr<0>>(vx, a, variant, w8, x8, y8, nb8, n8i, rb8i, s);
        case LAYOUT_Q3R:  return wrq_q8_t<dec_q3r<0> >(vx, a, variant, w8, x8, y8, nb8, n8i, rb8i, s);
        case LAYOUT_Q5KR: return wrq_q8_t<dec_q5kr<0>>(vx, a, variant, w8, x8, y8, nb8, n8i, rb8i, s);
        case LAYOUT_Q6R2:
            if (a.row_bytes % 16 == 0) return wrq_q8_t<dec_q6r2<0, 16>>(vx, a, variant, w8, x8, y8, nb8, n8i, rb8i, s);
            if (a.row_bytes % 8 == 0)  return wrq_q8_t<dec_q6r2<0, 8> >(vx, a, variant, w8, x8, y8, nb8, n8i, rb8i, s);
            return false;
        case LAYOUT_Q6R3:
            if (a.row_bytes % 16 == 0) return wrq_q8_t<dec_q6r3<0, 16>>(vx, a, variant, w8, x8, y8, nb8, n8i, rb8i, s);
            if (a.row_bytes % 8 == 0)  return wrq_q8_t<dec_q6r3<0, 8> >(vx, a, variant, w8, x8, y8, nb8, n8i, rb8i, s);
            return false;
        default: return false;
    }
}

} // namespace p100e::fast
