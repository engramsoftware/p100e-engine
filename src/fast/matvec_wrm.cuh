// p100e fast: the wrm kernel templates (see matvec_wrm.cu for the design and the bit-identity argument). Instantiated
// per decoder in matvec_wrm_<layout>.cu (one translation unit each: the instantiations of all layouts in one file
// made ptxas run out of host memory), dispatched by launch_wrm (matvec_wrm.cu).
#pragma once
#include "matvec.cuh"
#include "matvec_internal.cuh"
#include "matvec_dec.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace p100e::fast {

template <int K> using wic = std::integral_constant<int, K>;

// decoders with D::kAcc (dec_q6h) fold the accumulate into their tail: acc = D::fin_acc(acc, ...); the others keep
// acc += D::fin(...) (the same code as before this trait)
template <class D, class = void> struct wrm_has_acc : std::false_type {};
template <class D> struct wrm_has_acc<D, std::void_t<decltype(D::kAcc)>> : std::bool_constant<D::kAcc> {};

// Shared memory, group-major records (all offsets inside a record are compile-time, one base address per group):
// staged group g = S uint4: column j's 4 chunks at [4j .. 4j+3], then the (amax, sum) float2 of the NC columns at
// [4NC ..]; S odd (padded) so 32 lanes reading 32 consecutive records with LDS.128 are bank-conflict free.
template <int NC> struct wrm_rec { static constexpr int S = NC == 0 ? 1 : (4*NC + (NC + 1)/2) | 1; };

static __device__ __forceinline__ const char * wrm_pin(const char * p) {   // opaque: computed once, kept in a register
    asm volatile("" : "+l"(p));
    return p;
}

// MODE: see the top of the file. NS: staged columns for MODE 3 (MODE 1 / 2: NC, MODE 0: 0).
// RR (MODE 2 only, 1 = as before): tile rounds handled per pass over the K phases. A block keeps the accumulators of its RR
// tiles (one per round) in registers and stages each K phase once for all of them, instead of re-staging every phase for
// every round (n_in 8704 at 4 columns: 3 rounds x 2 phases = 6 stagings -> 2). Per (row, column) the group order and the
// adds are unchanged, so the bits are too.
template <class D, int NW, int RW, int NC, int MODE, int NS, bool PF, int RR = 1>
static __global__ void __launch_bounds__(WARP_SIZE*NW, 1)
mv_wrm(const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads, const mv_dst dst,
       const int ngroups, const int nrows, const int row_bytes, const int kpg) {
    constexpr int NST = MODE == 0 ? 0 : MODE == 3 ? NS : NC;   // staged columns
    constexpr int S = wrm_rec<NST>::S;
    extern __shared__ uint4 wm_sm[];
    const int lane = threadIdx.x;
    const int tid  = threadIdx.x + WARP_SIZE*threadIdx.y;
    const int nb = ngroups/8;
    const int ntiles = (nrows + RW - 1)/RW;
    D dec(lane);
    dec.pin();

    auto stage = [&](const int g0, const int gn) {
#pragma unroll
        for (int j = 0; j < NST; ++j) {
            const uint4 * ag = (const uint4 *) (aq + ((size_t) j*ngroups + g0)*16);
            for (int i = tid; i < 4*gn; i += WARP_SIZE*NW) wm_sm[(i >> 2)*S + 4*j + (i & 3)] = ag[i];
            const half2 * dg = ads + (size_t) j*ngroups + g0;
            for (int i = tid; i < gn; i += WARP_SIZE*NW) ((float2 *) (wm_sm + i*S + 4*NST))[j] = __half22float2(dg[i]);
        }
    };
    if constexpr (MODE == 1 || MODE == 3) {
        stage(0, ngroups);
        __syncthreads();
    }

    for (int bt0 = blockIdx.x; bt0*NW < ntiles; bt0 += gridDim.x*RR) {   // block-uniform (MODE 2 syncs inside)
        if constexpr (MODE != 2) {
            if (bt0*NW + threadIdx.y >= ntiles) break;   // RR == 1 here
        }
        bool active[RR];
        int  row0[RR];
        const char * xr[RR][RW];
        float acc[RR][NC][RW];
#pragma unroll
        for (int r = 0; r < RR; ++r) {
            const int tile = (bt0 + r*gridDim.x)*NW + threadIdx.y;
            active[r] = tile < ntiles;
            row0[r] = tile*RW;
#pragma unroll
            for (int i = 0; i < RW; ++i) xr[r][i] = wrm_pin(vx + (size_t) min(row0[r] + i, nrows - 1)*row_bytes);
#pragma unroll
            for (int j = 0; j < NC; ++j)
#pragma unroll
                for (int i = 0; i < RW; ++i) acc[r][j][i] = 0.0f;
        }

        // one group of every row: decode once, HFMA2 chains of all columns, epilogue per (row, column)
        auto group = [&](const typename D::raw (&r)[RW], const int t, const int ts, float (&acc)[NC][RW]) {
            typename D::sc sc[RW];
#pragma unroll
            for (int i = 0; i < RW; ++i) sc[i] = dec.prep(r[i]);
            half2 se[NC][RW], so[NC][RW];
#pragma unroll
            for (int j = 0; j < NC; ++j)
#pragma unroll
                for (int i = 0; i < RW; ++i) { se[j][i] = make_half2(0.0f, 0.0f); so[j][i] = make_half2(0.0f, 0.0f); }
            const uint4 * rec = wm_sm + ts*S;
            auto step = [&](auto kc) {
                constexpr int K = decltype(kc)::value;
                half2 a[NC][4];
#pragma unroll
                for (int j = 0; j < NC; ++j) {
                    uint4 v;
                    if (j < NST) v = rec[4*j + K];
                    else         v = __ldg((const uint4 *) (aq + ((size_t) j*ngroups + t)*16) + K);
                    a[j][0] = as_h2(v.x); a[j][1] = as_h2(v.y); a[j][2] = as_h2(v.z); a[j][3] = as_h2(v.w);
                }
#pragma unroll
                for (int i = 0; i < RW; ++i) {
                    half2 w[4];
                    D::template chunk<K>(r[i], w);
#pragma unroll
                    for (int j = 0; j < NC; ++j) {
                        se[j][i] = __hfma2(w[0], a[j][0], se[j][i]);
                        so[j][i] = __hfma2(w[1], a[j][1], so[j][i]);
                        se[j][i] = __hfma2(w[2], a[j][2], se[j][i]);
                        so[j][i] = __hfma2(w[3], a[j][3], so[j][i]);
                    }
                }
            };
            step(wic<0>{}); step(wic<1>{}); step(wic<2>{}); step(wic<3>{});
            float2 ds[NC];
#pragma unroll
            for (int j = 0; j < NST; j += 2) {   // two staged columns' (amax, sum) per LDS.128
                const float4 v = *((const float4 *) (rec + 4*NST + j/2));
                ds[j] = make_float2(v.x, v.y);
                if (j + 1 < NST) ds[j + 1] = make_float2(v.z, v.w);
            }
#pragma unroll
            for (int j = NST; j < NC; ++j) ds[j] = __half22float2(__ldg(ads + (size_t) j*ngroups + t));
#pragma unroll
            for (int j = 0; j < NC; ++j)
#pragma unroll
                for (int i = 0; i < RW; ++i) {
                    if constexpr (wrm_has_acc<D>::value) acc[j][i] = D::fin_acc(acc[j][i], se[j][i], so[j][i], sc[i], ds[j].x, ds[j].y);
                    else acc[j][i] += D::fin(se[j][i], so[j][i], sc[i], ds[j].x, ds[j].y);
                }
        };

        const int kstep = MODE == 2 ? kpg : ngroups;
        for (int g0 = 0; g0 < ngroups; g0 += kstep) {
            const int g1 = min(ngroups, g0 + kstep);
            if constexpr (MODE == 2) {
                __syncthreads();
                stage(g0, g1 - g0);
                __syncthreads();
            }
#pragma unroll
            for (int rr = 0; rr < RR; ++rr) {
                if (!active[rr]) continue;
                if constexpr (PF) {
                    // software pipeline, two buffers: the next group's loads are issued before this group's arithmetic
                    typename D::raw ra[RW], rb[RW];
                    int t = g0 + lane;
                    if (t < g1) {
#pragma unroll
                        for (int i = 0; i < RW; ++i) dec.load(ra[i], xr[rr][i], t, nb);
                    }
                    while (t < g1) {
                        if (t + WARP_SIZE < g1) {
#pragma unroll
                            for (int i = 0; i < RW; ++i) dec.load(rb[i], xr[rr][i], t + WARP_SIZE, nb);
                        }
                        group(ra, t, t - g0, acc[rr]);
                        t += WARP_SIZE;
                        if (t >= g1) break;
                        if (t + WARP_SIZE < g1) {
#pragma unroll
                            for (int i = 0; i < RW; ++i) dec.load(ra[i], xr[rr][i], t + WARP_SIZE, nb);
                        }
                        group(rb, t, t - g0, acc[rr]);
                        t += WARP_SIZE;
                    }
                } else {
                    for (int t = g0 + lane; t < g1; t += WARP_SIZE) {
                        typename D::raw r[RW];
#pragma unroll
                        for (int i = 0; i < RW; ++i) dec.load(r[i], xr[rr][i], t, nb);
                        group(r, t, t - g0, acc[rr]);
                    }
                }
            }
        }

#pragma unroll
        for (int rr = 0; rr < RR; ++rr) {
            if (!active[rr]) continue;
            // every lane gets each (column, row) sum from the xor tree; lane j*RW + i keeps that one and stores it, so
            // there is one (predicated) store per lane instead of a divergent store per value
            float mine = 0.0f;
#pragma unroll
            for (int j = 0; j < NC; ++j)
#pragma unroll
                for (int i = 0; i < RW; ++i) {
                    const float v = warp_reduce_sum(acc[rr][j][i]);
                    if (lane == j*RW + i) mine = v;
                }
            const int j = lane / RW, i = lane - (lane / RW)*RW;
            if (lane < NC*RW && row0[rr] + i < nrows) {
                if (dst.np == 1) dst.y[0][(size_t) j*nrows + row0[rr] + i] = mine;
                else             mv_store(dst, j, row0[rr] + i, mine);
            }
        }
    }
}

// one launch of NC columns. kpg: groups per K phase (MODE 2), 0 = the largest multiple of 32 that fits, split evenly
template <class D, int NW, int RW, int NC, int MODE, int NS, bool PF, int RR = 1>
static bool wrm_launch1(const char * vx, const a16_args & a, const mv_dst & d, int kpg, cudaStream_t s) {
    const int ng = a.ngroups;
    constexpr int NST = MODE == 0 ? 0 : MODE == 3 ? NS : NC;
    constexpr int S = wrm_rec<NST>::S;
    size_t smem = 0;
    if constexpr (MODE == 1 || MODE == 3) kpg = ng;
    if constexpr (MODE == 2) {
        if (kpg <= 0) {
            int kmax = 0;
            for (int k = 32; k < ng; k += 32) if ((size_t) k*S*16 <= 48*1024) kmax = k;
            if (kmax == 0) return false;
            const int nph = (ng + kmax - 1)/kmax;
            kpg = ((ng + nph - 1)/nph + 31)/32*32;
        }
        if (kpg % 32 != 0 || kpg >= ng) return false;
    }
    if constexpr (MODE != 0) {
        smem = (size_t) kpg*S*16;
        if (smem > 48*1024) return false;
    }
    const int ntiles = (a.nrows + RW - 1)/RW;
    const int need = (ntiles + NW - 1)/NW;   // blocks of NW tiles
    int grid = need;
    if constexpr (MODE != 0) {
        // persistent grid: as many blocks as fit at once (cached per device and smem size), no more than needed
        struct entry { int dev; size_t smem; int nblk; };
        static entry cache[16];
        static int ncache = 0;
        int dev = 0;
        cudaGetDevice(&dev);
        int nblk = 0;
        for (int i = 0; i < ncache; ++i) if (cache[i].dev == dev && cache[i].smem == smem) { nblk = cache[i].nblk; break; }
        if (nblk == 0) {
            int per_sm = 0, nsm = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, mv_wrm<D, NW, RW, NC, MODE, NS, PF, RR>, WARP_SIZE*NW, smem);
            cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
            nblk = std::max(1, per_sm)*nsm;
            if (ncache < 16) cache[ncache++] = entry{dev, smem, nblk};
        }
        if constexpr (MODE == 2 && RR == 1 && NC >= 3) {
            // K phases, more tile rounds than resident blocks: one pass over the phases for up to 4 rounds at once (mv_wrm RR);
            // mv_tune.wrm_ph: 1 = off (restage every phase per round), 0 = auto (<= 4 rounds), N = at most N rounds.
            // Wide launches (NC > 4): at most 2 rounds - the accumulators of RR x NC x RW rows plus the NC x 4 half2 of
            // the activation chunks would spill above that (__launch_bounds__(512, 1): 128 registers per lane)
            constexpr int RMAX = NC > 4 ? 2 : 4;
            const int cap = mv_tune.wrm_ph <= 0 ? RMAX : std::min(mv_tune.wrm_ph, RMAX);
            const int rounds = std::min(cap, (need + nblk - 1)/nblk);
            if (rounds == 2) return wrm_launch1<D, NW, RW, NC, MODE, NS, PF, 2>(vx, a, d, kpg, s);
            if constexpr (RMAX >= 3) {
                if (rounds == 3) return wrm_launch1<D, NW, RW, NC, MODE, NS, PF, 3>(vx, a, d, kpg, s);
                if (rounds == 4) return wrm_launch1<D, NW, RW, NC, MODE, NS, PF, 4>(vx, a, d, kpg, s);
            }
        }
        grid = std::max(1, std::min(nblk, (need + RR - 1)/RR));
    }
    mv_wrm<D, NW, RW, NC, MODE, NS, PF, RR><<<grid, dim3(WARP_SIZE, NW), smem, s>>>(vx, a.aq, a.ads, d, ng, a.nrows,
                                                                              a.row_bytes, kpg);
    return true;
}

// KIND 1 (staged): MODE 1 when the NC columns fit in shared memory, else K phases. KIND 3 (hybrid, NS staged columns):
// MODE 3, or KIND 1 when NC <= NS. KIND 0: MODE 0.
template <class D, int NW, int RW, int KIND, int NS, bool PF, int NC>
static bool wrm_kind(const char * vx, const a16_args & a, const mv_dst & d, int kpg, cudaStream_t s) {
    if constexpr (KIND == 0) {
        return wrm_launch1<D, NW, RW, NC, 0, 0, PF>(vx, a, d, 0, s);
    } else if constexpr (KIND == 3 && NC > NS) {
        if ((size_t) a.ngroups*wrm_rec<NS>::S*16 > 48*1024) return false;
        return wrm_launch1<D, NW, RW, NC, 3, NS, PF>(vx, a, d, 0, s);
    } else {
        if ((size_t) a.ngroups*wrm_rec<NC>::S*16 <= 48*1024 && kpg <= 0) return wrm_launch1<D, NW, RW, NC, 1, 0, PF>(vx, a, d, 0, s);
        return wrm_launch1<D, NW, RW, NC, 2, 0, PF>(vx, a, d, kpg, s);
    }
}

// WIDE: the geometry is also instantiated for 5..8 columns per launch (the batched multi-session decode: two MTP
// chats x (1 + 3 drafts) = 8 rows in ONE weight pass instead of two 4-column launches). The epilogue keeps one
// (column, row) sum per lane, so NC*RW <= 32 is required; geometries that do not need the wide case stay at <= 4 to
// keep the instantiation count (ptxas host memory, build time) down.
template <class D0, int NW, int RW, int KIND, int NS, bool PF, int LP, bool WIDE>
static bool wrm_nc(const char * vx, const a16_args & a, const mv_dst & d, int nc, int kpg, cudaStream_t s) {
    using D = typename D0::template rebind<LP>;
    static_assert(!WIDE || 8*RW <= 32, "wrm wide: NC*RW must fit one lane per (column, row) sum");
    switch (nc) {
        case 1: return wrm_kind<D, NW, RW, KIND, NS, PF, 1>(vx, a, d, kpg, s);
        case 2: return wrm_kind<D, NW, RW, KIND, NS, PF, 2>(vx, a, d, kpg, s);
        case 3: return wrm_kind<D, NW, RW, KIND, NS, PF, 3>(vx, a, d, kpg, s);
        case 4: return wrm_kind<D, NW, RW, KIND, NS, PF, 4>(vx, a, d, kpg, s);
        default: break;
    }
    if constexpr (WIDE) {
        switch (nc) {
            case 5: return wrm_kind<D, NW, RW, KIND, NS, PF, 5>(vx, a, d, kpg, s);
            case 6: return wrm_kind<D, NW, RW, KIND, NS, PF, 6>(vx, a, d, kpg, s);
            case 7: return wrm_kind<D, NW, RW, KIND, NS, PF, 7>(vx, a, d, kpg, s);
            case 8: return wrm_kind<D, NW, RW, KIND, NS, PF, 8>(vx, a, d, kpg, s);
            default: return false;
        }
    }
    return false;
}

// geometry ids (mv_tune.wrm_geo / P100E_MV_WRM_GEO; 0 = default by shape): id -> (KIND, NW, RW, PF, NS, LP, WIDE)
//   staged (MODE 1 / 2 by fit): 1..10; unstaged (MODE 0): 11..13; hybrid (MODE 3, NS staged columns): 21..27
//   (LP 1 = weights loaded L2-only, ld.global.cg, so they do not evict the cached activation columns; LP 2 = .cs)
//   ids 5, 7, 8 (16x1 pf1, 8x4 pf1, 4x4 pf1) were dropped: never the best on any shape
//   WIDE 1: also 5..8 columns per launch (wrm_nc); 26 / 27 = hybrid with 4 staged columns (n_in 5120 x 4 columns is
//   48.6 KB, the most that fits) for the wide case, the other columns through the read-only cache
#define WRM_GEOS(X) \
    X(1, 1, 16, 2, false, 0, 0, 1) X(2, 1, 16, 2, true, 0, 0, 0) X(3, 1, 8, 4, false, 0, 0, 1) X(4, 1, 8, 2, true, 0, 0, 0) \
    X(6, 1, 12, 2, true, 0, 0, 0)  X(9, 1, 16, 2, false, 0, 1, 1) X(10, 1, 12, 2, false, 0, 0, 0) \
    X(11, 0, 1, 1, true, 0, 0, 1)  X(12, 0, 2, 1, true, 0, 0, 0) X(13, 0, 1, 2, true, 0, 0, 1) \
    X(21, 3, 16, 2, false, 2, 0, 1) X(22, 3, 16, 2, false, 2, 1, 1) X(23, 3, 16, 2, false, 1, 1, 0) X(24, 3, 16, 2, false, 3, 1, 0) \
    X(25, 3, 16, 2, false, 2, 2, 0) X(26, 3, 16, 2, false, 4, 0, 1) X(27, 3, 16, 2, false, 4, 1, 1)


// default geometry by shape (P100 rank-0 shards): staged 16 warps x 2 rows without the
// software pipeline (geo 1) wins nearly every shape at T = 2..4; tiny n_out: one warp per row, unstaged (geo 11)
static int wrm_default_geo(const a16_args & a, int nc) {
    if (a.nrows <= 64) return 11;
    // the nc columns do not fit in shared memory (n_in 8704, ffn_down): 3 columns hybrid (2 staged, weights L2-only;
    // q4_K 79 vs 96 us K-phased at T = 3), 4 columns K-phased (111 vs 121 us hybrid)
    const int S = (4*nc + (nc + 1)/2) | 1;
    if ((size_t) a.ngroups*S*16 > 48*1024 && nc == 3) return 22;
    // 5..8 columns (wide): mv_tune.wrm_wide_geo (P100E_MV_WRM_WGEO) until the P100 sweep picks the default
    if (nc > 4 && mv_tune.wrm_wide_geo > 0) return mv_tune.wrm_wide_geo;
    return 1;
}

// Rows up to which 5..8 columns run as ONE launch (P100E_MV_WRM_WROWS; 0 = 64 x SMs, 3584 on the P100). Measured
// (P100 rank 0, T = 8): the 8-column activations of n_in 5120 are
// 94 KB of a16 fp16, twice the 48 KB a block can stage, so the wide launch is K-phased and restages every phase from L2
// for every pass of tile rounds (RR <= 2 at NC > 4: the registers). Shapes with more tile rounds than 2 x the resident
// blocks pay that restage 2-3 times and lose to two 4-column launches (ffn gate/up q4_K 181 vs 2 x 71 us, ffn_down
// 193 vs 157, attn_qkv 124 vs 112); shapes done in one pass win (attn_gate 3072 rows 75 vs 81 us, ssm_beta/alpha 24
// rows 10.9 vs 15.7, q8 attn_k/v 512 rows 36 vs 52). Per step: 2 x T4 48.25 ms, wide everywhere 55.7, this rule 47.1.
static int wrm_wide_rows() {
    static const int v = [] {
        if (const char * e = getenv("P100E_MV_WRM_WROWS")) return atoi(e);
        int dev = 0, nsm = 56;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        return 64*nsm;   // 2 rounds (RR) x one block per SM x 16 warps x 2 rows
    }();
    return v;
}

// columns per launch: mv_tune.wrm_nc (P100E_MV_WRM_NC; 8 = up to 8 columns in one launch where wrm_wide_rows allows,
// 4 = the old chunking everywhere)
template <class D>
static bool wrm_t(const char * vx, const a16_args & a, int ncols, int geo, int kpg, cudaStream_t s) {
    const mv_dst d0 = a.parts ? *a.parts : mv_dst_one(a.dst, a.nrows);
    const int step = ncols > 4 && mv_tune.wrm_nc > 4 && a.nrows <= wrm_wide_rows() ? std::min(8, mv_tune.wrm_nc) : 4;
    for (int c0 = 0; c0 < ncols; c0 += step) {
        const int nc = std::min(step, ncols - c0);
        a16_args b = a;
        b.aq = a.aq + (size_t) c0*a.ngroups*16; b.ads = a.ads + (size_t) c0*a.ngroups;
        mv_dst d = d0;   // column c0 of every part
        for (int p = 0; p < d.np; ++p) d.y[p] += (size_t) c0*(d.r0[p + 1] - d.r0[p]);
        const int g = geo > 0 ? geo : wrm_default_geo(b, nc);
        bool ok = false;
        switch (g) {
#define X(ID, KIND, NW, RW, PF, NS, LP, WIDE) case ID: ok = wrm_nc<D, NW, RW, KIND, NS, PF, LP, WIDE>(vx, b, d, nc, kpg, s); break;
            WRM_GEOS(X)
#undef X
            default: ok = false;
        }
        if (!ok) return false;
    }
    return true;
}

// wrm_t for a decoder with no other kernel (dec_q6h: a16_gemv has no wrc fallback for it). The default or tuned geometry
// can refuse a shape: geometry 22 (3 columns that do not fit in shared memory) needs n_in <= 10922, so q6r3 rows of the
// 1-GPU ffn_down (n_in 17408) at T = 3 had no kernel. Every geometry gives the same bits, so retry K-phased (geo 1),
// then unstaged (geo 11, no shared memory, 1..8 columns, always runs). A retry redoes any column chunk the failed try
// already launched: same bits into the same buffers.
template <class D>
static bool wrm_t_any(const char * vx, const a16_args & a, int ncols, int geo, int kpg, cudaStream_t s) {
    return wrm_t<D>(vx, a, ncols, geo, kpg, s) || wrm_t<D>(vx, a, ncols, 1, 0, s) || wrm_t<D>(vx, a, ncols, 11, 0, s);
}

// per-layout runners (matvec_wrm_<layout>.cu): T = 1..8 columns, false = not handled
bool wrm_run_q4kr(const char * vx, const a16_args & a, int ncols, int geo, int kpg, cudaStream_t s);
bool wrm_run_q3r (const char * vx, const a16_args & a, int ncols, int geo, int kpg, cudaStream_t s);
bool wrm_run_q5kr(const char * vx, const a16_args & a, int ncols, int geo, int kpg, cudaStream_t s);
bool wrm_run_q6r3(const char * vx, const a16_args & a, int ncols, int geo, int kpg, cudaStream_t s);   // VA from the row stride
bool wrm_run_q6r2(const char * vx, const a16_args & a, int ncols, int geo, int kpg, cudaStream_t s);
// q6r3 with the dec_q6h tail (P100E_MV_Q6_H16 = 1 / 2, matvec_dec_q6h.cuh), T = 1..8
bool wrm_run_q6h1(const char * vx, const a16_args & a, int ncols, int geo, int kpg, cudaStream_t s);
bool wrm_run_q6h2(const char * vx, const a16_args & a, int ncols, int geo, int kpg, cudaStream_t s);

} // namespace p100e::fast
