// p100e fast: "wrm" - warp rows over NC = 1..8 activation columns with the weight loads AND the dequant shared by the
// columns (the MTP verify batches T = 2..4, prefill chunks of <= 8 columns), and q8_wrm, the q8_0 counterpart.
//
// Why: the P100 (GP100) issues one warp instruction per cycle on each of its 2 schedulers per SM, and the decode GEMVs
// are close to that limit already at T = 1. wrcp (matvec_wr.cu) runs dot() once per (row, column): the compiler shares
// the half2 dequant across the columns but re-runs the scale/header decode per column and re-derives the 64-bit row
// addresses every iteration (~219 instructions per row and 32-group step at T = 3 vs ~121 at T = 1, SASS of
// mv_wrcp<q4kr, 16, 2, 3>), so T = 3 cost ~1.6x T = 1 although the weights are read once.
// Here every (row, group) is decoded once: prep() (scales / bias) once per row, then chunk by chunk (4 half2 weights
// at a time) the HFMA2s of every column, then fin() per (row, column). Per (row, column, group) that is the 16 HFMA2
// and the fp32 epilogue, nothing else (mv_wrm<q4kr, 16, 2, 4>: ~180 instructions per row-group step vs ~260).
//
// Bit identity: per (row, column) the fp16 partial sums se (even slots 0, 2, .., 14) and so (odd slots) are the same
// HFMA2 chains in the same order as dot(), and fin() is dot()'s fp32 expression (matvec_dec.cuh), so every group adds
// the same value; a lane walks its groups t = lane, lane + 32, ... in order and the row closes with the same xor warp
// tree. Column j of a call therefore equals the T = 1 warp-rows kernels (wr / wrp / wrq) on column j, bit for bit,
// whatever NC, geometry, staging mode or phase split.
//
// Activation staging (shared memory, group-major records, see wrm_rec):
//   MODE 1: all groups of the NC columns staged once per block, persistent row tiles (as wrcp);
//   MODE 2: K phases of kpg groups (a multiple of 32, so a lane's group order is unchanged): per round of tiles the
//           block stages phase by phase (2 syncs per phase) - n_in 8704 (ffn_down) at 3-4 columns, which do not fit
//           in 48 KB (wrcp: 2 + 2 columns = the weights read twice);
//   MODE 3: hybrid - the first NS columns staged (all groups), the others read through the read-only cache, every
//           block's warps walking the same groups at about the same time (no syncs after the staging);
//   MODE 0: no staging, activations through the read-only cache, one tile per warp: tiny n_out (the 24-row
//           ssm_alpha / ssm_beta shards: wrcp put all 24 rows on ONE block of 16 warps).
// Output: mv_dst, so one launch over a concatenated weight (gate|up, qkv|gate|beta|alpha) can write every part to its
// own [T][n_part] buffer (matvec_prequantized_parts).
#include "matvec_wrm.cuh"

namespace p100e::fast {

const char * wrm_geo_name(int g) {
    static char buf[64];
    switch (g) {
#define X(ID, KIND, NW, RW, PF, NS, LP, WIDE) case ID: \
        snprintf(buf, sizeof buf, "wrm %s nw%d rw%d pf%d ns%d lp%d", KIND == 1 ? "st" : KIND == 0 ? "ldg" : "hyb", NW, RW, \
                 (int) PF, KIND == 3 ? NS : KIND == 1 ? 4 : 0, LP); \
        return buf;
        WRM_GEOS(X)
#undef X
        default: return "?";
    }
}

// T = 1..8 columns (up to 4 per launch) on the warp-rows layouts; false = not handled (the caller falls back to wrcp)
bool launch_wrm(int layout, const char * vx, const a16_args & a, int ncols, cudaStream_t s) {
    const int geo = mv_tune.wrm_geo, kpg = mv_tune.wrm_kpg;
    switch (layout) {
        case LAYOUT_Q4KR: return wrm_run_q4kr(vx, a, ncols, geo, kpg, s);
        case LAYOUT_Q3R:  return wrm_run_q3r(vx, a, ncols, geo, kpg, s);
        case LAYOUT_Q5KR: return wrm_run_q5kr(vx, a, ncols, geo, kpg, s);
        case LAYOUT_Q6R2: return wrm_run_q6r2(vx, a, ncols, geo, kpg, s);
        case LAYOUT_Q6R3:
            // P100E_MV_Q6_H16 = 1 / 2: the dec_q6h tail (matvec_dec_q6h.cuh); a16_gemv routes T = 1 here too
            if (mv_tune.q6h == 1) return wrm_run_q6h1(vx, a, ncols, geo, kpg, s);
            if (mv_tune.q6h == 2) return wrm_run_q6h2(vx, a, ncols, geo, kpg, s);
            return wrm_run_q6r3(vx, a, ncols, geo, kpg, s);
        default: return false;
    }
}

// ------------------------------------------------------------------------------------------------- q8_0, T columns
// Q8_0 repacked row (matvec.cu): [qs plane 32*nb][d plane nb halves]; chunk c = 16 weights = one uint4 at 16c, scale
// d[c >> 1]. Per (row, column) exactly the sums of q8r_gemv / q8_wr / q8_wrt: lane l takes the chunks l, l + 32, ...
// in order, acc = fma(d, chunk FMA chain, acc), then the xor warp tree - so column j equals the T = 1 kernels on
// column j. The q8_0 verify shards (24-row ssm_alpha / ssm_beta, 512-row attn_k / attn_v) are latency bound at
// T >= 2 (a few warps per SM, one DRAM round trip per chunk step): U chunk steps of weight loads are issued at once.
static __device__ __forceinline__ float q8m_chunk(const uint4 q, const float4 (&xv)[4]) {   // = q8_chunk (matvec_wr.cu)
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

template <int NW, int RW, int T, int U>
static __global__ void __launch_bounds__(WARP_SIZE*NW)
q8_wrm(const char * __restrict__ w, const float * __restrict__ x, const mv_dst dst, const int nb, const int n_in,
       const int n_out, const int row_bytes) {
    const int lane = threadIdx.x;
    const int nch  = 2*nb;
    const int row0 = (blockIdx.x*NW + threadIdx.y)*RW;
    if (row0 >= n_out) return;
    const char * xr[RW];
#pragma unroll
    for (int i = 0; i < RW; ++i) xr[i] = wrm_pin(w + (size_t) min(row0 + i, n_out - 1)*row_bytes);
    float acc[RW][T];
#pragma unroll
    for (int i = 0; i < RW; ++i)
#pragma unroll
        for (int j = 0; j < T; ++j) acc[i][j] = 0.0f;
    for (int c0 = lane; c0 < nch; c0 += WARP_SIZE*U) {
        uint4 q[U][RW];
        unsigned short db[U][RW];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int c = c0 + WARP_SIZE*u;
            if (U == 1 || c < nch) {
#pragma unroll
                for (int i = 0; i < RW; ++i) {
                    q[u][i]  = __ldg((const uint4 *) (xr[i] + 16*c));
                    db[u][i] = __ldg((const unsigned short *) (xr[i] + 32*nb + 2*(c >> 1)));
                }
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int c = c0 + WARP_SIZE*u;
            if (U == 1 || c < nch) {
#pragma unroll
                for (int j = 0; j < T; ++j) {
                    float4 xv[4];
                    const float4 * xp = (const float4 *) (x + (size_t) j*n_in + 16*c);
#pragma unroll
                    for (int k = 0; k < 4; ++k) xv[k] = __ldg(xp + k);
#pragma unroll
                    for (int i = 0; i < RW; ++i) {
                        const float d = __half2float(*((const half *) &db[u][i]));
                        acc[i][j] = fmaf(d, q8m_chunk(q[u][i], xv), acc[i][j]);
                    }
                }
            }
        }
    }
    float mine = 0.0f;   // see mv_wrm: lane i*T + j keeps and stores the (row i, column j) sum
#pragma unroll
    for (int i = 0; i < RW; ++i)
#pragma unroll
        for (int j = 0; j < T; ++j) {
            const float v = warp_reduce_sum(acc[i][j]);
            if (lane == i*T + j) mine = v;
        }
    const int i = lane / T, j = lane - (lane / T)*T;
    if (lane < RW*T && row0 + i < n_out) {
        if (dst.np == 1) dst.y[0][(size_t) j*n_out + row0 + i] = mine;
        else             mv_store(dst, j, row0 + i, mine);
    }
}

// q8 geometry ids (mv_tune.q8_geo): id -> (NW, RW, U)
#define Q8M_GEOS(X) \
    X(31, 4, 2, 1) X(32, 4, 2, 2) X(33, 4, 2, 4) X(34, 4, 1, 2) X(35, 4, 1, 4) X(36, 8, 1, 2) X(37, 2, 2, 2) \
    X(38, 1, 1, 2) X(39, 1, 1, 4) X(40, 2, 1, 4) X(41, 8, 2, 2) X(42, 1, 1, 8) X(43, 2, 4, 1) X(44, 2, 4, 2)     X(45, 4, 4, 1) X(46, 1, 4, 2) X(47, 2, 3, 2)

const char * q8m_geo_name(int g) {
    static char buf[48];
    switch (g) {
#define X(ID, NW, RW, U) case ID: snprintf(buf, sizeof buf, "q8wrm nw%d rw%d u%d", NW, RW, U); return buf;
        Q8M_GEOS(X)
#undef X
        default: return "?";
    }
}

template <int NW, int RW, int U>
static bool q8m_launch(const char * w, const float * x, const mv_dst & d, int T, int nb, int n_in, int n_out, int rb,
                       cudaStream_t s) {
    const dim3 grid((n_out + NW*RW - 1)/(NW*RW)), blk(WARP_SIZE, NW);
    switch (T) {
        case 2: q8_wrm<NW, RW, 2, U><<<grid, blk, 0, s>>>(w, x, d, nb, n_in, n_out, rb); return true;
        case 3: q8_wrm<NW, RW, 3, U><<<grid, blk, 0, s>>>(w, x, d, nb, n_in, n_out, rb); return true;
        case 4: q8_wrm<NW, RW, 4, U><<<grid, blk, 0, s>>>(w, x, d, nb, n_in, n_out, rb); return true;
        // 5..8: the batched multi-session step (two MTP chats x 4 rows) in one weight pass; lane i*T + j keeps the
        // (row, column) sum, so RW*T <= 32 (every Q8M geometry has RW <= 4)
        case 5: q8_wrm<NW, RW, 5, U><<<grid, blk, 0, s>>>(w, x, d, nb, n_in, n_out, rb); return true;
        case 6: q8_wrm<NW, RW, 6, U><<<grid, blk, 0, s>>>(w, x, d, nb, n_in, n_out, rb); return true;
        case 7: q8_wrm<NW, RW, 7, U><<<grid, blk, 0, s>>>(w, x, d, nb, n_in, n_out, rb); return true;
        case 8: q8_wrm<NW, RW, 8, U><<<grid, blk, 0, s>>>(w, x, d, nb, n_in, n_out, rb); return true;
        default: return false;
    }
}

// default q8 geometry (P100 sweep)
static int q8m_default_geo(int64_t n_out, int T) {
    // the f32 activation reads (through L1/L2) dominate, more rows per warp share them
    // (512-row attn_k at T = 2 / 3 / 4: q8_wrt 25.7 / 34.0 / 40.3 us, nw2 rw3 u2 18.1 / 21.8 / 28.4 us);
    // tiny shards: one warp per row, 4 chunk steps of loads in flight
    if (n_out <= 64) return 39;
    // T = 5..8 (two batched MTP chats): the 512-row attn_k / attn_v shards want 4 rows per warp, one warp per block
    // (T = 8: nw1 rw4 u2 36.0 us vs nw2 rw3 u2 43.0)
    if (T > 4) return 46;
    return 47;
}

bool launch_q8_wrm(const char * w, const float * x, const mv_dst & d, int T, int64_t n_in, int64_t n_out, size_t row_bytes,
                   cudaStream_t s) {
    const int nb = (int) (n_in/32), no = (int) n_out, rb = (int) row_bytes;
    if (T < 2 || T > 8 || n_in % 256 != 0 || row_bytes % 16 != 0 || ((uintptr_t) x) % 16 != 0) return false;
    const int g = mv_tune.q8_geo > 0 ? mv_tune.q8_geo : q8m_default_geo(n_out, T);
    switch (g) {
#define X(ID, NW, RW, U) case ID: return q8m_launch<NW, RW, U>(w, x, d, T, nb, (int) n_in, no, rb, s);
        Q8M_GEOS(X)
#undef X
        default: return false;
    }
}

} // namespace p100e::fast
