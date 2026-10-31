// Internal shared pieces of the fast matvec kernels (not part of the public API).
#pragma once
#include "common.cuh"      // third_party/ggml: ggml block structs, WARP_SIZE
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define FP16_AVAILABLE 1
#ifndef GGML_UNUSED_VARS
#define GGML_UNUSED_VARS(...)
#endif

#define FM_ROWS   4     // output rows per CUDA block in the HFMA2 kernels
#define FM_NWARPS 4

namespace p100e::fast {

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, mask, 32);
    }
    return x;
}

// Output of a launch over a weight concatenated from np <= 4 parts (row ranges [r0[p], r0[p+1]), r0[0] = 0,
// r0[np] = n_out): row r of column j of part p goes to y[p][j*(r0[p+1] - r0[p]) + r - r0[p]]. np = 1 is the plain
// [T][n_out] output.
struct mv_dst {
    float * y[4];
    int     r0[5];
    int     np;
};
static inline mv_dst mv_dst_one(float * y, int nrows) {
    mv_dst d{};
    d.y[0] = y; d.r0[0] = 0; d.r0[1] = nrows; d.np = 1;
    return d;
}
static __device__ __forceinline__ void mv_store(const mv_dst & d, const int j, const int row, const float v) {
    int p = 0;
#pragma unroll
    for (int k = 1; k < 4; ++k) p += k < d.np && row >= d.r0[k];
    float * y = p == 0 ? d.y[0] : p == 1 ? d.y[1] : p == 2 ? d.y[2] : d.y[3];
    const int b = p == 0 ? d.r0[0] : p == 1 ? d.r0[1] : p == 2 ? d.r0[2] : d.r0[3];
    const int e = p == 0 ? d.r0[1] : p == 1 ? d.r0[2] : p == 2 ? d.r0[3] : d.r0[4];
    y[(size_t) j*(e - b) + (row - b)] = v;
}

// All launchers: activations in the a16 layout (aq: 16 half2 per 32-element group, ads: (amax, sum/amax) half2 per group),
// columns strided by ngroups*16 / ngroups, dst columns strided by nrows. ncols = 1..4 unless noted.
struct a16_args {
    const half2 * aq;
    const half2 * ads;
    float *       dst;
    int           ngroups;     // n_in / 32
    int           nrows;       // n_out
    int           row_bytes;
    const mv_dst * parts = nullptr;   // launch_wrm only: per-part outputs instead of dst (nullptr = dst)
};

void a16_quantize(const float * x, half2 * aq, half2 * ads, int64_t s11, int nblocks, int ncols, bool vec4, cudaStream_t s);

void launch_q4r (const char * vx, const a16_args & a, int ncols /*1..8*/, cudaStream_t s);
void launch_q4kr(const char * vx, const a16_args & a, int ncols, cudaStream_t s);
void launch_q3r (const char * vx, const a16_args & a, int ncols, cudaStream_t s);
void launch_iq1s(const char * vx, const a16_args & a, int ncols /*1..4*/, cudaStream_t s);   // IQ1_S raw blocks (matvec_iq1.cu)
void launch_q10 (const char * vx, const a16_args & a, int ncols /*1..4*/, cudaStream_t s);   // Q1_0 raw blocks (matvec_q10.cu)
void launch_q20 (const char * vx, const a16_args & a, int ncols /*1..4*/, cudaStream_t s);   // Q2_0 raw blocks (matvec_q20.cu)
// grouped expert GEMV (MoE): group g = expert row set w (nrows x n_in, raw Q1_0 rows; device or mapped host memory),
// a16 column col (0.. of aq / ads, laid out as a.ngroups*16 / a.ngroups half2 per column), output column out of dst
struct mv_group { const void * w; int col; int out; };
void launch_q10_grouped (const mv_group * tab /*device, G entries*/, int G, const a16_args & a, cudaStream_t s);
void launch_q4kr_grouped(const mv_group * tab /*device, G entries*/, int G, const a16_args & a, cudaStream_t s);   // Q4_K q4kr layout
void launch_q4r_grouped (const mv_group * tab /*device, G entries*/, int G, const a16_args & a, cudaStream_t s);   // Q4_0 q4r layout
void launch_q20_grouped (const mv_group * tab /*device, G entries*/, int G, const a16_args & a, cudaStream_t s);   // Q2_0
void launch_iq2sr_grouped(const mv_group * tab /*device, G entries*/, int G, const a16_args & a, cudaStream_t s);  // IQ2_S iq2sr layout
// NC-column groups (nc 1..4): group g = a16 columns tab[g].col .. col+nc-1 (one expert's routed rows, contiguous) ->
// output columns tab[g].out .. out+nc-1; the weight rows are read once per group instead of once per column. Each
// column's bits equal the nc = 1 launch (nwarps chosen from ngroups only).
void launch_iq2sr_grouped_nc(const mv_group * tab, int G, const a16_args & a, int nc, cudaStream_t s);
void launch_q4kr_grouped_nc (const mv_group * tab, int G, const a16_args & a, int nc, cudaStream_t s);
void launch_q4r_grouped_nc  (const mv_group * tab, int G, const a16_args & a, int nc, cudaStream_t s);
void launch_iq1m(const char * vx, const a16_args & a, int ncols /*1..4*/, cudaStream_t s);   // IQ1_M raw blocks
void launch_iq3r(bool s_variant /*IQ3_S*/, const char * vx, const a16_args & a, int ncols, cudaStream_t s);   // IQ3_XXS / IQ3_S (matvec_iq3r.cu)
void launch_q2kr(const char * vx, const a16_args & a, int ncols, cudaStream_t s);   // Q2_K (matvec_q2kr.cu)
void launch_iq2xxsr(const char * vx, const a16_args & a, int ncols, cudaStream_t s);   // IQ2_XXS / IQ2_XS / IQ2_S (matvec_iq2r.cu)
void launch_iq2xsr (const char * vx, const a16_args & a, int ncols, cudaStream_t s);
void launch_iq2sr  (const char * vx, const a16_args & a, int ncols, cudaStream_t s);
bool iq2r_repack(int layout, void * d, int64_t nrows, int64_t n_in, size_t row_bytes, cudaStream_t s);
bool iq2r_to_f16(int layout, const char * base, int64_t nrows, int64_t n_in, int64_t rb, half * dst, cudaStream_t s);
void launch_q6r2(const char * vx, const a16_args & a, int ncols, int VA /*4,8,16*/, cudaStream_t s);
void launch_q5k (const char * vx, const a16_args & a, int ncols, cudaStream_t s);
void launch_iq4xs(const char * vx, const a16_args & a, int ncols, cudaStream_t s);
void launch_q41 (const char * vx, const a16_args & a, int ncols /*1..8*/, cudaStream_t s);   // Q4_1 raw blocks (matvec_q41.cu)
void launch_q8r_f32(const char * w, const float * x, float * y, int T, int64_t n_in, int64_t n_out, size_t row_bytes, cudaStream_t s);
bool type_info(uint32_t type, int & blck, int & bytes);   // elements per block, bytes per block

// generic: any dequantizable type, f32 activations x[T][n_in], y[T][n_out]
bool generic_supported(uint32_t type);
int64_t generic_group(uint32_t type);   // n_in must be a multiple of this
void launch_generic(uint32_t type, const char * w, const float * x, float * y, int T, int64_t n_in, int64_t n_out,
                    size_t row_bytes, cudaStream_t s);

// T = 1 kernels tuned for the P100 shard shapes (matvec_wr.cu).  variant: see the tables there; false = not handled.
bool launch_wr(int layout, const char * vx, const a16_args & a, int variant, cudaStream_t s);
// K-split kernel for T = 1..4 columns on the layouts that have no legacy kernel (Q6R3, Q5KR); also used by the
// default T = 1 choice for tiny n_out.
bool launch_ks_cols(int layout, const char * vx, const a16_args & a, int ncols, cudaStream_t s);
bool launch_q8_t1(const char * w, const float * x, float * y, int64_t n_in, int64_t n_out, size_t row_bytes, int variant,
                  cudaStream_t s);
// Q8R, T = 2..4 columns, n_out <= 2048 (the MTP verify's ssm_alpha/beta and attn_k/v shards): q8_wr's warp-rows
// geometry with T accumulators per row, the same per-row sums as q8r_gemv / q8_wr; false = not handled
bool launch_q8_tn(const char * w, const float * x, float * y, int T, int64_t n_in, int64_t n_out, size_t row_bytes,
                  cudaStream_t s);
// T = 1 with the activation quantize fused into the GEMV (x = f32 activations); variants 301.. / 401..
bool launch_fused_t1(int layout, const char * vx, const float * x, float * y, int ngroups, int nrows, int row_bytes,
                     int variant, cudaStream_t s);
// T = 1 persistent warp rows on a prequantized activation (a.aq / a.ads); variants 501..524 (= the 301..324 geometry)
bool launch_wrq(int layout, const char * vx, const a16_args & a, int variant, cudaStream_t s);
// the inline-PTX twin of the wrp / wrq kernels (matvec_wrx.cu): variants 303/306/314/320 and their 5xx twins, when the
// type's switch is on (wrx_level(type) > 0; type 0..3 = q3_K, q4_K, q5_K, q6_K); bit-identical rows; false = not handled
bool launch_wrpx(int layout, const char * vx, const float * x, const half2 * aq, const half2 * ads, float * y, int ngroups,
                 int nrows, int row_bytes, int variant, cudaStream_t s);
int wrx_level(int type_idx);
// one launch for a wrq weight (variant 5xx, a.aq / a.ads) and a Q8R weight on f32 x8 (decode step 6a, matvec_wr.cu);
// both outputs bit-identical to their own launches; false = not handled (wrx twin on, PF geometry, shape)
bool launch_wrq_q8(int layout, const char * vx, const a16_args & a, int variant, const char * w8, const float * x8,
                   float * y8, int64_t n_in8, int64_t n8, size_t rb8, cudaStream_t s);
// warp rows over 1..8 activation columns, up to 4 per pass (column j at a.aq + j*ngroups*16, a.ads + j*ngroups, output
// column j at a.dst + j*nrows); every column's rows are bit-identical to the T = 1 warp-rows kernels (wr / wrp / wrq)
bool launch_wrc(int layout, const char * vx, const a16_args & a, int ncols, cudaStream_t s);
// the same contract as launch_wrc (column j bit-identical to the T = 1 warp-rows kernels), with every weight decoded
// once for all columns (matvec_wrm.cu); geometry from mv_tune.wrm_geo (0 = default by shape); false = not handled
bool launch_wrm(int layout, const char * vx, const a16_args & a, int ncols, cudaStream_t s);
const char * wrm_geo_name(int g);
// Q8R, T = 2..4 columns, any n_out: warp rows with U groups of loads in flight per lane (matvec_wrm.cu), the same
// per-row sums as q8r_gemv / q8_wr / q8_wrt; geometry from mv_tune.q8_geo (0 = default); false = not handled
bool launch_q8_wrm(const char * w, const float * x, const mv_dst & d, int T, int64_t n_in, int64_t n_out, size_t row_bytes,
                   cudaStream_t s);
const char * q8m_geo_name(int g);
const char * fused_variant_name(int v);
int wr_variant_count();
const char * wr_variant_name(int v);
const char * ks_variant_name(int v);
const char * q8_variant_name(int v);

// benchmark / tuning hook (host global, default {0}): variant -1 = always the legacy K-split kernels (every T), 0 = the
// default per-shape choice (matvec.cu), > 0 = force that variant id for T = 1 where it applies (ids 101..105 / 201..204
// / 401..404 are K-split kernels: their rows differ in the last bits from the warp-rows family, debug only).
struct mv_tune_t {
    int  variant    = 0;
    bool skip_quant = false;   // bench only: reuse the a16 activation already in scratch (time the GEMV alone)
    // multi-column warp rows (T >= 2): wrm = 1 the shared-dequant kernel (matvec_wrm.cu), 0 = wrcp (P100E_MV_WRM);
    // wrm_geo = geometry id (0 = default by shape, P100E_MV_WRM_GEO); wrm_kpg = groups per K phase (0 = auto)
    int  wrm = 1, wrm_geo = 0, wrm_kpg = 0;
    // K-phased wrm (n_in 8704 at 3-4 columns): tile rounds per pass over the phases, 0 = auto (<= 4), 1 = old restaging (P100E_MV_WRM_PH)
    int  wrm_ph = 0;
    // columns per wrm launch for T > 4 (the batched multi-session decode step, up to 8 rows): 8 = one weight pass
    // (P100E_MV_WRM_NC), 4 = the old two launches; wrm_wide_geo = geometry for 5..8 columns (0 = geo 1, P100E_MV_WRM_WGEO)
    int  wrm_nc = 8, wrm_wide_geo = 0;
    // Q8R at T = 2..4: q8_geo = 0 the default q8_wrm geometry, > 0 a geometry id, -1 = q8_wrt (P100E_MV_Q8_GEO)
    int  q8_geo = 0;
    // Q8R at T = 1 on shards of more than 2048 rows and n_in <= 5120 (ssm_out 3072 -> 5120, a Q8_0 model's 5120-wide weights):
    // the T = 1 variant id, 215 (q8wr nw4 rw2 staged) by default, 0 = the legacy q8r_gemv (P100E_MV_Q8_BIG); every q8wr
    // id (211..218) gives the legacy kernel's bits
    int  q8_big = 215;
    // inline-PTX mv_wrp (matvec_wrx.cu) per type q3_K, q4_K, q5_K, q6_K: 0 = off, 1 / 2 = on (groups per lane per step);
    // -1 = from the environment (P100E_MVWRP_Q3 / _Q4 / _Q5 / _Q6, default off)
    int  wrx[4] = {-1, -1, -1, -1};
    // Q6_K (q6r3) fp32 tail of the HFMA2 dot (P100E_MV_Q6_H16, matvec_dec_q6h.cuh): 0 = the dec_q6r3 tail,
    // 1 = lane fold, 2 = scale fold (default). 1 / 2 change the numerics vs 0 and run EVERY q6r3 GEMV, T = 1 included, on mv_wrm, so the
    // rows stay T-invariant
    int  q6h = 2;
};
extern mv_tune_t mv_tune;
// the T = 1 variant matvec() picks by default for a weight with f32 activations (0 = legacy kernel)
int default_t1_variant(int layout, uint32_t type, int64_t n_in, int64_t n_out);
// the T = 1 variant matvec_prequantized() picks by default (a16 layouts; 0 = legacy kernel)
int default_pq_variant(int layout, uint32_t type, int64_t n_in, int64_t n_out);

// repack (in place, one CUDA block per row)
bool repack_launch(int layout, void * d, int64_t nrows, int64_t n_in, size_t row_bytes, cudaStream_t s);

} // namespace p100e::fast
