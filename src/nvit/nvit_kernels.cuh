// nvit CUDA kernels (see nvit.h). Patches are in clip's 2x2 merge order everywhere: patch p belongs to merged token
// g = p / 4 (raster over the (ny/32) x (nx/32) token grid) at sub-position (p % 4) = (dy, dx) = ((p%4)/2, (p%4)%2).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace p100e::nvit {

constexpr int E = 1152, NH = 16, HD = 72, FF = 4304, MERGE_IN = 4608, OUT = 5120, PATCH_K = 768, POS_SIDE = 48;

// patches p0 .. p0+np-1 of a planar [3][ny][nx] image -> A [np][768] (k = ic*256 + kh*16 + kw, the conv kernel order)
void im2col_merge(const float * img, int nx, int ny, int p0, int np, float * A, cudaStream_t s);

// x[i][c] = (x[i][c] + bias[c]) + pos(c, patch p0 + i): the 48x48 learned grid bilinearly resized with align corners to
// the (nx/16) x (ny/16) patch grid exactly as ggml-cuda's upscale kernel (float math).
void add_bias_pos(float * x, const float * bias, const float * pos, int pw, int ph, int p0, int np, cudaStream_t s);

// ROWS consecutive rows per output row (1: LayerNorm per patch; 4: post-LN of a merge group concatenated to 4*E):
//   x[r] = x[r] + (add[r] + badd)          (add / badd may be null; x written back when add != null)
//   v    = (x[r] - mean) * rsqrt(var + eps) * w + b    (two-pass mean / variance)
//   x16[g][...] = f16(v * 2^-e), cs[g] = 2^e with max|v| * 2^-e in [8, 16) (fast::fold_gemm's activation format)
// rows = number of input rows (multiple of ROWS). x16 == null: only the add. y32 (optional): v in f32 [rows][E].
void add_ln_prescale(int ROWS, float * x, const float * add, const float * badd, const float * w, const float * b,
                     float eps, int rows, half * x16, int64_t ldx, float * cs, float * y32, cudaStream_t s);

// Attention operands are f16 in 64-row tiles (u32 = two halves, 2304 u32 = 9216 B per tile and head):
//   Q, K: [tile][d2 = 0..35][r = 0..63] = (x[64 tile + r][2 d2], x[64 tile + r][2 d2 + 1])
//   V:    [tile][k2 = 0..31][d = 0..71] = (v[64 tile + 2 k2][d], v[64 tile + 2 k2 + 1][d])
// Head h of a buffer starts at h * hs (u32). Rows past the last valid one are zero.
constexpr int TILE = 64, TILE_U32 = 2304;

// y [T][3*E] (q | k | v rows of the qkv matmul for patches p0 .. p0+T-1, patch grid width pw) + bias ->
//   Q = rope(q) * qscale (local query rows qrow0 ..), K = rope(k), V = v (global rows p0 ..), tiled as above.
// clip's 2D vision rope (ggml rope_multi VISION, sections 18/18, theta 10000): dims j and j+36 rotate together, pairs
// 0..17 by the patch row, 18..35 by the patch column. p0 and qrow0 are multiples of 64; the last tile of the call is
// zero-filled past T.
void qkv_rope(const float * y, const float * bias, int T, int p0, int pw, float qscale, uint32_t * qt, int64_t q_hs,
              int qrow0, uint32_t * kt, uint32_t * vt, int64_t kv_hs, cudaStream_t s);

// v = gelu_tanh(y[t][k] + bias[k]) (ggml's formula) for k < K, prescaled into x16 [T][Kpad] (zeros in K..Kpad-1)
void bias_gelu_prescale(const float * y, int64_t ldy, const float * bias, int T, int K, int Kpad, half * x16,
                        int64_t ldx, float * cs, cudaStream_t s);

// y[t][n] += bias[n]
void add_bias(float * y, const float * bias, int T, int n, cudaStream_t s);

// full (maskless) attention, head dim 72: local queries qrow0 .. qrow0+nq-1 (tiled Q, pre-scaled by the softmax scale
// * log2(e); qrow0 % 64 == 0) against keys / values 0 .. N-1 (tiled K / V); out[i][h * 72 + d] f32 (row stride ldo) for
// heads 0..nh-1. 64 x 64 tiles per block of 128 threads: S = Q K^T with half2 products (f16 chains of 18 dim pairs folded
// into fp32), fp32 online softmax, P rounded to f16, O += P V with f16 chains over the 64 keys of a tile folded into fp32.
// Split key range (two GPUs: own keys while the peer's K / V are still in flight): phase 1 over key tiles [kt0, kt1)
// writes the running max / sum per (query, head) to st [nq][16][2] and the unnormalised output to out; phase 2 resumes
// from them over another key range and normalises. phase 0: one pass (kt0 = 0, kt1 = -1: all tiles).
void attn72(const uint32_t * qt, int64_t q_hs, int qrow0, int nq, const uint32_t * kt, const uint32_t * vt, int64_t kv_hs,
            int N, int nh, float * out, int64_t ldo, cudaStream_t s, int kt0 = 0, int kt1 = -1, int phase = 0,
            float * st = nullptr);

// ---- fp32 oracle (debug reference path, encoder::opts::oracle) ----
// y [T][3*E] + bias -> q, k, v f32 [h][N][72] at rows p0.. (rope as qkv_rope, q not scaled)
void qkv_rope_f32(const float * y, const float * bias, int T, int p0, int pw, float * q, float * k, float * v, int64_t N,
                  cudaStream_t s);
// fp32 attention (one warp per query, expf softmax): out[i][h*72 + d] for queries 0..N-1, scale = 1/sqrt(72)
void attn_f32(const float * q, const float * k, const float * v, int N, float * out, cudaStream_t s);
// y[t][k] = gelu_tanh(y[t][k] + bias[k]) in place
void bias_gelu_f32(float * y, const float * bias, int T, int K, cudaStream_t s);

// f16 rows -> f32 (debug dumps): y[t][k] = x16[t][k] * cs[t]
void unprescale(const half * x16, int64_t ldx, const float * cs, int T, int K, float * y, cudaStream_t s);

} // namespace p100e::nvit
