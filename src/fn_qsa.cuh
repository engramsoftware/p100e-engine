// Flash-Next QSA (Qwen sparse attention): the lightning indexer picks, per query, a budget of whole blocks of `ratio`
// tokens (plus the incomplete tail, always visible); attention then runs over just those cells. Mirrors the llama.cpp
// reference (the fork's src/models/qwen4exp.cpp build_qsa_top_k / build_attn_qsa, gather path):
//  - raw indexer keys k_proj(x) [D] per token (cached as f16 there -> rounded to f16 here); a block's key is the mean of
//    its `ratio` raw keys, rms-normed (indexer k_norm), roped at the block's first position
//  - query: q_proj(x) [H][D], rms-normed per head, roped at the token position; score(block) = sum_h relu(q_h . k_b)
//  - visibility for query position q: blocks fully before tail_start = (q+1)/ratio*ratio score; the tail cells
//    [tail_start, q] are always in; width = top_k + ratio - 1 cells; ties: lower block first; the last selected block
//    may be partial (its lowest cells)
//  - attention over the selected cells with the usual softmax / scale (the gathered mask is 0 for every selected cell)
// All in f32, single stream, contiguous positions (cell index == position).
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace p100e::qsa {

// round in place to f16 precision (the reference caches the raw keys as f16)
void round_f16(float * x, int64_t n, cudaStream_t s);

// block keys for blocks [b0, b1): mean over the ratio raw keys of each block. The key of position p comes from
// ring[p % ring_n] when p < n_past (carried over from earlier chunks) else chunk[p - n_past]. out: [b1 - b0][D]
void pool_blocks(const float * ring, const float * chunk, int n_past, int b0, int b1, int ratio, int ring_n, int D, float * out, cudaStream_t s);

// keep the raw keys of the last ring_n positions of [n_past, n_past + T) in ring[p % ring_n]
void save_tail(const float * chunk, int n_past, int T, int ring_n, int D, float * ring, cudaStream_t s);

// combine the per-head block scores: S [m * H][NB] (row t*H + h) -> c [m][NB]: sum_h relu(S) for blocks fully before
// the tail of query pos[t], -inf otherwise
void combine(const float * S, float * c, const int32_t * pos, int m, int H, int NB, int ratio, cudaStream_t s);

// per row t: select the top blocks of c[t][0..NB) and write the visible cell list sel[t][0..n_t) (tail cells, then the
// chosen blocks' cells, the last chosen block possibly partial), n_t -> seln[t]; W = top_k + ratio - 1 (sel row stride)
void select(const float * c, int NB, const int32_t * pos, int m, int ratio, int W, int32_t * sel, int32_t * seln, cudaStream_t s);

// attention over the gathered cells: q [T][Hq][D], caches [n_ctx][Hkv][D] (f32 or f16: kv_f16), sel/seln as above,
// out [T][Hq][D]. D == 256; heads processed 4 per block (share one K/V read; 4 must divide Hq / Hkv)
void attention(const float * q, const void * kc, const void * vc, int kv_mode, const int32_t * sel, const int32_t * seln, float * out,
               int T, int Hq, int Hkv, int D, int W, float scale, cudaStream_t s);
// KV storage modes: 0 f32, 1 f16, 2 q8 (blocks of 32: half scale + 2 pad + 32 int8 = 36 B; 1.125 B per value)
size_t kv_row_bytes(int kv_mode, int D);   // bytes per (cell, kv head) row
void kv_store(int kv_mode, const float * k, const float * v, void * kc, void * vc, int T, int p0, int Hkv, int D, cudaStream_t s);
void kv_window_f32(int kv_mode, const void * kc, const void * vc, float * kf, float * vf, int N, int Hkv, int D, cudaStream_t s);

// f16 KV cache: store the batch's f32 K/V rows at positions p0..p0+T-1; read rows [0, N) back as f32 (dense paths)
void kv_store_f16(const float * k, const float * v, void * kc, void * vc, int T, int p0, int row, cudaStream_t s);
void kv_window_f32(const void * kc, const void * vc, float * kf, float * vf, int N, int row, cudaStream_t s);

// block positions b * ratio for rope, filled once: pos [NB]
void block_positions(int32_t * pos, int NB, int ratio, cudaStream_t s);

} // namespace p100e::qsa
