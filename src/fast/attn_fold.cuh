// Long-context prefill attention as two fp16 "fold" GEMMs per key chunk (scheme of the Kmic-68 llama.cpp fork's
// fattn-gemm.cu fold path, MIT; written for p100e's layouts). Per kv head (rows n = hh * T + t, the 6 q heads of the
// group x T tokens) and key chunk (<= 2048 keys):
//   qk:  S tile = K Q^T for 128 keys x 128 rows (half2 HFMA2 chains over D, one fp32 fold), causal mask from the
//        positions (query t sees keys 0..p0+t, nothing materialised), per (tile, row) max m_t, P = 2^(S' - m_t) in
//        f16 (<= 1/8), per (tile, row) fp32 sum l_t. Fully masked tiles skip the GEMM.
//   pv:  O = O * 2^(M_old - M) + sum_t 2^(m_t - M) (V P_t), fp16 chains over the 128 keys of a tile folded into fp32
//        with the tile's factor; the running output, max and sum stay fp32 across chunks.
// The key range may be cut into nsplit independent streams (own O / m / l, merged at the end) for parallelism at
// small T. Q is prescaled by scale/4 before the f16 conversion (fp16 headroom), the logits are in log2 units.
// f16, q8_0 and q4_0 caches (chunk dequantized to f16 per call). Precision: f16 products, fp32 accumulation across tiles.
#pragma once
#include <cstddef>
#include "fast/attn.cuh"

namespace p100e::fast {

// scratch bytes for T query tokens, Hkv kv heads, key chunk `chunk` and nsplit streams
size_t attn_fold_scratch_bytes(int T, int Hkv, int chunk, int nsplit);

// Same contract as attn_prefill: q, out [T][Hq][D] f32, D == 256, Hq == 6 * Hkv, query t at position p0 + t sees cache
// positions 0..p0+t (the cache must hold 0..p0+T-1). chunk <= 0 / nsplit <= 0: automatic (2048 keys; splits so the
// PV grid has >= ~4 waves), reduced until the buffers fit in scratch_bytes. Returns false (nothing launched) for an
// unsupported cache type / shape or too little scratch.
bool attn_fold(const float * q, const kv_cache & c, float * out, int T, int p0, int Hq, float scale, void * scratch,
               size_t scratch_bytes, cudaStream_t s, int chunk = 0, int nsplit = 0);

} // namespace p100e::fast
