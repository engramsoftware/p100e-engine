// Decode attention (GQA, split-K) with f16 / q8_0 / q4_0 KV caches for the Qwen3.8 full-attention layers.
// Supported shape: D == 256, Hq == 6 * Hkv, T = 1..4 query tokens (MTP verify), causal.
// Cache layout: [n_ctx][Hkv][D] per K and V. f16 = plain halves; q4_0 = ggml block_q4_0
// (fp16 scale d, 16 bytes: byte i low nibble = value i, high nibble = value i+16; value = (nib-8)*d),
// 8 blocks per (position, kv head); q8_0 = ggml q8_0 quantization (per 32 values: f16 scale d = amax/127,
// int8 q = round(x/d)), stored per (position, kv head) as the 8 scales then the 256 int8 (272 bytes, the same size
// as 8 ggml blocks, 16-byte aligned quants; fast/kv_q8.cuh).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace p100e::fast {

constexpr uint32_t KV_F16  = 1;   // == GGML_TYPE_F16
constexpr uint32_t KV_Q4_0 = 2;   // == GGML_TYPE_Q4_0
constexpr uint32_t KV_Q8_0 = 8;   // == GGML_TYPE_Q8_0 (quantization; in-row layout above)

struct kv_cache {
    uint32_t type = KV_F16;
    int n_ctx = 0, Hkv = 0, D = 0;   // n_ctx: positions per sequence (paged: the per-sequence cap)
    void * k = nullptr;
    void * v = nullptr;
    // Paged layout (pt != nullptr): the cache of a sequence is a list of pages of 2^psh positions in a
    // shared pool. Position p of query row t lives at k / v + pt[(rtab ? rtab[t] : 0) + (p >> psh)] * pstride +
    // (p & (2^psh - 1)) * kv_row_bytes (pt = device page table(s), rtab = device per-row offsets into pt: rows of
    // different sequences in one call). pt == nullptr: contiguous [n_ctx][row] as before. Only addresses change, never
    // the arithmetic: every kernel gives the paged cache the bits of the contiguous one.
    const int32_t * pt = nullptr;
    const int32_t * rtab = nullptr;
    int psh = 0;
    size_t pstride = 0;
};

size_t kv_row_bytes(uint32_t type, int Hkv, int D);           // bytes of one position (all kv heads), K or V
kv_cache kv_alloc(uint32_t type, int n_ctx, int Hkv, int D);  // cudaMalloc, returns k == nullptr on failure
void kv_free(kv_cache &);
// k, v: [T][Hkv][D] f32 -> cache positions p0..p0+T-1 (converted / quantized)
void kv_append(kv_cache &, const float * k, const float * v, int p0, int T, cudaStream_t s);
// q, out: [T][Hq][D] f32. Query t sits at position p0+t and sees positions 0..p0+t. scale applied to q.k.
// Uses a per-device scratch buffer (allocated on the first call at its largest size, MAX_SPLIT x T = 4, so the
// pointer never changes afterwards: do a warm-up call before graph capture); calls on different streams of one
// device must not overlap. Returns false (out untouched) when no kernel could run: unsupported shape, paged cache
// that attn_decode2 refused, or a scratch allocation failure; the caller must not use out then.
bool attn_decode(const float * q, const kv_cache &, float * out, int T, int p0, int Hq, float scale, cudaStream_t s);

// Prefill attention, any T >= 1 (meant for T > 4), f16 / q8_0 / q4_0 cache (returns false for other cache types / shapes
// so the caller can fall back to attn_decode). Same contract as attn_decode: q, out [T][Hq][D] f32, query t at
// position p0+t sees cache positions 0..p0+t (cache must already hold positions 0..p0+T-1). fp32 math on the f16
// cache, online softmax over 32-key tiles; one block per (2 query tokens x 6 q heads of one kv head), no scratch.
bool attn_prefill(const float * q, const kv_cache &, float * out, int T, int p0, int Hq, float scale, cudaStream_t s);
// the same with an explicit kernel variant (0 = attn_prefill_k, 1..3 = attn_pf2_k RT 4 / 2 / 1; all bitwise equal)
bool attn_prefill_variant(const float * q, const kv_cache &, float * out, int T, int p0, int Hq, float scale, cudaStream_t s,
                          int variant);

// CUDA-graph friendly variants: the first position p0 is read from device memory (*p0_dev) when the kernels run, so
// a captured graph stays valid while n_past grows. attn_decode_dev sizes its grid for p0 + T <= max_n; blocks past the
// split count of the actual n exit at once. Same split plan and summation order as attn_decode (bit-identical output).
void kv_append_dev(kv_cache &, const float * k, const float * v, const int32_t * p0_dev, int T, cudaStream_t s);
bool attn_decode_dev(const float * q, const kv_cache &, float * out, int T, const int32_t * p0_dev, int max_n, int Hq,
                     float scale, cudaStream_t s);   // false: as attn_decode
// Multi-row decode: T = 1..8 query rows, row t at position pos[t] (device) of its own sequence (paged
// cache: page table pt + rtab[t]); row t sees that sequence's positions 0..pos[t]. Each row is bitwise the T = 1
// attn_decode call at pos[t] on its sequence (fast/attn_dec2.cu). Returns false for an unsupported shape / type.
bool attn_decode_rows(const float * q, const kv_cache &, float * out, int T, const int32_t * pos, int Hq, float scale,
                      cudaStream_t s);
constexpr int ATTN_MAX_ROWS = 8;   // rows per attn_decode_rows call

// Benchmark only: finite pseudo-random cache rows at positions p0..p1-1 of K and V (f16 values in [-1, 1); q8_0 and
// q4_0 random quants with a small fixed scale). Attention cost does not depend on the values.
void kv_fill_random(kv_cache &, int p0, int p1, uint32_t seed, cudaStream_t s);


} // namespace p100e::fast
