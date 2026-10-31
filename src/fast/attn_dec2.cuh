// Decode attention v2 for T = 1..4 query tokens (MTP verify), D = 256, Hq = 6 * Hkv, f16 / q8_0 / q4_0 cache
// (fast/attn_dec2.cu). Same contract as fast::attn_decode: q, out [T][Hq][D] f32, query t at position p0 + t sees
// cache positions 0..p0+t, scale applied to q.k. p0_dev != nullptr: p0 is read on the device (CUDA graph replays).
// The grid and the key partition never depend on T, p0 or the context length, and token t's block runs exactly the
// T = 1 computation at p0 + t: row t of a T-token call is bitwise equal to the T = 1 call at any depth.
// Used by attn_decode / attn_decode_dev unless P100E_ATTN_DEC=1 (the v1 kernels).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "fast/attn.cuh"

namespace p100e::fast {

// returns false (nothing launched) for an unsupported shape / type or a failed scratch allocation
// pos_rows != nullptr: row t sits at position pos_rows[t] (device) instead of p0 + t, and reads its own
// sequence's pages (c.rtab); T <= ATTN_MAX_ROWS (8). Paged caches (c.pt) need pages of >= 32 positions.
bool attn_decode2(const float * q, const kv_cache & c, float * out, int T, int p0, const int32_t * p0_dev, int Hq,
                  float scale, cudaStream_t s, const int32_t * pos_rows = nullptr);
int attn_dec2_nsplit(int Hkv);   // fixed split-stream count of this device (P100E_DEC2_SPLIT overrides)
void attn_dec2_set_nsplit(int n);   // benchmarks only: override the split count (0 = default); never between graph captures
bool attn_dec2_reload_env();       // re-read P100E_DEC2_LEAN; true if it changed (captured graphs must be dropped)

} // namespace p100e::fast
