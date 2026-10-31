// q8_0 / q4_0 KV cache rows (device helpers, include from .cu files only).
// Quantization = ggml quantize_row_q8_0_ref per 32 values: d = amax / 127 (fp32), q = roundf(x / d), d stored as f16;
// value = q * f16(d). In-row layout per (position, kv head), D values = D/32 blocks: the D/32 f16 scales first, then
// the D int8 quants ([nb][2 B] | [D][1 B]) - same bytes per head as ggml's [nb][d | qs[32]] (34 B per block), but
// the quants are 16-byte aligned for D == 256 (272 B per head = 17 x 16 B), so the kernels use vector loads.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

namespace p100e::fast {

__device__ __forceinline__ float q8_warp_amax(float a) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));
    return a;
}

// One full warp quantizes block b (32 values, lane = value index) of one head row at `head` (nb blocks per head).
// Every lane must call it (warp shuffle). Bit-identical for the same inputs wherever it runs (append / attn_prep).
__device__ __forceinline__ void q8_quant_block_warp(float x, int lane, char * head, int b, int nb) {
    const float amax = q8_warp_amax(fabsf(x));
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    reinterpret_cast<int8_t *>(head + nb * 2)[b * 32 + lane] = (int8_t) (int) roundf(x * id);
    if (lane == 0) reinterpret_cast<__half *>(head)[b] = __float2half(d);
}

// q4_0 (ggml block_q4_0, 18 B: f16 d, byte i = nibble of value i | nibble of value i+16 << 4), ggml
// quantize_row_q4_0_ref: mx = the value of largest |x| (first one on ties), d = mx / -8, q = min(15, (int)(x / d + 8.5)).
// One full warp per block (lane = value index), x*id and +8.5 rounded separately (no contraction), so every caller
// (fast::kv_append, fast::attn_prep) writes the same bytes. blk = the 18-byte block.
__device__ __forceinline__ void q4_quant_block_warp(float x, int lane, char * blk) {
    float a = fabsf(x);
    int idx = lane;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float oa = __shfl_xor_sync(0xffffffff, a, o);
        const int oi = __shfl_xor_sync(0xffffffff, idx, o);
        if (oa > a || (oa == a && oi < idx)) { a = oa; idx = oi; }
    }
    const float xm = __shfl_sync(0xffffffff, x, idx);
    const float mx = a > 0.0f ? xm : 0.0f;   // the sequential scan starts from max = +0 and needs |x| > 0 to move
    const float d = mx / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const int q = min(15, (int) __fadd_rn(__fmul_rn(x, id), 8.5f));
    const int qhi = __shfl_down_sync(0xffffffff, q, 16);
    if (lane < 16) blk[2 + lane] = (char) (q | (qhi << 4));
    if (lane == 0) *reinterpret_cast<__half *>(blk) = __float2half(d);
}

// 4 int8 packed in a word -> floats (sign-extended)
__device__ __forceinline__ float q8_lo(uint32_t w, int i) { return (float) ((int32_t) (w << (24 - 8 * i)) >> 24); }

} // namespace p100e::fast
