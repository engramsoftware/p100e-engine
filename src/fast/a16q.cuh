// Device helper: the a16 activation quantize of fast::matvec (matvec_a16.cu, quantize_a16<vec4 = true>, A16_QG = 8),
// for kernels that produce an activation and quantize it in the same pass (fused producers of matvec_prequantized).
//
// Layout (per token column t of n_in values, ng = n_in / 32 groups):  aq = qbuf, ads = aq + T*ng*16
//   aq [t*ng*16 + g*16 + slot] : 16 half2 per 32-value group (see quantize_a16 for the slot order)
//   ads[t*ng + g]              : (amax, sum(v/amax)) as half2
// One 32-value group is owned by 8 consecutive lanes (lane & 7 == l), lane l holding values 4l..4l+3 as a float4.
// The arithmetic is the same instruction sequence as quantize_a16 (fmax tree, IEEE 1/amax, FMUL then a sequential
// FADD chain from +0, xor-butterfly sums over the 8 lanes), written with __fmul_rn/__fadd_rn so no FMA contraction
// can differ, so the bytes are identical to fast::matvec_quantize of the same f32 values.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace p100e::fast {

// v = values 4l..4l+3 of the group (l = lane & 7, 1-D blocks). qs = the group's 16 half2 slots, ads = the group's
// scale slot. All 8 lanes of the group must call this together (xor shuffles inside 8-lane segments; mask covers
// them); store = false runs the shuffles only (for lanes past the end of the data, pointers are not touched).
static __device__ __forceinline__ void a16_quant8(const float4 v4, half2 * __restrict__ qs, half2 * __restrict__ ads,
                                                  const bool store = true, const unsigned mask = 0xffffffffu) {
    const int l = threadIdx.x & 7;
    float v[4] = {v4.x, v4.y, v4.z, v4.w};
    float amax = 0.0f;
    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))));
#pragma unroll
    for (int s = 4; s > 0; s >>= 1) amax = fmaxf(amax, __shfl_xor_sync(mask, amax, s, 8));
    amax = amax < 65520.0f ? amax : 65504.0f;   // same clamp as quantize_a16: a finite f16 scale (bit-identical below 65520)
    const float id = amax > 0.0f ? 1.0f / amax : 0.0f;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = __fmul_rn(v[i], id);
        sum  = __fadd_rn(sum, v[i]);
    }
#pragma unroll
    for (int s = 4; s > 0; s >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(mask, sum, s, 8));
    if (!store) return;
    if (l == 0) *ads = make_half2(__float2half(amax), __float2half(sum));
    const int off = l >= 4 ? 1 : 0;
    const int k   = l - 4 * off;
    qs[4 * k + 0 + off] = make_half2(__float2half(v[0]), __float2half(v[2]));
    qs[4 * k + 2 + off] = make_half2(__float2half(v[1]), __float2half(v[3]));
}

// pointers of group g of token column t in a qbuf for T columns of n_in values
static __device__ __forceinline__ half2 * a16_qs(void * qbuf, int t, int g, int ng) {
    return reinterpret_cast<half2 *>(qbuf) + ((size_t) t * ng + g) * 16;
}
static __device__ __forceinline__ half2 * a16_ads(void * qbuf, int t, int g, int ng, int T) {
    return reinterpret_cast<half2 *>(qbuf) + (size_t) T * ng * 16 + (size_t) t * ng + g;
}

} // namespace p100e::fast
