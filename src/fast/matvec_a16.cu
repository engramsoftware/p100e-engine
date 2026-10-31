#include "matvec_internal.cuh"
#include <algorithm>
#include <cstdlib>

namespace p100e::fast {

#define A16_QK 32
#define A16_QG 8
#define A16_QBLOCK 128

// Quantize the f32 activations.  A16_QG lanes cooperate on one block of A16_QK elements.
//
// Originally one thread owned a whole block.  This kernel launches as often as the GEMV itself
// (18,094 times in one trace window) yet cost 4.78 us per call, and TRIPLING its work
// (gridX 2 -> 6) did not change that time -- so it was limited neither by SM count nor by
// bandwidth, but by the dependency chain inside a thread: a 32-step sequential max followed by a
// 32-step sequential sum.  Spreading a block over G lanes cuts the chain by G and lands at
// 2.0-2.6 us/call, essentially the 1.9 us launch floor measured for small kernels on this card.
//
// G = 8 is optimal: elements 4k..4k+3 and 4k+16..4k+19 each fall inside a single lane, so the
// output permutation below stays lane-local and needs no shuffle.  Splitting across the whole warp
// (G = 32) needs a shuffle to collect the output and is slower.  Measured us/call
// (4096x5 / 12288x5): G=1 4.88/6.29, G=2 2.63/3.62, G=4 2.29/2.99, G=8 2.25/2.64, G=32 2.62/3.90.
//
// Keeping one thread per block and turning the sum into a tree instead is 20% slower (5.84 us at
// 4096x5): 32 partial sums have to live in registers, and registers bind before the chain does.
// Break the chain by splitting threads, not by restructuring within a thread.
template <bool vec4>
static __global__ void quantize_a16(
        const float * __restrict__ x, half2 * __restrict__ aq, half2 * __restrict__ ads,
        const int s01, const int nblocks) {

    constexpr int G = A16_QG;       // lanes per block
    constexpr int E = A16_QK/G;     // elements per lane

    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    const int ib  = tid/G;          // block index within the row
    const int l   = tid % G;        // lane within the block
    const int row = blockIdx.y;
    if (ib >= nblocks) {
        return;
    }

    // ib is shared by all G lanes, so the tail drops out G lanes at a time.  G divides 32 and the
    // block size is a multiple of 32, so a segment never straddles a warp and shuffle partners are
    // always live.
    const unsigned int gmask = ((1u << G) - 1) << ((threadIdx.x & (WARP_SIZE - 1)) & ~(G - 1));

    const float * xp = x + (size_t) row*s01 + ib*A16_QK + l*E;

    float v[E];
    float amax = 0.0f;
    if (vec4) {
#pragma unroll
        for (int i = 0; i < E/4; ++i) {
            const float4 t = ((const float4 *) xp)[i];
            v[4*i + 0] = t.x; v[4*i + 1] = t.y; v[4*i + 2] = t.z; v[4*i + 3] = t.w;
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(t.x), fabsf(t.y)), fmaxf(fabsf(t.z), fabsf(t.w))));
        }
    } else {
#pragma unroll
        for (int i = 0; i < E; ++i) {
            v[i] = xp[i];
            amax = fmaxf(amax, fabsf(v[i]));
        }
    }
#pragma unroll
    for (int s = G/2; s > 0; s >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(gmask, amax, s, G));
    }
    // the scale is stored as f16: an amax that would round to inf (>= 65520) becomes 65504, the values scale by its
    // inverse (so they stay exact, |v| may pass 1), and the output stays finite instead of inf / NaN for every row.
    // Every group whose f16 amax is finite keeps its bits (matvec_dec.cuh a16_group does the same)
    amax = amax < 65520.0f ? amax : 65504.0f;

    const float id = amax > 0.0f ? 1.0f/amax : 0.0f;

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < E; ++i) {
        v[i] *= id;
        sum  += v[i];
    }
#pragma unroll
    for (int s = G/2; s > 0; s >>= 1) {
        sum += __shfl_xor_sync(gmask, sum, s, G);
    }

    if (l == 0) {
        ads[row*nblocks + ib] = make_half2(__float2half(amax), __float2half(sum));
    }

    // Lanes holding elements 0..15 write outputs 4k+0 / 4k+2; those holding 16..31 write 4k+1 / 4k+3.
    const int off   = l >= G/2 ? 1 : 0;
    const int kbase = (E*l - off*16)/4;

    half2 * qs = aq + (size_t) row*nblocks*(A16_QK/2) + ib*(A16_QK/2);
#pragma unroll
    for (int e = 0; e < E/4; ++e) {
        const int k = kbase + e;
        qs[4*k + 0 + off] = make_half2(__float2half(v[4*e + 0]), __float2half(v[4*e + 2]));
        qs[4*k + 2 + off] = make_half2(__float2half(v[4*e + 1]), __float2half(v[4*e + 3]));
    }
}

void a16_quantize(const float * x, half2 * aq, half2 * ads, int64_t s11, int nblocks, int ncols, bool vec4, cudaStream_t stream) {
    const dim3 grid((nblocks*A16_QG + A16_QBLOCK - 1)/A16_QBLOCK, ncols, 1);
    if (vec4) {
        quantize_a16<true> <<<grid, A16_QBLOCK, 0, stream>>>(x, aq, ads, (int) s11, nblocks);
    } else {
        quantize_a16<false><<<grid, A16_QBLOCK, 0, stream>>>(x, aq, ads, (int) s11, nblocks);
    }
}

} // namespace p100e::fast
