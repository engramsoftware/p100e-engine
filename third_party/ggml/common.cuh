// Minimal stand-in for ggml-cuda/common.cuh, just enough for the vendored convert.cu /
// dequantize.cuh (all ggml quant types -> f16/f32/bf16). Helper bodies copied from the
// P100 fork (98ab7e5, ggml-cuda/common.cuh), MIT. No ggml runtime needed.
#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "ggml.h"

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#define WARP_SIZE 32
#define GGML_CUDA_CC_PASCAL 600
#define NO_DEVICE_CODE

typedef void (*dequantize_kernel_t)(const void * vx, const int64_t ib, const int iqs, float2 & v);

template<typename dst_t>
using dequantize_kq_t = void (*)(const void * vx, const int64_t ib, dst_t * y, const int tid);

static bool fp16_available(const int cc) { return cc >= GGML_CUDA_CC_PASCAL; }

struct ggml_cuda_device_info {
    struct { int cc; } devices[16];
};

int ggml_cuda_get_device();
const ggml_cuda_device_info & ggml_cuda_info();

static __device__ __forceinline__ float ggml_cuda_e8m0_to_fp32(uint8_t x) {
#if CUDART_VERSION >= 12080
    const nv_bfloat16 e = __nv_cvt_e8m0_to_bf16raw(x);
    return (float) e;
#else
    uint32_t bits;
    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint32_t) x << 23;
    }
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
#endif
}

static __device__ __forceinline__ float ggml_cuda_ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || (x == 0x7F && x != 0xFF)) {
        return 0.0f;
    }
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = ldexpf((float) man, -9);
    } else {
        raw = ldexpf(1.0f + (float) man / 8.0f, exp - 7);
    }
    return static_cast<float>(raw / 2);
}

static const uint3 init_fastdiv_values(uint64_t d_64) {
    GGML_ASSERT(d_64 != 0);
    GGML_ASSERT(d_64 <= std::numeric_limits<uint32_t>::max());
    uint32_t d = (uint32_t)d_64;
    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }
    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    return make_uint3(mp, L, d);
}

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
    const uint32_t hi = __umulhi(n, fastdiv_values.x);
    return (hi + n) >> fastdiv_values.y;
}

static __device__ __forceinline__ uint2 fast_div_modulo(uint32_t n, const uint3 fastdiv_values) {
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    const uint32_t mod_val = n - div_val * fastdiv_values.z;
    return make_uint2(div_val, mod_val);
}
