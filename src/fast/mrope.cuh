// Qwen3-VL / qwen35 interleaved M-RoPE (llama.cpp LLAMA_ROPE_TYPE_IMROPE, ggml-cuda rope_multi with is_imrope).
// Rotary pair j (0 <= j < n_rot / 2, NEOX pairing j / j + n_rot / 2) takes the position of one of three streams by
// its sector = j % sect (sect = s0 + s1 + s2 + s3, the GGUF rope.dimension_sections):
//   sector % 3 == 1 && sector < 3 * s1 -> h (row), sector % 3 == 2 && sector < 3 * s2 -> w (column),
//   sector % 3 == 0 && sector < 3 * s0 -> t, otherwise the 4th stream (0 here; unused by sections [11, 11, 10, 0]).
// theta = pos * freq_base^(-2 j / n_rot) as k::rope_neox. For text rows t = h = w = p and every pair rotates by p, i.e.
// plain NEOX at p. Image token i of an nx x ny grid placed at P: t = P, h = P + i / nx, w = P + i % nx (mtmd).
#pragma once
#include <cstdint>

namespace p100e::fast {

struct mrope_pos {
    const int32_t * p = nullptr;   // device [3][stride] positions (t, h, w) of the rows; nullptr = plain NEOX
    int stride = 0;                // row stride between the three streams
    int s0 = 0, s1 = 0, s2 = 0;    // sections (in rotary pairs)
    int sect = 0;                  // sum of all sections
};

#ifdef __CUDACC__
// position of rotary pair j of row t
__device__ __forceinline__ int mrope_pick(const mrope_pos & m, int t, int j) {
    const int sector = j % m.sect;
    int s;
    if (sector % 3 == 1 && sector < 3 * m.s1) s = 1;
    else if (sector % 3 == 2 && sector < 3 * m.s2) s = 2;
    else if (sector % 3 == 0 && sector < 3 * m.s0) s = 0;
    else return 0;
    return m.p[s * m.stride + t];
}
#endif

} // namespace p100e::fast
