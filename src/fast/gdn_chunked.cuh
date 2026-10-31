// p100e::fast chunked gated delta net for prefill (the fork's GGML_CUDA_GDN_CHUNKED, from Kmic-68).
// Same inputs and outputs as fast::gdn_scan (src/fast/ssm.cuh) on the gdn_prep outputs: y [T][C] (q scaled | k | v),
// decay [T][Hv] (= exp(g)), beta [T][Hv], z [T][ldz], state [Hv][128][128] (row j = v dim, column i = k dim) read and
// written in place, out [T][Hv][128] = the gated-RMSNormed output. Not bit-identical to the sequential scan: the chunk
// form (64-token chunks, forward substitution on the residual, fp64 log-decay prefix sums, two-level fp32 sums) has
// its own rounding; tests/fast_gdn_chunk_test.cu compares both with an fp64 sequential reference.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace p100e::fast {

// scratch for gdn_scan_chunked at T tokens (bytes)
size_t gdn_chunked_scratch_bytes(int T, int Hk, int Hv);

// mode 1: all fp32 (the fork's default chunked mode), 2: K K^T / Q K^T and the state update in fp64.
// Returns false (nothing launched) when it does not apply: D != 128, Hv % Hk != 0, T < 64 (less than one full chunk),
// misaligned pointers / strides, or scratch smaller than gdn_chunked_scratch_bytes.
bool gdn_scan_chunked(const float * y, const float * decay, const float * beta, const float * z, const float * norm_w,
                      float * state, float * out, int T, int Hk, int Hv, int D, float eps, void * scratch,
                      size_t scratch_bytes, cudaStream_t s, int mode = 1, int ldz = 0);

// accuracy reference (tests): the sequential recurrence with an fp64 state (the fork's GGML_CUDA_GDN_REF), raw output
// (no norm) into out [T][Hv][128], final state in place. Slow.
void gdn_scan_ref_f64(const float * y, const float * decay, const float * beta, float * state, float * out, int T,
                      int Hk, int Hv, cudaStream_t s);

} // namespace p100e::fast
