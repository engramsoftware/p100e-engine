// p100e fast quantized matrix-vector products for decode (T = 1..8 columns).
// Standalone: no ggml runtime. Kernels are ported from the P100 llama.cpp fork (q4r/kq/mmvq-f16, MIT).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace p100e::fast {

enum layout_t : int {
    LAYOUT_RAW  = 0,   // untouched GGUF bytes (generic kernel, or kq kernels that read the standard layout)
    LAYOUT_Q4R  = 1,   // Q4_0  [qs plane][d plane]
    LAYOUT_Q4KR = 2,   // Q4_K  [qs plane, Q4_0 nibble order][header plane]
    LAYOUT_Q3R  = 3,   // Q3_K  [A,B,H words][scales][d]
    LAYOUT_Q6R2 = 4,   // Q6_K  [ql2][qh2][scales][d]
    LAYOUT_Q8R  = 5,   // Q8_0  [qs plane][d plane]
    LAYOUT_KQRAW = 6,  // Q5_K / IQ4_XS: untouched GGUF bytes, fast a16 kernel
    LAYOUT_Q6R3 = 7,   // Q6_K  q6r2 planes, qh2 bits re-ordered for a 2-op decode per half2 (matvec_pack_body.inc)
    LAYOUT_Q5KR = 8,   // Q5_K  [qs plane, Q4_0 nibble order][qh plane, 1 u32 per group][header plane]
    LAYOUT_IQ1S = 10,  // IQ1_S untouched GGUF bytes, a16 GEMV with the 2048-entry grid in shared memory (matvec_iq1.cu)
    LAYOUT_IQ1M = 11,  // IQ1_M likewise (per-16 scales)
    LAYOUT_IQ2XXSR = 12, // IQ2_XXS [group plane: 8 B qs per 32 values][d plane]  (matvec_iq2r.cu)
    LAYOUT_IQ2XSR  = 13, // IQ2_XS  [group plane][scales plane][d plane]
    LAYOUT_IQ2SR   = 14, // IQ2_S   [group plane: grid-low + sign bytes][qh plane][scales plane][d plane]
    LAYOUT_IQ3XXSR = 15, // IQ3_XXS [qs plane 64B/sb][signs+scale plane 32B/sb][d plane 2B/sb]
    LAYOUT_IQ3SR = 16,   // IQ3_S   [qs 64][signs 32][qh 8][scales 4][d 2] planes (matvec_iq3r.cu)
    LAYOUT_Q10 = 17,     // Q1_0    untouched GGUF 18-byte blocks, a16 GEMV with a 256-entry sign table (matvec_q10.cu)
    LAYOUT_Q20 = 18,     // Q2_0    untouched GGUF 18-byte blocks (64 weights), a16 GEMV with a 256-entry code table (matvec_q20.cu)
    LAYOUT_Q2KR = 9,   // Q2_K  [qs plane: A,B u32 per group in the a16 slot order][header plane: scales[16], d, dmin per sb]
};

struct packed_weight {
    uint32_t type = 0;      // ggml_type
    int64_t  n_in = 0, n_out = 0;
    void *   data = nullptr;
    size_t   bytes = 0;     // size of the weight in bytes (row stride x n_out; the raw GGUF tensor when stride == 0)
    int      layout = LAYOUT_RAW;
    size_t   stride = 0;    // row stride in bytes; 0 = the raw row bytes. A larger stride = padded rows (pack_row_stride:
                            // Q6_K rows padded to 8 / 16 bytes for q6r3, P100E_Q6_PAD); the pad bytes are never read
};

// the row stride of w in bytes (w.stride, or the raw row bytes of its type and n_in)
size_t row_stride(const packed_weight & w);

// The row stride a loader should lay the raw rows of a [n_out][n_in] weight out with before pack_weight: the raw row
// bytes, or, for Q6_K rows that are 4- but not 8-byte aligned (n_in/256 % 4 == 2, e.g. the n_in 8704 ffn_down shard), the raw
// row bytes rounded up to the P100E_Q6_PAD alignment so the weight packs to q6r3 (the warp-rows kernels) instead of
// q6r2 (legacy kernel) / raw. P100E_Q6_PAD=0 or P100E_MV_OLD_LAYOUTS=1: always the raw row bytes. Read once.
size_t pack_row_stride(uint32_t ggml_type, int64_t n_in, int64_t n_out);

// true for every type matvec() can run (dedicated fast kernel or the generic on-the-fly dequant GEMV)
bool has_fast_path(uint32_t ggml_type);

// d_raw = raw GGUF bytes of an [n_out][n_in] weight (row-major rows of n_in elements) on the device.
// Repacks IN PLACE where the type has a fast layout (returned data == d_raw), otherwise leaves the bytes untouched.
// Shapes/alignments a fast kernel cannot take fall back to the generic kernel (layout RAW).
// stride (0 = raw row bytes): the byte distance of the rows in d_raw, each row's raw GGUF bytes at its start; only
// Q6_K takes a padded stride (pack_row_stride), any other type with one is refused (bytes 0).
packed_weight pack_weight(uint32_t type, void * d_raw, int64_t n_in, int64_t n_out, cudaStream_t s, size_t stride = 0);

// scratch (device) needed by matvec for T <= 8 columns of n_in
size_t matvec_scratch_bytes(int64_t n_in, int T);

// y[T][n_out] = x[T][n_in] * W^T, T = 1..8; x and y are f32 device pointers (y must not alias x).
// Row results are invariant: for the warp-rows layouts (q4kr, q3r, q5kr, q6r2/q6r3 with an 8-byte aligned row stride)
// and q8r, a row's value does not depend on T, n_out or the T = 1 kernel variant (column j of a T-column call equals the
// T = 1 call; a weight concatenated with others row-wise gives each part's rows unchanged) - see matvec.cu.
// scratch: T = 1 may quantize inside the GEMV and leave scratch untouched (do not read it afterwards).
void matvec(const packed_weight & w, const float * x, float * y, int T, void * scratch, cudaStream_t s);

// re-read P100E_MVWRP_Q3 / _Q4 / _Q5 / _Q6 (the inline-PTX T = 1 warp-rows kernels, matvec_wrx.cu); true when a level
// changed (kernels captured in CUDA graphs before then still run the old choice)
bool wrx_reload_env();

// name of the kernel family matvec() will use for w (for reports)
const char * kernel_name(const packed_weight & w);

// ---- prequantized activations (quantize once, reuse for every matvec that reads the same x) ----
// matvec() quantizes x into the "a16" format (f16 values scaled per 32-group + per-group (amax, sum)) before the
// GEMV kernel of most layouts. These split that step out so one quantized x feeds several weights, or a fused
// producer kernel (fast/a16q.cuh) writes it directly. The qbuf layout for T columns of n_in values (ng = n_in/32):
// aq = (half2*) qbuf [T][ng][16], ads = aq + T*ng*16 [T][ng]; size matvec_scratch_bytes(n_in, T).
// true when matvec(w, ...) quantizes x (a16 kernel); false for f32-activation kernels (q8r, generic)
bool matvec_uses_a16(const packed_weight & w);
// qbuf = a16(x), x [T][n_in] f32, T = 1..8; bit-identical to the quantize matvec() runs
void matvec_quantize(const float * x, int64_t n_in, int T, void * qbuf, cudaStream_t s);
// same result as matvec(w, x, y, T, ...) when qbuf holds a16(x) for this n_in and T (bit for bit): a16 layouts read
// qbuf only, f32 layouts read x only (pass both). Aborts on a layout without a kernel (never leaves y unwritten).
void matvec_prequantized(const packed_weight & w, const float * x, const void * qbuf, float * y, int T, cudaStream_t s);
// w = up to 4 parts concatenated row-wise (rows [0, n_0) part 0, then part 1, ...): part p's rows of the T columns go to
// parts[p].y as [T][n_p]; the bits of matvec_prequantized on each part alone. One launch for T >= 2 on the warp-rows
// layouts and q8r (the MTP verify's gate|up, qkv|gate|beta|alpha, k|v), per-part launches on row views otherwise.
struct mv_out { float * y; int64_t n; };
// T = 1, one launch for two weights reading the same input: a (a16 warp-rows layout) on the prequantized qbuf -> ya and
// b (q8_0, Q8R) on the f32 x -> yb, each bit-identical to matvec_prequantized on it alone (decode step 6a,
// P100E_MERGE_INPROJ). false = not handled (nothing launched): run the two calls.
bool matvec_pair_t1(const packed_weight & a, const void * qbuf, float * ya, const packed_weight & b, const float * x,
                    float * yb, cudaStream_t s);
void matvec_prequantized_parts(const packed_weight & w, const float * x, const void * qbuf, const mv_out * parts, int nparts,
                               int T, cudaStream_t s);

} // namespace p100e::fast
