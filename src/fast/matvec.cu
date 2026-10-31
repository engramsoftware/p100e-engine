// p100e::fast public API: weight packing, activation quantize and kernel dispatch.
#include "matvec.cuh"
#include "matvec_internal.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace p100e::fast {

#include "matvec_pack_body.inc"

// Q8_0 (34-byte blocks = 17 u16): standard block b = [d][qs 16 u16].  Repacked row: [qs plane 16*nb u16][d plane nb u16],
// so every block's 32 quants are one 16-byte-aligned pair of uint4 loads when the row stride is a multiple of 16.
template <bool pack>
static __global__ void q8r_convert(const char * src, char * dst, const int nb, const size_t row_bytes) {
    extern __shared__ uint16_t row8[];
    const uint16_t * s = (const uint16_t *) (src + blockIdx.x*row_bytes);
    uint16_t       * d = (uint16_t       *) (dst + blockIdx.x*row_bytes);
    const int n16 = 17*nb;
    for (int i = threadIdx.x; i < n16; i += blockDim.x) row8[i] = s[i];
    __syncthreads();
    for (int i = threadIdx.x; i < n16; i += blockDim.x) {
        if (pack) {
            if (i < 16*nb) { const int b = i >> 4; d[i] = row8[17*b + 1 + (i & 15)]; }
            else           { d[i] = row8[17*(i - 16*nb)]; }
        } else {
            const int b = i / 17, k = i - 17*b;
            d[i] = k == 0 ? row8[16*nb + b] : row8[16*b + k - 1];
        }
    }
}

bool repack_launch(int layout, void * dp, int64_t nrows, int64_t n_in, size_t row_bytes, cudaStream_t s) {
    char * d = (char *) dp;
    const unsigned g = (unsigned) nrows;
    switch (layout) {
        case LAYOUT_Q4R: {
            const int nb = (int) (n_in/32);
            q4r_convert<true><<<g, 256, 18*(size_t) nb, s>>>(d, d, nb, row_bytes);
        } break;
        case LAYOUT_Q4KR: {
            const int nb = (int) (n_in/256);
            q4kr_convert<true><<<g, 256, 144*(size_t) nb, s>>>(d, d, nb, row_bytes);
        } break;
        case LAYOUT_Q3R: {
            const int nb = (int) (n_in/256);
            q3r_convert<true><<<g, 256, 110*(size_t) nb, s>>>(d, d, nb, row_bytes);
        } break;
        case LAYOUT_Q6R2:
        case LAYOUT_Q6R3: {
            const int nb = (int) (n_in/256);
            q6r_convert<true><<<g, 256, 210*(size_t) nb, s>>>(d, d, nb, row_bytes);
            q6r2_permute<true><<<g, 256, 192*(size_t) nb, s>>>(d, d, nb, row_bytes);
            if (layout == LAYOUT_Q6R3) q6r3_convert<true><<<g, 256, 0, s>>>(d, nb, row_bytes);
        } break;
        case LAYOUT_Q2KR: {
            const int nb = (int) (n_in/256);
            q2kr_convert<true><<<g, 256, 84*(size_t) nb, s>>>(d, d, nb, row_bytes);
        } break;
        case LAYOUT_IQ2XXSR: case LAYOUT_IQ2XSR: case LAYOUT_IQ2SR:
            if (row_bytes > 48*1024) return false;
            return iq2r_repack(layout, d, nrows, n_in, row_bytes, s);
        case LAYOUT_IQ3XXSR: {
            const int nb = (int) (n_in/256);
            iq3xxsr_convert<true><<<g, 256, 98*(size_t) nb, s>>>(d, d, nb, row_bytes);
        } break;
        case LAYOUT_IQ3SR: {
            const int nb = (int) (n_in/256);
            iq3sr_convert<true><<<g, 256, 110*(size_t) nb, s>>>(d, d, nb, row_bytes);
        } break;
        case LAYOUT_Q5KR: {
            const int nb = (int) (n_in/256);
            q5kr_convert<true><<<g, 256, 176*(size_t) nb, s>>>(d, d, nb, row_bytes);
        } break;
        case LAYOUT_Q8R: {
            const int nb = (int) (n_in/32);
            q8r_convert<true><<<g, 256, 34*(size_t) nb, s>>>(d, d, nb, row_bytes);
        } break;
        default: return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

static size_t row_bytes_of(uint32_t type, int64_t n_in) {
    int blck, bytes;
    if (!type_info(type, blck, bytes)) return 0;
    return (size_t) (n_in/blck)*bytes;
}

size_t row_stride(const packed_weight & w) { return w.stride ? w.stride : row_bytes_of(w.type, w.n_in); }

static bool old_layouts_on() {
    static const bool v = getenv("P100E_MV_OLD_LAYOUTS") && atoi(getenv("P100E_MV_OLD_LAYOUTS")) != 0;
    return v;
}

// P100E_Q6_PAD: 0 = off (Q6_K rows keep the raw stride: nb % 4 == 2 stays q6r2 on the legacy kernel), 8 / 16 = pad the
// rows to that alignment, unset or 1 = the default alignment (8; 16 gives the same bits).
// Only even nb (raw stride 4-byte aligned, today q6r2) is padded: odd nb (e.g. the 4-GPU ffn_down shard, n_in 4352) stays
// raw on the generic f32-activation GEMV, whose numerics differ more (f32 vs a16 activations); pack_weight takes a padded
// stride for it too (tests), it is just not the loader's default.
static int q6_pad_align() {
    static const int v = [] {
        const char * e = getenv("P100E_Q6_PAD");
        if (!e || !*e) return 8;
        // a whole number only: atoi would read "on" / "x" as 0 and silently turn the padding off
        char * end = nullptr;
        const long x = strtol(e, &end, 10);
        if (end == e || *end != '\0') {
            fprintf(stderr, "p100e: P100E_Q6_PAD=%s ignored (a number: 0, 1, 16), keeping 1\n", e);
            return 8;
        }
        return x == 0 ? 0 : x == 16 ? 16 : 8;
    }();
    return v;
}

size_t pack_row_stride(uint32_t type, int64_t n_in, int64_t n_out) {
    const size_t rb = row_bytes_of(type, n_in);
    const int al = q6_pad_align();
    if (type != GGML_TYPE_Q6_K || al == 0 || old_layouts_on() || rb == 0 || n_in % 256 != 0 || n_out % 4 != 0 || rb % 8 == 0 ||
        (n_in/256) % 2 != 0)
        return rb;
    const size_t ps = (rb + al - 1)/al*al;
    return ps <= 48*1024 ? ps : rb;
}

// model-coverage switches, read once: P100E_MV_FLOAT=0 -> F32 / F16 / BF16 weights have no fast
// path (the classic path; with such a decode weight the fused decode, graphs and P2P allreduce turn off, as before the
// port), P100E_MV_Q41=0 -> Q4_1 on the generic f32 GEMV (as before the port)
static bool env_on(const char * n) { const char * e = getenv(n); return !e || !*e || atoi(e) != 0; }
static bool float_on() { static const bool v = env_on("P100E_MV_FLOAT"); return v; }
static bool q41_on() { static const bool v = env_on("P100E_MV_Q41"); return v; }
// P100E_MV_Q2KR=0 -> Q2_K stays on the generic f32 GEMV (the MTP draft block's Q2_K tensors; matvec_q2kr.cu)
static bool iq1sr_on() { static const bool v = env_on("P100E_MV_IQ1SR"); return v; }   // =0: IQ1_S on the generic GEMV
static bool iq1mr_on() { static const bool v = env_on("P100E_MV_IQ1MR"); return v; }   // =0: IQ1_M on the generic GEMV
static bool q2kr_on() { static const bool v = env_on("P100E_MV_Q2KR"); return v; }

// P100E_MV_IQ2XXSR / _IQ2XSR / _IQ2SR = 0 -> that type stays on the generic f32 GEMV (matvec_iq2r.cu)
static bool iq2xxsr_on() { static const bool v = env_on("P100E_MV_IQ2XXSR"); return v; }
static bool iq2xsr_on()  { static const bool v = env_on("P100E_MV_IQ2XSR"); return v; }
static bool iq2sr_on()   { static const bool v = env_on("P100E_MV_IQ2SR"); return v; }
// P100E_MV_IQ3XXSR=0 / P100E_MV_IQ3SR=0 -> IQ3_XXS / IQ3_S stay on the generic f32 GEMV (matvec_iq3r.cu)
static bool iq3xxsr_on() { static const bool v = env_on("P100E_MV_IQ3XXSR"); return v; }
static bool iq3sr_on() { static const bool v = env_on("P100E_MV_IQ3SR"); return v; }
static bool q10_on() { static const bool v = env_on("P100E_MV_Q10"); return v; }   // =0: Q1_0 on the generic GEMV
static bool q20_on() { static const bool v = env_on("P100E_MV_Q20"); return v; }   // =0: Q2_0 on the generic GEMV

bool has_fast_path(uint32_t type) {
    if ((type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) && !float_on()) return false;
    return generic_supported(type);
}

packed_weight pack_weight(uint32_t type, void * d_raw, int64_t n_in, int64_t n_out, cudaStream_t s, size_t stride) {
    packed_weight w;
    w.type = type; w.n_in = n_in; w.n_out = n_out; w.data = d_raw; w.layout = LAYOUT_RAW;
    const size_t rb0 = row_bytes_of(type, n_in);
    // rb = the row stride from here on: every packer and kernel takes it as the distance of the rows only (a row's
    // planes are located from n_in), so the pad bytes past a row's raw bytes are never read as weights or rewritten
    const size_t rb = stride ? stride : rb0;
    if (rb != rb0) w.stride = rb;
    w.bytes = rb*(size_t) n_out;

    int blck = 0, bytes = 0;
    if (!type_info(type, blck, bytes) || n_in % blck != 0) { w.bytes = 0; return w; }   // unsupported: matvec() will refuse
    if (rb < rb0 || (rb != rb0 && type != GGML_TYPE_Q6_K)) {   // only Q6_K takes a padded stride (pack_row_stride)
        fprintf(stderr, "fast::pack_weight: row stride %zu refused for type %u (raw row %zu bytes)\n", rb, type, rb0);
        w.bytes = 0;
        return w;
    }
    if (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) {
        // float GEMV (matvec_generic.cu float_gemv): 4 weights per load, rows 8 / 16-byte aligned; otherwise no fast path
        // (data = null: the caller keeps the classic path for this weight)
        if (n_in % 4 != 0 || ((uintptr_t) d_raw) % 16 != 0) { w.data = nullptr; w.bytes = 0; }
        return w;
    }

    const int64_t nb256 = n_in/256;
    // P100E_MV_OLD_LAYOUTS=1: Q6_K as q6r2 and Q5_K raw (the layouts before q6r3 / q5kr), for A/B checks. With
    // P100E_MV_VARIANT=-1 as well (legacy kernels for every T) the results are bit-identical to the pre-warp-rows engine
    // (logit hashes, every step).
    const bool old_layouts = old_layouts_on();
    const bool common = n_in % 256 == 0 && n_out % 4 == 0 && rb <= 48*1024 && ((uintptr_t) d_raw) % 16 == 0;
    int layout = LAYOUT_RAW;
    switch ((ggml_type) type) {
        // Q4_0: the q4r row is [16-byte code blocks x nb][2-byte scales x nb], so any nb with 8-byte aligned rows works
        // (18*nb % 8 == 0 <=> n_in % 128 == 0); the Flash-Next expert ffn_down has n_in 640
        case GGML_TYPE_Q4_0: if (n_in % 128 == 0 && n_out % 4 == 0 && rb <= 48*1024 && ((uintptr_t) d_raw) % 16 == 0) layout = LAYOUT_Q4R; break;
        case GGML_TYPE_Q4_K: if (common) layout = LAYOUT_Q4KR; break;
        // q2kr: 8-byte qs loads per 32-value group need rb % 8 == 0 (even block count)
        case GGML_TYPE_Q2_K: if (common && q2kr_on() && rb % 8 == 0) layout = LAYOUT_Q2KR; break;
        // IQ1_S / IQ1_M: the raw GGUF rows stay as they are (2-byte aligned 50 / 56-byte blocks, no repack), the kernel
        // (matvec_iq1.cu) decodes through a shared-memory grid
        case GGML_TYPE_IQ1_S: if (common && iq1sr_on()) layout = LAYOUT_IQ1S; break;
        case GGML_TYPE_IQ1_M: if (common && iq1mr_on()) layout = LAYOUT_IQ1M; break;
        // Q1_0: 128-weight blocks (the expert-split ffn_down has n_in 640), raw rows, 2-byte aligned (matvec_q10.cu)
        case GGML_TYPE_Q1_0: if (q10_on() && n_in % 128 == 0 && n_out % 4 == 0 && rb <= 48*1024 && ((uintptr_t) d_raw) % 16 == 0) layout = LAYOUT_Q10; break;
        case GGML_TYPE_Q2_0: if (q20_on() && n_in % 64 == 0 && n_out % 4 == 0 && rb <= 48*1024 && ((uintptr_t) d_raw) % 16 == 0) layout = LAYOUT_Q20; break;
        // iq2*r: 4-byte aligned rows (even block count); 8-byte group loads when rb % 8 == 0
        case GGML_TYPE_IQ2_XXS: if (common && iq2xxsr_on() && nb256 % 2 == 0) layout = LAYOUT_IQ2XXSR; break;
        case GGML_TYPE_IQ2_XS:  if (common && iq2xsr_on()  && nb256 % 2 == 0) layout = LAYOUT_IQ2XSR; break;
        case GGML_TYPE_IQ2_S:   if (common && iq2sr_on()   && nb256 % 2 == 0) layout = LAYOUT_IQ2SR; break;
        // iq3xxsr / iq3sr: 4-byte plane loads per 32-value group need rb % 4 == 0 (even block count); the smem unpack path
        // stages two copies of a row (2*rb <= 48 KiB)
        case GGML_TYPE_IQ3_XXS: if (common && iq3xxsr_on() && nb256 % 2 == 0 && 2*rb <= 48*1024) layout = LAYOUT_IQ3XXSR; break;
        case GGML_TYPE_IQ3_S:   if (common && iq3sr_on()   && nb256 % 2 == 0 && 2*rb <= 48*1024) layout = LAYOUT_IQ3SR; break;
        case GGML_TYPE_Q3_K: if (common && nb256 % 2 == 0) layout = LAYOUT_Q3R; break;
        // q6r3 rows must be 8-byte aligned for its decoders (raw stride 210*nb: nb % 4 == 0); with the raw stride nb % 4
        // == 2 (e.g. the n_in 8704 ffn_down shard) stays q6r2, whose legacy kernel also takes a 4-byte aligned stride.
        // A padded stride (pack_row_stride, P100E_Q6_PAD) makes any nb 8-byte aligned: q6r3 on the warp-rows kernels.
        case GGML_TYPE_Q6_K:
            if (common && (rb != rb0 || nb256 % 2 == 0)) layout = old_layouts || rb % 8 != 0 ? LAYOUT_Q6R2 : LAYOUT_Q6R3;
            break;
        // q5kr: packed_to_f16 stages two copies of a row in shared memory (2*rb <= 48 KiB), else the raw kq kernel
        case GGML_TYPE_Q5_K: if (common && rb % 8 == 0) layout = old_layouts || 2*rb > 48*1024 ? LAYOUT_KQRAW : LAYOUT_Q5KR; break;
        case GGML_TYPE_IQ4_XS: if (common && rb % 8 == 0) layout = LAYOUT_KQRAW; break;
        // Q4_1: raw blocks, the fork's a16 HFMA2 GEMV (matvec_q41.cu); 4 rows per CUDA block
        case GGML_TYPE_Q4_1:
            if (q41_on() && n_in % 32 == 0 && n_out % 4 == 0 && rb % 4 == 0 && ((uintptr_t) d_raw) % 4 == 0) layout = LAYOUT_KQRAW;
            break;
        case GGML_TYPE_Q8_0: if (n_in % 256 == 0 && rb <= 48*1024 && ((uintptr_t) d_raw) % 16 == 0) layout = LAYOUT_Q8R; break;
        default: break;
    }
    if (layout != LAYOUT_RAW && layout != LAYOUT_KQRAW && layout != LAYOUT_IQ1S && layout != LAYOUT_IQ1M && layout != LAYOUT_Q10 && layout != LAYOUT_Q20) {
        // repack_launch reports failure through cudaGetLastError() after its in-place kernels ran, so an error some
        // earlier call left pending would label repacked bytes RAW (garbage rows on the slow generic kernel: the second
        // tp_model in a process). Clear it first and say so. (A sticky error stays: then the kernels do not
        // run, the bytes stay raw and RAW is right.)
        const cudaError_t pending = cudaGetLastError();
        if (pending != cudaSuccess)
            fprintf(stderr, "fast::pack_weight: cleared a CUDA error left pending by an earlier call: %s\n", cudaGetErrorString(pending));
        if (!repack_launch(layout, d_raw, n_out, n_in, rb, s)) layout = LAYOUT_RAW;
    }
    w.layout = layout;
    return w;
}

size_t matvec_scratch_bytes(int64_t n_in, int T) {
    const size_t ng = (size_t) (n_in/32);
    return (size_t) T*(ng*16*sizeof(half2) + ng*sizeof(half2)) + 256;
}

const char * kernel_name(const packed_weight & w) {
    switch (w.layout) {
        case LAYOUT_Q4R:   return "q4r";
        case LAYOUT_Q4KR:  return "q4kr";
        case LAYOUT_Q3R:   return "q3r";
        case LAYOUT_Q2KR:  return "q2kr";
        case LAYOUT_IQ1S:  return "iq1s-a16";
        case LAYOUT_IQ1M:  return "iq1m-a16";
        case LAYOUT_Q10:   return "q10-a16";
        case LAYOUT_Q20:   return "q20-a16";
        case LAYOUT_IQ2XXSR: return "iq2xxsr";
        case LAYOUT_IQ2XSR:  return "iq2xsr";
        case LAYOUT_IQ2SR:   return "iq2sr";
        case LAYOUT_IQ3XXSR: return "iq3xxsr";
        case LAYOUT_IQ3SR: return "iq3sr";
        case LAYOUT_Q6R2:  return "q6r2";
        case LAYOUT_Q6R3:  return "q6r3";
        case LAYOUT_Q5KR:  return "q5kr";
        case LAYOUT_Q8R:   return "q8r";
        case LAYOUT_KQRAW: return w.type == GGML_TYPE_IQ4_XS ? "kq-iq4xs" : w.type == GGML_TYPE_Q4_1 ? "q41-a16" : "kq-q5k";
        default:           return w.type == GGML_TYPE_F32 || w.type == GGML_TYPE_F16 || w.type == GGML_TYPE_BF16 ? "float" : "generic";
    }
}

bool matvec_uses_a16(const packed_weight & w) {
    return w.bytes != 0 && w.layout != LAYOUT_RAW && w.layout != LAYOUT_Q8R;
}

void matvec_quantize(const float * x, int64_t n_in, int T, void * qbuf, cudaStream_t s) {
    const int ng = (int) (n_in/32);
    half2 * aq  = (half2 *) qbuf;
    half2 * ads = aq + (size_t) T*ng*16;
    const bool vec4 = n_in % 4 == 0 && ((uintptr_t) x) % 16 == 0;
    a16_quantize(x, aq, ads, n_in, ng, T, vec4, s);
}

mv_tune_t mv_tune = [] {
    mv_tune_t t;
    // debug: P100E_MV_VARIANT = -1 forces the legacy kernels (every T), > 0 one T = 1 variant id (matvec_wr.cu)
    if (const char * e = getenv("P100E_MV_VARIANT")) t.variant = atoi(e);
    // T >= 2 on the warp-rows layouts: P100E_MV_WRM=0 -> wrcp (the kernel before matvec_wrm.cu); P100E_MV_WRM_GEO /
    // P100E_MV_WRM_KPG: geometry id / K-phase size for tuning (every choice gives the same bits)
    if (const char * e = getenv("P100E_MV_WRM")) t.wrm = atoi(e);
    if (const char * e = getenv("P100E_MV_WRM_GEO")) t.wrm_geo = atoi(e);
    if (const char * e = getenv("P100E_MV_WRM_KPG")) t.wrm_kpg = atoi(e);
    if (const char * e = getenv("P100E_MV_WRM_PH")) t.wrm_ph = atoi(e);
    if (const char * e = getenv("P100E_MV_WRM_NC")) t.wrm_nc = atoi(e);
    if (const char * e = getenv("P100E_MV_WRM_WGEO")) t.wrm_wide_geo = atoi(e);
    // q8_0 at T = 2..4: P100E_MV_Q8_GEO = -1 -> q8_wrt (the kernel before q8_wrm), > 0 a q8_wrm geometry id
    if (const char * e = getenv("P100E_MV_Q8_GEO")) t.q8_geo = atoi(e);
    // Q6_K HFMA2 tail (matvec_dec_q6h.cuh): a whole number 0 (the dec_q6r3 bits), 1 or 2 (default); anything
    // else keeps the default 2
    if (const char * e = getenv("P100E_MV_Q6_H16")) {
        char * end = nullptr;
        const long v = strtol(e, &end, 10);
        if (end != e && *end == '\0' && v >= 0 && v <= 2) t.q6h = (int) v;
        else fprintf(stderr, "p100e: P100E_MV_Q6_H16=%s ignored (0, 1 or 2), keeping %d\n", e, t.q6h);
    }
    // q8_0 at T = 1 on shards of more than 2048 rows: P100E_MV_Q8_BIG = 0 -> the legacy q8r_gemv,
    // 211..218 a q8wr geometry (all the same bits); other values are ignored (default 215)
    // (a whole-string number only: atoi would read "on" / "" / "x" as 0 and silently pick the legacy kernel)
    if (const char * e = getenv("P100E_MV_Q8_BIG")) {
        char * end = nullptr;
        const long v = strtol(e, &end, 10);
        if (end != e && *end == '\0' && (v == 0 || (v >= 211 && v <= 218))) t.q8_big = (int) v;
        else fprintf(stderr, "p100e: P100E_MV_Q8_BIG=%s ignored (0 or 211..218), keeping %d\n", e, t.q8_big);
    }
    return t;
}();

// ---- kernel families and bit identity ----------------------------------------------------------------------------
// The a16 layouts with warp-rows decoders (q4kr, q3r, q5kr, and q6r2 / q6r3 with an 8-byte aligned row stride) run
// every T on the warp-rows family: a row is summed lane-strided over its 32-value groups (lane l: groups l, l + 32,
// ... in order) and closed with an xor warp tree, in wr (1..28), wrp (301..324, the quantize fused in), wrq (501..524,
// prequantized activation, persistent), wrm (matvec_wrm*.cu, the T >= 2 default: up to 4 columns per pass with the
// weight decode shared by the columns) and wrcp / wrc (P100E_MV_WRM=0, and the fallback). Each group's contribution is
// the same decoder arithmetic, and the fused quantize writes the same bits as quantize_a16, so all of them give the
// same bits for a row: a row's value depends on (layout, n_in, the row, its activation column) only, not on T, n_out, the
// variant, or where the activation was quantized (separate kernel, fused producer, inside the GEMV). That keeps
//   - a concatenated weight (one launch over gate|up, qkv|gate|beta|alpha, ...) equal to its parts (the per-op views),
//   - matvec() equal to matvec_prequantized(),
//   - column j of a T-column call equal to the T = 1 call on that column (prefill chunks, the MTP verify).
// Q8R likewise: q8_wr (211..218) sums exactly like the legacy q8r_gemv (lane-strided 16-value chunks, fma(d, s, acc),
// xor tree), which serves T > 1 (q8_wrt for T = 2..4 on shards of <= 2048 rows, the same sums). The K-split variants (101..105, 201..204, 401..404) sum in another order: debug only.
// The remaining layouts (q4r, kq raw, q6r2 with a 4-byte aligned stride, generic) keep their legacy kernels.
static bool wr_family(int layout, size_t rb) {
    switch (layout) {
        case LAYOUT_Q4KR: case LAYOUT_Q3R: case LAYOUT_Q5KR: return true;
        case LAYOUT_Q6R2: case LAYOUT_Q6R3: return rb % 8 == 0;
        default: return false;
    }
}

// Default T = 1 kernel for f32 activations (matvec(): the quantize fused into the GEMV), per layout and shard shape,
// from the P100 sweeps of p100e-fast-matvec-bench (rank-0 shards, us per
// call incl. the activation quantize, legacy -> chosen). Ids: see mv_tune_t; 0 = the legacy kernel.
int default_t1_variant(int layout, uint32_t type, int64_t n_in, int64_t n_out) {
    (void) type;
    const bool tiny = n_out <= 64;
    switch (layout) {
        case LAYOUT_Q8R:
            if (tiny)          return 218;   // ssm_alpha/beta 24 rows: 11.7 -> 7.1 us (q8wr nw1 rw1)
            if (n_out <= 2048) return 214;   // attn_k/v 512 rows: 19.4 -> 14.4 us (q8wr nw8 rw2 staged)
            // ssm_out 3072 -> 5120 (Q6_K): 215. Only up to
            // n_in 5120: 215 stages x in 64 B per 16 inputs of shared memory (<= 20.6 KB, >= 3 blocks per 64 KB SM);
            // wider rows (ffn_down 8704, eh_proj 10240) would run 1-2 blocks per SM and keep the legacy kernel
            return n_in <= 5120 ? mv_tune.q8_big : 0;
        case LAYOUT_Q4KR:
            if (tiny)          return 315;   // 24 rows: 9.2-9.6 -> 7.1-7.2 us
            if (n_in >= 8192)  return 320;   // ffn_down 8704 -> 5120: 63.2 -> 54.9 us
            if (n_out == 6144) return 303;   // attn_q: 42.4 -> 40.3 us
            return 306;                      // ffn_gate/up 55.2/54.5 -> 52.0/51.7, attn_qkv 39.8 -> 33.9, attn_gate 29.3 -> 24.0,
                                             // attn_output 28.2 -> 22.8 us
        case LAYOUT_Q3R:
            return n_in >= 8192 ? 320 : 306; // ffn_down 65.3 -> 55.4, ffn_gate/up 70.2/65.1 -> 54.7/54.6 us
        case LAYOUT_Q6R2:
        case LAYOUT_Q6R3:
            if (tiny)           return 315;
            if (n_out >= 32768) return 5;    // output.weight (124160 rows): q6r2 kq 1009.8 -> q6r3 wr 931.4 us
            if (n_out == 6144)  return 303;  // attn_q: 64.9 -> 54.9 us
            return 314;                      // attn_qkv 58.9 -> 45.5, attn_gate 37.4 -> 30.3 us (q6r2 kq: 54.3 / 38.0)
        case LAYOUT_Q5KR:
            return tiny ? 315 : 306;         // ssm_out 3072 -> 5120: raw kq 38.7 -> q5kr fused 26.9 us
        default: return 0;
    }
}

// Default T = 1 kernel for a prequantized activation (matvec_prequantized(): the fused decode, whose producers write
// the a16 activation). Concatenated weights get their total n_out here.
int default_pq_variant(int layout, uint32_t type, int64_t n_in, int64_t n_out) {
    const int v = default_t1_variant(layout, type, n_in, n_out);
    return v >= 300 && v < 400 ? v + 200 : v;   // wrp geometry -> its wrq twin; wr ids stay
}

// y[T][n_out] from the a16 activation of T columns (aq [T][ng][16], ads [T][ng])
static void a16_gemv(const packed_weight & w, const half2 * aq, const half2 * ads, float * y, int T, cudaStream_t s) {
    const size_t rb = row_stride(w);
    const char * wd = (const char *) w.data;
    const int ng = (int) (w.n_in/32);
    auto args = [&](int c0) {
        a16_args a;
        a.aq = aq + (size_t) c0*ng*16; a.ads = ads + (size_t) c0*ng; a.dst = y + (size_t) c0*w.n_out;
        a.ngroups = ng; a.nrows = (int) w.n_out; a.row_bytes = (int) rb;
        return a;
    };
    if (mv_tune.variant >= 0 && wr_family(w.layout, rb)) {
        if (mv_tune.q6h != 0 && w.layout == LAYOUT_Q6R3) {
            // P100E_MV_Q6_H16: every T on mv_wrm with the dec_q6h tail (T-invariant); never a dec_q6r3 kernel
            if (launch_wrm(w.layout, wd, args(0), T, s)) return;
            fprintf(stderr, "p100e::fast::matvec: P100E_MV_Q6_H16: no q6h kernel for this q6r3 shape\n");
            abort();
        }
        if (T == 1) {
            const int v = mv_tune.variant > 0 ? mv_tune.variant : default_pq_variant(w.layout, w.type, w.n_in, w.n_out);
            const a16_args a = args(0);
            if (v >= 501 && v < 600 && launch_wrq(w.layout, wd, a, v, s)) return;
            if (v >= 301 && v < 400 && launch_wrq(w.layout, wd, a, v + 200, s)) return;
            if (v > 0 && v < 200 && launch_wr(w.layout, wd, a, v, s)) return;
        }
        if (mv_tune.wrm && launch_wrm(w.layout, wd, args(0), T, s)) return;
        if (launch_wrc(w.layout, wd, args(0), T, s)) return;
        fprintf(stderr, "p100e::fast::matvec: no warp-rows kernel for layout %d\n", w.layout);
        abort();
    }
    if (w.layout == LAYOUT_Q6R3 || w.layout == LAYOUT_Q5KR) {
        // no legacy kernel for these layouts: K split with the legacy geometry, up to 4 columns per launch
        for (int c0 = 0; c0 < T; c0 += 4) {
            if (!launch_ks_cols(w.layout, wd, args(c0), std::min(4, T - c0), s)) {
                fprintf(stderr, "p100e::fast::matvec: unsupported %s row stride %zu\n", kernel_name(w), rb);
                abort();
            }
        }
        return;
    }
    if (w.layout == LAYOUT_KQRAW && w.type == GGML_TYPE_Q4_1) {   // up to 8 columns per launch
        launch_q41(wd, args(0), T, s);
        return;
    }
    const int maxc = w.layout == LAYOUT_Q4R ? 8 : 4;
    for (int c0 = 0; c0 < T; c0 += maxc) {
        const int nc = std::min(maxc, T - c0);
        const a16_args a = args(c0);
        switch (w.layout) {
            case LAYOUT_Q4R:   launch_q4r (wd, a, nc, s); break;
            case LAYOUT_Q4KR:  launch_q4kr(wd, a, nc, s); break;
            case LAYOUT_Q3R:   launch_q3r (wd, a, nc, s); break;
            case LAYOUT_Q2KR:  launch_q2kr(wd, a, nc, s); break;
            case LAYOUT_IQ1S:  launch_iq1s(wd, a, nc, s); break;
            case LAYOUT_IQ1M:  launch_iq1m(wd, a, nc, s); break;
            case LAYOUT_Q10:   launch_q10(wd, a, nc, s); break;
            case LAYOUT_Q20:   launch_q20(wd, a, nc, s); break;
            case LAYOUT_IQ2XXSR: launch_iq2xxsr(wd, a, nc, s); break;
            case LAYOUT_IQ2XSR:  launch_iq2xsr (wd, a, nc, s); break;
            case LAYOUT_IQ2SR:   launch_iq2sr  (wd, a, nc, s); break;
            case LAYOUT_IQ3XXSR: launch_iq3r(false, wd, a, nc, s); break;
            case LAYOUT_IQ3SR: launch_iq3r(true, wd, a, nc, s); break;
            case LAYOUT_Q6R2:  launch_q6r2(wd, a, nc, rb % 16 == 0 ? 16 : rb % 8 == 0 ? 8 : 4, s); break;
            case LAYOUT_KQRAW:
                if (w.type == GGML_TYPE_IQ4_XS) launch_iq4xs(wd, a, nc, s); else launch_q5k(wd, a, nc, s);
                break;
            default:
                // a layout without a kernel here must never leave y unwritten
                fprintf(stderr, "p100e::fast::matvec: no a16 kernel for layout %d\n", w.layout);
                abort();
        }
    }
}

void matvec_prequantized(const packed_weight & w, const float * x, const void * qbuf, float * y, int T, cudaStream_t s) {
    if (!matvec_uses_a16(w)) { matvec(w, x, y, T, nullptr, s); return; }
    if (T < 1 || T > 8) { fprintf(stderr, "p100e::fast::matvec_prequantized: unsupported T\n"); abort(); }
    const int ng = (int) (w.n_in/32);
    const half2 * aq  = (const half2 *) qbuf;
    const half2 * ads = aq + (size_t) T*ng*16;
    a16_gemv(w, aq, ads, y, T, s);
}

bool matvec_pair_t1(const packed_weight & a, const void * qbuf, float * ya, const packed_weight & b, const float * x,
                    float * yb, cudaStream_t s) {
    if (mv_tune.variant != 0 || b.layout != LAYOUT_Q8R || b.bytes == 0 || !matvec_uses_a16(a)) return false;
    if (mv_tune.q6h != 0 && a.layout == LAYOUT_Q6R3) return false;   // the q6h tail has no fused q8 pair kernel
    const size_t rba = row_stride(a);
    if (!wr_family(a.layout, rba)) return false;
    const int vb = default_t1_variant(b.layout, b.type, b.n_in, b.n_out);
    if (vb < 211 || vb > 218) return false;   // only the warp-rows q8 kernels share q8_wr's per-row sums
    const int va = default_pq_variant(a.layout, a.type, a.n_in, a.n_out);
    const int ng = (int) (a.n_in/32);
    a16_args g;
    g.aq = (const half2 *) qbuf; g.ads = g.aq + (size_t) ng*16; g.dst = ya;
    g.ngroups = ng; g.nrows = (int) a.n_out; g.row_bytes = (int) rba;
    return launch_wrq_q8(a.layout, (const char *) a.data, g, va, (const char *) b.data, x, yb, b.n_in, b.n_out,
                         row_stride(b), s);
}

// Several parts concatenated row-wise in one packed weight, outputs per part ([T][n_p] at parts[p].y). Every row's value
// depends only on (layout, n_in, the row, its column) (see above), so one launch writing each part's rows to its own
// buffer gives the bits of separate launches on the parts. One launch: T >= 2 on the warp-rows layouts (wrm) and q8r
// (q8_wrm); T = 1 with the parts back to back in memory (the plain launch); else one launch per part on row views.
void matvec_prequantized_parts(const packed_weight & w, const float * x, const void * qbuf, const mv_out * parts, int nparts,
                               int T, cudaStream_t s) {
    int64_t rows = 0;
    bool contiguous = true;
    for (int p = 0; p < nparts; ++p) {
        if (p > 0 && parts[p].y != parts[p - 1].y + parts[p - 1].n) contiguous = false;
        rows += parts[p].n;
    }
    if (nparts < 1 || nparts > 4 || rows != w.n_out || w.bytes == 0 || T < 1 || T > 8) {
        fprintf(stderr, "p100e::fast::matvec_prequantized_parts: bad parts\n");
        abort();
    }
    if (nparts == 1 || (T == 1 && contiguous)) { matvec_prequantized(w, x, qbuf, parts[0].y, T, s); return; }
    const size_t rb = row_stride(w);
    const char * wd = (const char *) w.data;
    mv_dst d{};
    d.np = nparts;
    for (int p = 0; p < nparts; ++p) { d.y[p] = parts[p].y; d.r0[p + 1] = d.r0[p] + (int) parts[p].n; }
    static const bool q8_tn = !getenv("P100E_Q8_TN") || atoi(getenv("P100E_Q8_TN")) != 0;
    if (T >= 2 && mv_tune.variant >= 0) {
        if (matvec_uses_a16(w) && wr_family(w.layout, rb) && (mv_tune.wrm || (mv_tune.q6h != 0 && w.layout == LAYOUT_Q6R3))) {
            const int ng = (int) (w.n_in/32);
            a16_args a;
            a.aq = (const half2 *) qbuf; a.ads = a.aq + (size_t) T*ng*16; a.dst = parts[0].y;
            a.ngroups = ng; a.nrows = (int) w.n_out; a.row_bytes = (int) rb; a.parts = &d;
            if (launch_wrm(w.layout, wd, a, T, s)) return;
        }
        if (w.layout == LAYOUT_Q8R && T <= 8 && q8_tn && mv_tune.q8_geo >= 0 && (T <= 4 || mv_tune.wrm_nc > 4) &&
            launch_q8_wrm(wd, x, d, T, w.n_in, w.n_out, rb, s)) return;
    }
    int64_t r0 = 0;
    for (int p = 0; p < nparts; ++p) {   // one launch per part on a row view
        packed_weight v = w;
        v.data = (char *) w.data + (size_t) r0*rb;
        v.n_out = parts[p].n;
        v.bytes = rb*(size_t) parts[p].n;
        matvec_prequantized(v, x, qbuf, parts[p].y, T, s);
        r0 += parts[p].n;
    }
}

void matvec(const packed_weight & w, const float * x, float * y, int T, void * scratch, cudaStream_t s) {
    if (w.bytes == 0 || T < 1 || T > 8) { fprintf(stderr, "p100e::fast::matvec: unsupported type/shape/T\n"); abort(); }
    const size_t rb = row_stride(w);
    const char * wd = (const char *) w.data;

    if (w.layout == LAYOUT_RAW) {
        launch_generic(w.type, wd, x, y, T, w.n_in, w.n_out, rb, s);
        return;
    }
    if (w.layout == LAYOUT_Q8R) {
        if (T == 1 && mv_tune.variant >= 0) {
            const int v = mv_tune.variant > 0 ? mv_tune.variant : default_t1_variant(w.layout, w.type, w.n_in, w.n_out);
            if (v >= 200 && v < 300 && launch_q8_t1(wd, x, y, w.n_in, w.n_out, rb, v, s)) return;
        }
        // P100E_Q8_TN=0: small shards at T = 2..4 on the legacy kernel too
        static const bool q8_tn = !getenv("P100E_Q8_TN") || atoi(getenv("P100E_Q8_TN")) != 0;
        // T = 2..8 on q8_wrm (one weight pass; column j = the T = 1 call on column j). P100E_MV_WRM_NC <= 4 or
        // P100E_MV_Q8_GEO=-1: T = 5..8 (batched multi-session decode) as column chunks of <= 4 on the T-invariant
        // kernels (q8_wrm / q8_tn / the T = 1 kernels), as before
        if (T > 1 && T <= 8 && q8_tn && mv_tune.variant >= 0 && mv_tune.q8_geo >= 0 && (T <= 4 || mv_tune.wrm_nc > 4) &&
            launch_q8_wrm(wd, x, mv_dst_one(y, (int) w.n_out), T, w.n_in, w.n_out, rb, s)) return;
        if (T > 4 && q8_tn && mv_tune.variant >= 0) {
            for (int c0 = 0; c0 < T; c0 += 4)
                matvec(w, x + (size_t) c0*w.n_in, y + (size_t) c0*w.n_out, std::min(4, T - c0), scratch, s);
            return;
        }
        if (T > 1 && q8_tn && mv_tune.variant >= 0 && launch_q8_tn(wd, x, y, T, w.n_in, w.n_out, rb, s)) return;
        launch_q8r_f32(wd, x, y, T, w.n_in, w.n_out, rb, s);
        return;
    }

    // a16 path: T = 1 with the quantize fused into the GEMV where the layout has it; otherwise quantize the
    // activations into scratch once, then the GEMV kernel (note: the fused path does not write scratch)
    const int ng = (int) (w.n_in/32);
    if (T == 1 && mv_tune.variant >= 0 && wr_family(w.layout, rb) && !(mv_tune.q6h != 0 && w.layout == LAYOUT_Q6R3)) {
        const int v = mv_tune.variant > 0 ? mv_tune.variant : default_t1_variant(w.layout, w.type, w.n_in, w.n_out);
        if (v >= 300 && v < 500 && launch_fused_t1(w.layout, wd, x, y, ng, (int) w.n_out, (int) rb, v, s)) return;
    }
    half2 * aq  = (half2 *) scratch;
    half2 * ads = aq + (size_t) T*ng*16;
    const bool vec4 = w.n_in % 4 == 0 && ((uintptr_t) x) % 16 == 0;
    if (!mv_tune.skip_quant) a16_quantize(x, aq, ads, w.n_in, ng, T, vec4, s);
    a16_gemv(w, aq, ads, y, T, s);
}

} // namespace p100e::fast
