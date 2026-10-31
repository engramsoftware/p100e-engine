// Q4_0 weights repacked in place for sm_60 (GP100).
//
// block_q4_0 is 18 bytes, so its 16 quant bytes are never 4-byte aligned in a useful way: MMVQ reads them
// with 16-bit loads, and even the aligned-load kernel in mmvq-f16-sm60.cu has to realign every int with a
// funnel shift.  Standalone microbenchmark on P100 (p100/q4dec/q4dec-bench.cu), us per call, 4 rows/block:
//     shape (rows x K)   MMVQ q8_1   18-byte half2 kernel   repacked   pure read of the same bytes
//     4096  x 4096         29.93          25.74             23.15          20.59
//     12288 x 4096         73.40          65.78             57.63          51.67
//     16384 x 5120        118.21         106.46             88.63          82.76
//
// Repacked layout, per row (same bytes, same row stride 18*nb, so nothing else about the tensor changes):
//     [ qs of block 0 .. nb-1 : 16*nb bytes ][ d of block 0 .. nb-1 : 2*nb bytes ]
// With ne00 % 256 == 0 the row stride is a multiple of 16, so every block's quants are 16-byte aligned.
// The values are bit-identical; only their order in memory changes.
//
// Lifetime.  A tensor is examined once, at its first MUL_MAT on this device, and the decision never
// changes: repacked, or never repacked.  The first sight must not be inside CUDA graph capture (the
// repack would be captured and replayed), so a tensor first seen during capture stays unpacked for good.
// Graphs captured after the repack use the repacked kernels; none captured before can exist, because
// the first sight is the first time the tensor is used at all.
//
// Consumers.  Widths 1..8 (decode and the MTP verify batch) use the kernel below.  Anything else that
// reads a repacked tensor -- wider MUL_MATs (prefill) or any other op -- gets a temporary copy in the
// standard layout for the duration of that op (ggml_cuda_q4r_unpacked_scope).
//
// GGML_Q4R=0 disables repacking (tensors seen afterwards stay standard).
//
// The same machinery holds Q6_K (q6r, read by kq-sm60.cu) and, with GGML_KQR_Q4K=1 (default off), Q4_K (q4kr: qs
// plane in Q4_0 nibble order + header plane, read by mul_mat_vec_q4kr in kq-sm60.cu, bit-identical to kq's Q4_K
// kernel).  GGML_KQR_VERBOSE=1 logs every first-sight decision and, once after each batch of new tensors, the
// PACKED/NEVER counts per device and type.

#include "q4r-sm60.cuh"
#include "mmvq-f16-sm60.cuh"
#include "kq-sm60.cuh"
#include "convert.cuh"

#include <cstdlib>
#include <map>
#include <mutex>

#define Q4R_ROWS   4
#define Q4R_QK     32
#define Q4R_NWARPS 4

// ---------------------------------------------------------------- layouts
//
// The types this file can hold in a repacked layout.  Every site that acts on a repacked tensor (type gate,
// eligibility, pack/unpack converter, to_f16, GEMV dispatch) goes through q4r_layout_of and aborts on a type it
// does not handle, so a type added to one site but not to the others fails loudly instead of being converted or
// read as Q4_0.

enum q4r_layout : int8_t { Q4R_LAYOUT_NONE = 0, Q4R_LAYOUT_Q4_0 = 1, Q4R_LAYOUT_Q6_K = 2, Q4R_LAYOUT_Q4_K = 3,
                           Q4R_LAYOUT_Q3_K = 4 };
#define Q4R_NLAYOUTS 5

// GGML_KQR_Q4K=1 makes Q4_K repackable (q4kr).  Unset or 0: Q4_K has no layout here, so every site below treats it
// exactly as before this switch existed (never looked up, never packed; kq-sm60.cu runs it on the standard layout).
static bool q4r_q4k_on() {
    static const bool on = getenv("GGML_KQR_Q4K") && atoi(getenv("GGML_KQR_Q4K")) != 0;
    return on;
}

// GGML_Q3R=1 makes Q3_K repackable (q3r, kquant PLAN phase 3).  Unset or 0: Q3_K has no layout here (generic MMVQ).
static bool q4r_q3k_on() {
    static const bool on = getenv("GGML_Q3R") && atoi(getenv("GGML_Q3R")) != 0;
    return on;
}

static q4r_layout q4r_layout_of(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0: return Q4R_LAYOUT_Q4_0;
        case GGML_TYPE_Q6_K: return Q4R_LAYOUT_Q6_K;
        case GGML_TYPE_Q4_K: return q4r_q4k_on() ? Q4R_LAYOUT_Q4_K : Q4R_LAYOUT_NONE;
        case GGML_TYPE_Q3_K: return q4r_q3k_on() ? Q4R_LAYOUT_Q3_K : Q4R_LAYOUT_NONE;
        default:             return Q4R_LAYOUT_NONE;
    }
}

static const char * q4r_layout_name(const q4r_layout layout) {
    switch (layout) {
        case Q4R_LAYOUT_Q4_0: return "q4_0 (q4r)";
        case Q4R_LAYOUT_Q6_K: return "q6_K (q6r)";
        case Q4R_LAYOUT_Q4_K: return "q4_K (q4kr)";
        case Q4R_LAYOUT_Q3_K: return "q3_K (q3r)";
        default:              return "none";
    }
}

// ---------------------------------------------------------------- state

enum q4r_state : int8_t { Q4R_NEVER = 0, Q4R_PACKED = 1 };

static std::mutex                          q4r_mutex;
static std::map<const void *, q4r_state>   q4r_states;
static thread_local int                    q4r_bypass = 0; // > 0 while an unpacked copy is in use

// GGML_KQR_VERBOSE counters (guarded by q4r_mutex): first-sight decisions per device and layout.  Cumulative over
// the process; printed once after each batch of new first sights (at the next MUL_MAT that finds a known tensor).
enum { Q4R_WHY_ENV = 0, Q4R_WHY_CAPTURE = 1, Q4R_WHY_DEVICE = 2, Q4R_WHY_ELIGIBLE = 3, Q4R_NWHY = 4 };
static const char * const q4r_why_name[Q4R_NWHY] = { "disabled by env", "seen during capture", "device", "not eligible" };
struct q4r_counts {
    int64_t packed = 0;
    int64_t packed_bytes = 0;
    int64_t never[Q4R_NWHY] = {};
    int64_t never_bytes = 0;
};
static q4r_counts q4r_stats[GGML_CUDA_MAX_DEVICES][Q4R_NLAYOUTS];
static bool       q4r_stats_dirty = false;

static bool q4r_verbose() {
    static const bool verbose = getenv("GGML_KQR_VERBOSE") && atoi(getenv("GGML_KQR_VERBOSE")) != 0;
    return verbose;
}

// caller holds q4r_mutex
static void q4r_print_counts_locked() {
    for (int dev = 0; dev < GGML_CUDA_MAX_DEVICES; ++dev) {
        for (int l = 1; l < Q4R_NLAYOUTS; ++l) {
            const q4r_counts & c = q4r_stats[dev][l];
            const int64_t never = c.never[0] + c.never[1] + c.never[2] + c.never[3];
            if (c.packed == 0 && never == 0) {
                continue;
            }
            GGML_LOG_INFO("q4r: device %d %s: PACKED %lld (%.1f MiB), NEVER %lld (%.1f MiB: %s %lld, %s %lld, %s %lld, %s %lld)\n",
                dev, q4r_layout_name((q4r_layout) l), (long long) c.packed, c.packed_bytes/1048576.0, (long long) never,
                c.never_bytes/1048576.0, q4r_why_name[0], (long long) c.never[0], q4r_why_name[1], (long long) c.never[1],
                q4r_why_name[2], (long long) c.never[2], q4r_why_name[3], (long long) c.never[3]);
        }
    }
    q4r_stats_dirty = false;
}

static bool q4r_lookup(const void * p, q4r_state & s) {
    std::lock_guard<std::mutex> lock(q4r_mutex);
    auto it = q4r_states.find(p);
    if (it == q4r_states.end()) {
        return false;
    }
    s = it->second;
    return true;
}

void ggml_cuda_q4r_forget_range(const void * base, size_t size) {
    std::lock_guard<std::mutex> lock(q4r_mutex);
    const char * b = (const char *) base;
    auto it = q4r_states.lower_bound(base);
    while (it != q4r_states.end() && (const char *) it->first < b + size) {
        it = q4r_states.erase(it);
    }
}

bool ggml_cuda_q4r_is_repacked(const ggml_tensor * t) {
    if (t == nullptr || q4r_layout_of(t->type) == Q4R_LAYOUT_NONE || q4r_bypass > 0) {
        return false;
    }
    q4r_state s;
    return q4r_lookup(t->data, s) && s == Q4R_PACKED;
}

// ---------------------------------------------------------------- layout conversion

// One CUDA block per row.  The row is staged in shared memory as 16-bit words (block_q4_0 is 2-aligned).
template <bool pack>
static __global__ void q4r_convert(const char * src, char * dst, const int nb, const size_t row_bytes) {
    extern __shared__ uint16_t row[];
    const uint16_t * s = (const uint16_t *) (src + blockIdx.x*row_bytes);
    uint16_t       * d = (uint16_t       *) (dst + blockIdx.x*row_bytes);
    const int n16 = 9*nb;
    for (int i = threadIdx.x; i < n16; i += blockDim.x) {
        row[i] = s[i];
    }
    __syncthreads();
    // standard: block b = u16 [9b] d, [9b+1 .. 9b+8] qs.   repacked: qs of b = u16 [8b .. 8b+7], d of b = u16 [8nb + b].
    for (int i = threadIdx.x; i < n16; i += blockDim.x) {
        if (pack) {
            if (i < 8*nb) { const int b = i >> 3; d[i] = row[9*b + 1 + (i & 7)]; }
            else          { const int b = i - 8*nb; d[i] = row[9*b]; }
        } else {
            const int b = i / 9, k = i - 9*b;
            d[i] = k == 0 ? row[8*nb + b] : row[8*b + k - 1];
        }
    }
}

// Q6_K (210 bytes = 105 u16 per block): standard block b = [ql 64][qh 32][scales 8][d 1] u16.
// Repacked row: [ql of all blocks 64*nb][qh 32*nb][scales 8*nb][d nb] u16.  With an even block count the
// row stride is a multiple of 4 bytes, so every segment of every block can be read with 32-bit loads.
template <bool pack>
static __global__ void q6r_convert(const char * src, char * dst, const int nb, const size_t row_bytes) {
    extern __shared__ uint16_t row[];
    const uint16_t * s = (const uint16_t *) (src + blockIdx.x*row_bytes);
    uint16_t       * d = (uint16_t       *) (dst + blockIdx.x*row_bytes);
    const int n16 = 105*nb;
    for (int i = threadIdx.x; i < n16; i += blockDim.x) {
        row[i] = s[i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n16; i += blockDim.x) {
        int src_i;
        if (pack) {
            if      (i < 64*nb)  { const int b = i >> 6;                         src_i = 105*b +      (i & 63); }
            else if (i < 96*nb)  { const int j = i - 64*nb; const int b = j >> 5; src_i = 105*b + 64 + (j & 31); }
            else if (i < 104*nb) { const int j = i - 96*nb; const int b = j >> 3; src_i = 105*b + 96 + (j &  7); }
            else                 { const int b = i - 104*nb;                      src_i = 105*b + 104; }
        } else {
            const int b = i / 105, k = i - 105*b;
            if      (k <  64) { src_i = 64*b + k; }
            else if (k <  96) { src_i = 64*nb + 32*b + (k - 64); }
            else if (k < 104) { src_i = 96*nb +  8*b + (k - 96); }
            else              { src_i = 104*nb + b; }
        }
        d[i] = row[src_i];
    }
}

// q6r2 (GGML_Q6R2=1): a second step on the q6r row that rewrites the ql and qh planes (same sizes, same offsets)
// so that each 32-value group G = 8*sb + g reads only its own bits: ql2[16G + 4i + p] = low4(4i+p) | low4(16+4i+p) << 4
// (i = 0..3, p = 0..3) and qh2[8G + i] = sum_p hi2(4i+p) << 2p (i = 0..7).  Group g = 4n + t covers values
// 128n + 32t + l of the super-block; in q6r its low4(l) is nibble t>>1 of ql[128sb + 64n + 32(t&1) + l] and its hi2(l)
// is bits 2t..2t+1 of qh[128nb + 64sb + 32n + l].  scales and d stay where q6r put them.  One CUDA block per row.
static bool q6r2_on() {
    static const bool on = getenv("GGML_Q6R2") && atoi(getenv("GGML_Q6R2")) != 0;
    return on;
}

bool ggml_cuda_q6r2_on() {
    return q6r2_on();
}

template <bool pack>
static __global__ void q6r2_permute(const char * src, char * dst, const int nb, const size_t row_bytes) {
    extern __shared__ uint8_t pl[]; // the 192*nb bytes of the ql and qh planes of the source row
    const uint8_t * s = (const uint8_t *) (src + blockIdx.x*row_bytes);
    uint8_t       * d = (uint8_t       *) (dst + blockIdx.x*row_bytes);
    const int nplane = 192*nb;
    for (int i = threadIdx.x; i < nplane; i += blockDim.x) {
        pl[i] = s[i];
    }
    if (src != dst) {
        for (int i = nplane + threadIdx.x; i < 210*nb; i += blockDim.x) {
            d[i] = s[i];
        }
    }
    __syncthreads();
    const uint8_t * ql = pl;
    const uint8_t * qh = pl + 128*nb;
    if (pack) {
        // ql2: 16 bytes per group
        for (int o = threadIdx.x; o < 128*nb; o += blockDim.x) {
            const int G = o >> 4, j = o & 15, sb = G >> 3, g = G & 7, n = g >> 2, t = g & 3;
            const int l0 = j, l1 = 16 + j;   // j = 4i + p
            const int base = 128*sb + 64*n + 32*(t & 1), sh = 4*(t >> 1);
            d[o] = ((ql[base + l0] >> sh) & 0xF) | (((ql[base + l1] >> sh) & 0xF) << 4);
        }
        // qh2: 8 bytes per group
        for (int o = threadIdx.x; o < 64*nb; o += blockDim.x) {
            const int G = o >> 3, i = o & 7, sb = G >> 3, g = G & 7, n = g >> 2, t = g & 3;
            const uint8_t * h = qh + 64*sb + 32*n + 4*i;
            uint8_t v = 0;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                v |= ((h[p] >> (2*t)) & 3) << (2*p);
            }
            d[128*nb + o] = v;
        }
    } else {
        // q6r ql byte (sb, n, a, l) = low4 of group (n, a) | low4 of group (n, a+2) << 4
        for (int o = threadIdx.x; o < 128*nb; o += blockDim.x) {
            const int sb = o >> 7, r = o & 127, n = r >> 6, a = (r >> 5) & 1, l = r & 31;
            const int j = l & 15, hi = l >> 4;
            const int G0 = 8*sb + 4*n + a, G2 = G0 + 2;
            const int lo4 = (ql[16*G0 + j] >> (4*hi)) & 0xF;
            const int hi4 = (ql[16*G2 + j] >> (4*hi)) & 0xF;
            d[o] = lo4 | (hi4 << 4);
        }
        // q6r qh byte (sb, n, l) = sum_t hi2 of group (n, t) << 2t
        for (int o = threadIdx.x; o < 64*nb; o += blockDim.x) {
            const int sb = o >> 6, r = o & 63, n = r >> 5, l = r & 31, i = l >> 2, p = l & 3;
            uint8_t v = 0;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                v |= ((qh[8*(8*sb + 4*n + t) + i] >> (2*p)) & 3) << (2*t);
            }
            d[128*nb + o] = v;
        }
    }
}

// q6r2 row -> standard Q6_K row in one pass (the unpacked-scope path): rebuild the q6r row in shared memory with the
// q6r2_permute<false> mapping, then write the standard layout with the q6r_convert<false> mapping.
static __global__ void q6r2_unpack(const char * src, char * dst, const int nb, const size_t row_bytes) {
    extern __shared__ uint8_t sm[];
    uint8_t * a = sm;            // source row (q6r2), 210*nb bytes
    uint8_t * b = sm + 210*nb;   // rebuilt q6r row
    const uint8_t * s = (const uint8_t *) (src + blockIdx.x*row_bytes);
    uint16_t      * d = (uint16_t      *) (dst + blockIdx.x*row_bytes);
    for (int i = threadIdx.x; i < 210*nb; i += blockDim.x) {
        a[i] = s[i];
        if (i >= 192*nb) {
            b[i] = s[i];
        }
    }
    __syncthreads();
    const uint8_t * ql = a;
    const uint8_t * qh = a + 128*nb;
    for (int o = threadIdx.x; o < 128*nb; o += blockDim.x) {
        const int sb = o >> 7, r = o & 127, n = r >> 6, al = (r >> 5) & 1, l = r & 31;
        const int j = l & 15, hi = l >> 4;
        const int G0 = 8*sb + 4*n + al, G2 = G0 + 2;
        b[o] = ((ql[16*G0 + j] >> (4*hi)) & 0xF) | (((ql[16*G2 + j] >> (4*hi)) & 0xF) << 4);
    }
    for (int o = threadIdx.x; o < 64*nb; o += blockDim.x) {
        const int sb = o >> 6, r = o & 63, n = r >> 5, l = r & 31, i = l >> 2, p = l & 3;
        uint8_t v = 0;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            v |= ((qh[8*(8*sb + 4*n + t) + i] >> (2*p)) & 3) << (2*t);
        }
        b[128*nb + o] = v;
    }
    __syncthreads();
    const uint16_t * row = (const uint16_t *) b;
    for (int i = threadIdx.x; i < 105*nb; i += blockDim.x) {
        const int blk = i / 105, k = i - 105*blk;
        int src_i;
        if      (k <  64) { src_i = 64*blk + k; }
        else if (k <  96) { src_i = 64*nb + 32*blk + (k - 64); }
        else if (k < 104) { src_i = 96*nb +  8*blk + (k - 96); }
        else              { src_i = 104*nb + blk; }
        d[i] = row[src_i];
    }
}

// Q4_K (144 bytes = 36 u32 per super-block): standard block sb = u32 [36sb] d,dmin, [36sb+1 .. +3] scales[12],
// [36sb+4 .. +35] qs[128].  In qs, 64-value chunk j (0..3) is bytes 32j..32j+31; value 32g + l of group g = 2j + p
// (l = 0..31) is nibble p of byte 32j + l.
// Repacked row (q4kr): [qs plane 32*nb u32][header plane 4*nb u32].  Group g of sb is u32 [32sb + 4g .. +3]; its
// byte i has value i of the group in the low nibble and value i + 16 in the high nibble (Q4_0 order).  The header
// of sb (d, dmin, scales, unchanged) is u32 [32nb + 4sb .. +3].  A pure nibble/word permutation: pack o unpack = id.
// Every word is 4-aligned (row stride 144*nb, base 16-aligned per q4r_tensor_eligible), so the row is staged as u32.
template <bool pack>
static __global__ void q4kr_convert(const char * src, char * dst, const int nb, const size_t row_bytes) {
    extern __shared__ uint32_t row32[];
    const uint32_t * s = (const uint32_t *) (src + blockIdx.x*row_bytes);
    uint32_t       * d = (uint32_t       *) (dst + blockIdx.x*row_bytes);
    const int n32 = 36*nb;
    for (int i = threadIdx.x; i < n32; i += blockDim.x) {
        row32[i] = s[i];
    }
    __syncthreads();
    for (int o = threadIdx.x; o < n32; o += blockDim.x) {
        uint32_t v;
        if (pack) {
            if (o < 32*nb) {
                // output word w of group g of sb: low nibbles from raw qs word 8j + w (values 4w..4w+3 of the
                // group), high nibbles from raw qs word 8j + 4 + w (values 4w+16..4w+19), both at nibble p
                const int sb = o >> 5;
                const int g  = (o >> 2) & 7;
                const int w  = o & 3;
                const int j  = g >> 1;
                const int p  = 4*(g & 1);
                const uint32_t lo = row32[36*sb + 4 + 8*j + w];
                const uint32_t hi = row32[36*sb + 8 + 8*j + w];
                v = ((lo >> p) & 0x0F0F0F0Fu) | (((hi >> p) & 0x0F0F0F0Fu) << 4);
            } else {
                const int k = o - 32*nb;
                v = row32[36*(k >> 2) + (k & 3)];
            }
        } else {
            const int sb = o / 36;
            const int r  = o - 36*sb;
            if (r < 4) {
                v = row32[32*nb + 4*sb + r];
            } else {
                // raw qs word qi = 8j + 4hh + w: bytes 32j + 16hh + 4w .. +3 hold values 16hh + 4w .. +3 of group 2j
                // (low nibble) and of group 2j + 1 (high nibble); in the repacked words W(g) = u32 [32sb + 4g + w]
                // those values are the low (hh = 0) or high (hh = 1) nibbles.
                const int qi = r - 4;
                const int j  = qi >> 3;
                const int hh = (qi >> 2) & 1;
                const int w  = qi & 3;
                const uint32_t w0 = row32[32*sb + 8*j     + w];
                const uint32_t w1 = row32[32*sb + 8*j + 4 + w];
                v = hh == 0 ? ((w0 & 0x0F0F0F0Fu) | ((w1 & 0x0F0F0F0Fu) << 4))
                            : (((w0 >> 4) & 0x0F0F0F0Fu) | (w1 & 0xF0F0F0F0u));
            }
        }
        d[o] = v;
    }
}

// Q3_K (110 bytes per super-block): standard block sb = hmask[32] qs[64] scales[12] d[2].  Value v (0..31) of
// 32-value group g (n = g >> 2, j = g & 3) is q2 = (qs[32n + v] >> 2j) & 3, h = (hmask[v] >> (4n + j)) & 1, weight
// q2 - (h ? 0 : 4), scale index 2g + (v >> 4) (dequantize_q3_K in dequantize.cuh).
// Repacked row (q3r): [group plane 24*nb u32][scales plane 3*nb u32][d plane nb u16] = the same 110*nb bytes.
// Group t = 8sb + g is u32 [3t] = A, [3t + 1] = B, [3t + 2] = H.  In the a16 slot order of the kq kernels, slot
// s = 4k + r (k, r = 0..3) holds values lo = 4k + 16(r & 1) + (r >> 1) (low half) and lo + 2 (high half); its q2 bits
// are at bit 2r + 8(k & 1) (+ 16 for the high half) of A (k < 2) or B (k >= 2), and its h bit at bit 2r + 2 (+ 16) of
// the k-th h variant {H, H >> 1, rotr(H, 8), rotr(H, 9)}, i.e. bit (2r + 2 + 16 half + {0, 1, 8, 9}[k]) mod 32 of H.
// Both tilings cover their words exactly (checked by kquant\q3r_model.py, which also checks the decode).  The scales
// and d of sb are the original bytes.  A pure bit permutation: pack o unpack = id.  Needs an even block count
// (row bytes % 4 == 0) for the 32-bit loads.
static __device__ __forceinline__ int q3r_slot_lo(const int s) {
    return 4*(s >> 2) + 16*(s & 1) + ((s & 3) >> 1);
}
static __device__ __forceinline__ int q3r_qpos(const int s, const int half) {   // bit in A (s < 8) or B
    return 2*(s & 3) + 8*((s >> 2) & 1) + 16*half;
}
static __device__ __forceinline__ int q3r_hpos(const int s, const int half) {   // bit in H
    const int k = s >> 2;
    const int off = k == 0 ? 0 : k == 1 ? 1 : k == 2 ? 8 : 9;
    return (2*(s & 3) + 2 + 16*half + off) & 31;
}
// value v of a group -> (slot, half)
static __device__ __forceinline__ void q3r_slot_of(const int v, int & s, int & half) {
    const int u = v & 15;
    const int r = 2*(u & 1) + (v >> 4);
    half = (u & 3) >> 1;
    s = 4*(u >> 2) + r;
}

template <bool pack>
static __global__ void q3r_convert(const char * src, char * dst, const int nb, const size_t row_bytes) {
    extern __shared__ uint16_t row16[];
    const uint8_t * row = (const uint8_t *) row16;
    const uint16_t * s16 = (const uint16_t *) (src + blockIdx.x*row_bytes);
    uint8_t        * d   = (uint8_t        *) (dst + blockIdx.x*row_bytes);
    const int n16 = 55*nb;
    for (int i = threadIdx.x; i < n16; i += blockDim.x) {
        row16[i] = s16[i];
    }
    __syncthreads();
    const int nbytes = 110*nb;
    for (int o = threadIdx.x; o < nbytes; o += blockDim.x) {
        uint32_t v = 0;
        if (pack) {
            if (o < 96*nb) {
                const int wi = o >> 2, bi = o & 3;
                const int sb = wi/24, rr = wi - 24*sb, g = rr/3, kind = rr - 3*g;
                const uint8_t * blk = row + 110*sb;
                const int n = g >> 2, j = g & 3;
                for (int q = 0; q < 8; ++q) {
                    const int p = 8*bi + q;
                    int s = -1, half = 0;
                    if (kind < 2) {
                        const int pp = p & 15;
                        half = p >> 4;
                        s = 4*(2*kind + (pp >> 3)) + ((pp & 7) >> 1);
                        const int val = q3r_slot_lo(s) + 2*half;
                        const int q2 = (blk[32 + 32*n + val] >> (2*j)) & 3;
                        v |= (uint32_t) ((q2 >> (pp & 1)) & 1) << q;
                    } else {
                        for (int ss = 0; ss < 16 && s < 0; ++ss) {
                            for (int hh = 0; hh < 2; ++hh) {
                                if (q3r_hpos(ss, hh) == p) { s = ss; half = hh; break; }
                            }
                        }
                        const int val = q3r_slot_lo(s) + 2*half;
                        v |= (uint32_t) ((blk[val] >> (4*n + j)) & 1) << q;
                    }
                }
            } else if (o < 108*nb) {
                const int k = o - 96*nb;
                v = row[110*(k/12) + 96 + k%12];
            } else {
                const int k = o - 108*nb;
                v = row[110*(k >> 1) + 108 + (k & 1)];
            }
        } else {
            const int sb = o/110, r = o - 110*sb;
            const uint32_t * gw = (const uint32_t *) row;   // repacked row as u32 (row bytes % 4 == 0)
            if (r < 32) {
                // hmask byte r: bit t = h of value r of group t
                for (int t = 0; t < 8; ++t) {
                    int s, half;
                    q3r_slot_of(r, s, half);
                    const uint32_t H = gw[3*(8*sb + t) + 2];
                    v |= ((H >> q3r_hpos(s, half)) & 1u) << t;
                }
            } else if (r < 96) {
                // qs byte: bits 2j..2j+1 = q2 of value (q & 31) of group 4n + j
                const int q = r - 32, n = q >> 5, val = q & 31;
                for (int j = 0; j < 4; ++j) {
                    int s, half;
                    q3r_slot_of(val, s, half);
                    const uint32_t W = gw[3*(8*sb + 4*n + j) + (s >> 3)];
                    v |= ((W >> q3r_qpos(s, half)) & 3u) << (2*j);
                }
            } else if (r < 108) {
                v = row[96*nb + 12*sb + (r - 96)];
            } else {
                v = row[108*nb + 2*sb + (r - 108)];
            }
        }
        d[o] = (uint8_t) v;
    }
}

static void q4r_convert_launch(bool pack, ggml_type type, const void * src, void * dst, int64_t nrows, int64_t ne00, size_t row_bytes, cudaStream_t stream) {
    switch (q4r_layout_of(type)) {
        case Q4R_LAYOUT_Q3_K: {
            const int nb = (int) (ne00/QK_K);
            const size_t smem = 110*(size_t) nb;
            if (pack) {
                q3r_convert<true> <<<(unsigned) nrows, 256, smem, stream>>>((const char *) src, (char *) dst, nb, row_bytes);
            } else {
                q3r_convert<false><<<(unsigned) nrows, 256, smem, stream>>>((const char *) src, (char *) dst, nb, row_bytes);
            }
        } break;
        case Q4R_LAYOUT_Q6_K: {
            const int nb = (int) (ne00/QK_K);
            const size_t smem = 210*(size_t) nb;
            if (pack) {
                q6r_convert<true> <<<(unsigned) nrows, 256, smem, stream>>>((const char *) src, (char *) dst, nb, row_bytes);
                if (q6r2_on()) {
                    q6r2_permute<true><<<(unsigned) nrows, 256, 192*(size_t) nb, stream>>>((const char *) dst, (char *) dst, nb, row_bytes);
                }
            } else if (q6r2_on() && 2*smem <= 48*1024) {
                q6r2_unpack<<<(unsigned) nrows, 256, 2*smem, stream>>>((const char *) src, (char *) dst, nb, row_bytes);
            } else if (q6r2_on()) {
                // rows too long for the one-pass shared-memory budget
                q6r2_permute<false><<<(unsigned) nrows, 256, 192*(size_t) nb, stream>>>((const char *) src, (char *) dst, nb, row_bytes);
                q6r_convert<false><<<(unsigned) nrows, 256, smem, stream>>>((const char *) dst, (char *) dst, nb, row_bytes);
            } else {
                q6r_convert<false><<<(unsigned) nrows, 256, smem, stream>>>((const char *) src, (char *) dst, nb, row_bytes);
            }
        } break;
        case Q4R_LAYOUT_Q4_0: {
            const int nb = (int) (ne00/32);
            const size_t smem = 18*(size_t) nb;
            if (pack) {
                q4r_convert<true> <<<(unsigned) nrows, 256, smem, stream>>>((const char *) src, (char *) dst, nb, row_bytes);
            } else {
                q4r_convert<false><<<(unsigned) nrows, 256, smem, stream>>>((const char *) src, (char *) dst, nb, row_bytes);
            }
        } break;
        case Q4R_LAYOUT_Q4_K: {
            const int nb = (int) (ne00/QK_K);
            const size_t smem = 144*(size_t) nb;
            if (pack) {
                q4kr_convert<true> <<<(unsigned) nrows, 256, smem, stream>>>((const char *) src, (char *) dst, nb, row_bytes);
            } else {
                q4kr_convert<false><<<(unsigned) nrows, 256, smem, stream>>>((const char *) src, (char *) dst, nb, row_bytes);
            }
        } break;
        default:
            GGML_ABORT("q4r: no repacked layout for type %s (%s)", ggml_type_name(type), pack ? "pack" : "unpack");
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------- GEMV on the repacked layout

// Same activation format and LOP3 half2 expansion as mul_mat_vec_q4_0_a16 in mmvq-f16-sm60.cu; the only
// difference is the weight load: one aligned 8-byte load per thread per row instead of three 4-byte loads
// and two funnel shifts, plus a scale read from the row's scale tail.
template <int ncols_dst, int nwarps>
static __global__ void __launch_bounds__(WARP_SIZE*nwarps, (ncols_dst >= 6 ? 2 : 4)*(Q4R_NWARPS/nwarps))
mul_mat_vec_q4r(
        const char * __restrict__ vx, const half2 * __restrict__ aq, const half2 * __restrict__ ads,
        float * __restrict__ dst, const int nblocks, const int row_bytes,
        const int stride_col_aq, const int stride_col_ads, const int stride_col_dst) {
#if defined(FP16_AVAILABLE)
    const int row0     = blockIdx.x*Q4R_ROWS;
    const int tid      = threadIdx.x + threadIdx.y*WARP_SIZE;
    const int nthreads = WARP_SIZE*nwarps;

    const half2 magic_lo = __float2half2_rn(1032.0f);
    const half2 magic_hi = __float2half2_rn(  72.0f);

    float sumf[ncols_dst][Q4R_ROWS] = {{0.0f}};

    const char * const xbase = vx + (size_t) row0*row_bytes;
    const int dtail = 16*nblocks;

    for (int t = tid; t < 2*nblocks; t += nthreads) {
        const int kb = t >> 1;
        const int kh = t &  1;

        half2 a[ncols_dst][8];
        float yd[ncols_dst];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const half2 * ap = aq + (size_t) j*stride_col_aq + kb*(Q4R_QK/2) + kh*8;
#pragma unroll
            for (int c2 = 0; c2 < 2; ++c2) {
                const int4 av = *((const int4 *) (ap + 4*c2));
                a[j][4*c2 + 0] = *((const half2 *) &av.x);
                a[j][4*c2 + 1] = *((const half2 *) &av.y);
                a[j][4*c2 + 2] = *((const half2 *) &av.z);
                a[j][4*c2 + 3] = *((const half2 *) &av.w);
            }
            yd[j] = __low2float(ads[j*stride_col_ads + kb]);
        }

#pragma unroll
        for (int i = 0; i < Q4R_ROWS; ++i) {
            const char * xr = xbase + i*row_bytes;
            const uint2 q = *((const uint2 *) (xr + 16*kb + 8*kh));
            const float d = __half2float(*((const half *) (xr + dtail + 2*kb)));

            half2 w[8];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int w32  = k == 0 ? (int) q.x : (int) q.y;
                const int w32s = w32 >> 8;
                int t0, t1, t2, t3;
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t0) : "r"(w32),  "n"(0x000f000f), "n"(0x64006400));
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t1) : "r"(w32),  "n"(0x00f000f0), "n"(0x54005400));
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t2) : "r"(w32s), "n"(0x000f000f), "n"(0x64006400));
                asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(t3) : "r"(w32s), "n"(0x00f000f0), "n"(0x54005400));
                w[4*k + 0] = __hsub2(*((const half2 *) &t0), magic_lo);
                w[4*k + 1] = __hsub2(*((const half2 *) &t1), magic_hi);
                w[4*k + 2] = __hsub2(*((const half2 *) &t2), magic_lo);
                w[4*k + 3] = __hsub2(*((const half2 *) &t3), magic_hi);
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                half2 s = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    s = __hfma2(w[c], a[j][c], s);
                }
                const float af = __half2float(__hadd(__low2half(s), __high2half(s)));
                sumf[j][i] += yd[j]*d*af;
            }
        }
    }

    if constexpr (nwarps == 1) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < Q4R_ROWS; ++i) {
                const float v = warp_reduce_sum(sumf[j][i]);
                if (threadIdx.x == i) {
                    dst[j*stride_col_dst + row0 + i] = v;
                }
            }
        }
    } else {
        __shared__ float tmp[nwarps][ncols_dst][Q4R_ROWS];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < Q4R_ROWS; ++i) {
                const float v = warp_reduce_sum(sumf[j][i]);
                if (threadIdx.x == 0) {
                    tmp[threadIdx.y][j][i] = v;
                }
            }
        }
        __syncthreads();
        if (threadIdx.y != 0) {
            return;
        }
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            if (threadIdx.x < Q4R_ROWS) {
                float v = 0.0f;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
                    v += tmp[w][j][threadIdx.x];
                }
                dst[j*stride_col_dst + row0 + threadIdx.x] = v;
            }
        }
    }
#else
    GGML_UNUSED_VARS(vx, aq, ads, dst, nblocks, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

template <int ncols_dst>
static void q4r_launch(const char * vx, const half2 * aq, const half2 * ads, float * dst, int nblocks, int nrows,
        int row_bytes, int stride_col_aq, int stride_col_ads, int stride_col_dst, cudaStream_t stream) {
    const dim3 grid(nrows/Q4R_ROWS, 1, 1);
    static const int forced = getenv("GGML_Q4R_NWARPS") ? atoi(getenv("GGML_Q4R_NWARPS")) : 0;
    int nw = forced == 1 || forced == 2 || forced == 4 ? forced : (ncols_dst == 1 ? 4 : 2);
    while (nw > 1 && 2*nblocks < WARP_SIZE*nw) {
        nw /= 2;
    }
    switch (nw) {
        case 1:
            mul_mat_vec_q4r<ncols_dst, 1><<<grid, dim3(WARP_SIZE, 1, 1), 0, stream>>>(
                vx, aq, ads, dst, nblocks, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
            break;
        case 2:
            mul_mat_vec_q4r<ncols_dst, 2><<<grid, dim3(WARP_SIZE, 2, 1), 0, stream>>>(
                vx, aq, ads, dst, nblocks, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
            break;
        default:
            mul_mat_vec_q4r<ncols_dst, 4><<<grid, dim3(WARP_SIZE, 4, 1), 0, stream>>>(
                vx, aq, ads, dst, nblocks, row_bytes, stride_col_aq, stride_col_ads, stride_col_dst);
            break;
    }
}

// Dequantize a repacked tensor straight to row-major f16 -- what cuBLAS needs for prefill anyway, so the
// wide path pays no extra pass over the weights.  Same arithmetic as dequantize_q4_0 (float d*(q-8), one
// rounding to half), so the result is bit-identical to the standard path.
// One thread per 4 quant bytes (one aligned 32-bit load) producing elements j..j+3 and j+16..j+19 of the
// block as two 8-byte stores.  A thread per byte measured 579 us/call against 399 for upstream's
// dequantize_block_q4_0 on the same tensors (nvprof, 7B pp512).  Flat 1D grid: the output head has more
// rows than gridDim.y allows.
static __global__ void q4r_to_f16(const char * __restrict__ vx, half * __restrict__ y, const int nb, const int row_bytes, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; // 4-byte group over the whole tensor
    if (i >= n) {
        return;
    }
    const int64_t row = i / (4*nb);
    const int     g   = (int) (i - row*4*nb);
    const int     kb  = g >> 2;
    const int     j   = 4*(g & 3);
    const char * xr = vx + row*row_bytes;
    const uint32_t q = *((const uint32_t *) (xr + 4*g));
    const float    d = __half2float(*((const half *) (xr + 16*nb + 2*kb)));
    half2 lo[2], hi[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const uint32_t b0 = (q >> (16*k)) & 0xFF, b1 = (q >> (16*k + 8)) & 0xFF;
        lo[k] = __floats2half2_rn(((int) (b0 & 0xF) - 8)*d, ((int) (b1 & 0xF) - 8)*d);
        hi[k] = __floats2half2_rn(((int) (b0 >>  4) - 8)*d, ((int) (b1 >>  4) - 8)*d);
    }
    half * yr = y + row*nb*32 + kb*32 + j;
    *((uint2 *) (yr     )) = *((const uint2 *) lo);
    *((uint2 *) (yr + 16)) = *((const uint2 *) hi);
}

// Repacked Q6_K to f16, same arithmetic as dequantize_block_q6_K.  Each thread reads 4 consecutive bytes of
// ql (twice) and qh with 32-bit loads and makes 16 elements (4 runs of 4) with 8-byte stores; 16 threads per
// super-block, 16 super-blocks per CUDA block.  One byte per thread measured 342-402 us/call against 292 for
// upstream (7B pp512, nvprof).
static __global__ void q6r_to_f16(const char * __restrict__ vx, half * __restrict__ y, const int nb, const int row_bytes, const int nsb) {
    const int gsb = blockIdx.x*16 + (threadIdx.x >> 4);
    if (gsb >= nsb) {
        return;
    }
    const int row = gsb / nb;
    const int b   = gsb - row*nb;
    const int ip  = (threadIdx.x >> 3) & 1;
    const int l   = 4*(threadIdx.x & 7);
    const char * xr = vx + (size_t) row*row_bytes;
    const uint32_t q0 = *((const uint32_t *) (xr + 128*b + 64*ip + l));
    const uint32_t q1 = *((const uint32_t *) (xr + 128*b + 64*ip + l + 32));
    const uint32_t qh = *((const uint32_t *) (xr + 128*nb + 64*b + 32*ip + l));
    const int8_t * sc = (const int8_t *) (xr + 192*nb + 16*b + 8*ip + l/16);
    const float    d  = __half2float(*((const half *) (xr + 208*nb + 2*b)));
    const float d0 = d*sc[0], d2 = d*sc[2], d4 = d*sc[4], d6 = d*sc[6];
    half2 o0[2], o1[2], o2[2], o3[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        int v[4][2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int sh = 8*(2*k + e);
            const int a0 = (q0 >> sh) & 0xFF, a1 = (q1 >> sh) & 0xFF, h = (qh >> sh) & 0xFF;
            v[0][e] = ((a0 & 0xF) | (((h >> 0) & 3) << 4)) - 32;
            v[1][e] = ((a1 & 0xF) | (((h >> 2) & 3) << 4)) - 32;
            v[2][e] = ((a0 >>  4) | (((h >> 4) & 3) << 4)) - 32;
            v[3][e] = ((a1 >>  4) | (((h >> 6) & 3) << 4)) - 32;
        }
        o0[k] = __floats2half2_rn(d0*v[0][0], d0*v[0][1]);
        o1[k] = __floats2half2_rn(d2*v[1][0], d2*v[1][1]);
        o2[k] = __floats2half2_rn(d4*v[2][0], d4*v[2][1]);
        o3[k] = __floats2half2_rn(d6*v[3][0], d6*v[3][1]);
    }
    half * yr = y + (size_t) gsb*QK_K + 128*ip + l;
    *((uint2 *) (yr +  0)) = *((const uint2 *) o0);
    *((uint2 *) (yr + 32)) = *((const uint2 *) o1);
    *((uint2 *) (yr + 64)) = *((const uint2 *) o2);
    *((uint2 *) (yr + 96)) = *((const uint2 *) o3);
}

// q6r2 to f16: one thread per 32-value group, same arithmetic as q6r_to_f16 (float d*sc, times q - 32, one rounding).
static __global__ void q6r2_to_f16(const char * __restrict__ vx, half * __restrict__ y, const int nb, const int row_bytes, const int64_t ngroups) {
    const int64_t gt = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (gt >= ngroups) {
        return;
    }
    const int row = (int) (gt / (8*nb));
    const int G   = (int) (gt - (int64_t) row*8*nb);
    const int sb = G >> 3, g = G & 7, n = g >> 2, t = g & 3;
    const uint8_t * xr = (const uint8_t *) (vx + (size_t) row*row_bytes);
    const uint8_t * ql = xr + 16*G;
    const uint8_t * qh = xr + 128*nb + 8*G;
    const int8_t  * sc = (const int8_t *) (xr + 192*nb + 16*sb + 8*n + 2*t);
    const float d  = __half2float(*((const half *) (xr + 208*nb + 2*sb)));
    const float d0 = d*sc[0], d1 = d*sc[1];
    half out[32];
#pragma unroll
    for (int l = 0; l < 32; ++l) {
        const int lo = (ql[l & 15] >> (4*(l >> 4))) & 0xF;
        const int hi = (qh[l >> 2] >> (2*(l & 3))) & 3;
        const int v  = (lo | (hi << 4)) - 32;
        out[l] = __float2half_rn((l < 16 ? d0 : d1)*v);
    }
    half * yr = y + (size_t) row*nb*QK_K + 128*n + 32*t + 256*sb;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        ((uint4 *) yr)[c] = ((const uint4 *) out)[c];
    }
}

// get_scale_min_k4 of dequantize.cuh / ggml-quants.c (q = the 12 scale bytes)
static __device__ __forceinline__ void q4kr_scale_min(const int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

// Repacked Q4_K to f16 with the arithmetic of dequantize_q4_K (dequantize.cuh): dall/dmin from the half2 header,
// d1 = dall*sc, m1 = dmin*m, element = d1*q - m1 in float with one rounding to half -- written as the same expression
// so the compiler contracts it the same way (FMUL, FMUL, FFMA(d1, q, -m1), F2F.F16 in both SASS), and the f16 copy
// (and so the cuBLAS prefill) is bit-identical to the standard path's.  One warp per super-block (8 per CUDA block,
// flat over the whole tensor like q6r_to_f16): lane = u32 w of group g of the qs plane, making elements 4w..4w+3
// (low nibbles) and 4w+16..4w+19 (high nibbles) of the group as two 8-byte stores; the header load is a broadcast.
static __global__ void q4kr_to_f16(const char * __restrict__ vx, half * __restrict__ y, const int nb, const int row_bytes, const int nsb) {
    const int gsb = blockIdx.x*8 + (threadIdx.x >> 5);
    if (gsb >= nsb) {
        return;
    }
    const int row  = gsb / nb;
    const int sb   = gsb - row*nb;
    const int lane = threadIdx.x & 31;
    const int g    = lane >> 2;
    const int w    = lane & 3;
    const char * xr = vx + (size_t) row*row_bytes;
    const uint32_t q = *((const uint32_t *) (xr + 128*sb + 4*lane));
    const uint8_t * hdr = (const uint8_t *) (xr + 128*nb + 16*sb);
    const half2 dm = *((const half2 *) hdr);

    const float dall = __low2half(dm);
    const float dmin = __high2half(dm);

    uint8_t sc, m;
    q4kr_scale_min(g, hdr + 4, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;

    half lo[4], hi[4];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint8_t ql = (q >> (8*l)) & 0xFF;
        lo[l] = ggml_cuda_cast<half>(d1 * (ql & 0xF) - m1);
        hi[l] = ggml_cuda_cast<half>(d1 * (ql >>  4) - m1);
    }
    half * yr = y + (size_t) gsb*QK_K + 32*g + 4*w;
    *((uint2 *) (yr     )) = *((const uint2 *) lo);
    *((uint2 *) (yr + 16)) = *((const uint2 *) hi);
}

// Repacked Q3_K to f16 with dequantize_q3_K's arithmetic (dl = d * (us - 32) in float, element = dl * (q2 - (h ? 0 : 4))
// with one rounding to half), so the f16 copy for the cuBLAS prefill is bit-identical to the standard path's.  One
// thread per 32-value group: 3 group words + 2 scale words + d, 32 halves out as four 16-byte stores.
static __global__ void q3r_to_f16(const char * __restrict__ vx, half * __restrict__ y, const int nb, const int row_bytes, const int64_t ngroups) {
    const int64_t gt = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (gt >= ngroups) {
        return;
    }
    const int64_t row = gt / (8*nb);
    const int t  = (int) (gt - row*8*nb);
    const int sb = t >> 3, g = t & 7;
    const char * xr = vx + row*row_bytes;
    const uint32_t * gw = (const uint32_t *) xr + 3*t;
    const uint32_t A = gw[0], B = gw[1], H = gw[2];
    const uint8_t * sc = (const uint8_t *) (xr + 96*nb + 12*sb);
    const float d_all = __half2float(*((const half *) (xr + 108*nb + 2*sb)));

    float dl[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int is = 2*g + k;
        const int8_t us = is <  4 ? (sc[is-0] & 0xF) | (((sc[is+8] >> 0) & 3) << 4) :
                          is <  8 ? (sc[is-0] & 0xF) | (((sc[is+4] >> 2) & 3) << 4) :
                          is < 12 ? (sc[is-8] >>  4) | (((sc[is+0] >> 4) & 3) << 4) :
                                    (sc[is-8] >>  4) | (((sc[is-4] >> 6) & 3) << 4);
        dl[k] = d_all * (us - 32);
    }
    const uint32_t V[4] = {H, H >> 1, __funnelshift_r(H, H, 8), __funnelshift_r(H, H, 9)};
    __align__(16) half out[32];
#pragma unroll
    for (int s = 0; s < 16; ++s) {
        const int k = s >> 2, r = s & 3;
        const uint32_t W = (k < 2 ? A : B) >> (8*(k & 1));
        const int lo = 4*k + 16*(r & 1) + (r >> 1);
#pragma unroll
        for (int half_i = 0; half_i < 2; ++half_i) {
            const int q2 = (W >> (2*r + 16*half_i)) & 3;
            const int h  = (V[k] >> (2*r + 2 + 16*half_i)) & 1;
            const int v  = lo + 2*half_i;
            out[v] = ggml_cuda_cast<half>(dl[v >> 4] * ((int8_t) q2 - (h ? 0 : 4)));
        }
    }
    uint4 * yo = (uint4 *) (y + (size_t) gt*32);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        yo[c] = ((const uint4 *) out)[c];
    }
}

void ggml_cuda_q4r_to_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream) {
    switch (q4r_layout_of(src0->type)) {
        case Q4R_LAYOUT_Q3_K: {
            const int nb = (int) (src0->ne[0]/QK_K);
            const int64_t ngroups = src0->ne[1]*nb*8;
            q3r_to_f16<<<(unsigned) ((ngroups + 255)/256), 256, 0, stream>>>((const char *) src0->data, dst, nb, (int) src0->nb[1], ngroups);
        } break;
        case Q4R_LAYOUT_Q6_K: {
            const int nb = (int) (src0->ne[0]/QK_K);
            const int nsb = (int) (src0->ne[1]*nb);
            if (q6r2_on()) {
                const int64_t ngroups = (int64_t) nsb*8;
                q6r2_to_f16<<<(unsigned) ((ngroups + 255)/256), 256, 0, stream>>>((const char *) src0->data, dst, nb, (int) src0->nb[1], ngroups);
            } else {
                q6r_to_f16<<<(unsigned) ((nsb + 15)/16), 256, 0, stream>>>((const char *) src0->data, dst, nb, (int) src0->nb[1], nsb);
            }
        } break;
        case Q4R_LAYOUT_Q4_0: {
            const int nb = (int) (src0->ne[0]/32);
            const int64_t n = src0->ne[1]*nb*4;
            q4r_to_f16<<<(unsigned) ((n + 255)/256), 256, 0, stream>>>((const char *) src0->data, dst, nb, (int) src0->nb[1], n);
        } break;
        case Q4R_LAYOUT_Q4_K: {
            const int nb  = (int) (src0->ne[0]/QK_K);
            const int nsb = (int) (src0->ne[1]*nb);
            q4kr_to_f16<<<(unsigned) ((nsb + 7)/8), 256, 0, stream>>>((const char *) src0->data, dst, nb, (int) src0->nb[1], nsb);
        } break;
        default:
            GGML_ABORT("q4r: to_f16 has no repacked layout for type %s", ggml_type_name(src0->type));
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------- entry points

// The backend context's device (REVIEW-1 F8), not the current-device global; the CUDA backend sets one to the
// other before it computes, so the result is the same, but the exclusion of DP4A cards no longer depends on it.
static bool q4r_device_ok(const int device) {
    const int cc = ggml_cuda_info().devices[device].cc;
    // pf-gemm, test only: GGML_PF_TEST_ANY_ARCH=1 also repacks on newer NVIDIA cards, so the repack-aware prefill
    // paths (ggml_cuda_q4r_to_f16 into the fold GEMM) can be checked off the P100. Unset: unchanged.
    static const bool any_arch = getenv("GGML_PF_TEST_ANY_ARCH") && atoi(getenv("GGML_PF_TEST_ANY_ARCH")) != 0;
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_PASCAL && (cc < GGML_CUDA_CC_DP4A || any_arch);
}

static bool q4r_tensor_eligible(const ggml_tensor * t) {
    const bool common = t->buffer && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
           t->view_src == nullptr && ggml_is_contiguous(t) && t->ne[2] == 1 && t->ne[3] == 1 &&
           t->ne[0] % 256 == 0 && t->ne[1] % Q4R_ROWS == 0 && t->nb[1] <= 48*1024 && ((uintptr_t) t->data) % 16 == 0;
    if (!common) {
        return false;
    }
    switch (q4r_layout_of(t->type)) {
        case Q4R_LAYOUT_Q4_0:
            return true;
        case Q4R_LAYOUT_Q6_K:
            // an even block count keeps every row 4-byte aligned for the 32-bit loads in kq-sm60.cu
            return (t->ne[0]/QK_K) % 2 == 0;
        case Q4R_LAYOUT_Q4_K:
            // 144-byte blocks: every row and every 16-byte group/header is 16-aligned for any block count, so no
            // parity rule (the 49- and 15-block axis-0 shards qualify).  Rows % 4 (the GEMV row tile) is in `common`;
            // ragged shards (e.g. 30-row ssm_alpha/beta) stay NEVER and keep today's path.
            return true;
        case Q4R_LAYOUT_Q3_K:
            // 110-byte blocks: an even block count keeps every row 4-aligned for the u32 group/scale loads; odd
            // counts (e.g. a 49-block ffn_down shard) stay NEVER on generic MMVQ.
            return (t->ne[0]/QK_K) % 2 == 0;
        default:
            return false;
    }
}

static bool q4r_kernel_applies(const ggml_tensor * src1, const ggml_tensor * dst) {
    return src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && src1->ne[1] >= 1 && src1->ne[1] <= 8 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 && ggml_is_contiguous(dst) && src1->nb[0] == sizeof(float);
}

static void q4r_gemv(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();
    const int nblocks = (int) (src0->ne[0]/Q4R_QK);
    const int64_t ne11 = src1->ne[1];

    ggml_cuda_pool_alloc<half2> aq_own (ctx.pool());
    ggml_cuda_pool_alloc<half2> ads_own(ctx.pool());
    const half2 * aqp  = nullptr;
    const half2 * adsp = nullptr;
    if (!ggml_cuda_a16_cached(ctx, src1, nblocks, &aqp, &adsp)) {
        aq_own.alloc ((size_t) ne11*nblocks*(Q4R_QK/2));
        ads_own.alloc((size_t) ne11*nblocks);
        const int64_t s11 = src1->nb[1]/sizeof(float);
        const bool vec4 = s11 % 4 == 0 && ((uintptr_t) src1->data) % 16 == 0;
        ggml_cuda_a16_quantize((const float *) src1->data, aq_own.get(), ads_own.get(), s11, nblocks, (int) ne11, vec4, stream);
        aqp  = aq_own.get();
        adsp = ads_own.get();
    }

    const int row_bytes      = (int) src0->nb[1];
    const int stride_col_aq  = nblocks*(Q4R_QK/2);
    const int stride_col_ads = nblocks;
    const int stride_col_dst = (int) (dst->nb[1]/sizeof(float));
    const char * vx = (const char *) src0->data;
    float * d = (float *) dst->data;
    const int nrows = (int) src0->ne[1];

#define Q4R_CASE(N) case N: q4r_launch<N>(vx, aqp, adsp, d, nblocks, nrows, row_bytes, \
                                          stride_col_aq, stride_col_ads, stride_col_dst, stream); break;
    switch (ne11) {
        Q4R_CASE(1) Q4R_CASE(2) Q4R_CASE(3) Q4R_CASE(4)
        Q4R_CASE(5) Q4R_CASE(6) Q4R_CASE(7) Q4R_CASE(8)
        default: GGML_ABORT("q4r: unsupported ncols");
    }
#undef Q4R_CASE
}

// First sight of src0->data on this device: decide PACKED or NEVER for the life of the process and, for PACKED,
// convert the tensor in place on ctx's stream.  REVIEW-1 F8: q4r_mutex is held from the re-check to the publish, so
// two threads (e.g. an MTP context sharing the weights) cannot both convert the same tensor (a double permutation
// is garbage), and for q4kr the stream is synchronized before PACKED becomes visible, so a kernel on another
// stream can never read the new layout before the conversion has run.  (Q4_0/Q6_K keep their previous timing: no
// sync, as before GGML_KQR_Q4K existed.)  First sight is never inside a capture (checked below), so the sync is
// legal.
static q4r_state q4r_first_sight(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const q4r_layout layout) {
    std::lock_guard<std::mutex> lock(q4r_mutex);
    {
        auto it = q4r_states.find(src0->data);
        if (it != q4r_states.end()) {
            return it->second; // decided by another thread since the unlocked lookup
        }
    }

    static const bool off     = getenv("GGML_Q4R") && atoi(getenv("GGML_Q4R")) == 0;
    static const bool off_kq  = getenv("GGML_KQ")  && atoi(getenv("GGML_KQ"))  == 0;
    static const bool off_q6  = (getenv("GGML_Q6R") && atoi(getenv("GGML_Q6R")) == 0) || off_kq;
    // q4kr replaces kq's Q4_K kernel, so the kq switches turn it off as well
    static const bool off_q4k = off_kq || (getenv("GGML_KQ_K") && atoi(getenv("GGML_KQ_K")) == 0);
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &cap));
    bool type_off = true;
    switch (layout) {
        case Q4R_LAYOUT_Q4_0: type_off = off;     break;
        case Q4R_LAYOUT_Q6_K: type_off = off_q6;  break;
        case Q4R_LAYOUT_Q4_K: type_off = off_q4k; break;
        case Q4R_LAYOUT_Q3_K: type_off = false;   break; // GGML_Q3R gates the layout itself
        default: GGML_ABORT("q4r: no env switch for type %s", ggml_type_name(src0->type));
    }
    // GGML_KQR_VERBOSE=1: one log line per first sight (PACKED, or NEVER and why), so a test or a model run
    // can show which tensors on which device were repacked.
    const bool verbose = q4r_verbose();
    const int  why     = type_off                           ? Q4R_WHY_ENV      :
                         cap != cudaStreamCaptureStatusNone ? Q4R_WHY_CAPTURE  :
                         !q4r_device_ok(ctx.device)         ? Q4R_WHY_DEVICE   :
                         !q4r_tensor_eligible(src0)         ? Q4R_WHY_ELIGIBLE : -1;
    q4r_counts & counts = q4r_stats[ctx.device][layout];
    q4r_stats_dirty = true;
    if (why >= 0) {
        if (verbose) {
            GGML_LOG_INFO("q4r: %s %s [%lld x %lld] on device %d: NEVER (%s)\n", src0->name, ggml_type_name(src0->type),
                          (long long) src0->ne[0], (long long) src0->ne[1], ctx.device, q4r_why_name[why]);
        }
        counts.never[why]  += 1;
        counts.never_bytes += (int64_t) ggml_nbytes(src0);
        q4r_states[src0->data] = Q4R_NEVER;
        return Q4R_NEVER;
    }
    q4r_convert_launch(true, src0->type, src0->data, src0->data, src0->ne[1], src0->ne[0], src0->nb[1], ctx.stream());
    if (layout == Q4R_LAYOUT_Q4_K || layout == Q4R_LAYOUT_Q3_K) {
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    }
    q4r_states[src0->data] = Q4R_PACKED;
    counts.packed       += 1;
    counts.packed_bytes += (int64_t) ggml_nbytes(src0);
    if (verbose) {
        GGML_LOG_INFO("q4r: %s %s [%lld x %lld] on device %d: PACKED\n", src0->name, ggml_type_name(src0->type),
                      (long long) src0->ne[0], (long long) src0->ne[1], ctx.device);
    }
    return Q4R_PACKED;
}

// pf-gemm: the first part of ggml_cuda_q4r_mul_mat (lookup, first sight), for MUL_MATs the fold GEMM computes
// without passing through it. Same decisions, same verbose output.
bool ggml_cuda_q4r_settle(ggml_backend_cuda_context & ctx, const ggml_tensor * t) {
    const q4r_layout layout = q4r_layout_of(t->type);
    if (layout == Q4R_LAYOUT_NONE || q4r_bypass > 0) {
        return false;
    }
    q4r_state state;
    if (!q4r_lookup(t->data, state)) {
        state = q4r_first_sight(ctx, t, layout);
    }
    return state == Q4R_PACKED;
}

int ggml_cuda_q4r_mul_mat(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const q4r_layout layout = q4r_layout_of(src0->type);
    if (layout == Q4R_LAYOUT_NONE || q4r_bypass > 0) {
        return GGML_CUDA_Q4R_NONE;
    }

    q4r_state state;
    if (!q4r_lookup(src0->data, state)) {
        state = q4r_first_sight(ctx, src0, layout);
    } else if (q4r_verbose()) {
        // a known tensor: a batch of first sights (if any) is over, print the counts once
        std::lock_guard<std::mutex> lock(q4r_mutex);
        if (q4r_stats_dirty) {
            q4r_print_counts_locked();
        }
    }
    if (state != Q4R_PACKED) {
        return GGML_CUDA_Q4R_NONE;
    }

    // Any width the packed kernel does not take: the caller runs the normal path on a standard-layout copy
    // (or, for wide products, on an f16 copy made by ggml_cuda_q4r_to_f16).
    switch (layout) {
        case Q4R_LAYOUT_Q6_K:
            return ggml_cuda_kq_q6r_mul_mat(ctx, src0, src1, dst) ? GGML_CUDA_Q4R_DONE : GGML_CUDA_Q4R_UNPACK;
        case Q4R_LAYOUT_Q4_0:
            if (q4r_kernel_applies(src1, dst)) {
                q4r_gemv(ctx, src0, src1, dst);
                return GGML_CUDA_Q4R_DONE;
            }
            return GGML_CUDA_Q4R_UNPACK;
        case Q4R_LAYOUT_Q4_K:
            // widths 1..8 in kq-sm60.cu (5..8 as two passes); wider -> ggml_cuda_q4r_to_f16 + cuBLAS in the caller
            return ggml_cuda_kq_q4kr_mul_mat(ctx, src0, src1, dst) ? GGML_CUDA_Q4R_DONE : GGML_CUDA_Q4R_UNPACK;
        case Q4R_LAYOUT_Q3_K:
            // widths 1..8 in kq-sm60.cu (5..8 as two passes); wider -> q3r_to_f16 + cuBLAS in the caller
            return ggml_cuda_kq_q3r_mul_mat(ctx, src0, src1, dst) ? GGML_CUDA_Q4R_DONE : GGML_CUDA_Q4R_UNPACK;
        default:
            GGML_ABORT("q4r: no GEMV for repacked type %s", ggml_type_name(src0->type));
    }
}

ggml_cuda_q4r_unpacked_scope::ggml_cuda_q4r_unpacked_scope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) : ctx(ctx) {
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        ggml_tensor * s = dst->src[i];
        if (!ggml_cuda_q4r_is_repacked(s)) {
            continue;
        }
        const size_t n = ggml_nbytes(s);
        tmp[i]   = ctx.pool().alloc(n, &tmp_size[i]);
        q4r_convert_launch(false, s->type, s->data, tmp[i], s->ne[1], s->ne[0], s->nb[1], ctx.stream());
        srcs[i]  = s;
        saved[i] = s->data;
        s->data  = tmp[i];
    }
    ++q4r_bypass;
}

ggml_cuda_q4r_unpacked_scope::~ggml_cuda_q4r_unpacked_scope() {
    --q4r_bypass;
    // the pool is LIFO: release in reverse order
    for (int i = GGML_MAX_SRC - 1; i >= 0; --i) {
        if (srcs[i]) {
            srcs[i]->data = saved[i];
            ctx.pool().free(tmp[i], tmp_size[i]);
        }
    }
}
