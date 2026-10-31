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

// ---------------------------------------------------------------------------------------------------------
// Q2_K (84 bytes per super-block): standard block sb = scales[16] qs[64] d[2] dmin[2].  Value v (0..31) of 32-value group
// g (n = g >> 2, j = g & 3) is q2 = (qs[32n + v] >> 2j) & 3 (the same qs tiling as Q3_K), scale index 2g + (v >> 4);
// scales byte = (min << 4) | scale.  w = d*scale*q2 - dmin*min.
// Repacked row (q2kr): [group plane 16*nb u32][header plane 20*nb bytes] = the same 84*nb bytes.  Group t = 8sb + g is
// u32 [2t] = A, [2t + 1] = B holding the q2 bits in q3r's A / B tiling (q3r_slot_lo / q3r_qpos); sb's header (at
// 64*nb + 20*sb) is its 16 scale bytes then d, dmin.  A pure bit permutation: pack o unpack = id.  Needs rb % 8 == 0.
template <bool pack>
static __global__ void q2kr_convert(const char * src, char * dst, const int nb, const size_t row_bytes) {
    extern __shared__ uint16_t row16[];
    const uint8_t * row = (const uint8_t *) row16;
    const uint16_t * s16 = (const uint16_t *) (src + blockIdx.x*row_bytes);
    uint8_t        * d   = (uint8_t        *) (dst + blockIdx.x*row_bytes);
    const int n16 = 42*nb;
    for (int i = threadIdx.x; i < n16; i += blockDim.x) row16[i] = s16[i];
    __syncthreads();
    const int nbytes = 84*nb;
    for (int o = threadIdx.x; o < nbytes; o += blockDim.x) {
        uint32_t v = 0;
        if (pack) {
            if (o < 64*nb) {
                const int wi = o >> 2, bi = o & 3;
                const int t = wi >> 1, kind = wi & 1, sb = t >> 3, g = t & 7, n = g >> 2, j = g & 3;
                const uint8_t * blk = row + 84*sb;
                for (int q = 0; q < 8; ++q) {
                    const int p = 8*bi + q, pp = p & 15, half = p >> 4;
                    const int s = 4*(2*kind + (pp >> 3)) + ((pp & 7) >> 1);
                    const int val = q3r_slot_lo(s) + 2*half;
                    const int q2 = (blk[16 + 32*n + val] >> (2*j)) & 3;
                    v |= (uint32_t) ((q2 >> (pp & 1)) & 1) << q;
                }
            } else {
                const int k = o - 64*nb, sb = k/20, r = k - 20*sb;
                v = row[84*sb + (r < 16 ? r : 64 + r)];   // scales 0..15, then d, dmin at block bytes 80..83
            }
        } else {
            const int sb = o/84, r = o - 84*sb;
            const uint32_t * gw = (const uint32_t *) row;
            if (r < 16) {
                v = row[64*nb + 20*sb + r];
            } else if (r < 80) {
                const int q = r - 16, n = q >> 5, val = q & 31;
                for (int j = 0; j < 4; ++j) {
                    int s, half;
                    q3r_slot_of(val, s, half);
                    const uint32_t W = gw[2*(8*sb + 4*n + j) + (s >> 3)];
                    v |= ((W >> q3r_qpos(s, half)) & 3u) << (2*j);
                }
            } else {
                v = row[64*nb + 20*sb + 16 + (r - 80)];
            }
        }
        d[o] = (uint8_t) v;
    }
}

// ---------------------------------------------------------------------------------------------------------
// Q6_K, q6r3: the q6r2 planes with the 8 qh2 bytes of every group re-ordered for the decode.  q6r2 holds hi2 of
// value v (0..31) of the group at bits 2v..2v+1 of the 8 bytes read as a little-endian u64.  q6r3 holds two u32:
// HA (values 0..15) and HB (values 16..31, as u = v - 16); for u = 4k + j (k, j = 0..3) hi2 sits at bit
// 16*(j >> 1) + 2*(2k + (j & 1)).  The decode of a16 slot word k then needs one shift and one LOP3 per half2:
//     slot 4k+0 (u 4k, 4k+2 of HA) <- (HA << (4 - 4k)) & 0x00300030      slot 4k+2 (4k+1, 4k+3) <- (HA << (2 - 4k))
//     slot 4k+1 (u 4k, 4k+2 of HB) <- (HB << (8 - 4k)) & 0x03000300      slot 4k+3 (4k+1, 4k+3) <- (HB << (6 - 4k))
// (negative shifts are right shifts).  A pure bit permutation of each 8-byte group record: pack o unpack = id.
static __host__ __device__ __forceinline__ int q6r3_pos(const int u) {   // u = value index within its half (0..15)
    const int k = u >> 2, j = u & 3;
    return 16*(j >> 1) + 2*(2*k + (j & 1));
}

static __device__ __forceinline__ void q6r3_qh_permute(uint8_t * p, const bool pack) {
    uint8_t b[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) b[i] = p[i];
    const uint32_t H[2] = {(uint32_t) b[0] | (uint32_t) b[1] << 8 | (uint32_t) b[2] << 16 | (uint32_t) b[3] << 24,
                           (uint32_t) b[4] | (uint32_t) b[5] << 8 | (uint32_t) b[6] << 16 | (uint32_t) b[7] << 24};
    uint32_t o[2] = {0u, 0u};
#pragma unroll
    for (int v = 0; v < 32; ++v) {
        // q6r2: value v at bits 2v of the u64 = bits 2(v & 15) of word v >> 4; q6r3: bits q6r3_pos(v & 15) of word v >> 4
        const int from = pack ? 2*(v & 15) : q6r3_pos(v & 15);
        const int to   = pack ? q6r3_pos(v & 15) : 2*(v & 15);
        o[v >> 4] |= ((H[v >> 4] >> from) & 3u) << to;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t) (o[i >> 2] >> (8*(i & 3)));
}

// in place, one CUDA block per row: q6r2 -> q6r3 (pack) or back
template <bool pack>
static __global__ void q6r3_convert(char * d, const int nb, const size_t row_bytes) {
    uint8_t * qh = (uint8_t *) (d + blockIdx.x*row_bytes) + 128*nb;
    for (int G = threadIdx.x; G < 8*nb; G += blockDim.x) q6r3_qh_permute(qh + 8*G, pack);
}

// ---------------------------------------------------------------------------------------------------------
// Q5_K (176 bytes = 44 u32 per super-block): standard block sb = u32 [44sb] d,dmin, [+1 .. +3] scales[12],
// [+4 .. +11] qh[32] (bit g of byte l = high bit of value l of group g), [+12 .. +43] qs[128] (as Q4_K).
// Repacked row (q5kr): [qs plane 32*nb u32][qh plane 8*nb u32][header plane 4*nb u32] = 176*nb bytes.
//   qs plane: exactly the q4kr qs plane (group t = 8sb + g is u32 [4t .. 4t+3], byte i = value i | value i+16 << 4).
//   qh plane: u32 [32nb + t] = H, the 32 high bits of the group at q5kr_pos(v), chosen so that a16 slot word k
//     needs: slot 4k+0 (values 4k, 4k+2) (H << (4-k)) & 0x00100010, slot 4k+2 (4k+1, 4k+3) (H >> k) & 0x00100010,
//     slot 4k+1 (16+4k, 18+4k) (H >> k) & 0x01000100, slot 4k+3 (17+4k, 19+4k) (H >> (4+k)) & 0x01000100.
//   header plane: u32 [40nb + 4sb .. +3] = the original d, dmin, scales.
// A pure permutation: pack o unpack = id.
static __host__ __device__ __forceinline__ int q5kr_pos(const int v) {
    const int u = v & 15, k = u >> 2, j = u & 3;
    if (v < 16) return (j == 0 ? 0 : j == 1 ? 4 : j == 2 ? 16 : 20) + k;
    return (j == 0 ? 8 : j == 1 ? 12 : j == 2 ? 24 : 28) + k;
}

// output word o (0 .. 44nb-1) of the repacked (pack) or raw (unpack) row, from the other form staged as u32
static __device__ __forceinline__ uint32_t q5kr_word(const uint32_t * row32, const int nb, const int o, const bool pack) {
    const uint8_t * row8 = (const uint8_t *) row32;
    {
        uint32_t v = 0;
        if (pack) {
            if (o < 32*nb) {
                const int sb = o >> 5, g = (o >> 2) & 7, w = o & 3, j = g >> 1, p = 4*(g & 1);
                const uint32_t lo = row32[44*sb + 12 + 8*j + w];
                const uint32_t hi = row32[44*sb + 16 + 8*j + w];
                v = ((lo >> p) & 0x0F0F0F0Fu) | (((hi >> p) & 0x0F0F0F0Fu) << 4);
            } else if (o < 40*nb) {
                const int t = o - 32*nb, sb = t >> 3, g = t & 7;
                const uint8_t * qh = row8 + 176*sb + 16;
                for (int l = 0; l < 32; ++l) v |= (uint32_t) ((qh[l] >> g) & 1) << q5kr_pos(l);
            } else {
                const int k = o - 40*nb;
                v = row32[44*(k >> 2) + (k & 3)];
            }
        } else {
            const int sb = o / 44, r = o - 44*sb;
            if (r < 4) {
                v = row32[40*nb + 4*sb + r];
            } else if (r < 12) {
                // raw qh bytes 4(r-4) .. +3: bit g of byte l = bit q5kr_pos(l) of H(sb, g)
                for (int bi = 0; bi < 4; ++bi) {
                    const int l = 4*(r - 4) + bi;
                    uint32_t byte = 0;
                    for (int g = 0; g < 8; ++g) byte |= ((row32[32*nb + 8*sb + g] >> q5kr_pos(l)) & 1u) << g;
                    v |= byte << (8*bi);
                }
            } else {
                const int qi = r - 12, j = qi >> 3, hh = (qi >> 2) & 1, w = qi & 3;
                const uint32_t w0 = row32[32*sb + 8*j     + w];
                const uint32_t w1 = row32[32*sb + 8*j + 4 + w];
                v = hh == 0 ? ((w0 & 0x0F0F0F0Fu) | ((w1 & 0x0F0F0F0Fu) << 4))
                            : (((w0 >> 4) & 0x0F0F0F0Fu) | (w1 & 0xF0F0F0F0u));
            }
        }
        return v;
    }
}

template <bool pack>
static __global__ void q5kr_convert(const char * src, char * dst, const int nb, const size_t row_bytes) {
    extern __shared__ uint32_t row32[];
    const uint32_t * s = (const uint32_t *) (src + blockIdx.x*row_bytes);
    uint32_t       * d = (uint32_t       *) (dst + blockIdx.x*row_bytes);
    const int n32 = 44*nb;
    for (int i = threadIdx.x; i < n32; i += blockDim.x) row32[i] = s[i];
    __syncthreads();
    for (int o = threadIdx.x; o < n32; o += blockDim.x) d[o] = q5kr_word(row32, nb, o, pack);
}

// ---------------------------------------------------------------------------------------------------------
// IQ3_XXS (98 bytes per block: d[2] qs[64] ss[32]) and IQ3_S (110: d[2] qs[64] qh[8] signs[32] scales[4]): a pure plane split,
// so a 32-value group (8 per block) is read with aligned 4-byte loads from each plane (matvec_iq3r.cu).
//   iq3xxsr row: [qs 64*nb][ss 32*nb][d 2*nb]                      iq3sr row: [qs 64*nb][signs 32*nb][qh 8*nb][scales 4*nb][d 2*nb]
// pack o unpack = id.  Needs rb % 4 == 0 (even block count).
template <bool pack>
static __global__ void iq3xxsr_convert(const char * src, char * dst, const int nb, const size_t row_bytes) {
    extern __shared__ uint16_t row16[];
    const uint8_t * row = (const uint8_t *) row16;
    const uint16_t * s16 = (const uint16_t *) (src + blockIdx.x*row_bytes);
    uint8_t        * d   = (uint8_t        *) (dst + blockIdx.x*row_bytes);
    for (int i = threadIdx.x; i < 49*nb; i += blockDim.x) row16[i] = s16[i];
    __syncthreads();
    for (int o = threadIdx.x; o < 98*nb; o += blockDim.x) {
        uint32_t v;
        if (pack) {
            if (o < 64*nb)      { const int sb = o >> 6, r = o & 63;               v = row[98*sb + 2 + r]; }
            else if (o < 96*nb) { const int k = o - 64*nb, sb = k >> 5, r = k & 31; v = row[98*sb + 66 + r]; }
            else                { const int k = o - 96*nb;                           v = row[98*(k >> 1) + (k & 1)]; }
        } else {
            const int sb = o/98, r = o - 98*sb;
            if (r < 2)       v = row[96*nb + 2*sb + r];
            else if (r < 66) v = row[64*sb + r - 2];
            else             v = row[64*nb + 32*sb + r - 66];
        }
        d[o] = (uint8_t) v;
    }
}

template <bool pack>
static __global__ void iq3sr_convert(const char * src, char * dst, const int nb, const size_t row_bytes) {
    extern __shared__ uint16_t row16[];
    const uint8_t * row = (const uint8_t *) row16;
    const uint16_t * s16 = (const uint16_t *) (src + blockIdx.x*row_bytes);
    uint8_t        * d   = (uint8_t        *) (dst + blockIdx.x*row_bytes);
    for (int i = threadIdx.x; i < 55*nb; i += blockDim.x) row16[i] = s16[i];
    __syncthreads();
    for (int o = threadIdx.x; o < 110*nb; o += blockDim.x) {
        uint32_t v;
        if (pack) {
            if (o < 64*nb)       { const int sb = o >> 6, r = o & 63;                v = row[110*sb + 2 + r]; }
            else if (o < 96*nb)  { const int k = o - 64*nb, sb = k >> 5, r = k & 31;  v = row[110*sb + 74 + r]; }
            else if (o < 104*nb) { const int k = o - 96*nb, sb = k >> 3, r = k & 7;   v = row[110*sb + 66 + r]; }
            else if (o < 108*nb) { const int k = o - 104*nb, sb = k >> 2, r = k & 3;  v = row[110*sb + 106 + r]; }
            else                 { const int k = o - 108*nb;                           v = row[110*(k >> 1) + (k & 1)]; }
        } else {
            const int sb = o/110, r = o - 110*sb;
            if (r < 2)        v = row[108*nb + 2*sb + r];
            else if (r < 66)  v = row[64*sb + r - 2];
            else if (r < 74)  v = row[96*nb + 8*sb + r - 66];
            else if (r < 106) v = row[64*nb + 32*sb + r - 74];
            else              v = row[104*nb + 4*sb + r - 106];
        }
        d[o] = (uint8_t) v;
    }
}
