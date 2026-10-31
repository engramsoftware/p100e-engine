// attn_pf2.inc: included inside attn.cu's anonymous namespace (after attn_prefill_k).
//
// attn_prefill_k restructured for short chunks at depth, per-(query, head) arithmetic EXACTLY as attn_prefill_k (same
// 32-key tiles from key 0, same QK lane/dim partial sums and shuffle tree, same online-softmax warp reductions, same
// sequential PV chain over the tile's keys, same expression text so nvcc contracts the same fmas): the result is bitwise
// the old kernel's. What changes is how much of the GPU one chunk keeps busy:
//   * RT row triplets per block (NR = 3 RT rows of the flattened (token, head) rows of one kv head): RT = 4 is the old
//     block (2 tokens x 6 heads), RT = 2 one token (6 rows), RT = 1 half a token (3 rows). Rows are independent, so
//     more blocks never change a row; a block just stops at the last key its own rows see (the extra masked tiles the
//     old 2-token block ran for its first token add p = 0, fac = 1: a no-op).
//   * the tile stays in shared memory in its cache representation (f16 halves / q8_0 scales + int8; converted when used,
//     the same single float conversion / multiply as before), 16 / 8.5 KB instead of 32 KB of floats: two or more
//     blocks per SM.
//   * the next tile's global loads are issued into registers before the current tile's compute (K of tile t+1 during
//     softmax + PV of tile t, V of tile t during QK), so the loop no longer waits on memory twice per tile.
// q4_0 caches use the old kernel.
template <uint32_t KT> constexpr int PF2_NCH() { return KT == KV_F16 ? 4 : 3; }   // 16-byte chunks per thread per tile

template <uint32_t KT, int RT>
__global__ void __launch_bounds__(NTHR) attn_pf2_k(const float * __restrict__ q, const char * __restrict__ kc,
                                                   const char * __restrict__ vc, float * __restrict__ out, int T,
                                                   int p0, int Hq, int Hkv, float scale, const kv_pg pg) {
    constexpr int NR = 3 * RT, NRP = (NR + 3) & ~3, KPG = RT, NCH = PF2_NCH<KT>();
    constexpr size_t HB = head_bytes<KT>();                  // bytes of one (position, kv head) row == one raw smem row
    constexpr int NRAW = PF_KT * (int) (HB / 16);            // 16-byte chunks of a tile
    __shared__ __align__(16) float Qs[NR][DH];
    __shared__ __align__(16) uint4 Raw[NRAW];
    __shared__ __align__(16) float Ps[PF_KT][NRP];           // probabilities [key][row]
    __shared__ float Ss[NR][PF_KT + 1];                      // scores [row][key]
    __shared__ float mrow[NR], lrow[NR], fac[NR];

    const char * RawB = reinterpret_cast<const char *>(Raw);
    const int row0 = blockIdx.x * NR, hk = blockIdx.y, tid = threadIdx.x;
    const size_t rowb = (size_t) Hkv * HB;
    const char * kbase = kc + (size_t) hk * HB;
    const char * vbase = vc + (size_t) hk * HB;
    const int32_t * tab = kv_tab(pg, 0);
    const int tl = min(T - 1, (row0 + NR - 1) / GQA);        // last valid token of the block
    const int kend = p0 + tl + 1;                            // keys this block needs: [0, kend)

    for (int i = tid; i < NR * DH / 4; i += NTHR) {
        const int r = i / (DH / 4), d4 = i % (DH / 4);
        const int g = row0 + r;
        const int t = min(T - 1, g / GQA), h = hk * GQA + g % GQA;
        float4 v = reinterpret_cast<const float4 *>(q + ((size_t) t * Hq + h) * DH)[d4];
        v.x *= scale; v.y *= scale; v.z *= scale; v.w *= scale;
        reinterpret_cast<float4 *>(&Qs[r][0])[d4] = v;
    }
    if (tid < NR) { mrow[tid] = NEG_INF; lrow[tid] = 0.0f; }

    float acc[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) acc[r] = 0.0f;
    const int g8 = tid >> 3, l = tid & 7, rb = g8 % RT, kb = g8 / RT;

    uint4 rg[NCH];
    auto fetch = [&](const char * base, int k0) {   // global -> registers (zero past kend)
#pragma unroll
        for (int i = 0; i < NCH; ++i) {
            const int idx = tid + i * NTHR;
            uint4 u = make_uint4(0, 0, 0, 0);
            if constexpr (KT == KV_F16) {
                const int j = idx >> 5, c = idx & 31;
                if (k0 + j < kend) u = *reinterpret_cast<const uint4 *>(base + kv_off_tab(pg, tab, k0 + j, rowb) + c * 16);
            } else {
                if (idx < NRAW) {
                    const int j = idx / 17, c = idx - j * 17;
                    if (k0 + j < kend) u = *reinterpret_cast<const uint4 *>(base + kv_off_tab(pg, tab, k0 + j, rowb) + c * 16);
                }
            }
            rg[i] = u;
        }
    };
    auto stash = [&]() {   // registers -> shared memory (same chunk index as the fetch)
#pragma unroll
        for (int i = 0; i < NCH; ++i) {
            const int idx = tid + i * NTHR;
            if (KT == KV_F16 || idx < NRAW) Raw[idx] = rg[i];
        }
    };
    auto readk = [&](int kr, int d) -> float4 {   // key row kr of the tile, dims d..d+3, as floats
        if constexpr (KT == KV_F16) {
            const uint2 u = *reinterpret_cast<const uint2 *>(RawB + (size_t) kr * HB + d * 2);
            const __half2 * h2 = reinterpret_cast<const __half2 *>(&u);
            const float2 a = __half22float2(h2[0]), b = __half22float2(h2[1]);
            return make_float4(a.x, a.y, b.x, b.y);
        } else {
            const char * hb = RawB + (size_t) kr * HB;
            const float sc = h2f(reinterpret_cast<const uint16_t *>(hb)[d >> 5]);
            const uint32_t w = *reinterpret_cast<const uint32_t *>(hb + Q8_QOFF + d);
            return make_float4(__fmul_rn(q8_lo(w, 0), sc), __fmul_rn(q8_lo(w, 1), sc), __fmul_rn(q8_lo(w, 2), sc),
                               __fmul_rn(q8_lo(w, 3), sc));
        }
    };

    fetch(kbase, 0);
    for (int k0 = 0; k0 < kend; k0 += PF_KT) {
        __syncthreads();                       // Q / m,l init visible; previous tile's V and Ps consumed
        stash();                               // K of this tile
        fetch(vbase, k0);                      // V loads fly during QK
        __syncthreads();
        // ---- QK: 3 rows x KPG keys per group, 32 dims per lane ----
        float s[3][KPG];
#pragma unroll
        for (int a = 0; a < 3; ++a)
#pragma unroll
            for (int b = 0; b < KPG; ++b) s[a][b] = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = 4 * l + 32 * i;
            float4 qv[3], kv[KPG];
#pragma unroll
            for (int a = 0; a < 3; ++a) qv[a] = *reinterpret_cast<const float4 *>(&Qs[rb * 3 + a][d]);
#pragma unroll
            for (int b = 0; b < KPG; ++b) kv[b] = readk(kb * KPG + b, d);
#pragma unroll
            for (int a = 0; a < 3; ++a)
#pragma unroll
                for (int b = 0; b < KPG; ++b)
                    s[a][b] += qv[a].x * kv[b].x + qv[a].y * kv[b].y + qv[a].z * kv[b].z + qv[a].w * kv[b].w;
        }
#pragma unroll
        for (int a = 0; a < 3; ++a)
#pragma unroll
            for (int b = 0; b < KPG; ++b) {
                float v = s[a][b];
                v += __shfl_xor_sync(0xffffffff, v, 1);
                v += __shfl_xor_sync(0xffffffff, v, 2);
                v += __shfl_xor_sync(0xffffffff, v, 4);
                if (l == ((a * KPG + b) & 7)) {
                    const int r = rb * 3 + a, j = kb * KPG + b;
                    const int t = min(T - 1, (row0 + r) / GQA);
                    Ss[r][j] = k0 + j <= p0 + t ? v : NEG_INF;
                }
            }
        __syncthreads();                       // scores ready; K no longer needed
        stash();                               // V of this tile
        if (k0 + PF_KT < kend) fetch(kbase, k0 + PF_KT);   // next K flies during softmax + PV
        // ---- online softmax: warp per row, lane = key ----
        const int w = tid >> 5, lane = tid & 31;
        for (int r = w; r < NR; r += NTHR / 32) {
            const float sv = Ss[r][lane];
            const float mold = mrow[r];
            const float mnew = fmaxf(mold, pf_warp_max(sv));
            const float p = expf(sv - mnew);
            const float sum = pf_warp_sum(p);
            Ps[lane][r] = p;
            if (lane == 0) {
                const float f = expf(mold - mnew);
                fac[r] = f;
                lrow[r] = lrow[r] * f + sum;
                mrow[r] = mnew;
            }
        }
        __syncthreads();                       // V, Ps, fac ready
        // ---- PV: thread = dim ----
#pragma unroll
        for (int r = 0; r < NR; ++r) acc[r] *= fac[r];
        const int jn = min(PF_KT, kend - k0);
        auto pv_key = [&](int j) {
            float v;
            if constexpr (KT == KV_F16) {
                v = h2f(reinterpret_cast<const uint16_t *>(RawB + (size_t) j * HB)[tid]);
            } else {
                const char * hb = RawB + (size_t) j * HB;
                v = __fmul_rn((float) reinterpret_cast<const int8_t *>(hb + Q8_QOFF)[tid],
                              h2f(reinterpret_cast<const uint16_t *>(hb)[tid >> 5]));
            }
            const float4 pa = *reinterpret_cast<const float4 *>(&Ps[j][0]);
            acc[0] += pa.x * v;
            if constexpr (NR > 1) acc[1] += pa.y * v;
            if constexpr (NR > 2) acc[2] += pa.z * v;
            if constexpr (NR > 3) {
                acc[3] += pa.w * v;
                const float4 pb = *reinterpret_cast<const float4 *>(&Ps[j][4]);
                acc[4] += pb.x * v; acc[5] += pb.y * v;
                if constexpr (NR > 6) {
                    acc[6] += pb.z * v; acc[7] += pb.w * v;
                    const float4 pc = *reinterpret_cast<const float4 *>(&Ps[j][8]);
                    acc[8] += pc.x * v; acc[9] += pc.y * v; acc[10] += pc.z * v; acc[11] += pc.w * v;
                }
            }
        };
        // keys in order j = 0..jn-1 (the chain of every acc[r]); a full tile is unrolled so the smem loads of several keys
        // are in flight together (the fmas of one acc stay in the same order)
        if (jn == PF_KT) {
#pragma unroll 8
            for (int j = 0; j < PF_KT; ++j) pv_key(j);
        } else {
            for (int j = 0; j < jn; ++j) pv_key(j);
        }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const int g = row0 + r, t = g / GQA;
        if (t < T) out[((size_t) t * Hq + hk * GQA + g % GQA) * DH + tid] = acc[r] / lrow[r];
    }
}
