#include "ssm.cuh"
#include "a16q.cuh"

#include <cstdint>
#include <stdexcept>

namespace p100e::fast {

static __device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

static __device__ float block_sum(float v) {
    __shared__ float sh[32];
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    v = threadIdx.x < nw ? sh[threadIdx.x] : 0.0f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

// N block sums at once (one set of barriers): each value goes through exactly block_sum's reduction tree
template <int N>
static __device__ void block_sum_n(float (&v)[N]) {
    __shared__ float sh[N][32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
    for (int k = 0; k < N; ++k) v[k] = warp_sum(v[k]);
    __syncthreads();
    if (lane == 0) {
#pragma unroll
        for (int k = 0; k < N; ++k) sh[k][wid] = v[k];
    }
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
#pragma unroll
    for (int k = 0; k < N; ++k) {
        float u = threadIdx.x < nw ? sh[k][threadIdx.x] : 0.0f;
        if (wid == 0) u = warp_sum(u);
        v[k] = u;
    }
    if (threadIdx.x == 0) {   // sh[k][0] was read only by thread 0 itself
#pragma unroll
        for (int k = 0; k < N; ++k) sh[k][0] = v[k];
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < N; ++k) v[k] = sh[k][0];
}

static __device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }

// grid (C/D, max(1, T-2)), block D. Block y==0 does tokens 0..min(T,3)-1 (the ones that read the old
// conv state) and writes the new state; block y>=1 does token y+2 from x only. No races.
// SNAP: also writes the conv state after token t (t < T-1) to csnap + t*C*3 (rollback snapshots of a verify batch).
// SNAP also handles block y==0's tokens together (one block_sum_n for their q/k norms instead of one block_sum per
// token); token t's arithmetic is the same expressions and reduction tree in both instantiations, SNAP = false is the
// unchanged decode kernel (tests/fast_gdn_verify_test.cu checks the batched SNAP call against T sequential T = 1 calls
// bit-exactly).
// ldx / lda / ldb: row strides of x, a and b (the MTP verify reads them straight from concatenated projection outputs);
// y, decay and beta are written dense ([T][C], [T][Hv]).
// MS: several sequences in one launch, block z = sequence i of ms (gdn_multi): its rows start at
// row0, its conv state is slot `slot` of the pool, its snapshots start at slot snap0; T = its token count. Everything
// after the pointer setup is the single-sequence code: each sequence gets exactly the bits of its own call.
struct ms_args {
    const int4 * seqs;           // [nseq] (row0, T, state slot, first snapshot slot)
    float * cpool; size_t cslot;   // conv state of slot s = cpool + s * cslot
    float * spool; size_t sslot;   // ssm state of slot s = spool + s * sslot
    float * csnap; float * ssnap;  // snapshot slot 0 of this layer
    int Ttot;                      // rows of the whole call (qbuf layout)
};

template <bool SNAP, bool MS>
__global__ void gdn_prep_k(const float * __restrict__ x, float * cstate, const float * __restrict__ cw,
                           const float * a, const float * b, const float * __restrict__ dt_bias,
                           const float * __restrict__ ssm_a, float * __restrict__ y, float * decay, float * beta,
                           int T, int C, int Hk, int Hv, int D, float qscale, float eps, float * __restrict__ csnap,
                           int ldx, int lda, int ldb, const ms_args ms) {
    if constexpr (MS) {
        const int4 sd = ms.seqs[blockIdx.z];
        T = sd.y;
        if ((int) blockIdx.y >= (T > 3 ? T - 2 : 1)) return;   // whole block
        x += (int64_t) sd.x * ldx; a += (int64_t) sd.x * lda; b += (int64_t) sd.x * ldb;
        y += (int64_t) sd.x * C; decay += (int64_t) sd.x * Hv; beta += (int64_t) sd.x * Hv;
        cstate = ms.cpool + (size_t) sd.z * ms.cslot;
        if (SNAP) csnap = ms.csnap + (size_t) sd.w * C * 3;
    }
    const int chunk = blockIdx.x, c = chunk * D + threadIdx.x, by = blockIdx.y;
    const bool is_qk = chunk < 2 * Hk;
    const float oscale = chunk < Hk ? qscale : 1.0f;
    const float4 w = reinterpret_cast<const float4 *>(cw)[c];

    auto gates = [&](int t) {   // decay / beta of token t (chunk 0 only)
        for (int h = threadIdx.x; h < Hv; h += blockDim.x) {
            const int i = t * Hv + h, ia = t * lda + h, ib = t * ldb + h;
            const float zz = a[ia] + dt_bias[h];
            const float sp = zz > 20.0f ? zz : log1pf(expf(zz));
            decay[i] = expf(sp * ssm_a[h]);
            beta[i]  = 1.0f / (1.0f + expf(-b[ib]));
        }
    };
    auto emit = [&](int t, float v) {
        float yv = silu(v);
        if (is_qk) {
            const float ss = block_sum(yv * yv);
            yv *= oscale / fmaxf(sqrtf(ss), eps);
        }
        y[(int64_t) t * C + c] = yv;
        if (chunk == 0) gates(t);
    };

    if (by == 0) {
        float s0 = cstate[c * 3 + 0], s1 = cstate[c * 3 + 1], s2 = cstate[c * 3 + 2];
        const int n0 = T < 3 ? T : 3;
        if constexpr (SNAP) {
            // the verify batch: this block's (up to 3) tokens together - their x loads issued at once and the q/k
            // norms in one block_sum_n (each value through block_sum's tree), so each token's arithmetic is emit()'s
            float xt[3], yv[3];
#pragma unroll
            for (int t = 0; t < 3; ++t) xt[t] = t < n0 ? x[(int64_t) t * ldx + c] : 0.0f;
#pragma unroll
            for (int t = 0; t < 3; ++t) {
                yv[t] = 0.0f;
                if (t < n0) {
                    yv[t] = silu(s0 * w.x + s1 * w.y + s2 * w.z + xt[t] * w.w);
                    s0 = s1; s1 = s2; s2 = xt[t];
                    if (t < T - 1) {
                        float * cs = csnap + (int64_t) t * C * 3 + c * 3;
                        cs[0] = s0; cs[1] = s1; cs[2] = s2;
                    }
                }
            }
            if (is_qk) {
                float ss[3];
#pragma unroll
                for (int t = 0; t < 3; ++t) ss[t] = yv[t] * yv[t];
                block_sum_n<3>(ss);
#pragma unroll
                for (int t = 0; t < 3; ++t) yv[t] *= oscale / fmaxf(sqrtf(ss[t]), eps);
            }
#pragma unroll
            for (int t = 0; t < 3; ++t) if (t < n0) y[(int64_t) t * C + c] = yv[t];
            if (chunk == 0) for (int t = 0; t < n0; ++t) gates(t);
        } else {
            for (int t = 0; t < n0; ++t) {
                const float xt = x[(int64_t) t * ldx + c];
                emit(t, s0 * w.x + s1 * w.y + s2 * w.z + xt * w.w);
                s0 = s1; s1 = s2; s2 = xt;
            }
        }
        if (T > 3) {
            s0 = x[(int64_t) (T - 3) * ldx + c]; s1 = x[(int64_t) (T - 2) * ldx + c]; s2 = x[(int64_t) (T - 1) * ldx + c];
        }
        cstate[c * 3 + 0] = s0; cstate[c * 3 + 1] = s1; cstate[c * 3 + 2] = s2;
    } else {
        const int t = by + 2;
        const float * xc = x + c;
        emit(t, xc[(int64_t) (t - 3) * ldx] * w.x + xc[(int64_t) (t - 2) * ldx] * w.y +
                xc[(int64_t) (t - 1) * ldx] * w.z + xc[(int64_t) t * ldx] * w.w);
        if constexpr (SNAP) {
            if (t < T - 1) {   // state after token t = inputs t-2, t-1, t
                float * cs = csnap + (int64_t) t * C * 3 + c * 3;
                cs[0] = xc[(int64_t) (t - 2) * ldx]; cs[1] = xc[(int64_t) (t - 1) * ldx]; cs[2] = xc[(int64_t) t * ldx];
            }
        }
    }
}

// Recurrence. Block = R state rows of one v head (grid Hv x D/R), 4 threads per row j, each owning
// 32 columns as 8 float4 (float4 index m*4+p, so a row's 4 threads read 64 contiguous bytes per load).
// The block's rows stay in registers for all T tokens (state read once, written once). q/k/v/decay/beta
// for TC tokens are staged in shared memory per chunk (one coalesced load, then smem broadcasts), with
// the next chunk prefetched into registers while the current one is computed.
// R == D: the gated RMSNorm runs in the same kernel (warp per token row); otherwise out holds the raw
// output and gdn_norm_k runs after.
// SNAP: also writes the state after token t (t < T-1) to snap + t*Hv*D*D (rollback snapshots of a verify batch)
// straight from the registers: the state is read once and written T times, instead of T reads + T writes + T-1
// snapshot copies for T sequential T = 1 calls. Same per-token arithmetic as SNAP = false (the decode kernel).

template <bool SIG> static __device__ __forceinline__ float ogate(float v) { return SIG ? 1.0f / (1.0f + expf(-v)) : silu(v); }

template <int R, bool SNAP, bool MS, bool SIG = false>
__global__ void __launch_bounds__(4 * R)
gdn_scan_k(const float * __restrict__ y, const float * __restrict__ decay, const float * __restrict__ beta,
           const float * __restrict__ z, const float * __restrict__ nw, float * state, float * out,
           int T, int C, int Hk, int Hv, float eps, void * qbuf, float * __restrict__ snap, int ldz, const ms_args ms) {
    int qrow0 = 0, Tq = T;   // qbuf rows: this sequence's first row and the call's row count
    if constexpr (MS) {
        const int4 sd = ms.seqs[blockIdx.z];
        T = sd.y;
        y += (int64_t) sd.x * C; decay += (int64_t) sd.x * Hv; beta += (int64_t) sd.x * Hv;
        z += (int64_t) sd.x * ldz; out += (int64_t) sd.x * Hv * 128;
        state = ms.spool + (size_t) sd.z * ms.sslot;
        if (SNAP) snap = ms.ssnap + (size_t) sd.w * Hv * 128 * 128;
        qrow0 = sd.x; Tq = ms.Ttot;
    }
    constexpr int D = 128, NT = 4 * R, TC = 8;   // tokens per staged chunk (16 KB of q/k double-buffered)
    constexpr int NQK = TC * 2 * (D / 4);                        // float4s of q+k per chunk
    constexpr int NLD = (NQK + NT - 1) / NT;
    __shared__ float4 sqk[2][TC][2][D / 4];                       // [buf][t][q|k][float4]
    __shared__ float sv[2][TC][R], sdec[2][TC], sbet[2][TC];

    const int h = blockIdx.x, r0 = blockIdx.y * R, hk = h % Hk, tid = threadIdx.x;
    const int jl = tid >> 2, j = r0 + jl, p = tid & 3;
    float4 * Sp = reinterpret_cast<float4 *>(state + ((int64_t) h * D + j) * D);
    float4 S[8];
#pragma unroll
    for (int m = 0; m < 8; ++m) S[m] = Sp[m * 4 + p];

    const float * qb = y + hk * D;
    const float * kb = y + Hk * D + hk * D;
    const float * vb = y + 2 * Hk * D + h * D + r0;
    float * ob = out + (int64_t) h * D + j;

    // prefetch registers for one chunk
    float4 pqk[NLD];
    float pv[TC / 4], pd = 0.f, pb = 0.f;
    auto fetch = [&](int t0) {
#pragma unroll
        for (int l = 0; l < NLD; ++l) {
            const int idx = tid + l * NT;
            if (idx < NQK) {
                const int t = idx / (2 * D / 4), rem = idx % (2 * D / 4), w = rem / (D / 4), f = rem % (D / 4);
                if (t0 + t < T) pqk[l] = __ldg(reinterpret_cast<const float4 *>((w ? kb : qb) + (int64_t) (t0 + t) * C) + f);
            }
        }
#pragma unroll
        for (int l = 0; l < TC / 4; ++l) {
            const int idx = tid + l * NT, t = idx / R, r = idx % R;
            if (t0 + t < T) pv[l] = __ldg(vb + (int64_t) (t0 + t) * C + r);
        }
        if (tid < TC && t0 + tid < T) { pd = __ldg(decay + (t0 + tid) * Hv + h); pb = __ldg(beta + (t0 + tid) * Hv + h); }
    };
    auto stash = [&](int buf) {
#pragma unroll
        for (int l = 0; l < NLD; ++l) {
            const int idx = tid + l * NT;
            if (idx < NQK) (&sqk[buf][0][0][0])[idx] = pqk[l];
        }
#pragma unroll
        for (int l = 0; l < TC / 4; ++l) (&sv[buf][0][0])[tid + l * NT] = pv[l];
        if (tid < TC) { sdec[buf][tid] = pd; sbet[buf][tid] = pb; }
    };

    fetch(0);
    stash(0);
    __syncthreads();
    int buf = 0;
    for (int t0 = 0; t0 < T; t0 += TC) {
        const bool more = t0 + TC < T;
        if (more) fetch(t0 + TC);
        const int nt = T - t0 < TC ? T - t0 : TC;
        for (int tt = 0; tt < nt; ++tt) {
            const float4 * K4 = sqk[buf][tt][1];
            const float4 * Q4 = sqk[buf][tt][0];
            const float eg = sdec[buf][tt], bt = sbet[buf][tt], vj = sv[buf][tt][jl];
            float4 K[8];
            float s0 = 0.f, s1 = 0.f;
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                K[m] = K4[m * 4 + p];
                s0 += S[m].x * K[m].x + S[m].y * K[m].y;
                s1 += S[m].z * K[m].z + S[m].w * K[m].w;
            }
            float sk = s0 + s1;
            sk += __shfl_xor_sync(0xffffffff, sk, 1);
            sk += __shfl_xor_sync(0xffffffff, sk, 2);
            const float delta = (vj - eg * sk) * bt;
            float o0 = 0.f, o1 = 0.f;
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                const float4 q = Q4[m * 4 + p];
                S[m].x = S[m].x * eg + K[m].x * delta;
                S[m].y = S[m].y * eg + K[m].y * delta;
                S[m].z = S[m].z * eg + K[m].z * delta;
                S[m].w = S[m].w * eg + K[m].w * delta;
                o0 += S[m].x * q.x + S[m].y * q.y;
                o1 += S[m].z * q.z + S[m].w * q.w;
            }
            float o = o0 + o1;
            o += __shfl_xor_sync(0xffffffff, o, 1);
            o += __shfl_xor_sync(0xffffffff, o, 2);
            if (p == 0) ob[(int64_t) (t0 + tt) * Hv * D] = o;
            if constexpr (SNAP) {
                if (t0 + tt < T - 1) {
                    float4 * Sn = reinterpret_cast<float4 *>(snap + (int64_t) (t0 + tt) * Hv * D * D + ((int64_t) h * D + j) * D);
#pragma unroll
                    for (int m = 0; m < 8; ++m) Sn[m * 4 + p] = S[m];
                }
            }
        }
        if (more) { stash(buf ^ 1); __syncthreads(); buf ^= 1; }
    }
#pragma unroll
    for (int m = 0; m < 8; ++m) Sp[m * 4 + p] = S[m];

    if constexpr (R == D) {
        __syncthreads();   // makes this block's raw out rows visible to the norm pass
        const int lane = tid & 31, wid = tid >> 5;
        const float4 w4 = reinterpret_cast<const float4 *>(nw)[lane];
        for (int t = wid; t < T; t += NT / 32) {
            const int64_t off = ((int64_t) t * Hv + h) * D;
            float4 o4 = reinterpret_cast<float4 *>(out + off)[lane];
            const float4 z4 = reinterpret_cast<const float4 *>(z + (int64_t) t * ldz + h * D)[lane];
            const float ss = warp_sum(o4.x * o4.x + o4.y * o4.y + o4.z * o4.z + o4.w * o4.w);
            const float sc = rsqrtf(ss / D + eps);
            o4.x *= sc * w4.x * ogate<SIG>(z4.x); o4.y *= sc * w4.y * ogate<SIG>(z4.y);
            o4.z *= sc * w4.z * ogate<SIG>(z4.z); o4.w *= sc * w4.w * ogate<SIG>(z4.w);
            reinterpret_cast<float4 *>(out + off)[lane] = o4;
            if (qbuf) {   // a16 quantize of the out row for ssm_out (fast::matvec_prequantized); 4 groups per head
                const int ng = Hv * (D / 32), g = h * (D / 32) + (lane >> 3);
                a16_quant8(o4, a16_qs(qbuf, qrow0 + t, g, ng), a16_ads(qbuf, qrow0 + t, g, ng, Tq));
            }
        }
    }
}

// gated RMSNorm over rows of 128, a warp per row, in place on out [nrows][128] (row = t*Hv + h); z row t at z + t*ldz.
template <bool SIG>
__global__ void gdn_norm_k(float * out, const float * __restrict__ z, const float * __restrict__ nw, int nrows, float eps,
                           int Hv, int ldz) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= nrows) return;
    const int64_t off = (int64_t) row * 128;
    const float4 w4 = reinterpret_cast<const float4 *>(nw)[lane];
    float4 o4 = reinterpret_cast<float4 *>(out + off)[lane];
    const float4 z4 = reinterpret_cast<const float4 *>(z + (int64_t) (row / Hv) * ldz + (row % Hv) * 128)[lane];
    const float ss = warp_sum(o4.x * o4.x + o4.y * o4.y + o4.z * o4.z + o4.w * o4.w);
    const float sc = rsqrtf(ss / 128 + eps);
    o4.x *= sc * w4.x * ogate<SIG>(z4.x); o4.y *= sc * w4.y * ogate<SIG>(z4.y);
    o4.z *= sc * w4.z * ogate<SIG>(z4.z); o4.w *= sc * w4.w * ogate<SIG>(z4.w);
    reinterpret_cast<float4 *>(out + off)[lane] = o4;
}

void gdn_prep(const float * x, float * cstate, const GdnWeights & w, const float * a, const float * b,
              float * y, float * decay, float * beta, int T, int Hk, int Hv, int D, float eps, cudaStream_t s,
              float * cstate_snap, int ldx, int lda, int ldb) {
    if (D != 128) throw std::runtime_error("fast::gdn_prep: D must be 128");
    if (T <= 0) return;
    const int C = 2 * Hk * D + Hv * D;
    if (ldx <= 0) ldx = C;
    if (lda <= 0) lda = Hv;
    if (ldb <= 0) ldb = Hv;
    if (ldx < C || lda < Hv || ldb < Hv) throw std::runtime_error("fast::gdn_prep: row stride smaller than the row");
    // decay / beta [T][Hv] may overwrite alpha / beta in place only when those are dense (another token's input
    // would be clobbered otherwise)
    if (T > 1 && ((decay == a && lda != Hv) || (beta == b && ldb != Hv))) throw std::runtime_error("fast::gdn_prep: in-place gates need dense a / b");
    dim3 grid(C / D, T > 3 ? T - 2 : 1);
    if (cstate_snap && T > 1)
        gdn_prep_k<true, false><<<grid, D, 0, s>>>(x, cstate, w.conv_w, a, b, w.dt_bias, w.ssm_a, y, decay, beta,
                                                   T, C, Hk, Hv, D, rsqrtf((float) D), eps, cstate_snap, ldx, lda, ldb, ms_args{});
    else
        gdn_prep_k<false, false><<<grid, D, 0, s>>>(x, cstate, w.conv_w, a, b, w.dt_bias, w.ssm_a, y, decay, beta,
                                                    T, C, Hk, Hv, D, rsqrtf((float) D), eps, nullptr, ldx, lda, ldb, ms_args{});
}

template <int R, bool SIG>
static void scan_launch_g(const float * y, const float * decay, const float * beta, const float * z, const float * norm_w,
                          float * state, float * out, int T, int C, int Hk, int Hv, float eps, cudaStream_t s, void * qbuf,
                          float * snap, int ldz) {
    const dim3 grid(Hv, 128 / R);
    if (snap && T > 1)
        gdn_scan_k<R, true, false, SIG><<<grid, 4 * R, 0, s>>>(y, decay, beta, z, norm_w, state, out, T, C, Hk, Hv, eps, qbuf, snap, ldz,
                                                               ms_args{});
    else
        gdn_scan_k<R, false, false, SIG><<<grid, 4 * R, 0, s>>>(y, decay, beta, z, norm_w, state, out, T, C, Hk, Hv, eps, qbuf, nullptr,
                                                                ldz, ms_args{});
}
template <int R>
static void scan_launch(const float * y, const float * decay, const float * beta, const float * z, const float * norm_w,
                        float * state, float * out, int T, int C, int Hk, int Hv, float eps, cudaStream_t s, void * qbuf,
                        float * snap, int ldz, int gate) {
    if (gate) scan_launch_g<R, true>(y, decay, beta, z, norm_w, state, out, T, C, Hk, Hv, eps, s, qbuf, snap, ldz);
    else      scan_launch_g<R, false>(y, decay, beta, z, norm_w, state, out, T, C, Hk, Hv, eps, s, qbuf, snap, ldz);
}

void gdn_scan(const float * y, const float * decay, const float * beta, const float * z, const float * norm_w,
              float * state, float * out, int T, int Hk, int Hv, int D, float eps, cudaStream_t s, int rows, void * qbuf,
              float * state_snap, int ldz, int gate) {
    if (D != 128) throw std::runtime_error("fast::gdn_scan: D must be 128");
    if (T <= 0) return;
    const int C = 2 * Hk * D + Hv * D;
    if (ldz <= 0) ldz = Hv * D;
    if (ldz < Hv * D || ldz % 4) throw std::runtime_error("fast::gdn_scan: z row stride must be >= Hv*D and a multiple of 4");
    // z rows are read as float4 (z + t*ldz + h*D): the base must be 16-byte aligned too (like attn_prep's inputs)
    if (reinterpret_cast<uintptr_t>(z) & 15) throw std::runtime_error("fast::gdn_scan: z must be 16-byte aligned");
    if (rows <= 0) rows = T <= 4 ? 128 : 32;   // decode: fused norm, 1 launch; prefill: 4x the blocks
    if (qbuf && rows != 128) throw std::runtime_error("fast::gdn_scan: qbuf needs the fused-norm path (rows 128, T <= 4)");
    switch (rows) {
        case 128: scan_launch<128>(y, decay, beta, z, norm_w, state, out, T, C, Hk, Hv, eps, s, qbuf, state_snap, ldz, gate); return;
        case 64:  scan_launch<64>(y, decay, beta, z, norm_w, state, out, T, C, Hk, Hv, eps, s, nullptr, state_snap, ldz, gate); break;
        case 32:  scan_launch<32>(y, decay, beta, z, norm_w, state, out, T, C, Hk, Hv, eps, s, nullptr, state_snap, ldz, gate); break;
        case 16:  scan_launch<16>(y, decay, beta, z, norm_w, state, out, T, C, Hk, Hv, eps, s, nullptr, state_snap, ldz, gate); break;
        default: throw std::runtime_error("fast::gdn_scan: rows must be 16, 32, 64 or 128");
    }
    const int nrows = T * Hv;
    if (gate) gdn_norm_k<true><<<(nrows + 3) / 4, 128, 0, s>>>(out, z, norm_w, nrows, eps, Hv, ldz);
    else      gdn_norm_k<false><<<(nrows + 3) / 4, 128, 0, s>>>(out, z, norm_w, nrows, eps, Hv, ldz);
}

void gdn_layer_multi(const float * x, const float * a, const float * b, const float * z, const GdnWeights & w,
                     const gdn_multi & m, float * y, float * decay, float * beta, float * out, int Hk, int Hv, int D,
                     float eps, cudaStream_t s, void * qbuf) {
    if (D != 128) throw std::runtime_error("fast::gdn_layer_multi: D must be 128");
    if (m.nseq < 1 || m.maxT < 1 || m.maxT > 4 || m.Ttot < 1 || !m.seqs)
        throw std::runtime_error("fast::gdn_layer_multi: 1..4 tokens per sequence");
    if (reinterpret_cast<uintptr_t>(z) & 15) throw std::runtime_error("fast::gdn_layer_multi: z must be 16-byte aligned");
    if (w.gate) throw std::runtime_error("fast::gdn_layer_multi: sigmoid gate not supported (use gdn_layer)");
    const int C = 2 * Hk * D + Hv * D;
    const bool snap = m.conv_snap && m.ssm_snap;
    ms_args ms;
    ms.seqs = reinterpret_cast<const int4 *>(m.seqs);
    ms.cpool = m.conv_base; ms.cslot = m.conv_slot;
    ms.spool = m.ssm_base; ms.sslot = m.ssm_slot;
    ms.csnap = m.conv_snap; ms.ssnap = m.ssm_snap;
    ms.Ttot = m.Ttot;
    // prep: grid y covers the longest sequence (blocks past a sequence's own count exit)
    const dim3 gp(C / D, m.maxT > 3 ? m.maxT - 2 : 1, m.nseq);
    const float qs = rsqrtf((float) D);
    if (snap)
        gdn_prep_k<true, true><<<gp, D, 0, s>>>(x, nullptr, w.conv_w, a, b, w.dt_bias, w.ssm_a, y, decay, beta, 1, C, Hk, Hv,
                                                D, qs, eps, nullptr, C, Hv, Hv, ms);
    else
        gdn_prep_k<false, true><<<gp, D, 0, s>>>(x, nullptr, w.conv_w, a, b, w.dt_bias, w.ssm_a, y, decay, beta, 1, C, Hk, Hv,
                                                 D, qs, eps, nullptr, C, Hv, Hv, ms);
    // scan: rows 128 (fused norm + qbuf epilogue), as the T <= 4 single-sequence call
    const dim3 gs(Hv, 1, m.nseq);
    if (snap)
        gdn_scan_k<128, true, true><<<gs, 4 * 128, 0, s>>>(y, decay, beta, z, w.norm_w, nullptr, out, 1, C, Hk, Hv, eps, qbuf,
                                                          nullptr, Hv * D, ms);
    else
        gdn_scan_k<128, false, true><<<gs, 4 * 128, 0, s>>>(y, decay, beta, z, w.norm_w, nullptr, out, 1, C, Hk, Hv, eps, qbuf,
                                                           nullptr, Hv * D, ms);
}

// Decode step 6b (P100E_GDN_FUSE): gdn_prep_k + gdn_scan_k<128> of one T = 1 token in one launch. Grid Hv, 512 threads;
// block h first recomputes what the scan of head h reads from the prep: threads 0..127 the q chunk hk = h % Hk, 128..255
// the k chunk, 256..383 the v chunk of head h (conv + silu, q/k L2 norm through block_sum's exact tree per 128-thread
// group), thread 384 the gates of head h; the values go straight to the scan's shared buffers (the bits gdn_prep_k
// writes and gdn_scan_k loads), then the scan + gated norm + quantize run as in gdn_scan_k (T = 1, R = 128).
// Conv state, single writer: a v chunk belongs to one head (its block writes it); a q/k chunk is read by the Hv / Hk
// heads that share it, and only the last of them to have read it (device counter cnt[hk], fence + atomicAdd, reset by
// that block) writes the new state. y / decay / beta are written as gdn_prep_k does (q/k by the blocks h < Hk).
template <bool SIG>
__global__ void __launch_bounds__(512)
gdn_step1_k(const float * __restrict__ x, float * cstate, const float * __restrict__ cw, float * a, float * b,
            const float * __restrict__ dt_bias, const float * __restrict__ ssm_a, float * __restrict__ y, float * decay,
            float * beta, const float * __restrict__ z, const float * __restrict__ nw, float * state, float * out,
            int C, int Hk, int Hv, float qscale, float eps, void * qbuf, unsigned int * cnt) {
    constexpr int D = 128, R = 128, NT = 4 * R;
    __shared__ __align__(16) float sq[D], sk[D], sv[D];
    __shared__ float sdec, sbet;
    __shared__ float shn[16];
    __shared__ int last;
    const int h = blockIdx.x, hk = h % Hk, tid = threadIdx.x;
    const int part = tid >> 7, lt = tid & 127;
    // ---- prep part (gdn_prep_k, T = 1, by == 0)
    int c = 0;
    float n0 = 0.f, n1 = 0.f, n2 = 0.f, yv = 0.f;
    if (part < 3) {
        const int chunk = part == 0 ? hk : part == 1 ? Hk + hk : 2 * Hk + h;
        c = chunk * D + lt;
        const float4 w = reinterpret_cast<const float4 *>(cw)[c];
        const float s0 = cstate[c * 3 + 0], s1 = cstate[c * 3 + 1], s2 = cstate[c * 3 + 2];
        const float xt = x[c];
        yv = silu(s0 * w.x + s1 * w.y + s2 * w.z + xt * w.w);
        n0 = s1; n1 = s2; n2 = xt;
    } else if (lt == 0) {
        const float zz = a[h] + dt_bias[h];
        const float sp = zz > 20.0f ? zz : log1pf(expf(zz));
        sdec = expf(sp * ssm_a[h]);
        sbet = 1.0f / (1.0f + expf(-b[h]));
    }
    {   // q / k L2 norm: block_sum's tree over each 128-thread group (warp_sum, then the 4 warp sums in one warp_sum)
        const float v = warp_sum(yv * yv);
        if ((tid & 31) == 0) shn[tid >> 5] = v;
        __syncthreads();
        if (part < 2) {
            float u = (tid & 127) < 4 ? shn[part * 4 + (tid & 127)] : 0.0f;   // only warp 0 of the part reads (as block_sum)
            if (((tid >> 5) & 3) == 0) u = warp_sum(u);
            if ((tid & 127) == 0) shn[part * 4] = u;
        }
        __syncthreads();
        if (part < 2) yv *= (part == 0 ? qscale : 1.0f) / fmaxf(sqrtf(shn[part * 4]), eps);
    }
    if (part == 0) { sq[lt] = yv; if (h < Hk) y[c] = yv; }
    if (part == 1) { sk[lt] = yv; if (h < Hk) y[c] = yv; }
    if (part == 2) { sv[lt] = yv; y[c] = yv; cstate[c * 3 + 0] = n0; cstate[c * 3 + 1] = n1; cstate[c * 3 + 2] = n2; }
    if (part == 3 && lt == 0) { decay[h] = sdec; beta[h] = sbet; }
    // single writer of the shared q/k conv state: this block's reads of it are done (their values were used above)
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const unsigned int prev = atomicAdd(cnt + hk, 1u);
        last = prev == (unsigned int) (Hv / Hk - 1);
        if (last) { cnt[hk] = 0; __threadfence(); }
    }
    __syncthreads();
    if (last && part < 2) { cstate[c * 3 + 0] = n0; cstate[c * 3 + 1] = n1; cstate[c * 3 + 2] = n2; }
    // ---- scan part (gdn_scan_k<128>, T = 1)
    const int jl = tid >> 2, j = jl, p = tid & 3;
    float4 * Sp = reinterpret_cast<float4 *>(state + ((int64_t) h * D + j) * D);
    float4 S[8];
#pragma unroll
    for (int m = 0; m < 8; ++m) S[m] = Sp[m * 4 + p];
    float * ob = out + (int64_t) h * D + j;
    {
        const float4 * K4 = reinterpret_cast<const float4 *>(sk);
        const float4 * Q4 = reinterpret_cast<const float4 *>(sq);
        const float eg = sdec, bt = sbet, vj = sv[jl];
        float4 K[8];
        float s0 = 0.f, s1 = 0.f;
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            K[m] = K4[m * 4 + p];
            s0 += S[m].x * K[m].x + S[m].y * K[m].y;
            s1 += S[m].z * K[m].z + S[m].w * K[m].w;
        }
        float sk2 = s0 + s1;
        sk2 += __shfl_xor_sync(0xffffffff, sk2, 1);
        sk2 += __shfl_xor_sync(0xffffffff, sk2, 2);
        const float delta = (vj - eg * sk2) * bt;
        float o0 = 0.f, o1 = 0.f;
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            const float4 q = Q4[m * 4 + p];
            S[m].x = S[m].x * eg + K[m].x * delta;
            S[m].y = S[m].y * eg + K[m].y * delta;
            S[m].z = S[m].z * eg + K[m].z * delta;
            S[m].w = S[m].w * eg + K[m].w * delta;
            o0 += S[m].x * q.x + S[m].y * q.y;
            o1 += S[m].z * q.z + S[m].w * q.w;
        }
        float o = o0 + o1;
        o += __shfl_xor_sync(0xffffffff, o, 1);
        o += __shfl_xor_sync(0xffffffff, o, 2);
        if (p == 0) ob[0] = o;
    }
#pragma unroll
    for (int m = 0; m < 8; ++m) Sp[m * 4 + p] = S[m];
    __syncthreads();   // makes this block's raw out rows visible to the norm pass
    const int lane = tid & 31, wid = tid >> 5;
    if (wid == 0) {
        const float4 w4 = reinterpret_cast<const float4 *>(nw)[lane];
        const int64_t off = (int64_t) h * D;
        float4 o4 = reinterpret_cast<float4 *>(out + off)[lane];
        const float4 z4 = reinterpret_cast<const float4 *>(z + h * D)[lane];
        const float ss = warp_sum(o4.x * o4.x + o4.y * o4.y + o4.z * o4.z + o4.w * o4.w);
        const float sc = rsqrtf(ss / D + eps);
        o4.x *= sc * w4.x * ogate<SIG>(z4.x); o4.y *= sc * w4.y * ogate<SIG>(z4.y);
        o4.z *= sc * w4.z * ogate<SIG>(z4.z); o4.w *= sc * w4.w * ogate<SIG>(z4.w);
        reinterpret_cast<float4 *>(out + off)[lane] = o4;
        if (qbuf) {
            const int ng = Hv * (D / 32), g = h * (D / 32) + (lane >> 3);
            a16_quant8(o4, a16_qs(qbuf, 0, g, ng), a16_ads(qbuf, 0, g, ng, 1));
        }
    }
    (void) NT;
}

void gdn_layer(const float * x, const float * a, const float * b, const float * z, const GdnWeights & w,
               float * cstate, float * state, float * y, float * decay, float * beta, float * out,
               int T, int Hk, int Hv, int D, float eps, cudaStream_t s, void * qbuf, float * conv_snap, float * ssm_snap,
               int ldx, int lda, int ldb, int ldz, unsigned int * fuse_cnt) {
    // P100E_GDN_FUSE (fuse_cnt given): one launch for the plain T = 1 step (dense rows, decay / beta in place or not)
    if (fuse_cnt && T == 1 && D == 128 && !conv_snap && !ssm_snap && Hk > 0 && Hv % Hk == 0 && Hv / Hk <= 64 &&
        (ldx <= 0 || ldx == 2 * Hk * D + Hv * D) && (ldz <= 0 || ldz == Hv * D) && !(reinterpret_cast<uintptr_t>(z) & 15) &&
        static_cast<const float *>(y) != x) {   // other blocks still read x[c] after block h < Hk writes y[c]
        if (w.gate)
            gdn_step1_k<true><<<Hv, 512, 0, s>>>(x, cstate, w.conv_w, const_cast<float *>(a), const_cast<float *>(b), w.dt_bias,
                                                 w.ssm_a, y, decay, beta, z, w.norm_w, state, out, 2 * Hk * D + Hv * D, Hk, Hv,
                                                 rsqrtf((float) D), eps, qbuf, fuse_cnt);
        else
            gdn_step1_k<false><<<Hv, 512, 0, s>>>(x, cstate, w.conv_w, const_cast<float *>(a), const_cast<float *>(b), w.dt_bias,
                                                  w.ssm_a, y, decay, beta, z, w.norm_w, state, out, 2 * Hk * D + Hv * D, Hk, Hv,
                                                  rsqrtf((float) D), eps, qbuf, fuse_cnt);
        return;
    }
    gdn_prep(x, cstate, w, a, b, y, decay, beta, T, Hk, Hv, D, eps, s, conv_snap, ldx, lda, ldb);
    gdn_scan(y, decay, beta, z, w.norm_w, state, out, T, Hk, Hv, D, eps, s, 0, qbuf, ssm_snap, ldz, w.gate);
}

} // namespace p100e::fast
