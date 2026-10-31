// Fast gated-delta-net (linear attention / SSM) kernels for the Qwen3.8 hybrid layers.
// Covers everything between the projections: conv1d(k=4)+silu with state update, q/k L2 norm,
// decay/beta gates, the gated delta rule recurrence (fp32 state) and the gated output RMSNorm.
//
// Layouts are identical to the reference path (src/kernels.cu, model.cu):
//   x      [T][C]      qkv projection output, C = 2*Hk*D + Hv*D (q heads, k heads, v heads)
//   a, b   [T][Hv]     ssm_alpha / ssm_beta projections
//   z      [T][Hv*D]   attn_gate projection
//   cstate [C][3]      conv state (previous 3 inputs, oldest first)
//   state  [Hv][D][D]  delta-rule state, row j (v dim) x column i (k dim)
//   out    [T][Hv][D]  gated-normed output, ready for ssm_out
// v head h uses q/k head h % Hk. Tensor parallel: pass the per-GPU head counts (HKl, HVl) and the
// per-GPU buffers exactly as tp_model.cu builds them (its local qkv/conv rows keep this same layout
// with Hk = HKl, Hv = HVl, and v head h still maps to k head h % HKl), so the same calls work per GPU.
// Requires Dk == Dv == 128. Exact form (same math as the reference, fp32): sequential scan over T with
// the state held in registers, so the state is read once and written once per call for any T.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace p100e::fast {

struct GdnWeights {
    const float * conv_w;   // [C][4]
    const float * dt_bias;  // [Hv]
    const float * ssm_a;    // [Hv]
    const float * norm_w;   // [D]
    int gate = 0;           // output gate of the gated RMSNorm: 0 = silu(z) (Qwen3.8-27B), 1 = sigmoid(z) (Flash-Next)
};

// Step 1: conv1d+silu (+ conv state update), q/k L2-normalized (q also scaled by 1/sqrt(D)),
// decay = exp(softplus(a + dt_bias) * ssm_a), beta = sigmoid(b). y [T][C], decay/beta [T][Hv].
// decay/beta may alias a/b (dense a/b only).
// ldx / lda / ldb (0 = dense C / Hv / Hv): row strides of x, a and b, so they can be read in place from concatenated projection
// outputs [T][n_tot] (the MTP verify); y, decay and beta are always written dense.
// cstate_snap != null (T > 1): also writes the conv state after token t, t = 0..T-2, to cstate_snap + t*C*3 (the
// verify batch's rollback snapshots), the same values T sequential T = 1 calls leave in cstate after each token.
void gdn_prep(const float * x, float * cstate, const GdnWeights & w, const float * a, const float * b,
              float * y, float * decay, float * beta, int T, int Hk, int Hv, int D, float eps, cudaStream_t s,
              float * cstate_snap = nullptr, int ldx = 0, int lda = 0, int ldb = 0);

// Step 2: recurrence + gated RMSNorm. y/decay/beta from gdn_prep; out [T][Hv][D] (must not alias y).
void gdn_scan(const float * y, const float * decay, const float * beta, const float * z, const float * norm_w,
              float * state, float * out, int T, int Hk, int Hv, int D, float eps, cudaStream_t s, int rows = 0,
              void * qbuf = nullptr, float * state_snap = nullptr, int ldz = 0, int gate = 0);
// rows = state rows per block (16/32/64/128; 0 = auto: 128 for T<=4, else 32). 128 fuses the norm (1 launch); fewer rows give
// more blocks (better for few heads / long T) plus a separate norm launch.
// qbuf != null (rows-128 path only, i.e. T <= 4): also writes the a16 quantized out (fast::matvec_prequantized layout,
// n_in = Hv*D), the same bytes fast::matvec_quantize(out) would give.
// state_snap != null (T > 1): also writes the state after token t, t = 0..T-2, to state_snap + t*Hv*D*D from inside the
// scan (rollback snapshots; the state is still read once). Outputs, final state and snapshots are bit-identical to T
// sequential T = 1 calls with a state copy after each token (tests/fast_gdn_verify_test.cu).
// ldz (0 = dense Hv*D, else a multiple of 4): row stride of z.

// Both steps (2 launches). Scratch y [T][C]; decay/beta [T][Hv] (may alias a/b). conv_snap / ssm_snap: see above.
void gdn_layer(const float * x, const float * a, const float * b, const float * z, const GdnWeights & w,
               float * cstate, float * state, float * y, float * decay, float * beta, float * out,
               int T, int Hk, int Hv, int D, float eps, cudaStream_t s, void * qbuf = nullptr,
               float * conv_snap = nullptr, float * ssm_snap = nullptr, int ldx = 0, int lda = 0, int ldb = 0,
               int ldz = 0, unsigned int * fuse_cnt = nullptr);
// fuse_cnt (decode step 6b, P100E_GDN_FUSE): a zeroed device buffer of >= Hk counters owned by this stream; with it a plain
// T = 1 step (no snapshots, dense rows) is ONE launch (gdn_step1_k), bit-identical to the two (same PTX op order; engine
// check: p100e-run --ab "P100E_GDN_FUSE=0/P100E_GDN_FUSE=1" with a prompt and --gen, every decode step hashed).

// Several sequences in one call (batched multi-session decode / MTP verify). Rows of all sequences are
// stacked dense: x [Ttot][C], a / b [Ttot][Hv] (decay / beta may alias them), z [Ttot][Hv*D], y scratch [Ttot][C],
// out [Ttot][Hv][D]; sequence i owns rows row0..row0+T-1 (T = 1..4) and its states live in a slot pool. Device
// descriptor seqs [nseq][4] = (row0, T, state slot, first snapshot slot). Each sequence gets bitwise the result of its
// own gdn_layer call (T tokens, qbuf rows row0.. of a Ttot-row a16 buffer; snapshots: as conv_snap / ssm_snap there,
// starting at its snapshot slot). Graph friendly: only seqs (device) varies between steps of one (Ttot, nseq, maxT).
struct gdn_multi {
    const int32_t * seqs = nullptr;   // device [nseq][4]
    int nseq = 0, Ttot = 0, maxT = 0;   // maxT: upper bound of any sequence's T (sizes the prep grid)
    float * conv_base = nullptr; size_t conv_slot = 0;   // conv state of slot s = conv_base + s * conv_slot (floats)
    float * ssm_base = nullptr;  size_t ssm_slot = 0;    // ssm state of slot s = ssm_base + s * ssm_slot
    float * conv_snap = nullptr; float * ssm_snap = nullptr;   // this layer's snapshot slot 0 (nullptr: no snapshots);
                                                               // slot i at + i * C * 3 / + i * Hv * D * D
};
void gdn_layer_multi(const float * x, const float * a, const float * b, const float * z, const GdnWeights & w,
                     const gdn_multi & m, float * y, float * decay, float * beta, float * out, int Hk, int Hv, int D,
                     float eps, cudaStream_t s, void * qbuf = nullptr);

} // namespace p100e::fast
