#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const float *   beta_pre,
                                     float *         beta_sig,
                                     const float *   s_base,
                                     const int32_t * s_rows,
                                     int64_t         s_row_stride,
                                     float *         s_gather_dst) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
    // With the GET_ROWS folded in, read straight from the cache row and write the same values to
    // the folded GET_ROWS node's dst (node-preserving).  Each element is read once and written
    // once by its owning thread, so input and output may share an address.
    if (s_rows != nullptr) {
        const float * src = s_base + (int64_t) s_rows[sequence] * s_row_stride + h_idx * S_v * S_v + col * S_v;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            s_shard[r]  = src[i];
        }
        if (s_gather_dst != nullptr) {
            float * gd = s_gather_dst + state_in_offset + col * S_v;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                gd[i]       = s_shard[r];
            }
        }
    } else {
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        // Apply beta's sigmoid here when beta_sig != nullptr.  The expression must stay identical
        // to op_sigmoid in unary.cu -- that is what makes the result bit-identical.
        float beta_val;
        if (beta_sig != nullptr) {
            beta_val = 1.0f / (1.0f + expf(-beta_pre[gb_offset]));
            // Also write the folded SIGMOID node's dst; exactly one thread writes each element.
            if (lane == 0 && threadIdx.y == 0 && blockIdx.z == 0) {
                beta_sig[gb_offset] = beta_val;
            }
        } else {
            beta_val = *beta_t;
        }

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// Opt-in variants of the scalar-gate (non-KDA) recurrence above; the kernel above is untouched and
// stays the default.
//   pipe: Ported from Kmic-68/llama.cpp e5c264b71 (MIT), the "pipelined delta-net reduction" half of
//         that commit (GGML_PF_GDN_PIPE=1). attn[col] for token t and kv[col] for token t+1 both read
//         only the state after token t, so token t+1's kv partial is pulled forward and the two
//         reduce together as one float2 butterfly (one shuffle-chain latency instead of two on the
//         loop's critical path). Bit-identical per Kmic: same per-component offsets and order, same
//         accumulation over r.
//   walk: Ported from Kmic-68/llama.cpp 1b29f55de (MIT) (GGML_PF_GDN_WALK=1). Tokens are visited in
//         order, so the per-token q/k/v addresses and the g/beta offset are stepped instead of
//         recomputed (sm_60 has no native 64-bit multiply). Addressing only, so bit-identical.
// pf-gdn adaptation: both keep this fork's prologue (folded state GET_ROWS, node-preserving) and the
// folded beta sigmoid (beta_sig != nullptr), which Kmic's kernel does not have.
template <int S_v, bool keep_rs_t, bool pipe, bool walk>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_pf_cuda(const float * q,
                                        const float * k,
                                        const float * v,
                                        const float * g,
                                        const float * beta,
                                        const float * curr_state,
                                        float *       dst,
                                        float *       state,
                                        int64_t       H,
                                        int64_t       n_tokens,
                                        int64_t       n_seqs,
                                        int64_t       sq1,
                                        int64_t       sq2,
                                        int64_t       sq3,
                                        int64_t       sv1,
                                        int64_t       sv2,
                                        int64_t       sv3,
                                        int64_t       sb1,
                                        int64_t       sb2,
                                        int64_t       sb3,
                                        const uint3   neqk1_magic,
                                        const uint3   rq3_magic,
                                        float         scale,
                                        int64_t       state_slot_stride,
                                        int           K,
                                        const float *   beta_pre,
                                        float *         beta_sig,
                                        const float *   s_base,
                                        const int32_t * s_rows,
                                        int64_t         s_row_stride,
                                        float *         s_gather_dst) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];

    ggml_cuda_pdl_sync();
    // prologue identical to gated_delta_net_cuda (folded state GET_ROWS, node-preserving)
    if (s_rows != nullptr) {
        const float * src = s_base + (int64_t) s_rows[sequence] * s_row_stride + h_idx * S_v * S_v + col * S_v;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            s_shard[r]  = src[i];
        }
        if (s_gather_dst != nullptr) {
            float * gd = s_gather_dst + state_in_offset + col * S_v;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                gd[i]       = s_shard[r];
            }
        }
    } else {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            s_shard[r]  = curr_state[i];
        }
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
    auto snapshot = [&](const int t) {
        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * slot = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    slot[col * S_v + i] = s_shard[r];
                }
            }
        }
    };

    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
    float g_val    = 0.0f;
    float beta_val = 0.0f;
    float v_val    = 0.0f;

    // walk (1b29f55de): addresses of the token the pointers currently point at
    const float * q_w  = q + iq3 * sq3 + iq1 * sq1 + lane;
    const float * k_w  = k + iq3 * sq3 + iq1 * sq1 + lane;
    const float * v_w  = v + sequence * sv3 + h_idx * sv1 + col;
    int64_t       gb_w = sequence * sb3 + h_idx * sb1;

    // load token t's inputs (t is only used by the recomputing form)
    auto load_token = [&](const int t) {
        const float * q_t;
        const float * k_t;
        const float * v_t;
        int64_t       gb;
        if constexpr (walk) {
            q_t = q_w; k_t = k_w; v_t = v_w; gb = gb_w;
            q_w  += sq2;
            k_w  += sq2;
            v_w  += sv2;
            gb_w += sb2;
        } else {
            q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1 + lane;
            k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1 + lane;
            v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1 + col;
            gb  = sequence * sb3 + t * sb2 + h_idx * sb1;
        }
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            k_reg[r] = k_t[r * warp_size];
            q_reg[r] = q_t[r * warp_size];
        }
        g_val = expf(g[gb]);
        // folded beta sigmoid: the expression must stay identical to op_sigmoid in unary.cu
        if (beta_sig != nullptr) {
            beta_val = 1.0f / (1.0f + expf(-beta_pre[gb]));
            if (lane == 0 && threadIdx.y == 0 && blockIdx.z == 0) {
                beta_sig[gb] = beta_val;
            }
        } else {
            beta_val = beta[gb];
        }
        v_val = *v_t;
    };

    if constexpr (pipe) {
        // e5c264b71: software-pipelined, kv[col] for token t+1 shares token t's attn butterfly
        float kv_shard = 0.0f;
        load_token(0);
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard += s_shard[r] * k_reg[r];
        }
        // carried across iterations already reduced -- do not reduce it a second time
        float kv_col = warp_reduce_sum<warp_size>(kv_shard);

        for (int t = 0; t < n_tokens; t++) {
            // delta[col] = (v[col] - g * kv[col]) * beta
            const float delta_col = (v_val - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]    = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            // token t+1's kv partial reads only the state after token t
            float kv_next = 0.0f;
            if (t + 1 < n_tokens) {
                load_token(t + 1);
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_next += s_shard[r] * k_reg[r];
                }
            }

            const float2 red = warp_reduce_sum<warp_size>(make_float2(attn_partial, kv_next));
            kv_col = red.y;

            if (lane == 0) {
                attn_data[col] = red.x * scale;
            }

            attn_data += S_v * H;

            snapshot(t);
        }
    } else {
        // walk only: the shipped loop with stepped addresses
        for (int t = 0; t < n_tokens; t++) {
            load_token(t);

            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            const float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            const float delta_col = (v_val - g_val * kv_col) * beta_val;

            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]    = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            const float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }

            attn_data += S_v * H;

            snapshot(t);
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// launcher for gated_delta_net_pf_cuda; pf_variant: bit 0 = pipe, bit 1 = walk (non-zero)
template <bool keep_rs_t>
static void launch_gated_delta_net_pf(
        const int pf_variant,
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K,
        const float * beta_pre_d, float * beta_sig_d,
        const float * s_base_d, const int32_t * s_rows_d, int64_t s_row_stride, float * s_gather_dst_d,
        cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

#define GDN_PF_LAUNCH(SV, PIPE, WALK) \
    ggml_cuda_kernel_launch(gated_delta_net_pf_cuda<SV, keep_rs_t, PIPE, WALK>, launch_params, \
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, \
        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, \
        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, \
        beta_pre_d, beta_sig_d, s_base_d, s_rows_d, s_row_stride, s_gather_dst_d)
#define GDN_PF_LAUNCH_SV(SV) \
    switch (pf_variant) { \
        case 1:  GDN_PF_LAUNCH(SV, true,  false); break; \
        case 2:  GDN_PF_LAUNCH(SV, false, true);  break; \
        default: GDN_PF_LAUNCH(SV, true,  true);  break; \
    }
    switch (S_v) {
        case 16:  GDN_PF_LAUNCH_SV(16);  break;
        case 32:  GDN_PF_LAUNCH_SV(32);  break;
        case 64:  GDN_PF_LAUNCH_SV(64);  break;
        case 128: GDN_PF_LAUNCH_SV(128); break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
#undef GDN_PF_LAUNCH_SV
#undef GDN_PF_LAUNCH
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K,
        const float * beta_pre_d, float * beta_sig_d,
        const float * s_base_d, const int32_t * s_rows_d, int64_t s_row_stride, float * s_gather_dst_d,
        cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                beta_pre_d, beta_sig_d, s_base_d, s_rows_d, s_row_stride, s_gather_dst_d);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                beta_pre_d, beta_sig_d, s_base_d, s_rows_d, s_row_stride, s_gather_dst_d);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                beta_pre_d, beta_sig_d, s_base_d, s_rows_d, s_row_stride, s_gather_dst_d);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                beta_pre_d, beta_sig_d, s_base_d, s_rows_d, s_row_stride, s_gather_dst_d);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache,
        bool fuse_beta_sigmoid = false, const ggml_cuda_gated_delta_net_fused_gather * gather = nullptr) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    // When beta's sigmoid is folded in, src_beta (the SIGMOID node's dst) has not been written
    // yet, so read from its input and write the result to src_beta as well.
    const float * beta_pre_d = nullptr;
    float *       beta_sig_d = nullptr;
    if (fuse_beta_sigmoid) {
        GGML_ASSERT(src_beta->src[0] != nullptr);
        GGML_ASSERT(ggml_are_same_shape(src_beta, src_beta->src[0]));
        GGML_ASSERT(ggml_are_same_stride(src_beta, src_beta->src[0]));
        GGML_ASSERT(ggml_is_contiguous(src_beta->src[0]));
        beta_pre_d = (const float *) src_beta->src[0]->data;
        beta_sig_d = (float *) src_beta->data;
        GGML_ASSERT(beta_pre_d != nullptr && beta_sig_d != nullptr);
    }

    const float *   s_base_d       = nullptr;
    const int32_t * s_rows_d        = nullptr;
    int64_t         s_row_stride    = 0;
    float *         s_gather_dst_d  = nullptr;
    if (gather != nullptr) {
        GGML_ASSERT(gather->base != nullptr && gather->rows != nullptr && gather->row_stride > 0);
        s_base_d      = gather->base;
        s_rows_d      = gather->rows;
        s_row_stride  = gather->row_stride;
        s_gather_dst_d = gather->gather_dst;
    }

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K,
                beta_pre_d, beta_sig_d, s_base_d, s_rows_d, s_row_stride, s_gather_dst_d, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K,
                beta_pre_d, beta_sig_d, s_base_d, s_rows_d, s_row_stride, s_gather_dst_d, stream);
        }
    } else {
        // Ported from Kmic-68/llama.cpp 77e05601d (MIT): opt-in chunked prefill form / fp64 reference
        // (gdn-chunked.cu). Returns false unless GGML_CUDA_GDN_CHUNKED or GGML_CUDA_GDN_REF is set,
        // so by default the shipped recurrence below runs exactly as before.
        // pf-gdn: our beta-sigmoid and state-gather fusions are passed through.
        if (S_v == 128) {
            const ggml_cuda_gdn_alt_args a = { q_d, k_d, v_d, g_d, b_d,
                s_rows_d != nullptr ? s_base_d : s_d, dst_d, state_d,
                H, n_tokens, n_seqs, neqk1, rq3, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                scale, state_slot_stride, K, s_rows_d, s_row_stride,
                beta_pre_d, beta_sig_d, s_gather_dst_d };
            if (ggml_cuda_gdn_alt(ctx, a)) {
                return;
            }
        }
        // Opt-in recurrence variants (Kmic e5c264b71 pipelined reduction, 1b29f55de walked
        // addresses); unset = the shipped kernel below
        static const int pf_variant =
            ((getenv("GGML_PF_GDN_PIPE") && atoi(getenv("GGML_PF_GDN_PIPE")) != 0) ? 1 : 0) |
            ((getenv("GGML_PF_GDN_WALK") && atoi(getenv("GGML_PF_GDN_WALK")) != 0) ? 2 : 0);
        if (pf_variant != 0) {
            if (keep_rs) {
                launch_gated_delta_net_pf<true>(pf_variant, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K,
                    beta_pre_d, beta_sig_d, s_base_d, s_rows_d, s_row_stride, s_gather_dst_d, stream);
            } else {
                launch_gated_delta_net_pf<false>(pf_variant, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K,
                    beta_pre_d, beta_sig_d, s_base_d, s_rows_d, s_row_stride, s_gather_dst_d, stream);
            }
            return;
        }
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K,
                beta_pre_d, beta_sig_d, s_base_d, s_rows_d, s_row_stride, s_gather_dst_d, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K,
                beta_pre_d, beta_sig_d, s_base_d, s_rows_d, s_row_stride, s_gather_dst_d, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}

void ggml_cuda_op_gated_delta_net_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                        const ggml_cuda_gated_delta_net_fused_cache *  cache,
                                        bool                                           fuse_beta_sigmoid,
                                        const ggml_cuda_gated_delta_net_fused_gather * gather) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, cache, fuse_beta_sigmoid, gather);
}
