// p100e-native vision encoder for the Qwen3.8 mmproj (projector qwen3vl_merger): CPU preprocessing with mtmd's exact
// math, then the ViT (27 pre-LN blocks, hidden 1152, 16 heads x 72, GELU-tanh MLP 4304, 2D vision RoPE, full attention)
// and the 2x2 merger (4608 -> 4608 -> 5120) on one or two GPUs. No llama.cpp / ggml at runtime.
//
// Split over two GPUs: sequence-parallel (each GPU owns half of the patches for every op; the K / V rows of each layer
// are exchanged once per block over P2P so every GPU's attention sees all keys). Per layer and GPU that moves
// N/2 x 2 x 1152 f16 = 18.8 MB at 2040 tokens (8160 patches), against 2 all-reduces of N x 1152 (37.6 MB f16 / 75 MB f32)
// for a head / MLP-column split, and it needs no partial-sum precision tricks. Each GPU needs the whole layer's
// weights: they stay in pinned host memory as stored in the GGUF (Q8_0 / F16 / F32) and are uploaded one layer ahead
// into a double-buffered staging area, then dequantized to f16 there (opts::host_weights, default); the upload hides
// behind ~30 ms of compute per layer. Permanent VRAM per GPU is then the small f32 tensors (norms, biases, patch and
// position embeddings, ~16 MB). The activations live in scratch memory that the caller can lend (the LLM's prefill
// ring is idle during an encode), else the encoder allocates its own.
//
// Linear layers: f16 weights (Q8_0 dequantized with ggml's to-f16) x per-token prescaled f16 activations through
// fast::fold_gemm (half2 products, fp32 folds every 256 values of K). Attention: own D = 72 kernel (half2 products,
// fp32 online softmax, no mask). Norms, softmax state, residual stream: fp32.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace p100e::nvit {

// ---------------------------------------------------------------------------------------------------------------------
// preprocessing (CPU). Bit-identical to mtmd (fork 98ab7e5): stb_image decode to RGB (mtmd_helper_bitmap_init_from_buf),
// smart_resize to multiples of 32 px inside [min_tokens, max_tokens] x 1024 px (calc_size_preserved_ratio), Pillow-style
// fixed-point bicubic resize with PAD_CEIL letterboxing in black (img_tool::resize), /255, (x - 0.5) / 0.5.

struct rgb_image {
    int w = 0, h = 0;
    std::vector<uint8_t> px;   // [h][w][3]
};

// image limits (decode_image refuses them from the file header, preprocess throws): sides <= 32768 px, <= 2^28 px in
// all (805 MB of RGB), aspect ratio (long / short side) <= 200 like Qwen2-VL's smart_resize. A 1 x 100000 strip
// would otherwise become a 32 x 457888 px image (7x max_tokens, an O(N^2) attention over 57k patches)
constexpr int img_max_side = 32768;
constexpr int64_t img_max_pixels = (int64_t) 1 << 28;
constexpr int img_max_aspect = 200;
bool image_size_ok(int w, int h, std::string * err = nullptr);   // false + err when w x h is outside the limits

// stb_image (png / jpeg / bmp / gif / tga / psd / hdr / pic / pnm). false + err on failure (also: outside the limits).
bool decode_image(const void * bytes, size_t n, rgb_image & out, std::string * err = nullptr);
// smart_resize: output size (multiples of 32) for a w x h image with token limits [min_tok, max_tok]. Unlike mtmd's
// the area never exceeds max_tok x 1024 px from the max_tok side (mtmd's overshoots for a short side < 32 px after
// scaling); the min_tok side rounds up as mtmd does (a few rows / columns above max_tok only when min_tok ~ max_tok)
void smart_size(int w, int h, int min_tok, int max_tok, int & ow, int & oh);

struct image {                 // preprocessed: planar [3][ny][nx] f32 in [-1, 1] (clip's inp_raw layout)
    int nx = 0, ny = 0;
    std::vector<float> data;
    int n_tokens() const { return (nx / 32) * (ny / 32); }
};
image preprocess(const rgb_image & img, int min_tok, int max_tok);

// ---------------------------------------------------------------------------------------------------------------------
struct opts {
    std::string mmproj;                 // mmproj-Qwen3.8-27B-Q8_0.gguf
    std::vector<int> devices{0};        // 1 or 2 CUDA devices (2: sequence-parallel split)
    int max_tokens = 2048;              // largest image (merged tokens) the own scratch is sized for
    bool host_weights = true;           // false: raw weights resident in VRAM (~600 MB per GPU), no per-layer upload
    int chunks = 0;                     // patch chunks per GPU (pipeline streams); 0 = automatic
    int stream_priority = 1;            // 1: CUDA's least stream priority, 0: default priority. Both are 0 on current
                                        // GPUs (nothing ranks below the default): decode of other sessions stays ahead
                                        // only through tp_model's greatest-priority main streams and the pacing
    bool oracle = false;                // debug: fp32 reference path instead (weights dequantized to f32, SGEMM, fp32
                                        // attention; first device only; slow) - same dump hook
    bool verbose = false;
};

struct timing {
    double prep_ms = 0;        // host -> device image upload + patch embedding
    double layers_ms = 0;      // 27 blocks
    double merger_ms = 0;      // post-LN + merger + download
    double total_ms = 0;       // whole encode (host wall time, sync to sync)
};

// device memory lent by the caller for one encode (e.g. the LLM's idle prefill ring). One entry per device of
// opts::devices, in order; bytes >= scratch_bytes(n_tokens).
struct scratch_view {
    void * p = nullptr;
    size_t bytes = 0;
    int dev = -1;                       // CUDA device of p (-1: the entry's position = rank)
    std::shared_ptr<void> hold;         // the lender's lease (e.g. tp_model::idle_scratch): kept until the encode that
                                        // uses this memory returns, dropped at once when it is not used
};

class encoder {
public:
    explicit encoder(const opts & o);   // throws std::runtime_error
    ~encoder();
    encoder(const encoder &) = delete;
    encoder & operator=(const encoder &) = delete;

    int n_embd() const;                 // 5120 (projection_dim)
    int n_ranks() const;
    // scratch per device for an image of n_tokens merged tokens
    size_t scratch_bytes(int n_tokens) const;
    // VRAM the encoder keeps between encodes on rank r (weights / small tensors; plus its own scratch if allocated)
    size_t permanent_bytes(int rank) const;
    // free the encoder's own scratch (the next encode without borrowed scratch allocates it again)
    void release_scratch();

    // encode one preprocessed image. out = [ty * tx][n_embd] f32, token i at grid row i / tx, column i % tx.
    // borrow: optional scratch per device (else the encoder's own, allocated on first use for opts::max_tokens).
    // Synchronous (returns when the result is in out). Not thread-safe (one encode at a time).
    void encode(const image & img, std::vector<float> & out, int & tx, int & ty, timing * t = nullptr,
                const std::vector<scratch_view> * borrow = nullptr);

    // debug hook: when set, encode runs one chunk per GPU, synchronises after each step and calls dump(name, layer,
    // rows, cols, data) with host copies (rows = patches in merge order, or merged tokens) of: "inp_pos_emb" (layer -1),
    // per layer in dump_layers: "ln1", "Qcur_rope", "Kcur_rope" (rows = patches, cols = 16 x 72, unscaled),
    // "kqv_out", "ffn_inp", "layer_out"; then "post_ln" (layer 27) and "embd" (final, merged tokens).
    // optional source of borrowed scratch, asked at every encode that gets no `borrow` (e.g. tp_model::idle_scratch:
    // the LLM's prefill ring, idle between forwards; empty while a prefill runs). Entries are matched to the encoder's
    // devices by dev; when one is missing or too small the encoder uses its own scratch for that encode only (sized for
    // the image, freed when it returns, so the next encode borrows again). When even that does not fit in free VRAM
    // and the source lent nothing (busy), the encode polls the source until it lends (P100E_NVIT_SCRATCH_WAIT_S, 600)
    // instead of failing. Set it only when the lender can lend at all (else every encode logs a busy lender).
    std::function<std::vector<scratch_view>()> scratch_source;

    std::function<void(const std::string & name, int layer, int rows, int cols, const float * data)> dump;
    std::vector<int> dump_layers;

private:
    struct impl;
    std::unique_ptr<impl> p_;
};

} // namespace p100e::nvit
