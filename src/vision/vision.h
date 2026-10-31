// Image input for p100e: llama.cpp's mtmd / clip from the fork (same commit as
// the server, ggml-cuda sm_60) linked in as DLLs, used only to tokenize prompts with images, preprocess the images
// and run the vision encoder on one GPU. p100e then prefills the 5120-wide embedding rows with interleaved M-RoPE
// positions (tp_model::forward_image).
//
// The fork's DLLs (mtmd, llama, ggml, ggml-base, ggml-cuda, ggml-cpu, cuBLAS) are delay-loaded on the first
// vision_encoder: text-only use of the engine never needs them. They are looked up in vision_opts::dll_dir, then
// P100E_VISION_DLLS, the executable's directory and the fork build's bin directory.
//
// Tokenization matches llama-server's mtmd_tokenize: the prompt is split at marker(); each text part is tokenized on its
// own (parse_special, no BOS: Qwen adds none), each image becomes tokens(<|vision_start|>) + the image rows +
// tokens(<|vision_end|>).
//
// Embedding cache: the encoder output is kept in host RAM (LRU, vision_opts::cache_mb) under a 128-bit hash of the
// image file bytes + the preprocessing settings (token caps). A cache hit skips decoding, preprocessing and the
// encoder. image_embd::key is the same hash, for prompt-cache keys (two different images must never match the same
// cached prefix).
//
// Environment: the fork's ggml reads GGML_* variables when its CUDA backend initializes (inside the vision_encoder
// constructor) or at a kernel's first use. Before the DLLs load the constructor removes every inherited GGML_* variable
// and sets GGML_CUDA_GEMM_FOLD from vision_opts::fold_gemm (vision_pin_ggml_env), so the encoder always runs the
// configuration it was validated with, whatever launched the process. GGML_CUDA_P2P in particular is fatal: ggml then
// enables GPU peer access itself and aborts on "peer access is already enabled" when tp_model (loaded first, as the
// server does) has enabled it. P100E_VISION_GGML_ENV=keep keeps the inherited variables for experiments (GGML_CUDA_P2P
// is removed regardless).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <tuple>
#include <string>
#include <vector>

// p100e-server links the encoder as a DLL (p100e-vision.dll, cmake/vision.cmake): the server's static, CPU-only fork
// libraries (llama / mtmd / ggml) and the fork DLLs this code is linked against export the same names, so they must
// never meet in one module. P100E_VISION_BUILD_DLL: building that DLL; P100E_VISION_DLL: using it; neither: static
// (p100e-run, p100e-vision-test).
#if defined(_WIN32) && defined(P100E_VISION_BUILD_DLL)
#define P100E_VISION_API __declspec(dllexport)
#elif defined(_WIN32) && defined(P100E_VISION_DLL)
#define P100E_VISION_API __declspec(dllimport)
#else
#define P100E_VISION_API
#endif

namespace p100e {

struct vision_opts {
    std::string mmproj;          // mmproj GGUF (mmproj-Qwen3.8-27B-Q8_0.gguf)
    std::string text_model;      // the LLM GGUF, loaded vocab-only (tokenizer, n_embd and rope type for mtmd)
    int device = 1;              // CUDA device of the encoder (ggml backend "CUDA<n>", CUDA runtime order); -1 = CPU
    int min_tokens = 1024;       // image token limits (llama-server --image-min-tokens / --image-max-tokens); the
    int max_tokens = 2048;       // encoder's compute buffer is reserved for max_tokens at the warmup
    bool flash_attn = true;      // llama-server --flash-attn on
    bool warmup = true;          // encode a dummy image at load (reserves the compute buffer once)
    int n_threads = 4;           // CPU threads (preprocessing, CPU encoder)
    size_t cache_mb = 2048;      // embedding cache in host RAM (f32 rows, ~40 MiB per 2048-token image); 0 = off
    std::string dll_dir;         // directory with the fork's DLLs (empty: search, see above)
    // the fork's Pascal fold GEMM for the encoder's matmuls (GGML_CUDA_GEMM_FOLD=2; false: 0 = cuBLAS COMPUTE_16F):
    // fp16 products with fp32 accumulation, 2-4 % faster than cuBLAS here and more accurate. Only
    // ggml (the encoder) reads it. With P100E_VISION_GGML_ENV=keep an inherited GGML_CUDA_GEMM_FOLD wins.
    bool fold_gemm = true;
    bool verbose = false;        // forward llama.cpp / mtmd info logs (warnings and errors always go to stderr)
    // p100e-native encoder (src/nvit) instead of mtmd's: -1 = environment P100E_NVIT (unset or "1": native,
    // "0": mtmd), 0 = mtmd, 1 = native. Native runs on native_devices (empty: P100E_NVIT_DEVICES, else "0,1"; one GPU
    // or two, sequence-parallel); no mtmd context is created (the vocab-only text model still loads for tokenization).
    int native = -1;
    std::vector<int> native_devices;
};

// one encoded image (shared by the cache and the prompts that use it)
struct image_embd {
    int nx = 0, ny = 0;          // token grid (one token per 32 x 32 px of the resized image), n_tok = nx * ny
    int n_embd = 0;
    std::vector<float> data;     // [nx * ny][n_embd] f32, row-major in raster order (token i at row i / nx, col i % nx)
    uint64_t key[2] = {0, 0};    // content hash (file bytes + preprocessing settings)
    int src_w = 0, src_h = 0;    // decoded image size
    int n_tok() const { return nx * ny; }
    size_t bytes() const { return data.size() * sizeof(float); }
};

struct vision_timing {
    double hash_ms = 0, decode_ms = 0, resize_ms = 0, preprocess_ms = 0, encode_ms = 0;
    double wait_ms = 0;          // waiting for another thread's encode (cache miss only)
    bool cached = false;
    double total_ms() const { return hash_ms + wait_ms + decode_ms + resize_ms + preprocess_ms + encode_ms; }
};

// one piece of a multimodal prompt: a run of text tokens, or an image
struct mm_piece {
    std::vector<int32_t> tokens;                // text (empty for an image)
    std::shared_ptr<const image_embd> image;    // image (null for text)
    vision_timing t;                            // image only
};

struct image_input {
    std::string bytes;           // the encoded file (png / jpeg / bmp / gif ...: stb_image formats; WebP through mtmd's
                                 // helper = ffmpeg / ffprobe on the PATH, also in native mode)
    int max_tokens = 0;          // per-image token cap (0 = vision_opts::max_tokens; capped at it); see detail_max_tokens
};

class P100E_VISION_API vision_encoder {
public:
    explicit vision_encoder(const vision_opts & o);   // throws std::runtime_error (missing DLLs / files, load errors)
    ~vision_encoder();
    vision_encoder(const vision_encoder &) = delete;
    vision_encoder & operator=(const vision_encoder &) = delete;

    // prompt text with one marker() per image -> pieces in order (adjacent text merged). Images that are not in the
    // cache are decoded, preprocessed and encoded here (synchronously, on the encoder's GPU). Thread-safe: one encode
    // at a time, but the cache has its own lock, so an image that is already cached never waits for another
    // request's encode (an image being encoded by another thread is waited for and then taken from the cache).
    // marker: the media marker the prompt uses (empty: marker(); p100e-server passes its random per-process marker, as
    // llama-server does, so a literal "<__media__>" in user text stays text)
    std::vector<mm_piece> build(const std::string & prompt, const std::vector<image_input> & images,
                                const std::string & marker = "");
    // KV rows build() would produce for the same input (text tokens + <|vision_start|> / image rows / <|vision_end|>
    // per image), without encoding: cached images by their entry, others from the decoded size (mtmd's smart resize
    // under the image's token cap, as encode() sizes it). For the count_tokens endpoints; thread-safe, never waits
    // for an encode.
    int count_tokens(const std::string & prompt, const std::vector<image_input> & images,
                     const std::string & marker = "") const;
    // one image -> embeddings (cache first; same locking as build)
    std::shared_ptr<const image_embd> encode(const image_input & img, vision_timing * t = nullptr);

    static const char * marker();              // "<__media__>" (mtmd's default marker)
    int n_embd() const;                         // width of the embedding rows (= the LLM's n_embd; checked by forward_mm)
    int min_tokens() const;
    int max_tokens() const;
    // OpenAI image_url.detail -> per-image token cap: "low" -> min_tokens (1024); "high", "auto", "" and anything else
    // -> max_tokens. The cap is not a hard limit at min_tokens: an image is never cut below min_tokens (Qwen-VL needs
    // >= 1024 tokens for grounding), and the size the cap gives is rounded to 32 px per side and then up to at least
    // min_tokens, so e.g. a 1920x1080 image with "low" gets 43 x 25 = 1075 tokens.
    int detail_max_tokens(const std::string & detail) const;
    // tokenizer of the text model (vocab-only load)
    std::vector<int32_t> tokenize(const std::string & text, bool parse_special = true) const;
    std::string piece(int32_t id) const;       // detokenize one token (special tokens rendered)
    int32_t eos() const;                        // <|im_end|>
    bool is_eog(int32_t id) const;              // end-of-generation token

    struct cache_stats { size_t entries = 0, bytes = 0, hits = 0, misses = 0, evictions = 0; };
    cache_stats stats() const;
    void cache_clear();
    // the warmup log reported CLIP ops the GPU backend does not support (they would fall back to the CPU) or flash
    // attention falling back (llama.cpp issue #21272); the lines are in warnings()
    bool cpu_fallback() const;
    std::vector<std::string> warnings() const;
    std::string dll_dir() const;               // where the DLLs were loaded from
    bool native() const;                        // the p100e-native encoder is in use (vision_opts::native)
    // native encoder only: where its activations live during an encode - f() returns device regions, e.g.
    // tp_model::idle_scratch() (the LLM's prefill ring, idle between forwards; empty while a prefill runs). `hold` is the
    // lender's lease: kept until the encode returns. Without regions (or when one is too small) the encoder allocates
    // its own scratch for max_tokens on first use (logged).
    struct scratch_region { int dev = -1; void * p = nullptr; size_t bytes = 0; std::shared_ptr<void> hold; };
    void set_scratch_source(std::function<std::vector<scratch_region>()> f);

private:
    struct impl;
    std::unique_ptr<impl> p_;
};

// 128-bit content hash used for the cache keys (not cryptographic)
P100E_VISION_API void vision_hash128(const void * data, size_t n, uint64_t seed, uint64_t out[2]);

// Load the fork's DLLs now (dll_dir, else the search at the top of this file); returns their directory. The encoder and
// vision_mmproj_width do this themselves; it is needed before calling ggml / gguf functions directly (tests).
P100E_VISION_API std::string vision_load_dlls(const std::string & dll_dir = "");

// The encoder's GGML_* environment (see the top of this file; the vision_encoder constructor calls it before the DLLs
// load): removes every GGML_* variable (P100E_VISION_GGML_ENV=keep: only GGML_CUDA_P2P) and sets GGML_CUDA_GEMM_FOLD
// to 2 (fold_gemm) or 0 (with keep: only when it is not set). Changes the process environment (CRT and OS block).
// Returns one line per change ("removed NAME=value", "set NAME=value") for the log.
P100E_VISION_API std::vector<std::string> vision_pin_ggml_env(bool fold_gemm);

// Embedding width of an mmproj (clip.vision.projection_dim), read from its GGUF metadata (loads the fork's DLLs from
// dll_dir or the default search, like vision_encoder). Throws for a projector p100e cannot use: a Qwen3-VL deepstack
// mmproj (any true clip.vision.is_deepstack_layers or v.deepstack.* tensors: the encoder output is then
// projection_dim * (1 + n_deepstack) wide and the extra features go into early LLM layers, which p100e does not do), or
// one without projection_dim.
P100E_VISION_API int vision_mmproj_width(const std::string & mmproj, const std::string & dll_dir = "");

// Per-image token cap below the encoder's max (detail "low"): the size (multiples of 32 px) a w x h image is resampled to
// on the host before the encoder's own preprocessing. Both encoders (mtmd and native) use it, so they get the same
// pixels and token grid: the cap's smart size, raised back into [min_tok, max_tok] the way mtmd's smart_resize would
// (the cap's floor rounding lands below min_tok when cap == min_tok: 1920x1080 -> 1376x800 = 1075 tokens, not 1008).
// The result is a fixed point of smart_resize(min_tok, max_tok), so the encoder keeps it (one resample in total).
// Returns false (no host resample) when the cap does not change the plain [min_tok, max_tok] size.
bool vision_cap_size(int w, int h, int min_tok, int cap, int max_tok, int & tw, int & th);

// Whether a decoded w x h image is accepted (encode() and count_tokens() throw otherwise): sides <= 32768 px, <= 2^28 px
// in all, aspect ratio (long / short side) <= 200 like Qwen2-VL's smart_resize (nvit::image_size_ok, same limits). An
// extreme strip such as 1 x 100000 would otherwise ask for several times max_tokens. false + why when refused.
bool vision_image_ok(int w, int h, std::string * why = nullptr);

// Prompt-cache identity of a multimodal prompt, one 64-bit key per KV row: a text row is its token id (< 2^32), an
// image row is bit 63 | a hash of (image key, row index). Two prompts can share cached KV rows exactly as far as these
// keys agree: a different image (or the same file under another token cap) never matches, the same image re-sent does.
// The keys also fix the rotary positions (the grid follows from the key), so a key-equal prefix is position-equal.
P100E_VISION_API std::vector<uint64_t> mm_row_keys(const std::vector<mm_piece> & pieces);

} // namespace p100e
