// vision_encoder: llama.cpp mtmd / clip (our fork's DLLs, delay-loaded) behind a small API. See vision.h.
#include "vision.h"
#ifdef P100E_HAVE_NVIT
#include "nvit.h"
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "llama.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <list>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#ifndef _WIN32
extern char ** environ;
#endif

#ifndef P100E_FORK_BIN
#define P100E_FORK_BIN ""
#endif

namespace p100e {

// ---------------------------------------------------------------------------------------------------------------------
// hash: two 64-bit lanes over 8-byte words (multiply-xorshift mixing), length and seed folded in. Fast (GB/s), not
// cryptographic; collisions only matter for the cache identity, 128 bits are plenty.
static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 31; x *= 0x7fb5d329728ea185ull;
    x ^= x >> 27; x *= 0x81dadef4bc2dd44dull;
    x ^= x >> 33;
    return x;
}

void vision_hash128(const void * data, size_t n, uint64_t seed, uint64_t out[2]) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    uint64_t a = 0x9e3779b97f4a7c15ull ^ seed, b = 0xc2b2ae3d27d4eb4full ^ (seed * 0x165667b19e3779f9ull);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        uint64_t w0, w1;
        memcpy(&w0, p + i, 8);
        memcpy(&w1, p + i + 8, 8);
        a = (a ^ mix64(w0 + 0x9e3779b97f4a7c15ull)) * 0xff51afd7ed558ccdull;
        b = (b ^ mix64(w1 + 0x632be59bd9b4e019ull)) * 0xc4ceb9fe1a85ec53ull;
        a = (a << 29) | (a >> 35);
        b = (b << 31) | (b >> 33);
    }
    uint64_t tail[2] = {0, 0};
    memcpy(tail, p + i, n - i);
    a = mix64(a ^ mix64(tail[0] ^ n));
    b = mix64(b ^ mix64(tail[1] ^ (n * 0x9e3779b97f4a7c15ull)));
    out[0] = mix64(a + b);
    out[1] = mix64(b ^ (a << 1));
}

std::vector<uint64_t> mm_row_keys(const std::vector<mm_piece> & pieces) {
    std::vector<uint64_t> k;
    for (const auto & p : pieces) {
        if (!p.image) {
            for (int32_t t : p.tokens) k.push_back((uint64_t) (uint32_t) t);
            continue;
        }
        const uint64_t base = mix64(p.image->key[0] ^ mix64(p.image->key[1]));
        for (int i = 0; i < p.image->n_tok(); ++i) k.push_back((1ull << 63) | mix64(base + (uint64_t) i));
    }
    return k;
}

// ---------------------------------------------------------------------------------------------------------------------
// host resize for per-image token caps below the encoder's max (detail "low"): Pillow-style separable bicubic
// (a = -0.5) with the support widened by the downscale factor (antialias). The result has sides that are multiples of
// 32 px inside mtmd's token limits, so mtmd's own smart_resize keeps it as is.
static double bicubic(double x) {
    const double a = -0.5;
    if (x < 0) x = -x;
    if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
    return 0;
}

struct resample_k { std::vector<int> x0, n; std::vector<float> w; int maxn = 0; };
static resample_k resample_coeffs(int in, int out) {
    resample_k k;
    const double scale = (double) in / out, fs = std::max(scale, 1.0), support = 2.0 * fs;
    k.maxn = (int) std::ceil(support) * 2 + 1;
    k.x0.resize(out); k.n.resize(out); k.w.assign((size_t) out * k.maxn, 0.0f);
    for (int o = 0; o < out; ++o) {
        const double c = (o + 0.5) * scale;
        int lo = std::max(0, (int) (c - support + 0.5)), hi = std::min(in, (int) (c + support + 0.5));
        double sum = 0;
        std::vector<double> w(hi - lo);
        for (int x = lo; x < hi; ++x) { w[x - lo] = bicubic((x - c + 0.5) / fs); sum += w[x - lo]; }
        k.x0[o] = lo; k.n[o] = std::min(hi - lo, k.maxn);
        for (int j = 0; j < k.n[o]; ++j) k.w[(size_t) o * k.maxn + j] = (float) (sum != 0 ? w[j] / sum : 0);
    }
    return k;
}

static std::vector<uint8_t> resize_rgb(const uint8_t * src, int sw, int sh, int dw, int dh) {
    const resample_k kx = resample_coeffs(sw, dw), ky = resample_coeffs(sh, dh);
    std::vector<float> tmp((size_t) sh * dw * 3);
    for (int y = 0; y < sh; ++y)
        for (int o = 0; o < dw; ++o) {
            float acc[3] = {0, 0, 0};
            const float * w = &kx.w[(size_t) o * kx.maxn];
            const uint8_t * s = src + ((size_t) y * sw + kx.x0[o]) * 3;
            for (int j = 0; j < kx.n[o]; ++j) { acc[0] += w[j] * s[3 * j]; acc[1] += w[j] * s[3 * j + 1]; acc[2] += w[j] * s[3 * j + 2]; }
            float * d = &tmp[((size_t) y * dw + o) * 3];
            d[0] = acc[0]; d[1] = acc[1]; d[2] = acc[2];
        }
    std::vector<uint8_t> dst((size_t) dh * dw * 3);
    for (int o = 0; o < dh; ++o) {
        const float * w = &ky.w[(size_t) o * ky.maxn];
        for (int x = 0; x < dw; ++x) {
            float acc[3] = {0, 0, 0};
            for (int j = 0; j < ky.n[o]; ++j) {
                const float * s = &tmp[((size_t) (ky.x0[o] + j) * dw + x) * 3];
                acc[0] += w[j] * s[0]; acc[1] += w[j] * s[1]; acc[2] += w[j] * s[2];
            }
            for (int c = 0; c < 3; ++c) dst[((size_t) o * dw + x) * 3 + c] = (uint8_t) std::min(255.0f, std::max(0.0f, std::round(acc[c])));
        }
    }
    return dst;
}

// mtmd's smart_resize (mtmd-image.cpp calc_size_preserved_ratio, align 32 = patch 16 x merge 2), plus nvit's bound
// (nvit::smart_size): when the short side is raised to 32 px the long side is floored into max_tok instead of keeping
// mtmd's overshoot. Returns true for those shapes (mtmd's own preprocessing would give more than max_tok tokens).
static bool smart_size(int w, int h, int min_tok, int max_tok, int & ow, int & oh) {
    const int f = 32;
    const int min_px = min_tok * f * f, max_px = max_tok * f * f;
    auto rnd = [&](float x) { return (int) std::round(x / (float) f) * f; };
    auto cei = [&](float x) { return (int) std::ceil(x / (float) f) * f; };
    auto flo = [&](float x) { return (int) std::floor(x / (float) f) * f; };
    int wb = std::max(f, rnd((float) w)), hb = std::max(f, rnd((float) h));
    bool bounded = false;
    if (max_px > 0 && hb * wb > max_px) {
        const float beta = std::sqrt((float) h * w / max_px);
        hb = std::max(f, flo(h / beta)); wb = std::max(f, flo(w / beta));
        if (hb * wb > max_px) {
            const int other = std::max(f, max_px / f / f * f);
            if (wb == f) hb = other;
            else wb = other;
            bounded = true;
        }
    } else if (min_px > 0 && hb * wb < min_px) {
        const float beta = std::sqrt((float) min_px / ((float) h * w));
        hb = cei(h * beta); wb = cei(w * beta);
    }
    ow = wb; oh = hb;
    return bounded;
}

// same limits as nvit::image_size_ok (vision.h)
bool vision_image_ok(int w, int h, std::string * why) {
    const char * r = nullptr;
    if (w <= 0 || h <= 0) r = "empty image";
    else if (w > 32768 || h > 32768) r = "a side is above 32768 px";
    else if ((int64_t) w * h > ((int64_t) 1 << 28)) r = "more than 2^28 pixels";
    else if ((int64_t) std::max(w, h) > (int64_t) 200 * std::min(w, h)) r = "aspect ratio above 200:1";
    if (r && why) *why = std::to_string(w) + "x" + std::to_string(h) + ": " + r;
    return !r;
}

bool vision_cap_size(int w, int h, int min_tok, int cap, int max_tok, int & tw, int & th) {
    tw = th = 0;
    if (w <= 0 || h <= 0 || cap >= max_tok) return false;
    int cw, ch, mw, mh;
    smart_size(w, h, min_tok, cap, cw, ch);
    // the cap's floor rounding can land below min_tok (cap == min): take the size mtmd would then upscale that to, so
    // the image is resampled once and the encoder's smart_resize keeps it
    smart_size(cw, ch, min_tok, max_tok, cw, ch);
    smart_size(w, h, min_tok, max_tok, mw, mh);
    if (cw == mw && ch == mh) return false;
    tw = cw;
    th = ch;
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
#ifdef _WIN32
static std::wstring widen(const std::string & s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), w.data(), n);
    return w;
}
static bool file_exists(const std::string & p) {
    const DWORD a = GetFileAttributesW(widen(p).c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}
static std::string exe_dir() {
    wchar_t buf[MAX_PATH * 2];
    const DWORD n = GetModuleFileNameW(nullptr, buf, (DWORD) (sizeof buf / sizeof buf[0]));
    std::wstring w(buf, n);
    const size_t k = w.find_last_of(L"\\/");
    if (k != std::wstring::npos) w.resize(k);
    const int m = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(m, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), s.data(), m, nullptr, nullptr);
    return s;
}
#else
static bool file_exists(const std::string & p) { FILE * f = fopen(p.c_str(), "rb"); if (f) fclose(f); return f != nullptr; }
static std::string exe_dir() { return "."; }
#endif

// The DLLs are delay-loaded (linker /DELAYLOAD): point the loader at their directory before the first call.
static std::string load_dlls(const std::string & want) {
    static std::mutex mu;
    static std::string loaded;
    std::lock_guard<std::mutex> lk(mu);
    if (!loaded.empty()) return loaded;
#ifndef _WIN32
    // Linux: the fork's shared libraries are ordinary link-time dependencies (RPATH to the fork build, see
    // cmake/vision.cmake), already loaded with this module; there is nothing to locate
    (void) want;
    loaded = "(linked)";
    return loaded;
#endif
    std::vector<std::string> cand;
    if (!want.empty()) cand.push_back(want);
    if (const char * e = getenv("P100E_VISION_DLLS")) cand.push_back(e);
    cand.push_back(exe_dir());
    if (*P100E_FORK_BIN) cand.push_back(P100E_FORK_BIN);
    for (const auto & d : cand) {
        if (d.empty() || !file_exists(d + "\\mtmd.dll")) continue;
        if (!want.empty() && d != want) fprintf(stderr, "vision: no mtmd.dll in %s, using %s\n", want.c_str(), d.c_str());
#ifdef _WIN32
        if (!SetDllDirectoryW(widen(d).c_str())) throw std::runtime_error("vision: SetDllDirectory failed for " + d);
        // load them now (explicitly, so a missing dependency is an error here and not an SEH exception later)
        for (const char * dll : {"ggml-base.dll", "ggml.dll", "llama.dll", "mtmd.dll"}) {
            if (!LoadLibraryW(widen(d + "\\" + dll).c_str()))
                throw std::runtime_error("vision: cannot load " + d + "\\" + dll + " (error " + std::to_string(GetLastError()) + ")");
        }
#endif
        loaded = d;
        return loaded;
    }
    std::string msg = "vision: mtmd.dll not found (set --vision-dlls or P100E_VISION_DLLS); looked in:";
    for (const auto & d : cand) msg += " [" + d + "]";
    throw std::runtime_error(msg);
}

// ---------------------------------------------------------------------------------------------------------------------
// environment (vision.h): the UCRT's _putenv_s updates both its own table (getenv in this exe and in the fork's DLLs,
// which share the UCRT) and the OS block (SetEnvironmentVariable); an empty value removes the variable
static void env_set(const std::string & name, const char * value) {
#ifdef _WIN32
    _putenv_s(name.c_str(), value ? value : "");
#else
    if (value) setenv(name.c_str(), value, 1);
    else unsetenv(name.c_str());
#endif
}

static std::string upper(std::string s) {
    for (char & c : s) c = (char) toupper((unsigned char) c);
    return s;
}

// (name, value) of every GGML_* variable in the process environment
static std::vector<std::pair<std::string, std::string>> env_ggml_vars() {
    std::vector<std::pair<std::string, std::string>> v;
    auto take = [&](const std::string & kv) {
        const size_t eq = kv.find('=');
        if (eq == 0 || eq == std::string::npos) return;   // "=C:=C:\\..." per-drive entries
        const std::string name = kv.substr(0, eq);
        if (upper(name).compare(0, 5, "GGML_") == 0) v.emplace_back(name, kv.substr(eq + 1));
    };
#ifdef _WIN32
    if (wchar_t * blk = GetEnvironmentStringsW()) {
        for (const wchar_t * e = blk; *e; e += wcslen(e) + 1) {
            const int n = WideCharToMultiByte(CP_UTF8, 0, e, -1, nullptr, 0, nullptr, nullptr);
            std::string s(n > 1 ? n - 1 : 0, ' ');
            if (n > 1) WideCharToMultiByte(CP_UTF8, 0, e, -1, s.data(), n, nullptr, nullptr);
            take(s);
        }
        FreeEnvironmentStringsW(blk);
    }
#else
    for (char ** e = environ; e && *e; ++e) take(*e);
#endif
    return v;
}

std::vector<std::string> vision_pin_ggml_env(bool fold_gemm) {
    const char * mode = getenv("P100E_VISION_GGML_ENV");
    const bool keep = mode && std::string(mode) == "keep";
    std::vector<std::string> log;
    for (const auto & kv : env_ggml_vars()) {
        const std::string up = upper(kv.first);
        if (keep && up != "GGML_CUDA_P2P") continue;
        if (!keep && up == "GGML_CUDA_GEMM_FOLD") continue;   // set below
        env_set(kv.first, nullptr);
        log.push_back("removed " + kv.first + "=" + kv.second);
    }
    if (getenv("GGML_CUDA_P2P")) {   // in the CRT table only (set by this process outside the OS block)
        env_set("GGML_CUDA_P2P", nullptr);
        log.push_back("removed GGML_CUDA_P2P");
    }
    const char * fold = fold_gemm ? "2" : "0";
    const char * cur = getenv("GGML_CUDA_GEMM_FOLD");
    if (keep ? !cur : !(cur && std::string(cur) == fold)) {
        log.push_back(std::string("set GGML_CUDA_GEMM_FOLD=") + fold + (cur ? std::string(" (was ") + cur + ")" : std::string()));
        env_set("GGML_CUDA_GEMM_FOLD", fold);
    }
    return log;
}

// clip.vision.projection_dim of an mmproj (0 when missing); throws for a deepstack projector (vision.h)
static int mmproj_width_meta(const std::string & path) {
    gguf_init_params gp{true, nullptr};
    std::unique_ptr<gguf_context, void (*)(gguf_context *)> g(gguf_init_from_file(path.c_str(), gp), gguf_free);
    if (!g) throw std::runtime_error("vision: cannot read the GGUF metadata of " + path);
    const int64_t kd = gguf_find_key(g.get(), "clip.vision.is_deepstack_layers");
    if (kd >= 0 && gguf_get_kv_type(g.get(), kd) == GGUF_TYPE_ARRAY && gguf_get_arr_type(g.get(), kd) == GGUF_TYPE_BOOL) {
        const int8_t * b = static_cast<const int8_t *>(gguf_get_arr_data(g.get(), kd));
        for (size_t i = 0; i < gguf_get_arr_n(g.get(), kd); ++i)
            if (b[i])
                throw std::runtime_error("vision: " + path + " is a deepstack projector (clip.vision.is_deepstack_layers[" +
                                         std::to_string(i) + "] = true): its output is wider than the LLM's embedding "
                                         "and feeds early layers, which p100e does not support");
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(g.get()); ++i) {
        const char * n = gguf_get_tensor_name(g.get(), i);
        if (strncmp(n, "v.deepstack.", 12) == 0)
            throw std::runtime_error("vision: " + path + " has deepstack tensors (" + n + "): not supported by p100e");
    }
    int w = 0;
    const int64_t k = gguf_find_key(g.get(), "clip.vision.projection_dim");
    if (k >= 0) {
        const gguf_type t = gguf_get_kv_type(g.get(), k);
        if (t == GGUF_TYPE_INT32) w = gguf_get_val_i32(g.get(), k);
        else if (t == GGUF_TYPE_UINT32) w = (int) gguf_get_val_u32(g.get(), k);
    }
    return w;
}

std::string vision_load_dlls(const std::string & dll_dir) { return load_dlls(dll_dir); }

int vision_mmproj_width(const std::string & mmproj, const std::string & dll_dir) {
    load_dlls(dll_dir);
    const int w = mmproj_width_meta(mmproj);
    if (w <= 0) throw std::runtime_error("vision: no clip.vision.projection_dim in " + mmproj);
    return w;
}

// ---------------------------------------------------------------------------------------------------------------------
struct vision_encoder::impl {
    vision_opts o;
    std::string dlls;
    llama_model * model = nullptr;
    const llama_vocab * vocab = nullptr;
    mtmd_context * ctx = nullptr;
#ifdef P100E_HAVE_NVIT
    std::unique_ptr<nvit::encoder> nv;   // p100e-native encoder (vision_opts::native); mtmd ctx is null then
#endif
    int n_embd = 0;
    std::vector<int32_t> beg, end;   // tokens(<|vision_start|>), tokens(<|vision_end|>)
    // enc_mu: the mtmd context (decode, preprocess, encode: one image at a time); cache_mu: the LRU, its index and the
    // stats, held only for lookups / inserts and never across an encode (taken inside enc_mu, never the other way round)
    std::mutex enc_mu;
    mutable std::mutex cache_mu;

    // log routing: llama.cpp / ggml / mtmd messages arrive in pieces (GGML_LOG_LEVEL_CONT continues the last level)
    std::mutex log_mu;
    std::string line;
    int line_level = GGML_LOG_LEVEL_INFO;
    std::vector<std::string> warns;   // the first max_warns lines (load and warmup; every bad image would add more)
    static constexpr size_t max_warns = 256;
    bool fallback = false;

    // LRU embedding cache
    struct entry { uint64_t k0, k1; std::shared_ptr<const image_embd> e; };
    std::list<entry> lru;   // front = most recent
    struct kh { size_t operator()(const std::pair<uint64_t, uint64_t> & k) const { return (size_t) (k.first ^ (k.second * 31)); } };
    std::unordered_map<std::pair<uint64_t, uint64_t>, std::list<entry>::iterator, kh> idx;
    size_t bytes = 0;
    cache_stats st;
    bool logs_set = false;

    ~impl() {
        if (ctx) mtmd_free(ctx);
        if (model) llama_model_free(model);
        if (logs_set) {   // the callbacks point at this object
            llama_log_set(nullptr, nullptr);
            mtmd_helper_log_set(nullptr, nullptr);
        }
    }

    void log_line(int level, const std::string & s) {
        const bool warn = level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR;
        if (warn && warns.size() < max_warns) warns.push_back(s);
        if (s.find("unsupported operators") != std::string::npos || s.find("falling back to CPU") != std::string::npos ||
            s.find("flash attention not supported") != std::string::npos)
            fallback = true;
        // "graph splits = N": more than one split means part of the graph runs on another backend (the CPU)
        const size_t gs = s.find("graph splits = ");
        if (gs != std::string::npos && atoi(s.c_str() + gs + 15) > 1 && o.device >= 0) {
            fallback = true;
            if (warns.size() < max_warns) warns.push_back(s);
        }
        if (warn || o.verbose || gs != std::string::npos || s.find("compute buffer size") != std::string::npos ||
            s.find("flash attention is") != std::string::npos)
            fprintf(stderr, "vision| %s\n", s.c_str());
    }
    static void log_cb(enum ggml_log_level level, const char * text, void * ud) {
        impl * self = static_cast<impl *>(ud);
        std::lock_guard<std::mutex> lk(self->log_mu);
        if (level != GGML_LOG_LEVEL_CONT) self->line_level = level;
        for (const char * c = text; *c; ++c) {
            if (*c == '\n') { self->log_line(self->line_level, self->line); self->line.clear(); }
            else self->line += *c;
        }
    }

    // cache_get / cache_put: with cache_mu held
    std::shared_ptr<const image_embd> cache_get(uint64_t k0, uint64_t k1) {
        auto it = idx.find({k0, k1});
        if (it == idx.end()) return nullptr;
        lru.splice(lru.begin(), lru, it->second);
        return it->second->e;
    }
    void cache_put(const std::shared_ptr<const image_embd> & e) {
        const size_t cap = o.cache_mb << 20;
        if (cap == 0 || e->bytes() > cap) return;
        if (idx.count({e->key[0], e->key[1]})) return;
        while (!lru.empty() && bytes + e->bytes() > cap) {
            bytes -= lru.back().e->bytes();
            idx.erase({lru.back().k0, lru.back().k1});
            lru.pop_back();
            ++st.evictions;
        }
        lru.push_front({e->key[0], e->key[1], e});
        idx[{e->key[0], e->key[1]}] = lru.begin();
        bytes += e->bytes();
    }

    std::vector<int32_t> tok(const std::string & text, bool parse_special) const {
        std::vector<int32_t> r(text.size() + 8);
        int n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), r.data(), (int32_t) r.size(), false, parse_special);
        if (n < 0) {
            r.resize(-n);
            n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), r.data(), (int32_t) r.size(), false, parse_special);
        }
        if (n < 0) throw std::runtime_error("vision: tokenization failed");
        r.resize(n);
        return r;
    }

#ifdef P100E_HAVE_NVIT
    // native mode's fallback decoder: mtmd's helper (mtmd_helper_bitmap_init_from_buf, the mtmd path's decoder) for
    // formats stb_image does not read. Only WebP gets here: for anything else the helper has nothing beyond stb_image
    // for images, and with no mtmd context (native mode) its audio branch must not be reached.
    bool decode_mtmd(const std::string & bytes, nvit::rgb_image & rgb, std::string & err) const {
        const bool webp = bytes.size() >= 12 && !memcmp(bytes.data(), "RIFF", 4) && !memcmp(bytes.data() + 8, "WEBP", 4);
        if (!webp) { err = "not a WebP file either"; return false; }
        mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_buf(ctx, (const unsigned char *) bytes.data(),
                                                                          bytes.size(), false, mtmd_helper_init_opt_default());
        if (bw.video_ctx) mtmd_helper_video_free(bw.video_ctx);
        if (!bw.bitmap) { err = "WebP decode failed (it needs ffmpeg and ffprobe on the PATH)"; return false; }
        std::unique_ptr<mtmd_bitmap, void (*)(mtmd_bitmap *)> bmp(bw.bitmap, mtmd_bitmap_free);
        const int nx = (int) mtmd_bitmap_get_nx(bmp.get()), ny = (int) mtmd_bitmap_get_ny(bmp.get());
        const unsigned char * d = mtmd_bitmap_get_data(bmp.get());
        if (mtmd_bitmap_is_audio(bmp.get()) || !d || nx <= 0 || ny <= 0 ||
            mtmd_bitmap_get_n_bytes(bmp.get()) != (size_t) nx * ny * 3) {
            err = "mtmd returned no RGB image";
            return false;
        }
        rgb.w = nx;
        rgb.h = ny;
        rgb.px.assign(d, d + (size_t) nx * ny * 3);
        return true;
    }
#endif

    // the mtmd path: images mtmd cannot take. Beyond the limits (vision_image_ok), or a shape whose size mtmd's own
    // smart_resize would take past max_tokens (smart_size bounds it, mtmd does not): w x h as given to mtmd
    void check_mtmd_size(int w, int h) const {
        std::string why;
        if (!vision_image_ok(w, h, &why)) throw std::runtime_error("vision: image refused (" + why + ")");
        int cw, ch;
        if (smart_size(w, h, o.min_tokens, o.max_tokens, cw, ch))
            throw std::runtime_error("vision: image refused (" + std::to_string(w) + "x" + std::to_string(h) +
                                     ": too elongated for " + std::to_string(o.max_tokens) + " tokens)");
    }

    // locking: the hash needs none; the cache lookup takes cache_mu alone, so a cached image never waits for an encode;
    // a miss takes enc_mu and looks again (another thread may have just encoded the same image), encodes, and inserts
    // under cache_mu. Native mode decodes, resizes and preprocesses before enc_mu (CPU work that overlaps another
    // request's GPU encode; the result is the same), and takes it only around the second lookup and the encode.
    std::shared_ptr<const image_embd> encode(const image_input & in, vision_timing & t) {
        using clk = std::chrono::steady_clock;
        auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        const int cap = in.max_tokens > 0 ? std::min(in.max_tokens, o.max_tokens) : o.max_tokens;
        auto t0 = clk::now();
        uint64_t key[2];
        // settings in the seed: the same bytes with another cap or token minimum are another entry
        const uint64_t seed = ((uint64_t) (uint32_t) cap << 32) ^ (uint64_t) (uint32_t) o.min_tokens ^ 0x5649534e00000000ull;
        vision_hash128(in.bytes.data(), in.bytes.size(), seed, key);
        auto t1 = clk::now();
        t.hash_ms = ms(t0, t1);
        {
            std::lock_guard<std::mutex> lk(cache_mu);
            if (auto e = cache_get(key[0], key[1])) { t.cached = true; ++st.hits; return e; }
        }
#ifdef P100E_HAVE_NVIT
        if (nv) {   // native: decode, the mtmd path's host resample for a cap, mtmd-exact preprocessing, encode on the GPUs
            auto e = std::make_shared<image_embd>();
            e->key[0] = key[0]; e->key[1] = key[1];
            nvit::rgb_image rgb;
            std::string err;
            if (!nvit::decode_image(in.bytes.data(), in.bytes.size(), rgb, &err)) {
                // what stb_image cannot read, the mtmd path may (WebP through ffmpeg: fork built with MTMD_VIDEO, ffmpeg
                // on the PATH): decode it with mtmd's helper, then preprocess and encode natively as usual
                std::string err2;
                if (!decode_mtmd(in.bytes, rgb, err2))
                    throw std::runtime_error("vision: cannot decode the image (" + std::to_string(in.bytes.size()) +
                                             " bytes): " + err + "; mtmd's decoder: " + err2);
                fprintf(stderr, "vision: native decoder (stb_image) cannot read this image (%s); decoded through mtmd's "
                                "helper instead: %dx%d\n", err.c_str(), rgb.w, rgb.h);
            }
            {
                std::string why;   // decode_image checks its header; the WebP path's size is only known here
                if (!vision_image_ok(rgb.w, rgb.h, &why)) throw std::runtime_error("vision: image refused (" + why + ")");
            }
            auto t2 = clk::now();
            t.decode_ms = ms(t1, t2);
            e->src_w = rgb.w;
            e->src_h = rgb.h;
            int cw, ch;
            if (vision_cap_size(rgb.w, rgb.h, o.min_tokens, cap, o.max_tokens, cw, ch)) {   // same pixels as the mtmd path
                rgb.px = resize_rgb(rgb.px.data(), rgb.w, rgb.h, cw, ch);
                rgb.w = cw;
                rgb.h = ch;
            }
            auto t2r = clk::now();
            t.resize_ms = ms(t2, t2r);
            const nvit::image im = nvit::preprocess(rgb, o.min_tokens, o.max_tokens);   // = mtmd's preprocessing
            rgb = nvit::rgb_image();   // the decoded pixels are not needed while waiting for the encoder
            auto t3 = clk::now();
            t.preprocess_ms = ms(t2r, t3);
            std::lock_guard<std::mutex> elk(enc_mu);
            auto t4 = clk::now();
            t.wait_ms = ms(t3, t4);
            {
                std::lock_guard<std::mutex> lk(cache_mu);
                if (auto c = cache_get(key[0], key[1])) { t.cached = true; ++st.hits; return c; }
                ++st.misses;
            }
            nv->encode(im, e->data, e->nx, e->ny);
            e->n_embd = n_embd;
            t.encode_ms = ms(t4, clk::now());
            std::shared_ptr<const image_embd> ce = e;
            std::lock_guard<std::mutex> lk(cache_mu);
            cache_put(ce);
            return ce;
        }
#endif

        std::lock_guard<std::mutex> elk(enc_mu);
        t.wait_ms = ms(t1, clk::now());
        {
            std::lock_guard<std::mutex> lk(cache_mu);
            if (auto e = cache_get(key[0], key[1])) { t.cached = true; ++st.hits; return e; }
            ++st.misses;
        }
        t1 = clk::now();   // decode / resize / preprocess / encode times below: after the wait

        mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_buf(ctx, (const unsigned char *) in.bytes.data(),
                                                                          in.bytes.size(), false, mtmd_helper_init_opt_default());
        if (!bw.bitmap) throw std::runtime_error("vision: cannot decode the image (" + std::to_string(in.bytes.size()) + " bytes)");
        std::unique_ptr<mtmd_bitmap, void (*)(mtmd_bitmap *)> bmp(bw.bitmap, mtmd_bitmap_free);
        auto t2 = clk::now();
        t.decode_ms = ms(t1, t2);
        auto e = std::make_shared<image_embd>();
        e->src_w = (int) mtmd_bitmap_get_nx(bmp.get());
        e->src_h = (int) mtmd_bitmap_get_ny(bmp.get());
        e->key[0] = key[0]; e->key[1] = key[1];
        {
            std::string why;   // the limits first (the size math below assumes them)
            if (!vision_image_ok(e->src_w, e->src_h, &why)) throw std::runtime_error("vision: image refused (" + why + ")");
        }
        int cw, ch;   // per-image cap below the encoder's: resize on the host to the cap's smart size (vision_cap_size)
        if (vision_cap_size(e->src_w, e->src_h, o.min_tokens, cap, o.max_tokens, cw, ch)) {
            check_mtmd_size(cw, ch);   // the resampled image, as mtmd will size it
            std::vector<uint8_t> rs = resize_rgb(mtmd_bitmap_get_data(bmp.get()), e->src_w, e->src_h, cw, ch);
            bmp.reset(mtmd_bitmap_init((uint32_t) cw, (uint32_t) ch, rs.data()));
            if (!bmp) throw std::runtime_error("vision: bitmap init failed");
        } else {
            check_mtmd_size(e->src_w, e->src_h);   // mtmd sizes the image itself
        }
        char id[40];
        snprintf(id, sizeof id, "%016llx%016llx", (unsigned long long) key[0], (unsigned long long) key[1]);
        mtmd_bitmap_set_id(bmp.get(), id);
        auto t3 = clk::now();
        t.resize_ms = ms(t2, t3);

        // preprocess (mtmd_tokenize of the marker alone: [<|vision_start|>] [image] [<|vision_end|>])
        std::unique_ptr<mtmd_input_chunks, void (*)(mtmd_input_chunks *)> chunks(mtmd_input_chunks_init(), mtmd_input_chunks_free);
        const std::string m = mtmd_default_marker();
        mtmd_input_text txt{m.c_str(), m.size(), false, true};
        const mtmd_bitmap * bp = bmp.get();
        const int32_t rc = mtmd_tokenize(ctx, chunks.get(), &txt, &bp, 1);
        if (rc != 0) throw std::runtime_error("vision: mtmd_tokenize failed (" + std::to_string(rc) + ")");
        const mtmd_input_chunk * img = nullptr;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
            const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks.get(), i);
            if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) img = c;
        }
        if (!img) throw std::runtime_error("vision: no image chunk");
        auto t4 = clk::now();
        t.preprocess_ms = ms(t3, t4);

        const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(img);
        const size_t n = mtmd_image_tokens_get_n_tokens(it);
        const mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(it, 0, n - 1);
        e->nx = (int) last.x + 1;
        e->ny = (int) last.y + 1;
        if ((size_t) e->nx * e->ny != n) throw std::runtime_error("vision: unexpected token grid (not M-RoPE?)");
        if (mtmd_encode_chunk(ctx, img) != 0) throw std::runtime_error("vision: mtmd_encode_chunk failed");
        const float * out = mtmd_get_output_embd(ctx);
        e->n_embd = n_embd;
        e->data.assign(out, out + n * (size_t) n_embd);
        t.encode_ms = ms(t4, std::chrono::steady_clock::now());
        std::shared_ptr<const image_embd> ce = e;
        std::lock_guard<std::mutex> lk(cache_mu);
        cache_put(ce);
        return ce;
    }
};

vision_encoder::vision_encoder(const vision_opts & o) : p_(new impl) {
    impl & I = *p_;
    I.o = o;
    bool use_native = false;
#ifdef P100E_HAVE_NVIT
    {
        const char * env = getenv("P100E_NVIT");
        use_native = o.native == 1 || (o.native < 0 && !(env && atoi(env) == 0));
    }
#endif
    // before the DLLs load (ggml-cuda reads GGML_CUDA_P2P at its initialization inside this constructor, the others at
    // their first use): the validated environment, whatever launched the process (vision.h)
    for (const auto & l : vision_pin_ggml_env(o.fold_gemm)) fprintf(stderr, "vision: env %s\n", l.c_str());
    I.dlls = load_dlls(o.dll_dir);
    // the embedding width, and no deepstack projector (its output rows would be wider than projection_dim): checked
    // before mtmd loads the weights
    I.n_embd = mmproj_width_meta(o.mmproj);
    llama_log_set(impl::log_cb, &I);
    mtmd_helper_log_set(impl::log_cb, &I);
    I.logs_set = true;
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    mp.n_gpu_layers = 0;
    I.model = llama_model_load_from_file(o.text_model.c_str(), mp);
    if (!I.model) throw std::runtime_error("vision: cannot load the vocabulary of " + o.text_model);
    I.vocab = llama_model_get_vocab(I.model);

#ifdef P100E_HAVE_NVIT
    if (use_native) {
        nvit::opts no;
        no.mmproj = o.mmproj;
        no.max_tokens = o.max_tokens;
        no.verbose = o.verbose;
        no.devices = o.native_devices;
        if (no.devices.empty()) {
            const char * dv = getenv("P100E_NVIT_DEVICES");
            const std::string ds = dv && *dv ? dv : "0,1";
            size_t a = 0;
            while (a < ds.size()) {
                size_t b = ds.find(',', a);
                if (b == std::string::npos) b = ds.size();
                if (b > a) no.devices.push_back(atoi(ds.substr(a, b - a).c_str()));
                a = b + 1;
            }
        }
        try {
            I.nv = std::make_unique<nvit::encoder>(no);
        } catch (const std::exception & ex) {
            if (o.native == 1) throw;
            // automatic choice: fall back to mtmd (e.g. no P2P between the GPUs, mmproj of another model)
            fprintf(stderr, "vision: native encoder unavailable (%s), using mtmd\n", ex.what());
            I.nv.reset();
        }
        if (I.nv) {
            I.n_embd = I.nv->n_embd();
            // the wrapper mtmd puts around an image: <|vision_start|> ... <|vision_end|> (parse_special)
            I.beg = I.tok("<|vision_start|>", true);
            I.end = I.tok("<|vision_end|>", true);
            return;
        }
        // mtmd from here on: GGML_CUDA_GEMM_FOLD was pinned with the rest of the GGML_* environment above
    }
#endif
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = o.device >= 0;
    cp.device = nullptr;
    if (o.device >= 0) {
        const std::string dn = "CUDA" + std::to_string(o.device);
        cp.device = ggml_backend_dev_by_name(dn.c_str());
        if (!cp.device) throw std::runtime_error("vision: ggml backend device " + dn + " not found");
    }
    cp.print_timings = o.verbose;
    cp.n_threads = o.n_threads;
    cp.flash_attn_type = o.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.warmup = o.warmup;
    cp.image_min_tokens = o.min_tokens;
    cp.image_max_tokens = o.max_tokens;
    cp.media_marker = mtmd_default_marker();
    I.ctx = mtmd_init_from_file(o.mmproj.c_str(), I.model, cp);
    if (!I.ctx) throw std::runtime_error("vision: cannot load the mmproj " + o.mmproj);
    if (!mtmd_support_vision(I.ctx)) throw std::runtime_error("vision: the mmproj has no vision encoder");
    if (!mtmd_decode_use_mrope(I.ctx)) throw std::runtime_error("vision: the model does not use M-RoPE (unsupported)");
    // embedding width: the projector's output (read above; a vocab-only load leaves the text model's hparams at 0)
    if (I.n_embd <= 0) I.n_embd = llama_model_n_embd(I.model);
    if (I.n_embd <= 0) throw std::runtime_error("vision: cannot read clip.vision.projection_dim from " + o.mmproj);

    // <|vision_start|> / <|vision_end|> tokens: tokenize the marker with a placeholder bitmap (no pixels, no encode)
    std::unique_ptr<mtmd_bitmap, void (*)(mtmd_bitmap *)> ph(mtmd_bitmap_init(64, 64, nullptr), mtmd_bitmap_free);
    std::unique_ptr<mtmd_input_chunks, void (*)(mtmd_input_chunks *)> chunks(mtmd_input_chunks_init(), mtmd_input_chunks_free);
    const std::string m = mtmd_default_marker();
    mtmd_input_text txt{m.c_str(), m.size(), false, true};
    const mtmd_bitmap * bp = ph.get();
    if (mtmd_tokenize(I.ctx, chunks.get(), &txt, &bp, 1) != 0) throw std::runtime_error("vision: placeholder tokenize failed");
    bool seen_img = false;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks.get(), i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) { seen_img = true; continue; }
        size_t n = 0;
        const llama_token * t = mtmd_input_chunk_get_tokens_text(c, &n);
        (seen_img ? I.end : I.beg).insert((seen_img ? I.end : I.beg).end(), t, t + n);
    }
}

vision_encoder::~vision_encoder() = default;   // impl::~impl frees (also when the constructor threw)

// prompt -> the text between the media markers (one more part than markers); checks the image count
static std::vector<std::string> split_marker(const std::string & prompt, const std::string & marker, size_t n_images) {
    const std::string m = marker.empty() ? std::string(mtmd_default_marker()) : marker;
    std::vector<std::string> parts;
    size_t s = 0;
    for (;;) {
        const size_t k = prompt.find(m, s);
        if (k == std::string::npos) { parts.push_back(prompt.substr(s)); break; }
        parts.push_back(prompt.substr(s, k - s));
        s = k + m.size();
    }
    if (parts.size() - 1 != n_images)
        throw std::runtime_error("vision: " + std::to_string(parts.size() - 1) + " media markers but " +
                                 std::to_string(n_images) + " images");
    return parts;
}

int vision_encoder::count_tokens(const std::string & prompt, const std::vector<image_input> & images,
                                 const std::string & marker) const {
    impl & I = *p_;
    const std::vector<std::string> parts = split_marker(prompt, marker, images.size());
    size_t n = 0;
    for (const auto & t : parts) n += I.tok(t, true).size();
    for (const auto & img : images) {
        n += I.beg.size() + I.end.size();
        // same cap and cache key as encode()
        const int cap = img.max_tokens > 0 ? std::min(img.max_tokens, I.o.max_tokens) : I.o.max_tokens;
        const uint64_t seed = ((uint64_t) (uint32_t) cap << 32) ^ (uint64_t) (uint32_t) I.o.min_tokens ^ 0x5649534e00000000ull;
        uint64_t key[2];
        vision_hash128(img.bytes.data(), img.bytes.size(), seed, key);
        {
            std::lock_guard<std::mutex> lk(I.cache_mu);
            auto it = I.idx.find({key[0], key[1]});
            if (it != I.idx.end()) { n += (size_t) it->second->e->n_tok(); continue; }
        }
        int w, h;
#ifdef P100E_HAVE_NVIT
        if (!I.ctx) {
            // native mode has no mtmd context, and mtmd's helper reads it for anything that looks like audio (ID3 /
            // RIFF-WAVE / MP3 sync bytes): a null dereference. Size it with encode()'s own decoders instead.
            nvit::rgb_image rgb;
            std::string err, err2;
            if (!nvit::decode_image(img.bytes.data(), img.bytes.size(), rgb, &err) && !I.decode_mtmd(img.bytes, rgb, err2))
                throw std::runtime_error("vision: cannot decode the image (" + std::to_string(img.bytes.size()) + " bytes): " +
                                         err + "; mtmd's decoder: " + err2);
            w = rgb.w;
            h = rgb.h;
        } else
#endif
        {
            // placeholder decode: the size only (no hash, no pixels kept); the context is only read for audio files
            mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_buf(I.ctx, (const unsigned char *) img.bytes.data(),
                                                                              img.bytes.size(), true, mtmd_helper_init_opt_default());
            if (!bw.bitmap) throw std::runtime_error("vision: cannot decode the image (" + std::to_string(img.bytes.size()) + " bytes)");
            w = (int) mtmd_bitmap_get_nx(bw.bitmap);
            h = (int) mtmd_bitmap_get_ny(bw.bitmap);
            mtmd_bitmap_free(bw.bitmap);
        }
        // the images encode() refuses (size limits; on the mtmd path also the shapes mtmd would size past max_tokens)
        {
            std::string why;
            if (!vision_image_ok(w, h, &why)) throw std::runtime_error("vision: image refused (" + why + ")");
            int tw, th;
            if (I.ctx) {
                if (vision_cap_size(w, h, I.o.min_tokens, cap, I.o.max_tokens, tw, th)) I.check_mtmd_size(tw, th);
                else I.check_mtmd_size(w, h);
            }
        }
        // encode(): with a cap below the encoder's max the image is resized to the cap's size (moved back up to
        // min_tokens), which mtmd then keeps; otherwise mtmd's own smart resize under max_tokens
        int cw, ch;
        if (cap < I.o.max_tokens) {
            smart_size(w, h, I.o.min_tokens, cap, cw, ch);
            smart_size(cw, ch, I.o.min_tokens, I.o.max_tokens, cw, ch);
        } else {
            smart_size(w, h, I.o.min_tokens, I.o.max_tokens, cw, ch);
        }
        n += (size_t) (cw / 32) * (size_t) (ch / 32);
    }
    return (int) n;
}

std::vector<mm_piece> vision_encoder::build(const std::string & prompt, const std::vector<image_input> & images,
                                            const std::string & marker) {
    impl & I = *p_;   // no lock here: tokenization only reads the vocabulary; I.encode locks per image
    const std::vector<std::string> parts = split_marker(prompt, marker, images.size());
    std::vector<mm_piece> out;
    auto add_text = [&](const std::vector<int32_t> & t) {
        if (t.empty()) return;
        if (out.empty() || out.back().image) out.emplace_back();
        out.back().tokens.insert(out.back().tokens.end(), t.begin(), t.end());
    };
    for (size_t i = 0; i < parts.size(); ++i) {
        add_text(I.tok(parts[i], true));
        if (i + 1 == parts.size()) break;
        add_text(I.beg);
        mm_piece p;
        p.image = I.encode(images[i], p.t);
        out.push_back(std::move(p));
        add_text(I.end);
    }
    return out;
}

std::shared_ptr<const image_embd> vision_encoder::encode(const image_input & img, vision_timing * t) {
    vision_timing tt;
    auto e = p_->encode(img, tt);
    if (t) *t = tt;
    return e;
}

const char * vision_encoder::marker() { return "<__media__>"; }
int vision_encoder::n_embd() const { return p_->n_embd; }
int vision_encoder::min_tokens() const { return p_->o.min_tokens; }
int vision_encoder::max_tokens() const { return p_->o.max_tokens; }
int vision_encoder::detail_max_tokens(const std::string & d) const {
    if (d == "low") return std::min(p_->o.min_tokens, p_->o.max_tokens);
    return p_->o.max_tokens;
}
std::vector<int32_t> vision_encoder::tokenize(const std::string & text, bool parse_special) const {
    return p_->tok(text, parse_special);
}
std::string vision_encoder::piece(int32_t id) const {
    char buf[256];
    int n = llama_token_to_piece(p_->vocab, id, buf, sizeof buf, 0, true);
    if (n < 0) {
        std::string s(-n, '\0');
        n = llama_token_to_piece(p_->vocab, id, s.data(), (int32_t) s.size(), 0, true);
        return s.substr(0, std::max(n, 0));
    }
    return std::string(buf, n);
}
int32_t vision_encoder::eos() const { return llama_vocab_eos(p_->vocab); }
bool vision_encoder::is_eog(int32_t id) const { return llama_vocab_is_eog(p_->vocab, id); }
vision_encoder::cache_stats vision_encoder::stats() const {
    std::lock_guard<std::mutex> lk(p_->cache_mu);
    cache_stats s = p_->st;
    s.entries = p_->lru.size();
    s.bytes = p_->bytes;
    return s;
}
void vision_encoder::cache_clear() {
    std::lock_guard<std::mutex> lk(p_->cache_mu);
    p_->lru.clear(); p_->idx.clear(); p_->bytes = 0;
}
bool vision_encoder::cpu_fallback() const { return p_->fallback; }
std::vector<std::string> vision_encoder::warnings() const {
    std::lock_guard<std::mutex> lk(p_->log_mu);
    return p_->warns;
}
std::string vision_encoder::dll_dir() const { return p_->dlls; }
void vision_encoder::set_scratch_source(std::function<std::vector<scratch_region>()> f) {
#ifdef P100E_HAVE_NVIT
    if (!p_->nv) return;
    if (!f) { p_->nv->scratch_source = nullptr; return; }
    p_->nv->scratch_source = [f]() {
        std::vector<nvit::scratch_view> v;
        for (const auto & t : f()) {
            nvit::scratch_view sv;
            sv.dev = t.dev;
            sv.p = t.p;
            sv.bytes = t.bytes;
            sv.hold = t.hold;
            v.push_back(sv);
        }
        return v;
    };
#else
    (void) f;
#endif
}

bool vision_encoder::native() const {
#ifdef P100E_HAVE_NVIT
    return (bool) p_->nv;
#else
    return false;
#endif
}

} // namespace p100e
