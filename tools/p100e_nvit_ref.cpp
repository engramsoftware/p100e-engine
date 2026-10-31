// nvit_ref: the fork's mtmd encoder in-process (delay-loaded DLLs) with a ggml eval callback that captures the graph
// input (inp_raw) and named intermediate tensors. Includes only the fork's headers (see p100e_nvit_ref.h).
#include "p100e_nvit_ref.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <stdexcept>

#ifndef P100E_FORK_BIN
#define P100E_FORK_BIN ""
#endif

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
    char buf[MAX_PATH * 2];
    const DWORD n = GetModuleFileNameA(nullptr, buf, (DWORD) sizeof buf);
    std::string s(buf, n);
    const size_t k = s.find_last_of("\\/");
    return k == std::string::npos ? "." : s.substr(0, k);
}

// the DLLs are delay-loaded: point the loader at their directory before the first call
static std::string load_dlls(const std::string & want) {
    static std::string loaded;
    if (!loaded.empty()) return loaded;
    std::vector<std::string> cand;
    if (!want.empty()) cand.push_back(want);
    if (const char * e = getenv("P100E_VISION_DLLS")) cand.push_back(e);
    cand.push_back(exe_dir());
    if (*P100E_FORK_BIN) cand.push_back(P100E_FORK_BIN);
    for (const auto & d : cand) {
        if (d.empty() || !file_exists(d + "\\mtmd.dll")) continue;
        if (!SetDllDirectoryW(widen(d).c_str())) throw std::runtime_error("nvit_ref: SetDllDirectory failed for " + d);
        for (const char * dll : {"ggml-base.dll", "ggml.dll", "llama.dll", "mtmd.dll"})
            if (!LoadLibraryW(widen(d + "\\" + dll).c_str()))
                throw std::runtime_error("nvit_ref: cannot load " + d + "\\" + dll + " (error " + std::to_string(GetLastError()) + ")");
        loaded = d;
        return loaded;
    }
    throw std::runtime_error("nvit_ref: mtmd.dll not found (set P100E_VISION_DLLS)");
}

struct nvit_ref::impl {
    std::string dlls;
    llama_model * model = nullptr;
    mtmd_context * ctx = nullptr;
    bool verbose = false;
    // capture state (eval callback)
    bool capturing = false;
    std::set<std::string> names;
    nvit_ref_result * res = nullptr;
    std::string line;

    static void log_cb(enum ggml_log_level level, const char * text, void * ud) {
        impl * self = (impl *) ud;
        if (self->verbose || level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
    }
    static bool src_is_inp_raw(const ggml_tensor * t, int & k) {
        for (int i = 0; i < GGML_MAX_SRC; ++i)
            if (t->src[i] && strstr(t->src[i]->name, "inp_raw") != nullptr && t->src[i]->type == GGML_TYPE_F32 &&
                ggml_nelements(t->src[i]) % 3 == 0) { k = i; return true; }
        return false;
    }
    static bool eval_cb(ggml_tensor * t, bool ask, void * ud) {
        impl * self = (impl *) ud;
        if (!self->capturing) return false;
        static int dbg_n = getenv("NVIT_REF_DEBUG") ? 12 : 0;
        if (ask && dbg_n > 0) {
            --dbg_n;
            fprintf(stderr, "nvit_ref node %s op %s src:", t->name, ggml_op_desc(t));
            for (int i = 0; i < GGML_MAX_SRC; ++i) if (t->src[i]) fprintf(stderr, " [%s]", t->src[i]->name);
            fprintf(stderr, "\n");
        }
        int k = -1;
        const bool raw = self->res->inp_raw.empty() && src_is_inp_raw(t, k);
        const bool named = self->names.count(t->name) > 0;
        if (ask) return raw || named;
        if (raw) {
            const ggml_tensor * s = t->src[k];
            self->res->inp_raw.resize(ggml_nelements(s));
            ggml_backend_tensor_get(s, self->res->inp_raw.data(), 0, ggml_nbytes(s));
        }
        if (named && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)) {
            nvit_ref_tensor & o = self->res->t[t->name];
            o.cols = (int) t->ne[0];
            o.rows = (int) (ggml_nelements(t) / t->ne[0]);
            o.data.resize(ggml_nelements(t));
            ggml_backend_tensor_get(t, o.data.data(), 0, ggml_nbytes(t));
        }
        return true;
    }
};

nvit_ref::nvit_ref(const std::string & mmproj, const std::string & vocab_model, int device, int min_tok, int max_tok,
                   const std::string & dll_dir, bool verbose) : p_(new impl) {
    impl & I = *p_;
    I.verbose = verbose;
    I.dlls = load_dlls(dll_dir);
    llama_log_set(impl::log_cb, &I);
    mtmd_helper_log_set(impl::log_cb, &I);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    mp.n_gpu_layers = 0;
    I.model = llama_model_load_from_file(vocab_model.c_str(), mp);
    if (!I.model) throw std::runtime_error("nvit_ref: cannot load the vocabulary of " + vocab_model);
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = device >= 0;
    cp.device = nullptr;
    if (device >= 0) {
        const std::string dn = "CUDA" + std::to_string(device);
        cp.device = ggml_backend_dev_by_name(dn.c_str());
        if (!cp.device) throw std::runtime_error("nvit_ref: ggml device " + dn + " not found");
    }
    cp.print_timings = false;
    cp.n_threads = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.warmup = true;
    cp.image_min_tokens = min_tok;
    cp.image_max_tokens = max_tok;
    cp.media_marker = mtmd_default_marker();
    cp.cb_eval = impl::eval_cb;
    cp.cb_eval_user_data = &I;
    I.ctx = mtmd_init_from_file(mmproj.c_str(), I.model, cp);
    if (!I.ctx) throw std::runtime_error("nvit_ref: cannot load the mmproj " + mmproj);
}

nvit_ref::~nvit_ref() {
    if (!p_) return;
    if (p_->ctx) mtmd_free(p_->ctx);
    if (p_->model) llama_model_free(p_->model);
    llama_log_set(nullptr, nullptr);
    mtmd_helper_log_set(nullptr, nullptr);
}

std::string nvit_ref::dll_dir() const { return p_->dlls; }
double nvit_ref::vram_mib() const { return 0; }

void nvit_ref::encode(const std::string & bytes, int runs, const std::vector<std::string> & names, nvit_ref_result & out) {
    impl & I = *p_;
    mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_buf(I.ctx, (const unsigned char *) bytes.data(), bytes.size(),
                                                                      false, mtmd_helper_init_opt_default());
    if (!bw.bitmap) throw std::runtime_error("nvit_ref: cannot decode the image");
    std::unique_ptr<mtmd_bitmap, void (*)(mtmd_bitmap *)> bmp(bw.bitmap, mtmd_bitmap_free);
    std::unique_ptr<mtmd_input_chunks, void (*)(mtmd_input_chunks *)> chunks(mtmd_input_chunks_init(), mtmd_input_chunks_free);
    const std::string m = mtmd_default_marker();
    mtmd_input_text txt{m.c_str(), m.size(), false, true};
    const mtmd_bitmap * bp = bmp.get();
    if (mtmd_tokenize(I.ctx, chunks.get(), &txt, &bp, 1) != 0) throw std::runtime_error("nvit_ref: mtmd_tokenize failed");
    const mtmd_input_chunk * img = nullptr;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks.get(), i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) img = c;
    }
    if (!img) throw std::runtime_error("nvit_ref: no image chunk");
    const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(img);
    const size_t n = mtmd_image_tokens_get_n_tokens(it);
    const mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(it, 0, n - 1);
    out.tx = (int) last.x + 1;
    out.ty = (int) last.y + 1;
    out.n_embd = llama_model_n_embd(I.model);
    out.encode_ms.clear();
    I.capturing = false;
    for (int r = 0; r < std::max(1, runs); ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        if (mtmd_encode_chunk(I.ctx, img) != 0) throw std::runtime_error("nvit_ref: mtmd_encode_chunk failed");
        out.encode_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    // vocab-only loads leave n_embd 0: the projector width is the output row size
    if (out.n_embd <= 0) out.n_embd = 5120;
    const float * e = mtmd_get_output_embd(I.ctx);
    out.embd.assign(e, e + n * (size_t) out.n_embd);
    // capture run
    I.names = std::set<std::string>(names.begin(), names.end());
    I.res = &out;
    out.inp_raw.clear();
    out.t.clear();
    I.capturing = true;
    const int rc = mtmd_encode_chunk(I.ctx, img);
    I.capturing = false;
    if (rc != 0) throw std::runtime_error("nvit_ref: capture encode failed");
    const size_t npx = out.inp_raw.size() / 3;
    out.nx = out.tx * 32;
    out.ny = out.ty * 32;
    if (npx != (size_t) out.nx * out.ny) {
        fprintf(stderr, "nvit_ref: inp_raw has %zu px, token grid says %dx%d\n", npx, out.nx, out.ny);
    }
}
