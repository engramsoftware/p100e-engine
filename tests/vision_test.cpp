// p100e-vision-test: host-side checks of the image path (no LLM, no P100).
//   1. mm_track (src/mm_track.h): image rows, rope delta after a cut, cuts inside an image refused
//   2. vision_pin_ggml_env: inherited GGML_* removed (CRT and OS environment), GGML_CUDA_GEMM_FOLD pinned, keep mode
//   3. vision_mmproj_width: projection_dim read, deepstack projectors (flag array or v.deepstack.* tensors) rejected,
//      on GGUF files written here with the fork's gguf API (+ the real mmproj when --mmproj is given)
//   4. with --mmproj and --model (GPU, encoder only): detail "low" on 1920x1080 = 43 x 25 = 1075 tokens; a cached image
//      returns at once while another thread encodes; two threads encoding the same new image encode it once
// usage: p100e-vision-test [--mmproj f --model f [--dev N]] [--vision-dlls dir]
//   (--dev = CUDA runtime index; set CUDA_DEVICE_ORDER=PCI_BUS_ID for a stable order)
#include "vision.h"
#include "../src/mm_track.h"

#include "ggml.h"
#include "gguf.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace p100e;

static int g_checks = 0, g_fail = 0;
static void check(bool ok, const std::string & what) {
    ++g_checks;
    if (!ok) ++g_fail;
    printf("  %-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
    fflush(stdout);
}
template <class F> static bool throws(F f) {
    try { f(); } catch (const std::exception &) { return true; }
    return false;
}

// ---------------------------------------------------------------------------------------------------------------------
static void test_mm_track() {
    printf("mm_track\n");
    mm_track t;
    check(t.delta() == 0 && t.cut(7) == 0, "empty: delta 0, any cut fine");
    // text rows 0..9, then a 40x27 image at rows 10..1089 (rotary 10.., text after it at 10 + 40 = 50: delta 1090 - 50),
    // text 1090..1099, a 60x34 image at rows 1100..3139 (rotary 60.., text after at 60 + 60 = 120: delta 3140 - 120)
    t.add(10, 1090, 1040);
    t.add(1100, 3140, 3020);
    check(t.delta() == 3020, "delta after two images = the last image's");
    check(throws([&] { t.check_cut(11); }) && throws([&] { t.check_cut(1089); }) && throws([&] { t.check_cut(2000); }),
          "cuts strictly inside an image refused");
    check(!throws([&] { t.check_cut(10); }) && !throws([&] { t.check_cut(1090); }) && !throws([&] { t.check_cut(1100); }) &&
              !throws([&] { t.check_cut(3140); }) && !throws([&] { t.check_cut(5000); }),
          "cuts at image boundaries and in text allowed");
    {
        mm_track u = t;
        check(throws([&] { u.cut(1500); }) && u.spans.size() == 2 && u.delta() == 3020, "a refused cut changes nothing");
    }
    check(t.cut(4000) == 3020 && t.spans.size() == 2, "cut in the text after both images: delta unchanged");
    check(t.cut(3140) == 3020 && t.spans.size() == 2, "cut right after the second image keeps it");
    check(t.cut(1100) == 1040 && t.spans.size() == 1, "cut at the second image's first row drops it: first image's delta");
    check(t.cut(1095) == 1040 && t.spans.size() == 1, "cut in the text between the images");
    check(t.cut(10) == 0 && t.spans.empty(), "cut at the first image's first row: no image left, delta 0");
    check(throws([&] { mm_track v; v.add(10, 20, 5); v.add(15, 30, 9); }), "overlapping image rows rejected");
    check(throws([&] { mm_track v; v.add(10, 10, 0); }), "empty image rows rejected");
    // the delta formula tp_model uses: an image of nx x ny at rotary P takes nx * ny rows, the text after it is at
    // P + max(nx, ny); delta grows by nx * ny - max(nx, ny)
    {
        int rows = 0, rpos = 0;
        mm_track w;
        auto text = [&](int n) { rows += n; rpos += n; };
        auto image = [&](int nx, int ny) {
            const int r0 = rows;
            rows += nx * ny;
            rpos += std::max(nx, ny);
            w.add(r0, rows, rows - rpos);
        };
        text(4); image(40, 27); text(3); image(1, 1); text(2); image(27, 40); text(5);
        check(w.delta() == 1080 - 40 + 0 + 1080 - 40 && rows - rpos == w.delta(), "delta = sum of nx * ny - max(nx, ny)");
    }
}

// ---------------------------------------------------------------------------------------------------------------------
static void setenv_(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}
static bool os_has(const char * k) {
#ifdef _WIN32
    char buf[8];
    return GetEnvironmentVariableA(k, buf, sizeof buf) > 0 || GetLastError() != ERROR_ENVVAR_NOT_FOUND;
#else
    return getenv(k) != nullptr;
#endif
}
static std::string env_or(const char * k, const char * d = "(unset)") { const char * v = getenv(k); return v ? v : d; }

static void test_env() {
    printf("vision_pin_ggml_env\n");
    setenv_("P100E_VISION_GGML_ENV", nullptr);
    setenv_("GGML_CUDA_P2P", "1");
    setenv_("GGML_P100_CUDA_GRAPHS", "1");
    setenv_("ggml_cuda_fa_gemm", "1");   // Windows names are case-insensitive: ggml still sees it
    setenv_("GGML_CUDA_GEMM_FOLD", "0");
    setenv_("P100E_UNRELATED", "x");
    auto log = vision_pin_ggml_env(true);
    for (const auto & l : log) printf("       %s\n", l.c_str());
    check(!getenv("GGML_CUDA_P2P") && !os_has("GGML_CUDA_P2P"), "GGML_CUDA_P2P removed (CRT and OS block)");
    check(!getenv("GGML_P100_CUDA_GRAPHS") && !getenv("GGML_CUDA_FA_GEMM") && !getenv("ggml_cuda_fa_gemm"),
          "other inherited GGML_* removed (any case)");
    check(env_or("GGML_CUDA_GEMM_FOLD") == "2", "GGML_CUDA_GEMM_FOLD pinned to 2 (was 0)");
    check(env_or("P100E_UNRELATED") == "x", "non-GGML variables untouched");
    log = vision_pin_ggml_env(true);
    check(log.empty(), "second call: nothing to change");
    vision_pin_ggml_env(false);
    check(env_or("GGML_CUDA_GEMM_FOLD") == "0", "fold_gemm = false pins 0");

    setenv_("P100E_VISION_GGML_ENV", "keep");
    setenv_("GGML_CUDA_P2P", "1");
    setenv_("GGML_PF_VCVT", "1");
    setenv_("GGML_CUDA_GEMM_FOLD", "1");
    vision_pin_ggml_env(true);
    check(!getenv("GGML_CUDA_P2P") && env_or("GGML_PF_VCVT") == "1" && env_or("GGML_CUDA_GEMM_FOLD") == "1",
          "keep: only GGML_CUDA_P2P removed, an inherited GEMM_FOLD wins");
    setenv_("GGML_CUDA_GEMM_FOLD", nullptr);
    vision_pin_ggml_env(true);
    check(env_or("GGML_CUDA_GEMM_FOLD") == "2", "keep: GEMM_FOLD set when absent");
    setenv_("P100E_VISION_GGML_ENV", nullptr);
    setenv_("GGML_PF_VCVT", nullptr);
    setenv_("P100E_UNRELATED", nullptr);
}

// ---------------------------------------------------------------------------------------------------------------------
static std::string tmp_path(const char * name) {
#ifdef _WIN32
    char d[MAX_PATH];
    GetTempPathA(MAX_PATH, d);
    return std::string(d) + name;
#else
    return std::string("/tmp/") + name;
#endif
}

// a metadata-only mmproj stand-in: projection_dim (0: none), the deepstack flags, optionally one v.deepstack tensor
static std::string write_gguf(const char * name, int proj, const std::vector<int8_t> & flags, bool ds_tensor) {
    const std::string p = tmp_path(name);
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "clip");
    gguf_set_val_str(g, "clip.projector_type", "qwen3vl_merger");
    if (proj > 0) gguf_set_val_u32(g, "clip.vision.projection_dim", (uint32_t) proj);
    if (!flags.empty()) gguf_set_arr_data(g, "clip.vision.is_deepstack_layers", GGUF_TYPE_BOOL, flags.data(), flags.size());
    ggml_init_params ip{1 << 20, nullptr, false};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_set_name(t, ds_tensor ? "v.deepstack.3.fc1.weight" : "mm.1.bias");
    memset(t->data, 0, ggml_nbytes(t));
    gguf_add_tensor(g, t);
    const bool ok = gguf_write_to_file(g, p.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    if (!ok) throw std::runtime_error("cannot write " + p);
    return p;
}

static void test_mmproj(const std::string & mmproj, const std::string & dlls) {
    printf("vision_mmproj_width\n");
    vision_load_dlls(dlls);   // the gguf / ggml calls below go to the delay-loaded fork DLLs
    const std::string a = write_gguf("p100e_vt_plain.gguf", 5120, std::vector<int8_t>(27, 0), false);
    const std::string b = write_gguf("p100e_vt_dsflag.gguf", 5120, {0, 0, 1, 0}, false);
    const std::string c = write_gguf("p100e_vt_dstensor.gguf", 5120, {}, true);
    const std::string d = write_gguf("p100e_vt_noproj.gguf", 0, {}, false);
    int w = 0;
    check(!throws([&] { w = vision_mmproj_width(a, dlls); }) && w == 5120, "projection_dim 5120, 27 false flags: accepted");
    std::string msg;
    try { vision_mmproj_width(b, dlls); } catch (const std::exception & e) { msg = e.what(); }
    check(msg.find("deepstack") != std::string::npos && msg.find("[2]") != std::string::npos,
          "a true is_deepstack_layers entry: rejected (" + msg.substr(0, 60) + "...)");
    msg.clear();
    try { vision_mmproj_width(c, dlls); } catch (const std::exception & e) { msg = e.what(); }
    check(msg.find("v.deepstack.3.fc1.weight") != std::string::npos, "a v.deepstack.* tensor: rejected");
    check(throws([&] { vision_mmproj_width(d, dlls); }), "no projection_dim: rejected");
    check(throws([&] { vision_mmproj_width(tmp_path("p100e_vt_missing.gguf"), dlls); }), "missing file: rejected");
    for (const auto & f : {a, b, c, d}) remove(f.c_str());
    if (!mmproj.empty()) check(!throws([&] { w = vision_mmproj_width(mmproj, dlls); }) && w == 5120, "the real mmproj: 5120");
}

// ---------------------------------------------------------------------------------------------------------------------
// 24-bit BMP (stb_image reads it) with a pattern that depends on seed
static std::string make_bmp(int w, int h, uint32_t seed) {
    const int row = (w * 3 + 3) & ~3;
    const uint32_t data = (uint32_t) row * h, size = 54 + data;
    std::string b(size, '\0');
    auto p16 = [&](int o, uint32_t v) { b[o] = (char) (v & 255); b[o + 1] = (char) ((v >> 8) & 255); };
    auto p32 = [&](int o, uint32_t v) { p16(o, v & 0xffff); p16(o + 2, v >> 16); };
    b[0] = 'B'; b[1] = 'M';
    p32(2, size); p32(10, 54); p32(14, 40); p32(18, (uint32_t) w); p32(22, (uint32_t) h); p16(26, 1); p16(28, 24);
    p32(34, data);
    uint32_t s = seed * 2654435761u + 1;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            s = s * 1664525u + 1013904223u;
            char * px = &b[54 + (size_t) y * row + (size_t) x * 3];
            px[0] = (char) ((x * 255 / w + (s >> 28)) & 255);
            px[1] = (char) ((y * 255 / h + seed * 40) & 255);
            px[2] = (char) (((x / 64 + y / 64) & 1) ? 230 : 25);
        }
    return b;
}

static void test_encoder(const std::string & mmproj, const std::string & model, int dev, const std::string & dlls) {
    printf("vision_encoder (CUDA%d)\n", dev);
    vision_opts o;
    o.mmproj = mmproj;
    o.text_model = model;
    o.device = dev;
    o.dll_dir = dlls;
    vision_encoder enc(o);
    check(enc.n_embd() == 5120 && !enc.cpu_fallback(), "loaded: n_embd 5120, no CPU fallback");

    image_input low;
    low.bytes = make_bmp(1920, 1080, 1);
    low.max_tokens = enc.detail_max_tokens("low");
    vision_timing t;
    auto e = enc.encode(low, &t);
    printf("       detail low 1920x1080: %dx%d = %d tokens, encode %.0f ms\n", e->nx, e->ny, e->n_tok(), t.encode_ms);
    check(e->nx == 43 && e->ny == 25 && e->n_tok() == 1075, "detail low on 1920x1080 = 43 x 25 = 1075 tokens (not 1024)");

    image_input tiny_img;
    tiny_img.bytes = make_bmp(96, 64, 2);
    auto es = enc.encode(tiny_img, &t);
    check(!t.cached, "small image encoded");
    auto pieces = enc.build(std::string("<|im_start|>user\n") + vision_encoder::marker() + "What is this?<|im_end|>\n",
                            {tiny_img});
    check(pieces.size() == 3 && !pieces[0].image && pieces[1].image == es && pieces[1].t.cached && !pieces[2].image,
          "build: text, the cached image (same object), text");

    // a cached image while another thread encodes a new one: must not wait for that encode
    image_input big;
    big.bytes = make_bmp(1920, 1080, 3);   // full cap: ~2040 tokens
    std::atomic<bool> started{false}, done{false};
    double big_ms = 0;
    std::thread th([&] {
        started = true;
        vision_timing tb;
        enc.encode(big, &tb);
        big_ms = tb.encode_ms;
        done = true;
    });
    while (!started) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(150));   // the big encode is under way
    const auto a = std::chrono::steady_clock::now();
    vision_timing tc;
    auto ec = enc.encode(tiny_img, &tc);
    const double hit_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    const bool overlapped = !done;
    th.join();
    printf("       cached lookup %.2f ms during a %.0f ms encode (still running: %s)\n", hit_ms, big_ms, overlapped ? "yes" : "no");
    if (overlapped) check(tc.cached && ec == es && hit_ms < 50, "cached image returned while another thread encodes");
    else printf("  skip cached-during-encode check (the other encode finished within 150 ms)\n");

    // two threads, one new image: encoded once (the second waits, then hits the cache)
    image_input dup_img;
    dup_img.bytes = make_bmp(640, 480, 4);
    const auto s0 = enc.stats();
    std::shared_ptr<const image_embd> r1, r2;
    vision_timing t1, t2;
    std::thread x([&] { r1 = enc.encode(dup_img, &t1); });
    std::thread y([&] { r2 = enc.encode(dup_img, &t2); });
    x.join();
    y.join();
    const auto s1 = enc.stats();
    printf("       same image twice at once: misses +%zu, hits +%zu, waits %.0f / %.0f ms\n", s1.misses - s0.misses,
           s1.hits - s0.hits, t1.wait_ms, t2.wait_ms);
    check(r1 == r2 && s1.misses - s0.misses == 1 && s1.hits - s0.hits == 1 && (t1.cached != t2.cached),
          "the same new image from two threads is encoded once");
}

int main(int argc, char ** argv) {
    std::string mmproj, model, dlls;
    int dev = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value"); return argv[++i]; };
        if (a == "--mmproj") mmproj = next();
        else if (a == "--model") model = next();
        else if (a == "--dev") dev = std::stoi(next());
        else if (a == "--vision-dlls") dlls = next();
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    try {
        test_mm_track();
        test_env();
        test_mmproj(mmproj, dlls);
        if (!mmproj.empty() && !model.empty()) test_encoder(mmproj, model, dev, dlls);
        else printf("(encoder checks skipped: pass --mmproj and --model)\n");
    } catch (const std::exception & e) {
        printf("FAIL exception: %s\n", e.what());
        ++g_fail;
    }
    printf("%s: %d checks, %d failed\n", g_fail ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
