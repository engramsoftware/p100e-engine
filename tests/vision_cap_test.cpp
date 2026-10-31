// detail "low" (per-image token cap = min_tokens) must give the same token grid with both encoders.
// Part 1, no GPU: vision_cap_size targets (1920x1080 -> 1075 tokens, 2000x1000 -> 1058, the mtmd path's counts; the
// native path used to preprocess straight to the cap and got 1008 / 990, below min_tokens).
// Part 2, with --mmproj M --vocab V [--device D]: synthetic images through vision_encoder twice, native (nvit on D) and
// mtmd (CUDA D), at detail "low" and uncapped: grids must match the expected counts and each other; prints the cosine
// of the two embeddings (they differ by mtmd's own numerical error).
#include "vision.h"
#include "nvit.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace p100e;

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

// 24-bit BMP (stb_image decodes it, mtmd too) of a deterministic pattern: gradients + a grid + some noise
static std::string make_bmp(int w, int h) {
    const int row = (w * 3 + 3) & ~3;
    const uint32_t data = (uint32_t) row * h, size = 54 + data;
    std::string b(size, '\0');
    auto u32 = [&](size_t at, uint32_t v) { memcpy(&b[at], &v, 4); };
    auto u16 = [&](size_t at, uint16_t v) { memcpy(&b[at], &v, 2); };
    b[0] = 'B'; b[1] = 'M';
    u32(2, size); u32(10, 54); u32(14, 40); u32(18, (uint32_t) w); u32(22, (uint32_t) h); u16(26, 1); u16(28, 24);
    u32(34, data);
    uint32_t x = 2463534242u;
    for (int y = 0; y < h; ++y) {
        char * p = &b[54 + (size_t) (h - 1 - y) * row];   // bottom-up rows
        for (int i = 0; i < w; ++i) {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            const bool grid = (i % 97) < 3 || (y % 61) < 3;
            p[3 * i + 0] = (char) (grid ? 20 : (i * 255 / w + (x & 15)) & 255);        // B
            p[3 * i + 1] = (char) (grid ? 20 : (y * 255 / h + ((x >> 8) & 15)) & 255);  // G
            p[3 * i + 2] = (char) (grid ? 230 : ((i + y) * 255 / (w + h)) & 255);      // R
        }
    }
    return b;
}

int main(int argc, char ** argv) {
    std::string mmproj, vocab;
    int dev = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--mmproj")) mmproj = argv[i + 1];
        else if (!strcmp(argv[i], "--vocab")) vocab = argv[i + 1];
        else if (!strcmp(argv[i], "--device")) dev = atoi(argv[i + 1]);
        else { fprintf(stderr, "usage: %s [--mmproj M --vocab V [--device D]]\n", argv[0]); return 2; }
    }
    const int MIN = 1024, MAX = 2048;
    struct tc { int w, h, tok_low, tok_full; };
    const tc cases[] = {{1920, 1080, 1075, 2040}, {2000, 1000, 1058, 1953}};

    // ---- part 1: sizes ----
    for (const tc & c : cases) {
        int tw = 0, th = 0;
        const bool rs = vision_cap_size(c.w, c.h, MIN, MIN, MAX, tw, th);
        const int tok = tw / 32 * (th / 32);
        int fw, fh, ow, oh;
        nvit::smart_size(tw, th, MIN, MAX, fw, fh);          // the encoder's own preprocessing keeps it (one resample)
        nvit::smart_size(c.w, c.h, MIN, MIN, ow, oh);        // the old native path: straight to the cap
        int nw, nh;
        nvit::smart_size(c.w, c.h, MIN, MAX, nw, nh);
        printf("%dx%d low: resample %s to %dx%d = %d tokens (expected %d; old native path %dx%d = %d); full %dx%d = %d\n",
               c.w, c.h, rs ? "yes" : "no", tw, th, tok, c.tok_low, ow, oh, ow / 32 * (oh / 32), nw, nh, nw / 32 * (nh / 32));
        CHECK(rs && tok == c.tok_low, "%dx%d: %d tokens at detail low, expected %d", c.w, c.h, tok, c.tok_low);
        CHECK(tok >= MIN, "%dx%d: %d tokens < min %d", c.w, c.h, tok, MIN);
        CHECK(fw == tw && fh == th, "%dx%d: %dx%d is not kept by smart_resize (-> %dx%d)", c.w, c.h, tw, th, fw, fh);
        CHECK(nw / 32 * (nh / 32) == c.tok_full, "%dx%d: %d tokens uncapped, expected %d", c.w, c.h, nw / 32 * (nh / 32), c.tok_full);
    }
    {
        int tw, th;
        CHECK(!vision_cap_size(800, 600, MIN, MIN, MAX, tw, th), "800x600 (1036 tokens uncapped) needs no host resample");
        CHECK(!vision_cap_size(1920, 1080, MIN, MAX, MAX, tw, th), "cap == max: no host resample");
        CHECK(!vision_cap_size(1920, 1080, MIN, 4096, MAX, tw, th), "cap > max: no host resample");
    }
    // 1 x H and W x 1 strips: refused beyond 200:1, and the size never goes past max_tokens (mtmd's math gave 1 x 100000
    // -> 32 x 457888 = 14309 tokens at max 2048)
    {
        struct sc { int w, h; bool ok; };
        const sc strips[] = {{1, 100000, false}, {100000, 1, false}, {1, 150, true}, {150, 1, true}, {1, 201, false}, {201, 1, false}};
        for (const sc & c : strips) {
            CHECK(nvit::image_size_ok(c.w, c.h) == c.ok, "%dx%d: nvit::image_size_ok should be %d", c.w, c.h, (int) c.ok);
            CHECK(vision_image_ok(c.w, c.h) == c.ok, "%dx%d: vision_image_ok should be %d", c.w, c.h, (int) c.ok);
            int ow, oh;
            nvit::smart_size(c.w, c.h, MIN, MAX, ow, oh);
            const int tok = ow / 32 * (oh / 32);
            printf("%dx%d strip: %s, smart size %dx%d = %d tokens\n", c.w, c.h, c.ok ? "accepted" : "refused", ow, oh, tok);
            CHECK(ow % 32 == 0 && oh % 32 == 0 && tok >= 1 && tok <= MAX, "%dx%d: %d tokens, max %d", c.w, c.h, tok, MAX);
        }
        // the short side raised to 32 px within 200:1 (150:1, small max_tok): nvit and the host cap resample stay in bounds
        int ow, oh, tw, th;
        nvit::smart_size(20, 3000, 16, 64, ow, oh);
        CHECK(ow == 32 && oh == 2048, "20x3000 at max 64: %dx%d, expected 32x2048 (64 tokens)", ow, oh);
        nvit::smart_size(3000, 20, 16, 64, ow, oh);
        CHECK(ow == 2048 && oh == 32, "3000x20 at max 64: %dx%d, expected 2048x32 (64 tokens)", ow, oh);
        CHECK(vision_cap_size(20, 3000, 16, 32, 64, tw, th) && tw / 32 * (th / 32) <= 32,
              "20x3000 capped at 32 of 64: %dx%d = %d tokens", tw, th, tw / 32 * (th / 32));
    }

    // ---- part 2: both encoders ----
    if (!mmproj.empty() && !vocab.empty()) {
        for (const tc & c : cases) {
            const std::string bytes = make_bmp(c.w, c.h);
            std::shared_ptr<const image_embd> got[2][2];   // [native, mtmd][low, full]
            for (int native = 1; native >= 0; --native) {
                vision_opts vo;
                vo.mmproj = mmproj;
                vo.text_model = vocab;
                vo.device = dev;
                vo.min_tokens = MIN;
                vo.max_tokens = MAX;
                vo.cache_mb = 0;
                vo.native = native;
                vo.native_devices = {dev};
                try {
                    vision_encoder enc(vo);
                    CHECK(enc.native() == (native == 1), "encoder kind");
                    for (int full = 0; full < 2; ++full) {
                        image_input in;
                        in.bytes = bytes;
                        in.max_tokens = full ? 0 : enc.detail_max_tokens("low");
                        vision_timing t;
                        got[1 - native][full] = enc.encode(in, &t);
                        const auto & e = *got[1 - native][full];
                        printf("%s %dx%d %s: grid %dx%d = %d tokens (resize %.1f ms, preprocess %.1f ms, encode %.1f ms)\n",
                               native ? "native" : "mtmd  ", c.w, c.h, full ? "full" : "low ", e.nx, e.ny, e.n_tok(),
                               t.resize_ms, t.preprocess_ms, t.encode_ms);
                        CHECK(e.n_tok() == (full ? c.tok_full : c.tok_low), "%s %dx%d %s: %d tokens, expected %d",
                              native ? "native" : "mtmd", c.w, c.h, full ? "full" : "low", e.n_tok(), full ? c.tok_full : c.tok_low);
                    }
                } catch (const std::exception & ex) {
                    ++fails;
                    printf("FAIL: %s encoder: %s\n", native ? "native" : "mtmd", ex.what());
                }
            }
            for (int full = 0; full < 2; ++full) {
                const auto & a = got[0][full];
                const auto & b = got[1][full];
                if (!a || !b) continue;
                CHECK(a->nx == b->nx && a->ny == b->ny, "%dx%d %s: native grid %dx%d vs mtmd %dx%d", c.w, c.h,
                      full ? "full" : "low", a->nx, a->ny, b->nx, b->ny);
                if (a->data.size() != b->data.size()) continue;
                double d = 0, na = 0, nb = 0;
                for (size_t i = 0; i < a->data.size(); ++i) {
                    d += (double) a->data[i] * b->data[i];
                    na += (double) a->data[i] * a->data[i];
                    nb += (double) b->data[i] * b->data[i];
                }
                const double cs = d / std::sqrt(na * nb);
                printf("%dx%d %s: native vs mtmd embeddings cos %.5f\n", c.w, c.h, full ? "full" : "low ", cs);
                CHECK(cs > 0.98, "%dx%d %s: native vs mtmd cosine %.5f (different pixels?)", c.w, c.h, full ? "full" : "low", cs);
            }
        }
    } else {
        printf("(part 2 skipped: --mmproj and --vocab run both encoders)\n");
    }
    printf(fails ? "%d CHECK(S) FAILED\n" : "ALL CHECKS PASSED\n", fails);
    return fails ? 1 : 0;
}
