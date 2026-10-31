// nvit CPU preprocessing: a line-by-line port of the fork's (98ab7e5) mtmd image path for qwen3vl_merger, so the
// planar f32 image equals clip's inp_raw bit for bit:
//   mtmd-helper.cpp  mtmd_helper_bitmap_init_from_buf  -> stbi_load_from_memory(..., 3)
//   mtmd-image.cpp   mtmd_image_preprocessor_dyn_size::preprocess -> calc_size_preserved_ratio (align 32,
//                    min/max pixels = tokens x 1024), img_tool::resize (RESIZE_ALGO_BICUBIC, PAD_CEIL, pad 0,0,0)
//                    -> resize_pillow (Pillow fixed-point resampler)
//   clip-impl.h      clip_image_f32::from_u8 (/255.0f), normalize (mean = std = 0.5)
//   clip.cpp         clip_encode: interleaved RGB -> planar [3][ny][nx]
// The float / double types of every intermediate are kept as in the original (they decide rounding).
#include "nvit.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include "stb/stb_image.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>

namespace p100e::nvit {

bool image_size_ok(int w, int h, std::string * err) {
    const char * why = nullptr;
    if (w <= 0 || h <= 0) why = "empty image";
    else if (w > img_max_side || h > img_max_side) why = "a side is above 32768 px";
    else if ((int64_t) w * h > img_max_pixels) why = "more than 2^28 pixels";
    else if ((int64_t) std::max(w, h) > (int64_t) img_max_aspect * std::min(w, h)) why = "aspect ratio above 200:1";
    if (why && err) *err = std::to_string(w) + "x" + std::to_string(h) + ": " + why;
    return !why;
}

bool decode_image(const void * bytes, size_t n, rgb_image & out, std::string * err) {
    int nx = 0, ny = 0, nc = 0;
    // the header's size first: an image beyond the limits is refused before stb_image allocates its pixels
    if (n <= (size_t) INT_MAX && stbi_info_from_memory((const stbi_uc *) bytes, (int) n, &nx, &ny, &nc) &&
        !image_size_ok(nx, ny, err))
        return false;
    unsigned char * data = stbi_load_from_memory((const stbi_uc *) bytes, (int) n, &nx, &ny, &nc, 3);
    if (!data) {
        if (err) *err = std::string("stb_image: ") + (stbi_failure_reason() ? stbi_failure_reason() : "decode failed");
        return false;
    }
    out.w = nx;
    out.h = ny;
    out.px.assign(data, data + (size_t) nx * ny * 3);
    stbi_image_free(data);
    return true;
}

// img_tool::calc_size_preserved_ratio with align_size 32, longest_edge 0
static void calc_size(int width, int height, int align, int min_pixels, int max_pixels, int & ow, int & oh) {
    if (width <= 0 || height <= 0) { ow = oh = 0; return; }
    auto round_by_factor = [f = align](float x) { return static_cast<int>(std::round(x / static_cast<float>(f))) * f; };
    auto ceil_by_factor  = [f = align](float x) { return static_cast<int>(std::ceil(x / static_cast<float>(f))) * f; };
    auto floor_by_factor = [f = align](float x) { return static_cast<int>(std::floor(x / static_cast<float>(f))) * f; };
    int w_bar = std::max(align, round_by_factor(width));
    int h_bar = std::max(align, round_by_factor(height));
    if (max_pixels > 0 && h_bar * w_bar > max_pixels) {
        const auto beta = std::sqrt(static_cast<float>(height) * width / max_pixels);
        h_bar = std::max(align, floor_by_factor(height / beta));
        w_bar = std::max(align, floor_by_factor(width / beta));
        // not in mtmd: when the short side was raised to `align` (aspect above max_pixels / align^2), the long side
        // still had the full 1 / beta and the area went past max_pixels (1 x 100000 at 2048 tokens: 14309 tokens).
        // Floor the long side into max_pixels instead. Only those shapes change (others are <= max_pixels already).
        if (h_bar * w_bar > max_pixels) {
            const int other = std::max(align, max_pixels / align / align * align);
            if (w_bar == align) h_bar = other;
            else w_bar = other;
        }
    } else if (min_pixels > 0 && h_bar * w_bar < min_pixels) {
        const auto beta = std::sqrt(static_cast<float>(min_pixels) / (static_cast<float>(height) * width));
        h_bar = ceil_by_factor(height * beta);
        w_bar = ceil_by_factor(width * beta);
    }
    ow = w_bar;
    oh = h_bar;
}

void smart_size(int w, int h, int min_tok, int max_tok, int & ow, int & oh) {
    const int patch_area = 16 * 16 * 2 * 2;   // patch 16, merge 2
    calc_size(w, h, 32, min_tok * patch_area, max_tok * patch_area, ow, oh);
}

// ---- img_tool::resize_pillow (bicubic only) ----
static int precompute_weights(int inSize, int outSize, std::vector<int> & bounds, std::vector<int32_t> & weights) {
    const int PRECISION_BITS = 32 - 8 - 2;
    const double filter_support = 2.0;
    auto resample_filter = [](double x) -> double {
        if (x < 0.0) x = -x;
        constexpr double a = -0.5;
        if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
        if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
        return 0.0;
    };
    double support, scale, filterscale;
    double center, ww, ss;
    int xx, x, ksize, xmin, xmax;
    filterscale = scale = static_cast<double>(inSize) / outSize;
    if (filterscale < 1.0) filterscale = 1.0;
    support = filter_support * filterscale;
    ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    std::vector<double> pre_weights((size_t) outSize * ksize);
    bounds.resize((size_t) outSize * 2);
    for (xx = 0; xx < outSize; xx++) {
        center = (xx + 0.5) * scale;
        ww = 0.0;
        ss = 1.0 / filterscale;
        xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        xmax = static_cast<int>(center + support + 0.5);
        if (xmax > inSize) xmax = inSize;
        xmax -= xmin;
        for (x = 0; x < xmax; x++) {
            double w = resample_filter((x + xmin - center + 0.5) * ss);
            pre_weights[(size_t) xx * ksize + x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; x++) {
            if (ww != 0.0) pre_weights[(size_t) xx * ksize + x] /= ww;
        }
        for (; x < ksize; x++) pre_weights[(size_t) xx * ksize + x] = 0;
        bounds[(size_t) xx * 2 + 0] = xmin;
        bounds[(size_t) xx * 2 + 1] = xmax;
    }
    weights.resize((size_t) outSize * ksize);
    const double fxp_scale = std::ldexp(1.0, PRECISION_BITS);
    for (size_t i = 0; i < (size_t) outSize * ksize; i++) {
        const double rounded = pre_weights[i] * fxp_scale + (pre_weights[i] < 0 ? -0.5 : 0.5);
        weights[i] = static_cast<int32_t>(rounded);
    }
    return ksize;
}

static inline uint8_t clip8(int val) { return val < 0 ? 0 : val > 255 ? 255 : (uint8_t) val; }

static std::vector<uint8_t> resample_horizontal(const uint8_t * src, int in_nx, int in_ny, int out_nx, int ksize,
                                                const std::vector<int> & bounds, const std::vector<int32_t> & weights) {
    const int PRECISION_BITS = 32 - 8 - 2;
    std::vector<uint8_t> out((size_t) out_nx * in_ny * 3);
    for (int yy = 0; yy < in_ny; yy++) {
        const uint8_t * src_row = src + (size_t) yy * in_nx * 3;
        uint8_t * dst_row = out.data() + (size_t) yy * out_nx * 3;
        for (int xx = 0; xx < out_nx; xx++) {
            const int xmin = bounds[(size_t) xx * 2 + 0];
            const int xcnt = bounds[(size_t) xx * 2 + 1];
            const int32_t * k = &weights[(size_t) xx * ksize];
            const uint8_t * p = src_row + (size_t) xmin * 3;
            int32_t ss0 = 1 << (PRECISION_BITS - 1), ss1 = 1 << (PRECISION_BITS - 1), ss2 = 1 << (PRECISION_BITS - 1);
            for (int x = 0; x < xcnt; x++) {
                ss0 += p[0] * k[x];
                ss1 += p[1] * k[x];
                ss2 += p[2] * k[x];
                p += 3;
            }
            dst_row[xx * 3 + 0] = clip8(ss0 >> PRECISION_BITS);
            dst_row[xx * 3 + 1] = clip8(ss1 >> PRECISION_BITS);
            dst_row[xx * 3 + 2] = clip8(ss2 >> PRECISION_BITS);
        }
    }
    return out;
}

static std::vector<uint8_t> resample_vertical(const uint8_t * src, int in_nx, int out_ny, int ksize,
                                              const std::vector<int> & bounds, const std::vector<int32_t> & weight) {
    const int PRECISION_BITS = 32 - 8 - 2;
    const size_t row_elems = (size_t) in_nx * 3;
    std::vector<uint8_t> out(row_elems * out_ny);
    std::vector<int32_t> acc(row_elems);
    for (int yy = 0; yy < out_ny; yy++) {
        const int ymin = bounds[(size_t) yy * 2 + 0];
        const int ycnt = bounds[(size_t) yy * 2 + 1];
        const int32_t * k = &weight[(size_t) yy * ksize];
        std::fill(acc.begin(), acc.end(), 1 << (PRECISION_BITS - 1));
        for (int y = 0; y < ycnt; y++) {
            const uint8_t * src_row = src + (size_t) (ymin + y) * row_elems;
            const int32_t w = k[y];
            for (size_t i = 0; i < row_elems; i++) acc[i] += src_row[i] * w;
        }
        uint8_t * dst_row = out.data() + (size_t) yy * row_elems;
        for (size_t i = 0; i < row_elems; i++) dst_row[i] = clip8(acc[i] >> PRECISION_BITS);
    }
    return out;
}

static rgb_image resize_pillow(const rgb_image & img, int target_width, int target_height) {
    const int src_width = img.w, src_height = img.h;
    const bool need_horizontal = target_width != src_width;
    const bool need_vertical = target_height != src_height;
    std::vector<int> bounds_horiz, bounds_vert;
    std::vector<int32_t> weights_horiz, weights_vert;
    int ksize_horiz = 0, ksize_vert = 0;
    if (need_horizontal) ksize_horiz = precompute_weights(src_width, target_width, bounds_horiz, weights_horiz);
    if (need_vertical) ksize_vert = precompute_weights(src_height, target_height, bounds_vert, weights_vert);
    rgb_image dst;
    const uint8_t * src = img.px.data();
    if (need_horizontal && need_vertical) {
        auto temp = resample_horizontal(src, src_width, src_height, target_width, ksize_horiz, bounds_horiz, weights_horiz);
        dst.w = target_width; dst.h = target_height;
        dst.px = resample_vertical(temp.data(), target_width, target_height, ksize_vert, bounds_vert, weights_vert);
    } else if (need_horizontal) {
        dst.w = target_width; dst.h = src_height;
        dst.px = resample_horizontal(src, src_width, src_height, target_width, ksize_horiz, bounds_horiz, weights_horiz);
    } else if (need_vertical) {
        dst.w = src_width; dst.h = target_height;
        dst.px = resample_vertical(src, src_width, target_height, ksize_vert, bounds_vert, weights_vert);
    } else {
        dst = img;
    }
    return dst;
}

// img_tool::resize with PAD_CEIL and pad color {0, 0, 0}
static rgb_image resize_pad_ceil(const rgb_image & src, int tw, int th) {
    if (src.w == tw && src.h == th) return src;
    const float scale_w = static_cast<float>(tw) / src.w;
    const float scale_h = static_cast<float>(th) / src.h;
    const float scale = std::min(scale_w, scale_h);
    const int new_width = std::min(static_cast<int>(std::ceil(src.w * scale)), tw);
    const int new_height = std::min(static_cast<int>(std::ceil(src.h * scale)), th);
    const rgb_image resized = resize_pillow(src, new_width, new_height);
    rgb_image dst;
    dst.w = tw; dst.h = th;
    dst.px.assign((size_t) tw * th * 3, 0);
    const int offset_x = (tw - new_width) / 2;
    const int offset_y = (th - new_height) / 2;
    for (int y = 0; y < resized.h; ++y) {
        const int dy = y + offset_y;
        if (dy < 0 || dy >= th) continue;
        for (int x = 0; x < resized.w; ++x) {
            const int dx = x + offset_x;
            if (dx < 0 || dx >= tw) continue;
            const uint8_t * s = &resized.px[((size_t) y * resized.w + x) * 3];
            uint8_t * d = &dst.px[((size_t) dy * tw + dx) * 3];
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2];
        }
    }
    return dst;
}

image preprocess(const rgb_image & img, int min_tok, int max_tok) {
    if (img.w <= 0 || img.h <= 0 || img.px.size() != (size_t) img.w * img.h * 3)
        throw std::runtime_error("nvit::preprocess: bad image");
    std::string why;
    if (!image_size_ok(img.w, img.h, &why)) throw std::runtime_error("nvit::preprocess: image refused (" + why + ")");
    int tw = 0, th = 0;
    smart_size(img.w, img.h, min_tok, max_tok, tw, th);
    if (tw <= 0 || th <= 0) throw std::runtime_error("nvit::preprocess: bad target size");
    const rgb_image r = resize_pad_ceil(img, tw, th);
    image out;
    out.nx = tw;
    out.ny = th;
    const size_t n = (size_t) tw * th;
    out.data.resize(3 * n);
    const float mean[3] = {0.5f, 0.5f, 0.5f}, stdv[3] = {0.5f, 0.5f, 0.5f};
    for (size_t i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
            float v = (float) r.px[i * 3 + c] / 255.0f;    // from_u8
            v = (v - mean[c]) / stdv[c];                    // normalize
            out.data[c * n + i] = v;                        // planar (clip_encode)
        }
    }
    return out;
}

} // namespace p100e::nvit
