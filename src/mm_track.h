// Image rows of one context: tp_model's M-RoPE bookkeeping (host only, no CUDA; tests/vision_test.cpp).
//
// An image of nx x ny tokens placed at rotary position P takes nx * ny KV rows but advances the rotary position of the
// following text only by max(nx, ny), so text after it is roped at row - delta with delta growing by
// nx * ny - max(nx, ny) per image (tp_model::rope_delta()). Text rows never change the delta. To cut a context back
// (rollback; later the prompt cache) the delta at the cut must be known: it is the delta after the last image that
// stays, so it is enough to keep, per image, its KV rows and the delta of the text after it. A cut strictly inside an
// image is refused: its remaining rows would be a partial image (and the text after it would need a position that no
// llama.cpp prompt has).
//
// The sessions merge moves this per sequence (one mm_track per session, like rope_delta).
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace p100e {

struct mm_track {
    struct span {
        int r0, r1;    // KV rows [r0, r1) of the image
        int delta;     // rope delta of the text after it (KV row - rotary position)
    };
    std::vector<span> spans;   // in row order, non-overlapping

    // rope delta of text after every image recorded so far (0 without images)
    int delta() const { return spans.empty() ? 0 : spans.back().delta; }

    void add(int r0, int r1, int delta) {
        if (r1 <= r0 || r0 < 0 || (!spans.empty() && r0 < spans.back().r1))
            throw std::logic_error("mm_track: image rows " + std::to_string(r0) + ".." + std::to_string(r1) + " out of order");
        spans.push_back({r0, r1, delta});
    }

    // the image that a cut to the first n rows would split (r0 < n < r1), or nullptr
    const span * split_by(int n) const {
        for (const auto & s : spans)
            if (n > s.r0 && n < s.r1) return &s;
        return nullptr;
    }

    // throws (std::runtime_error) when keeping only the first n rows would split an image; changes nothing
    void check_cut(int n) const {
        if (const span * s = split_by(n))
            throw std::runtime_error("cannot cut the context to " + std::to_string(n) + " rows: inside the image at rows " +
                                     std::to_string(s->r0) + ".." + std::to_string(s->r1 - 1));
    }

    // keep the first n rows: images at or after row n are dropped; returns the rope delta of text from row n on.
    // Throws like check_cut (nothing changed then).
    int cut(int n) {
        check_cut(n);
        while (!spans.empty() && spans.back().r0 >= n) spans.pop_back();
        return delta();
    }

    void clear() { spans.clear(); }
};

} // namespace p100e
