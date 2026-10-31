// Tensor-parallel split of Qwen3.8 (qwen35) over R ranks: which part of every split dimension each
// rank holds. tp_model slices its weights, sizes its buffers / state / KV pages and launches its kernels from these
// ranges only, so this header is the one place that knows the rank layout (a later network split, ranks on another
// machine over TCP, takes the same table: every rank's ranges, its weights, the hash that names the split).
//
// Dimensions and their units (a rank always holds whole units, at least one of each dimension):
//  - attention: whole KV-head groups (a KV head and its GQA q heads; attn_q/k/v rows, attn_output columns), unit 1 KV head
//  - delta net: whole k-head groups (a k head, its q / k channels and its HV/HK v heads in every v group; attn_qkv /
//    attn_gate / ssm_beta / ssm_alpha rows, ssm_out columns), unit kh_unit k heads: the ssm_out column split must stay
//    block aligned (q4_K / q6_K: 256-column blocks over DV = 128 -> 2 k heads) and ssm_beta / ssm_alpha (one row per
//    v head) keep n_out % 4 for the prefill GEMM (HV/HK = 3 -> 4 k heads): Qwen3.8-27B moves k heads in fours
//  - FFN: ffn_gate / ffn_up rows and ffn_down columns in blocks of ff_unit (256: one K-quant superblock)
//  - vocab: output.weight rows in blocks of voc_unit (256 when the vocab allows: the fast GEMV / GEMM need n_out % 4)
//  - MTP eh_proj: input columns of [enorm(embed) | hnorm(h)] (2E) in blocks of eh_unit; never weighted (it is small)
// Equal weights and a dimension that divides by R: exactly dim / R per rank, the pre-'tp-uneven' split (R = 1, 2, 4 on
// Qwen3.8-27B: the same slices, kernels and launches as before). Otherwise the units are apportioned by weight: the
// coarse dimensions (KV heads, k-head groups) by largest remainder of units x weight share, then the fine ones (FFN,
// vocab blocks) so that every rank's parameter count lands on its weight share of the total (27B on 3 GPUs, equal
// weights: KV 2/1/1, k heads 4/8/4, FFN 23/20/25 blocks, vocab 400/235/335 blocks = 8.54 G parameters each). The
// balance counts weight parameters only: the KV cache follows the KV heads (2/1/1: GPU 0 holds half of it).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace p100e {

struct tp_range {
    int64_t b = 0, n = 0;   // [b, b + n)
    int64_t e() const { return b + n; }
};

// one rank's share of every split dimension
struct tp_rank_dims {
    tp_range kv;   // KV heads; q heads = kv * GQA (q below)
    tp_range q;    // attention q heads
    tp_range kh;   // delta-net k heads; v heads of v group g: [g * HK + kh.b, + kh.n) (HV / HK groups)
    tp_range ff;   // FFN rows of ffn_gate / ffn_up = input columns of ffn_down
    tp_range voc;  // output.weight rows (token ids)
    tp_range eh;   // MTP eh_proj input columns (0..2E: the embedding half, then the hidden half)
};

// what the split is computed from (tp_model::init fills it from the GGUF)
struct tp_split_in {
    int R = 1;
    std::vector<double> w;          // relative weights per rank (empty: equal)
    int H = 0, HKV = 0, HK = 0, HV = 0, FF = 0, VOCAB = 0, E = 0;
    int kh_unit = 1, ff_unit = 256, voc_unit = 256, eh_unit = 256;
    // parameters per unit, for the tie-break of equal remainders (balance the ranks' weight bytes)
    double c_kv = 1, c_kh = 1, c_ff = 1, c_voc = 1;
};

struct tp_split {
    std::vector<tp_rank_dims> rk;
    std::vector<double> w;          // the weights used (normalised to sum R)
    bool weighted = false;          // explicit unequal weights
    bool even = true;               // every range is dim / R (the pre-'tp-uneven' layout)

    // e.g. "KV heads 2/1/1, k heads 4/8/4, FFN 5888/5120/6400, vocab 102400/60160/85760, eh_proj columns 3584/3328/3328"
    std::string str() const {
        std::string s;
        auto part = [&](const char * name, tp_range tp_rank_dims::*f) {
            s += s.empty() ? "" : ", ";
            s += name;
            for (size_t r = 0; r < rk.size(); ++r) s += (r ? "/" : " ") + std::to_string((long long) (rk[r].*f).n);
        };
        part("KV heads", &tp_rank_dims::kv);
        part("k heads", &tp_rank_dims::kh);
        part("FFN", &tp_rank_dims::ff);
        part("vocab", &tp_rank_dims::voc);
        part("eh_proj columns", &tp_rank_dims::eh);
        return s;
    }
    // identity of the layout (session-file fingerprint, numerics hash): every rank's ranges in order
    uint64_t hash() const {
        uint64_t h = 1469598103934665603ull;
        auto add = [&](int64_t v) { for (int i = 0; i < 8; ++i) { h ^= (uint64_t) (v >> (8 * i)) & 0xff; h *= 1099511628211ull; } };
        add((int64_t) rk.size());
        for (const auto & d : rk)
            for (const tp_range & x : {d.kv, d.q, d.kh, d.ff, d.voc, d.eh}) { add(x.b); add(x.n); }
        return h;
    }
};

// "2,1,1" -> {2, 1, 1}: positive finite numbers, comma separated. Throws with a message naming the bad entry.
inline std::vector<double> tp_parse_split(const std::string & s) {
    std::vector<double> v;
    size_t i = 0;
    while (i <= s.size()) {
        const size_t j = std::min(s.find(',', i), s.size());
        const std::string t = s.substr(i, j - i);
        char * end = nullptr;
        const double x = t.empty() ? 0.0 : strtod(t.c_str(), &end);
        if (t.empty() || !end || *end || !std::isfinite(x) || x <= 0)
            throw std::runtime_error("tensor-parallel split '" + s + "': '" + t + "' is not a positive number (expected "
                                     "relative weights per GPU, e.g. 2,1,1)");
        v.push_back(x);
        i = j + 1;
    }
    return v;
}

inline tp_split tp_make_split(const tp_split_in & in) {
    const int R = in.R;
    if (R < 1) throw std::runtime_error("tensor-parallel split: no ranks");
    std::vector<double> w = in.w;
    if (w.empty()) w.assign(R, 1.0);
    if ((int) w.size() != R)
        throw std::runtime_error("tensor-parallel split: " + std::to_string(w.size()) + " weights for " + std::to_string(R) +
                                 " GPUs (give one weight per GPU, e.g. 2,1,1 for three)");
    double W = 0;
    for (double x : w) {
        if (!(x > 0) || !std::isfinite(x)) throw std::runtime_error("tensor-parallel split: weights must be positive");
        W += x;
    }
    for (double & x : w) x = x * R / W;   // normalised: sum R, 1 each when equal
    bool equal = true;
    for (double x : w) equal = equal && std::fabs(x - 1.0) < 1e-12;
    tp_split sp;
    sp.rk.resize(R);
    sp.w = w;
    sp.weighted = !equal;
    std::vector<double> load(R, 0.0);   // parameters assigned so far per rank (the balance of the fine dimensions)
    std::string wdesc;
    for (int r = 0; r < R; ++r) {
        char b[32];
        snprintf(b, sizeof b, "%s%g", r ? "," : "", in.w.empty() ? 1.0 : in.w[r]);
        wdesc += b;
    }
    // all parameters of the split dimensions, and each rank's target share of them
    const double c_all = in.HKV * in.c_kv + (double) in.HK / std::max(1, in.kh_unit) * in.c_kh +
                         (double) in.FF / std::max(1, in.ff_unit) * in.c_ff + (double) in.VOCAB / std::max(1, in.voc_unit) * in.c_voc;
    // n elements of one dimension in units of `unit` -> per-rank ranges (element offsets).
    //  - coarse (fill = false: KV heads, k-head groups): ideal = units x weight share (largest remainder; ties: the
    //    rank with the least parameters per weight so far, then the lower rank)
    //  - fine (fill = true: FFN, vocab blocks): ideal = what brings each rank to its weight share of all parameters,
    //    given what the coarse dimensions left it and the share of `rest` (parameters of the dimensions after this one),
    //    so the 256-row blocks even out the whole-head steps (27B on 3 GPUs: the GPU with 2 KV heads gets fewer FFN rows)
    auto apportion = [&](const char * what, int64_t n, int64_t unit, double cost, bool weigh, bool fill, double rest,
                         tp_range tp_rank_dims::*f) {
        std::vector<int64_t> cnt(R, 0);
        if ((equal || !weigh) && n % R == 0 && !(fill && !sp.even)) {
            // the even split (bit-identical to the old dim / R layout); a fine dimension after an uneven coarse one
            // still balances (equal shares there would keep the coarse imbalance)
            for (int r = 0; r < R; ++r) cnt[r] = n / R;
            if (weigh) for (int r = 0; r < R; ++r) load[r] += cost * (double) cnt[r] / unit;
        } else {
            sp.even = false;
            if (unit < 1 || n % unit)
                throw std::runtime_error(std::string("tensor-parallel split: ") + what + " " + std::to_string((long long) n) +
                                         " is not a multiple of its split unit " + std::to_string((long long) unit) +
                                         " (no uneven split of this model over " + std::to_string(R) + " GPUs)");
            const int64_t nu = n / unit;
            std::vector<double> ideal(R);
            for (int r = 0; r < R; ++r) ideal[r] = (double) nu * (weigh ? w[r] : 1.0) / R;
            const bool fl = fill && cost > 0 && nu >= R;
            if (fl) {
                // one unit each first (no rank without a share), the other nu - R by deficit to the weight share
                double s = 0;
                for (int r = 0; r < R; ++r) {
                    ideal[r] = std::max(0.0, (c_all * w[r] / R - load[r] - rest * w[r] / R) / cost - 1.0);
                    s += ideal[r];
                }
                for (int r = 0; r < R; ++r) ideal[r] = 1.0 + (s > 0 ? ideal[r] * (double) (nu - R) / s : (double) (nu - R) * w[r] / R);
            }
            int64_t given = 0;
            for (int r = 0; r < R; ++r) {
                cnt[r] = (int64_t) std::floor(ideal[r] + 1e-9);
                given += cnt[r];
            }
            for (; given < nu; ++given) {   // largest remainder; ties: least load per weight, then the lower rank
                int best = -1;
                for (int r = 0; r < R; ++r) {
                    if (best < 0) { best = r; continue; }
                    const double dr = ideal[r] - cnt[r], db = ideal[best] - cnt[best];
                    const double lr = (load[r] + cost * cnt[r]) / (weigh ? w[r] : 1.0);
                    const double lb = (load[best] + cost * cnt[best]) / (weigh ? w[best] : 1.0);
                    if (dr > db + 1e-9) best = r;
                    else if (dr > db - 1e-9 && weigh && lr < lb * (1 - 1e-9)) best = r;
                }
                ++cnt[best];
            }
            if (weigh) for (int r = 0; r < R; ++r) load[r] += cost * (double) cnt[r];
            for (int r = 0; r < R; ++r) cnt[r] *= unit;
        }
        int64_t b = 0;
        for (int r = 0; r < R; ++r) {
            if (cnt[r] < 1)
                throw std::runtime_error(std::string("tensor-parallel split ") + wdesc + ": rank " + std::to_string(r) +
                                         " (the GPU at position " + std::to_string(r) + " of the device list) would get no " +
                                         what + " (" +
                                         std::to_string((long long) (n / std::max<int64_t>(1, unit))) + " units of " +
                                         std::to_string((long long) unit) + " over " + std::to_string(R) +
                                         " GPUs); every GPU needs at least one unit of every dimension: give it more weight "
                                         "or use fewer GPUs");
            (sp.rk[r].*f) = tp_range{b, cnt[r]};
            b += cnt[r];
        }
    };
    if (in.HKV < 1 || in.H % in.HKV) throw std::runtime_error("tensor-parallel split: q heads not a multiple of the KV heads");
    if (in.HK < 1 || in.HV % in.HK) throw std::runtime_error("tensor-parallel split: v heads not a multiple of the k heads");
    const double c_voc_all = (double) in.VOCAB / std::max(1, in.voc_unit) * in.c_voc;
    apportion("KV heads", in.HKV, 1, in.c_kv, true, false, 0, &tp_rank_dims::kv);
    apportion("delta-net k heads", in.HK, in.kh_unit, in.c_kh, true, false, 0, &tp_rank_dims::kh);
    apportion("FFN rows", in.FF, in.ff_unit, in.c_ff, true, true, c_voc_all, &tp_rank_dims::ff);
    apportion("vocab rows", in.VOCAB, in.voc_unit, in.c_voc, true, true, 0, &tp_rank_dims::voc);
    apportion("MTP eh_proj columns", 2LL * in.E, in.eh_unit, 0, false, false, 0, &tp_rank_dims::eh);
    const int gqa = in.H / in.HKV;
    for (auto & d : sp.rk) d.q = tp_range{d.kv.b * gqa, d.kv.n * gqa};
    return sp;
}

} // namespace p100e
