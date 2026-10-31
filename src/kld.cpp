// KL-divergence evaluation against a llama-perplexity base file: see kld.h. The per-row math, the statistics and
// the log formats are copied from the llama.cpp fork's tools/perplexity/perplexity.cpp (at 98ab7e5).
#include "kld.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#define p100e_fseek _fseeki64
#define p100e_ftell _ftelli64
#else
#define p100e_fseek fseeko
#define p100e_ftell ftello
#endif

namespace p100e {

#define PM "\xc2\xb1"   // the plus-minus sign of perplexity.cpp's logs (UTF-8)
#define DP "\xce\x94p"  // "Delta p"

kld_base_file::kld_base_file(const std::string & path) {
    f_ = fopen(path.c_str(), "rb");
    if (!f_) throw std::runtime_error("kld: cannot open " + path);
    char check[8];
    if (fread(check, 1, 8, f_) != 8 || memcmp(check, "_logits_", 8) != 0)
        throw std::runtime_error("kld: " + path + " does not look like a file containing log-probabilities");
    if (fread(&n_ctx, 4, 1, f_) != 1 || fread(&n_vocab, 4, 1, f_) != 1 || fread(&n_chunk, 4, 1, f_) != 1)
        throw std::runtime_error("kld: failed reading n_ctx, n_vocab, n_chunk");
    if (n_ctx < 4 || n_vocab < 2 || n_chunk < 1) throw std::runtime_error("kld: bad header");
    tokens.resize((size_t) n_ctx * n_chunk);
    if (fread(tokens.data(), 4, tokens.size(), f_) != tokens.size()) throw std::runtime_error("kld: failed reading tokens");
    // the ids index the logits and the base rows (kld_row): one outside [0, n_vocab) is a corrupt or foreign file
    for (size_t i = 0; i < tokens.size(); ++i)
        if (tokens[i] < 0 || tokens[i] >= n_vocab)
            throw std::runtime_error("kld: token " + std::to_string(tokens[i]) + " at " + std::to_string(i) +
                                     " outside the vocabulary (n_vocab " + std::to_string(n_vocab) + ")");
    nv = 2 * ((n_vocab + 1) / 2) + 4;
    first = (int) n_ctx / 2;
    n_scored = (int) n_ctx - 1 - first;
    data_off = 20 + (uint64_t) tokens.size() * 4;
    p100e_fseek(f_, 0, SEEK_END);
    file_size = (uint64_t) p100e_ftell(f_);
    const uint64_t want = data_off + (uint64_t) n_chunk * n_scored * nv * 2;
    if (file_size != want)
        throw std::runtime_error("kld: file size " + std::to_string(file_size) + " != expected " + std::to_string(want));
}

kld_base_file::~kld_base_file() { if (f_) fclose(f_); }

void kld_base_file::read_chunk(int c, std::vector<uint16_t> & out) {
    if (c < 0 || c >= n_chunk) throw std::runtime_error("kld: chunk out of range");
    out.resize((size_t) n_scored * nv);
    if (p100e_fseek(f_, (int64_t) (data_off + (uint64_t) c * n_scored * nv * 2), SEEK_SET) != 0)
        throw std::runtime_error("kld: seek failed");
    if (fread(out.data(), 2, out.size(), f_) != out.size()) throw std::runtime_error("kld: failed reading log-probs of chunk " + std::to_string(c));
}

void kld_stats::add(const kld_stats & o) {
    sum_nll += o.sum_nll;
    sum_nll2 += o.sum_nll2;
    sum_nll_base += o.sum_nll_base;
    sum_nll_base2 += o.sum_nll_base2;
    sum_nll_nll_base += o.sum_nll_nll_base;
    sum_kld += o.sum_kld;
    sum_kld2 += o.sum_kld2;
    sum_p_diff += o.sum_p_diff;
    sum_p_diff2 += o.sum_p_diff2;
    sum_p_diff4 += o.sum_p_diff4;
    n_same_top += o.n_same_top;
    max_p_diff = std::max(max_p_diff, o.max_p_diff);
    count += o.count;
}

// perplexity.cpp: static std::pair<double, float> log_softmax(int n_vocab, const float * logits,
// const uint16_t * base_log_prob, int tok, kl_divergence_result & kld)
std::pair<double, float> kld_row(int n_vocab, const float * logits, const uint16_t * base_log_prob, int tok, kld_stats & kld) {
    float max_logit = logits[0];
    int imax = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            imax = i;
        }
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    const float log_sum_exp = log(sum_exp);
    const float * d = (const float *) base_log_prob;
    const float scale = d[0];
    const float min_log_prob = d[1];
    base_log_prob += 4;

    const float nll = max_logit + log_sum_exp - logits[tok];
    kld.sum_nll += nll;
    kld.sum_nll2 += nll * nll;

    const float nll_base = -(scale * base_log_prob[tok] + min_log_prob);
    kld.sum_nll_base += nll_base;
    kld.sum_nll_base2 += nll_base * nll_base;

    kld.sum_nll_nll_base += nll * nll_base;

    max_logit += log_sum_exp;
    double sum = 0;
    int imax_base = -1;
    float p_log_base_max = 0;
    for (int i = 0; i < n_vocab; ++i) {
        const float p_log_base = scale * base_log_prob[i] + min_log_prob;
        if (i == 0 || p_log_base > p_log_base_max) {
            p_log_base_max = p_log_base;
            imax_base = i;
        }
        if (p_log_base > -16.f) {
            const float p_base = expf(p_log_base);
            sum += p_base * (p_log_base - logits[i] + max_logit);
        }
    }
    kld.sum_kld += sum;
    kld.sum_kld2 += sum * sum;
    ++kld.count;
    if (imax == imax_base) {
        ++kld.n_same_top;
    }

    const float p_base = expf(-nll_base);
    const float p = expf(-nll);
    const float p_diff = p - p_base;
    kld.sum_p_diff += p_diff;
    const double p_diff2 = p_diff * p_diff;
    kld.sum_p_diff2 += p_diff2;
    kld.sum_p_diff4 += p_diff2 * p_diff2;
    kld.max_p_diff = std::max(kld.max_p_diff, std::fabs(p_diff));

    return std::make_pair(sum, p_diff);
}

void kld_rows(int n_vocab, const float * logits, const int32_t * tokens, int n_token, const uint16_t * base,
              kld_stats & st, float * kld_values, float * p_diff_values, int n_threads) {
    const int nv = 2 * ((n_vocab + 1) / 2) + 4;
    if (n_threads <= 0) n_threads = (int) std::max(1u, std::thread::hardware_concurrency());
    n_threads = std::max(1, std::min(n_threads, n_token));
    std::atomic<int> counter{0};
    std::mutex mutex;
    auto compute = [&]() {
        kld_stats local;
        for (int i; (i = counter.fetch_add(1)) < n_token;) {
            const std::pair<double, float> v = kld_row(n_vocab, logits + (size_t) i * n_vocab, base + (size_t) i * nv,
                                                       tokens[i + 1], local);
            kld_values[i] = (float) v.first;
            p_diff_values[i] = v.second;
        }
        std::lock_guard<std::mutex> lock(mutex);
        st.add(local);
    };
    std::vector<std::thread> workers;
    for (int t = 1; t < n_threads; ++t) workers.emplace_back(compute);
    compute();
    for (auto & w : workers) w.join();
}

static std::pair<double, double> mean_and_uncertainty(double sum, double sum2, size_t count) {
    if (count < 1) return std::make_pair(0., 0.);
    double f = sum / count;
    double df = sum2 / count - f * f;
    df = df > 0 && count > 10 ? sqrt(df / (count - 1)) : 0.;
    return std::make_pair(f, df);
}

static double covariance(double suma, double sumb, double sumab, size_t count) {
    if (count < 10) return 0.0;
    double var = sumab / count - (suma / count) * (sumb / count);
    var /= count - 1;
    return var;
}

void kld_print_header(FILE * out) {
    fprintf(out, "chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              " DP " RMS            Same top p\n");
}

void kld_print_chunk(FILE * out, int chunk, const kld_stats & kld) {
    fprintf(out, "%4d", chunk);
    auto log_ppl = mean_and_uncertainty(kld.sum_nll, kld.sum_nll2, kld.count);
    const double ppl_val = exp(log_ppl.first);
    const double ppl_unc = ppl_val * log_ppl.second;
    fprintf(out, "    %9.4lf " PM " %9.4lf", ppl_val, ppl_unc);

    auto log_ppl_base = mean_and_uncertainty(kld.sum_nll_base, kld.sum_nll_base2, kld.count);
    const double log_ppl_cov = covariance(kld.sum_nll, kld.sum_nll_base, kld.sum_nll_nll_base, kld.count);
    const double log_ppl_ratio_val = log_ppl.first - log_ppl_base.first;
    const double log_ppl_ratio_unc = sqrt(log_ppl.second * log_ppl.second + log_ppl_base.second * log_ppl_base.second - 2.0 * log_ppl_cov);
    fprintf(out, "    %10.5lf " PM " %10.5lf", log_ppl_ratio_val, log_ppl_ratio_unc);

    auto kl_div = mean_and_uncertainty(kld.sum_kld, kld.sum_kld2, kld.count);
    fprintf(out, "    %10.5lf " PM " %10.5lf", kl_div.first, kl_div.second);

    auto p_diff_mse = mean_and_uncertainty(kld.sum_p_diff2, kld.sum_p_diff4, kld.count);
    const double p_diff_rms_val = sqrt(p_diff_mse.first);
    const double p_diff_rms_unc = 0.5 / p_diff_rms_val * p_diff_mse.second;
    fprintf(out, "    %6.3lf " PM " %6.3lf %%", 100.0 * p_diff_rms_val, 100.0 * p_diff_rms_unc);

    double p_top_val = 1. * kld.n_same_top / kld.count;
    double p_top_unc = sqrt(p_top_val * (1 - p_top_val) / (kld.count - 1));
    fprintf(out, "    %6.3lf " PM " %6.3lf %%", 100.0 * p_top_val, 100.0 * p_top_unc);
    fprintf(out, "\n");
}

void kld_print_final(FILE * out, const kld_stats & kld, std::vector<float> kld_values, std::vector<float> p_diff_values) {
    if (kld.count < 100) return;   // perplexity.cpp: no statistics on so few values
    std::sort(kld_values.begin(), kld_values.end());
    std::sort(p_diff_values.begin(), p_diff_values.end());

    fprintf(out, "====== Perplexity statistics ======\n");
    auto log_ppl = mean_and_uncertainty(kld.sum_nll, kld.sum_nll2, kld.count);
    const double ppl_val = exp(log_ppl.first);
    const double ppl_unc = ppl_val * log_ppl.second;
    fprintf(out, "Mean PPL(Q)                   : %10.6lf " PM " %10.6lf\n", ppl_val, ppl_unc);
    auto log_ppl_base = mean_and_uncertainty(kld.sum_nll_base, kld.sum_nll_base2, kld.count);
    const double ppl_base_val = exp(log_ppl_base.first);
    const double ppl_base_unc = ppl_base_val * log_ppl_base.second;
    fprintf(out, "Mean PPL(base)                : %10.6lf " PM " %10.6lf\n", ppl_base_val, ppl_base_unc);
    const double log_ppl_cov = covariance(kld.sum_nll, kld.sum_nll_base, kld.sum_nll_nll_base, kld.count);
    const double log_ppl_cor = log_ppl_cov / (log_ppl.second * log_ppl_base.second);
    fprintf(out, "Cor(ln(PPL(Q)), ln(PPL(base))): %6.2lf%%\n", 100.0 * log_ppl_cor);
    const double log_ppl_ratio_val = log_ppl.first - log_ppl_base.first;
    const double log_ppl_ratio_unc = sqrt(log_ppl.second * log_ppl.second + log_ppl_base.second * log_ppl_base.second - 2.0 * log_ppl_cov);
    fprintf(out, "Mean ln(PPL(Q)/PPL(base))     : %10.6lf " PM " %10.6lf\n", log_ppl_ratio_val, log_ppl_ratio_unc);
    const double ppl_ratio_val = exp(log_ppl_ratio_val);
    const double ppl_ratio_unc = ppl_ratio_val * log_ppl_ratio_unc;
    fprintf(out, "Mean PPL(Q)/PPL(base)         : %10.6lf " PM " %10.6lf\n", ppl_ratio_val, ppl_ratio_unc);
    const double ppl_cov = ppl_val * ppl_base_val * log_ppl_cov;
    const double ppl_diff_val = ppl_val - ppl_base_val;
    const double ppl_diff_unc = sqrt(ppl_unc * ppl_unc + ppl_base_unc * ppl_base_unc - 2.0 * ppl_cov);
    fprintf(out, "Mean PPL(Q)-PPL(base)         : %10.6lf " PM " %10.6lf\n", ppl_diff_val, ppl_diff_unc);
    fprintf(out, "\n");

    fprintf(out, "====== KL divergence statistics ======\n");
    auto kl_div = mean_and_uncertainty(kld.sum_kld, kld.sum_kld2, kld.count);
    fprintf(out, "Mean    KLD: %10.6lf " PM " %10.6lf\n", kl_div.first, kl_div.second);
    auto kld_median = kld_values.size() % 2 == 0 ? 0.5f * (kld_values[kld_values.size() / 2] + kld_values[kld_values.size() / 2 - 1])
                                                 : kld_values[kld_values.size() / 2];
    auto percentile = [](const std::vector<float> & values, float fraction) {
        if (fraction <= 0) return values.front();
        if (fraction >= 1) return values.back();
        float p = fraction * (values.size() - 1);
        size_t ip = size_t(p);
        p -= ip;
        return (1 - p) * values[ip] + p * values[std::min(ip + 1, values.size() - 1)];
    };
    fprintf(out, "Maximum KLD: %10.6f\n", kld_values.back());
    fprintf(out, "99.9%%   KLD: %10.6f\n", percentile(kld_values, 0.999f));
    fprintf(out, "99.0%%   KLD: %10.6f\n", percentile(kld_values, 0.990f));
    fprintf(out, "95.0%%   KLD: %10.6f\n", percentile(kld_values, 0.950f));
    fprintf(out, "90.0%%   KLD: %10.6f\n", percentile(kld_values, 0.900f));
    fprintf(out, "Median  KLD: %10.6f\n", kld_median);
    fprintf(out, "10.0%%   KLD: %10.6f\n", percentile(kld_values, 0.100f));
    fprintf(out, " 5.0%%   KLD: %10.6f\n", percentile(kld_values, 0.050f));
    fprintf(out, " 1.0%%   KLD: %10.6f\n", percentile(kld_values, 0.010f));
    fprintf(out, " 0.1%%   KLD: %10.6f\n", percentile(kld_values, 0.001f));
    fprintf(out, "Minimum KLD: %10.6f\n", kld_values.front());
    fprintf(out, "\n");

    fprintf(out, "====== Token probability statistics ======\n");
    auto p_diff = mean_and_uncertainty(kld.sum_p_diff, kld.sum_p_diff2, kld.count);
    fprintf(out, "Mean    " DP ": %6.3lf " PM " %5.3lf %%\n", 100.0 * p_diff.first, 100.0 * p_diff.second);
    auto p_diff_median = p_diff_values.size() % 2 == 0 ? 0.5f * (p_diff_values[p_diff_values.size() / 2] + p_diff_values[p_diff_values.size() / 2 - 1])
                                                       : p_diff_values[p_diff_values.size() / 2];
    fprintf(out, "Maximum " DP ": %6.3lf%%\n", 100.0 * p_diff_values.back());
    fprintf(out, "99.9%%   " DP ": %6.3lf%%\n", 100.0 * percentile(p_diff_values, 0.999f));
    fprintf(out, "99.0%%   " DP ": %6.3lf%%\n", 100.0 * percentile(p_diff_values, 0.990f));
    fprintf(out, "95.0%%   " DP ": %6.3lf%%\n", 100.0 * percentile(p_diff_values, 0.950f));
    fprintf(out, "90.0%%   " DP ": %6.3lf%%\n", 100.0 * percentile(p_diff_values, 0.900f));
    fprintf(out, "75.0%%   " DP ": %6.3lf%%\n", 100.0 * percentile(p_diff_values, 0.750f));
    fprintf(out, "Median  " DP ": %6.3lf%%\n", 100.0 * p_diff_median);
    fprintf(out, "25.0%%   " DP ": %6.3lf%%\n", 100.0 * percentile(p_diff_values, 0.250f));
    fprintf(out, "10.0%%   " DP ": %6.3lf%%\n", 100.0 * percentile(p_diff_values, 0.100f));
    fprintf(out, " 5.0%%   " DP ": %6.3lf%%\n", 100.0 * percentile(p_diff_values, 0.050f));
    fprintf(out, " 1.0%%   " DP ": %6.3lf%%\n", 100.0 * percentile(p_diff_values, 0.010f));
    fprintf(out, " 0.1%%   " DP ": %6.3lf%%\n", 100.0 * percentile(p_diff_values, 0.001f));
    fprintf(out, "Minimum " DP ": %6.3lf%%\n", 100.0 * p_diff_values.front());
    auto p_diff_mse = mean_and_uncertainty(kld.sum_p_diff2, kld.sum_p_diff4, kld.count);
    const double p_diff_rms_val = sqrt(p_diff_mse.first);
    const double p_diff_rms_unc = 0.5 / p_diff_rms_val * p_diff_mse.second;
    fprintf(out, "RMS " DP "    : %6.3lf " PM " %5.3lf %%\n", 100.0 * p_diff_rms_val, 100.0 * p_diff_rms_unc);
    const double same_top_p = 1.0 * kld.n_same_top / kld.count;
    fprintf(out, "Same top p: %6.3lf " PM " %5.3lf %%\n", 100.0 * same_top_p, 100.0 * sqrt(same_top_p * (1.0 - same_top_p) / (kld.count - 1)));
}

void kld_print_summary(FILE * out, const char * tag, const kld_stats & kld, std::vector<float> kld_values) {
    if (kld.count < 2 || kld_values.empty()) return;
    std::sort(kld_values.begin(), kld_values.end());
    float p = 0.990f * (kld_values.size() - 1);   // perplexity.cpp's percentile(kld_values, 0.990f)
    const size_t ip = size_t(p);
    p -= ip;
    const float p99 = (1 - p) * kld_values[ip] + p * kld_values[std::min(ip + 1, kld_values.size() - 1)];
    auto kl_div = mean_and_uncertainty(kld.sum_kld, kld.sum_kld2, kld.count);
    auto log_ppl = mean_and_uncertainty(kld.sum_nll, kld.sum_nll2, kld.count);
    auto log_ppl_base = mean_and_uncertainty(kld.sum_nll_base, kld.sum_nll_base2, kld.count);
    const double top = 1.0 * kld.n_same_top / kld.count;
    fprintf(out, "KLD SUMMARY %s: rows %zu | Mean KLD %.6f +- %.6f | 99%% KLD %.6f | Same top p %.3f +- %.3f %% | "
            "PPL(Q) %.6f +- %.6f | PPL(base) %.6f +- %.6f\n", tag, kld.count, kl_div.first, kl_div.second, p99,
            100.0 * top, 100.0 * sqrt(top * (1.0 - top) / (kld.count - 1)), exp(log_ppl.first),
            exp(log_ppl.first) * log_ppl.second, exp(log_ppl_base.first), exp(log_ppl_base.first) * log_ppl_base.second);
}

// perplexity.cpp: nearest_int + static double log_softmax(int n_vocab, const float * logits, uint16_t * log_prob, int tok)
static inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

double kld_compress_row(int n_vocab, const float * logits, uint16_t * log_prob, int tok) {
    float max_logit = logits[0];
    float min_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
        min_logit = std::min(min_logit, logits[i]);
    }
    min_logit = std::max(min_logit, max_logit - 16);
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    const float log_sum_exp = log(sum_exp);
    const float min_log_prob = min_logit - max_logit - log_sum_exp;
    const float scale = (max_logit - min_logit) / 65535.f;
    float * d = (float *) log_prob;
    d[0] = scale;
    d[1] = min_log_prob;
    log_prob += 4;
    if (scale) {
        const float inv_scale = 1 / scale;
        for (int i = 0; i < n_vocab; ++i) {
            log_prob[i] = logits[i] > min_logit ? nearest_int(inv_scale * (logits[i] - min_logit)) : 0;
        }
    } else {
        std::memset(log_prob, 0, n_vocab * sizeof(uint16_t));
    }
    return max_logit + log_sum_exp - logits[tok];
}

} // namespace p100e
