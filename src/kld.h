// KL-divergence evaluation against a llama-perplexity base file (--kl-divergence-base), reproducing
// tools/perplexity/perplexity.cpp of the llama.cpp fork (at 98ab7e5) exactly:
//  - file: "_logits_" | uint32 n_ctx | int32 n_vocab | int32 n_chunk | int32 tokens[n_ctx * n_chunk] | per chunk
//    n_scored = n_ctx - 1 - n_ctx/2 rows of nv = 2*((n_vocab+1)/2) + 4 uint16: row = float scale, float min_log_prob
//    (4 uint16), then n_vocab quantized log-probs q (log p = scale*q + min_log_prob), padding to nv
//  - chunk c is evaluated from an empty context (tokens[c*n_ctx .. +n_ctx), the first one replaced by BOS when the
//    vocab adds BOS); rows at positions first = n_ctx/2 .. n_ctx-2 are scored, row i predicting token first+i+1
//  - per-row KLD / NLL / same-top / p_diff and the summary statistics are perplexity.cpp's formulas (host, threads)
// Host only (no CUDA): shared by p100e-run and tests/kld_test.cpp.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace p100e {

struct kld_base_file {
    uint32_t n_ctx = 0;
    int n_vocab = 0, n_chunk = 0;
    int nv = 0;                      // uint16 per row: 2*((n_vocab+1)/2) + 4
    int first = 0;                   // n_ctx / 2
    int n_scored = 0;                // rows per chunk: n_ctx - 1 - first
    std::vector<int32_t> tokens;     // [n_chunk][n_ctx]
    uint64_t data_off = 0;           // byte offset of chunk 0's rows
    uint64_t file_size = 0;

    explicit kld_base_file(const std::string & path);   // reads the header + tokens, checks the size
    ~kld_base_file();
    kld_base_file(const kld_base_file &) = delete;
    kld_base_file & operator=(const kld_base_file &) = delete;
    // chunk c's rows -> out (resized to n_scored * nv). Thread-safe w.r.t. nothing: one reader at a time.
    void read_chunk(int c, std::vector<uint16_t> & out);

private:
    FILE * f_ = nullptr;
};

// perplexity.cpp's kl_divergence_result
struct kld_stats {
    double sum_nll = 0.0, sum_nll2 = 0.0, sum_nll_base = 0.0, sum_nll_base2 = 0.0, sum_nll_nll_base = 0.0;
    double sum_kld = 0.0, sum_kld2 = 0.0, sum_p_diff = 0.0, sum_p_diff2 = 0.0, sum_p_diff4 = 0.0;
    float max_p_diff = 0.0f;
    size_t n_same_top = 0, count = 0;
    void add(const kld_stats & o);
};

// one row: perplexity.cpp's log_softmax(n_vocab, logits, base_log_prob, tok, kld) verbatim; returns (KLD, p_diff)
std::pair<double, float> kld_row(int n_vocab, const float * logits, const uint16_t * base_log_prob, int tok, kld_stats & st);

// rows 0..n_token-1: logits [n_token][n_vocab], base [n_token][nv], tokens[i+1] = the token row i predicts.
// n_threads workers (<= 0: hardware concurrency). kld_values / p_diff_values [n_token] (per-row results).
void kld_rows(int n_vocab, const float * logits, const int32_t * tokens, int n_token, const uint16_t * base,
              kld_stats & st, float * kld_values, float * p_diff_values, int n_threads);

// perplexity.cpp's log lines: the chunk table header, one chunk line, and the final statistics block
void kld_print_header(FILE * out);
void kld_print_chunk(FILE * out, int chunk, const kld_stats & st);
void kld_print_final(FILE * out, const kld_stats & st, std::vector<float> kld_values, std::vector<float> p_diff_values);
// one grep-able line: KLD SUMMARY <tag>: mean KLD +- err, 99% KLD, same top p +- err, PPL(Q), PPL(base)
void kld_print_summary(FILE * out, const char * tag, const kld_stats & st, std::vector<float> kld_values);

// writer side (tests): perplexity.cpp's log_softmax(n_vocab, logits, log_prob, tok) compression of one row into
// nv uint16; returns the row's NLL of tok
double kld_compress_row(int n_vocab, const float * logits, uint16_t * log_prob, int tok);

} // namespace p100e
