// p100e-server sampler: llama.cpp's common_sampler (fork common/sampling.cpp @ 98ab7e5) on host logit rows.
//
// common_sampler reads logits from a llama_context (llama_get_logits_ith); p100e has none, so this is a port of the
// same object with set_logits() taking a float row: the same chain built from the same llama_sampler_* objects (logit
// bias + suppress tokens, penalties / dry / top_n_sigma / top_k / typ_p / top_p / min_p / xtc / temperature, dist or
// mirostat), the same lazy grammar (tool calls) and reasoning-budget sampler, the same top-k prefilter and the same
// grammar rejection-resampling. sample_and_accept_n() is common_sampler_sample_and_accept_n (sampling.cpp:851-879):
// the MTP verify rule that keeps the output distribution exact.
#pragma once

#include "common.h"
#include "llama.h"

#include <string>
#include <vector>

struct srv_sampler;

// note: can mutate params (backend_sampling is cleared), like common_sampler_init
srv_sampler * srv_sampler_init(const llama_model * model, common_params_sampling & params);
void srv_sampler_free(srv_sampler * s);

void srv_sampler_accept(srv_sampler * s, llama_token token, bool is_generated);
void srv_sampler_accept_prompt(srv_sampler * s, const llama_tokens & prompt);
void srv_sampler_reset(srv_sampler * s);

// sample from one row of n_vocab logits (common_sampler_sample)
llama_token srv_sampler_sample(srv_sampler * s, const float * logits, bool grammar_first = false);

// rows = draft.size() + 1 rows of n_vocab logits: sample row i, accept it, stop at the first sample that differs from
// draft[i] (it is still returned). Returns 1 .. draft.size() + 1 tokens, all accepted into the sampler.
std::vector<llama_token> srv_sampler_sample_and_accept_n(srv_sampler * s, const float * rows, const llama_tokens & draft,
                                                          bool grammar_first = false);

// true when the next sample() is the argmax of the raw logits whatever they are (temperature <= 0 and no sampler that
// changes or reorders logits before it, no active grammar, no forced reasoning-budget token): the caller may take the
// GPU argmax instead of copying the logits. accept() must still be called with the token.
bool srv_sampler_is_greedy(const srv_sampler * s);

bool srv_sampler_reasoning_budget_force(srv_sampler * s);

// candidates of the last sample() (sorted by probability when do_sort)
llama_token_data_array * srv_sampler_get_candidates(srv_sampler * s, bool do_sort);

std::string srv_sampler_print(const srv_sampler * s);

// the model's recommended sampling defaults from GGUF metadata (general.sampling.*), unless set on the command line
// (common.cpp common_init_sampler_from_model, static there)
void srv_init_sampler_from_model(const llama_model * model, common_params_sampling & sparams);

struct srv_sampler_deleter {
    void operator()(srv_sampler * s) { srv_sampler_free(s); }
};
typedef std::unique_ptr<srv_sampler, srv_sampler_deleter> srv_sampler_ptr;
