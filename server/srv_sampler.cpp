// p100e-server sampler: port of the fork's common/sampling.cpp (98ab7e5, MIT) onto host logit rows. Kept line for
// line where it decides anything (chain construction, prefilter, grammar / reasoning-budget handling, rejection
// resampling, sample-and-accept), so a request samples exactly as it does in llama-server; only the logits
// source differs (a float row instead of llama_get_logits_ith; no backend sampling).
#include "srv_sampler.h"

#include "log.h"
#include "reasoning-budget.h"
#include "sampling.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>

// ---- prefilter (sampling.cpp:112-249) ----

static void srv_prefilter(const float * logits, int n_vocab, int nkeep, std::vector<int32_t> & out) {
    const int cap = 4*nkeep;

    out.clear();
    out.reserve(cap);

    float thr = -INFINITY;

    const auto probe = [&](int i) {
        if (logits[i] > thr) {
            out.push_back(i);

            if ((int) out.size() == cap) {
                std::nth_element(out.begin(), out.begin() + nkeep - 1, out.end(),
                        [logits](int32_t a, int32_t b) { return logits[a] > logits[b]; });
                out.resize(nkeep);
                thr = logits[out[nkeep - 1]];
            }
        }
    };

    int i0 = 0;
    for (; i0 + 64 <= n_vocab; i0 += 64) {
        int any = 0;
        for (int j = 0; j < 64; ++j) {
            any |= (int) (logits[i0 + j] > thr);
        }
        if (!any) {
            continue;
        }
        for (int j = 0; j < 64; ++j) {
            probe(i0 + j);
        }
    }
    for (; i0 < n_vocab; ++i0) {
        probe(i0);
    }
}

static int srv_prefilter_nkeep(const common_params_sampling & params, const llama_vocab * vocab) {
    static const bool enabled = getenv("LLAMA_SAMPLER_PREFILTER")
        ? atoi(getenv("LLAMA_SAMPLER_PREFILTER")) != 0 : true;

    const int n_vocab = llama_vocab_n_tokens(vocab);

    int32_t n_suppress = 0;
    llama_vocab_get_suppress_tokens(vocab, &n_suppress);

    if (!enabled || params.mirostat != 0 || params.top_k <= 0 ||
        !params.logit_bias.empty() || n_suppress > 0) {
        return 0;
    }

    const bool penalties_off =
        params.penalty_last_n == 0 ||
        (params.penalty_repeat == 1.0f && params.penalty_freq == 0.0f && params.penalty_present == 0.0f);

    if (!penalties_off &&
        (params.penalty_last_n < 0 || params.penalty_repeat < 1.0f ||
         params.penalty_freq < 0.0f || params.penalty_present < 0.0f)) {
        return 0;
    }

    const bool dry_off = params.dry_multiplier == 0.0f || params.dry_base < 1.0f || params.dry_penalty_last_n == 0;

    bool seen_top_k = false;
    for (const auto & cnstr : params.samplers) {
        if (cnstr == COMMON_SAMPLER_TYPE_TOP_K) {
            seen_top_k = true;
            break;
        }
        switch (cnstr) {
            case COMMON_SAMPLER_TYPE_PENALTIES:                              break;
            case COMMON_SAMPLER_TYPE_DRY:         if (!dry_off) return 0;     break;
            case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: if (params.top_n_sigma > 0.0f) return 0; break;
            default: return 0;
        }
    }

    if (!seen_top_k) {
        return 0;
    }

    const int margin = penalties_off ? 0 : params.penalty_last_n;
    const int nkeep  = std::min<int64_t>(n_vocab, (int64_t) params.top_k + margin);

    if (nkeep <= 0 || 8*nkeep >= n_vocab) {
        return 0;
    }

    return nkeep;
}

// the next sample is the raw argmax: temperature <= 0 (the temperature sampler keeps only the first maximum) and
// nothing before it in the chain can change, drop or reorder the maximum
static bool srv_greedy_chain(const common_params_sampling & params, const llama_vocab * vocab) {
    int32_t n_suppress = 0;
    llama_vocab_get_suppress_tokens(vocab, &n_suppress);
    if (params.temp > 0.0f || params.dynatemp_range > 0.0f || params.mirostat != 0) return false;
    if (!params.logit_bias.empty() || n_suppress > 0 || params.n_probs > 0) return false;
    bool has_temp = false;
    for (const auto & c : params.samplers) {
        switch (c) {
            case COMMON_SAMPLER_TYPE_PENALTIES:
                if (params.penalty_last_n != 0 && (params.penalty_repeat != 1.0f || params.penalty_freq != 0.0f ||
                                                   params.penalty_present != 0.0f)) return false;
                break;
            case COMMON_SAMPLER_TYPE_DRY:
                if (params.dry_multiplier != 0.0f && params.dry_base >= 1.0f && params.dry_penalty_last_n != 0) return false;
                break;
            case COMMON_SAMPLER_TYPE_TYPICAL_P:   if (params.typ_p < 1.0f) return false; break;
            case COMMON_SAMPLER_TYPE_XTC:         if (params.xtc_probability > 0.0f) return false; break;
            case COMMON_SAMPLER_TYPE_TEMPERATURE: has_temp = true; break;
            case COMMON_SAMPLER_TYPE_TOP_K:
            case COMMON_SAMPLER_TYPE_TOP_P:
            case COMMON_SAMPLER_TYPE_MIN_P:
            case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: break;   // keep the maximum
            default: return false;                         // infill, adaptive-p, ...
        }
    }
    return has_temp;
}

struct srv_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;

    const llama_vocab * vocab;
    int n_vocab;

    std::vector<llama_token_data> cur;
    llama_token_data_array cur_p;

    int pf_nkeep;
    std::vector<int32_t> pf_idx;
    bool greedy_chain;
    int prompt_n;   // max prompt tail tokens needed by active history samplers (0 = none, -1 = all)

    void reset() {
        llama_sampler_reset(chain);
    }

    void set_logits(const float * logits, bool prefilter = false) {
        if (prefilter && pf_nkeep > 0 && pf_nkeep < n_vocab) {
            srv_prefilter(logits, n_vocab, pf_nkeep, pf_idx);
        } else {
            pf_idx.clear();
        }

        if (!pf_idx.empty()) {
            cur.resize(pf_idx.size());
            for (size_t i = 0; i < pf_idx.size(); ++i) {
                const llama_token token_id = pf_idx[i];
                cur[i] = llama_token_data{token_id, logits[token_id], 0.0f};
            }
        } else {
            cur.resize(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }
};

srv_sampler * srv_sampler_init(const llama_model * model, common_params_sampling & params) {
    if (!std::isfinite(params.penalty_repeat) ||
        params.penalty_repeat <= 0.0f ||
        !std::isfinite(1.0f/params.penalty_repeat)) {
        throw std::invalid_argument("penalty_repeat must be finite and greater than 0");
    }
    if (!std::isfinite(params.penalty_freq)) {
        throw std::invalid_argument("penalty_freq must be finite");
    }
    if (!std::isfinite(params.penalty_present)) {
        throw std::invalid_argument("penalty_present must be finite");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparams);

    std::vector<llama_sampler *> samplers;

    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
        throw std::invalid_argument("llguidance grammars are not supported");
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    trigger_patterns.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
             if (params.grammar_lazy) {
                 grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                         trigger_patterns_c.data(), trigger_patterns_c.size(),
                         trigger_tokens.data(), trigger_tokens.size());
             } else {
                 grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
             }
        }
    }
    if (!grmr && !grammar_str.empty()) {
        throw std::runtime_error("failed to parse grammar");
    }

    // tokens of the generation prompt the template already placed in the prompt
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && std::isspace(piece[0]) && !std::isspace(params.generation_prompt[0])) {
                continue;
            }
            prefill_tokens.push_back(tokens[i]);
        }
    }

    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(grmr, token);
        }
    }

    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0 || params.reasoning_control)) {
        rbudget = common_reasoning_budget_init(
            vocab,
            {params.reasoning_budget_start},
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(rbudget, token);
        }
    }

    {
        std::vector<llama_logit_bias> merged = params.logit_bias;

        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        for (int32_t i = 0; i < n_suppress; ++i) {
            merged.push_back({ suppress[i], -INFINITY });
        }

        if (!merged.empty()) {
            samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), merged.size(), merged.data()));
        }
    }

    if (params.mirostat == 0) {
        bool use_adaptive_p = false;

        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }
                        samplers.push_back(llama_sampler_init_dry(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    samplers.push_back(llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    use_adaptive_p = true;
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        if (use_adaptive_p) {
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            samplers.push_back(llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    params.backend_sampling = false;   // the logits are always on the host here

    int prompt_n = 0;
    if (params.mirostat == 0) {
        for (const auto & cnstr : params.samplers) {
            if (cnstr == COMMON_SAMPLER_TYPE_PENALTIES) {
                const bool active = params.penalty_last_n != 0 &&
                    (params.penalty_repeat != 1.0f || params.penalty_freq != 0.0f || params.penalty_present != 0.0f);
                if (active) {
                    prompt_n = (params.penalty_last_n < 0 || prompt_n < 0) ? -1 : std::max<int>(prompt_n, params.penalty_last_n);
                }
            } else if (cnstr == COMMON_SAMPLER_TYPE_DRY) {
                const bool active = params.dry_multiplier != 0.0f && params.dry_base >= 1.0f && params.dry_penalty_last_n != 0;
                if (active) {
                    prompt_n = (params.dry_penalty_last_n < 0 || prompt_n < 0) ? -1 : std::max<int>(prompt_n, params.dry_penalty_last_n);
                }
            }
        }
    }

    auto * result = new srv_sampler {
        /* .params       = */ params,
        /* .grmr         = */ grmr,
        /* .rbudget      = */ rbudget,
        /* .chain        = */ chain,
        /* .vocab        = */ vocab,
        /* .n_vocab      = */ llama_vocab_n_tokens(vocab),
        /* .cur          = */ {},
        /* .cur_p        = */ {},
        /* .pf_nkeep     = */ srv_prefilter_nkeep(params, vocab),
        /* .pf_idx       = */ {},
        /* .greedy_chain = */ srv_greedy_chain(params, vocab),
        /* .prompt_n     = */ prompt_n,
    };

    return result;
}

void srv_sampler_free(srv_sampler * s) {
    if (!s) {
        return;
    }
    llama_sampler_free(s->grmr);
    llama_sampler_free(s->rbudget);
    llama_sampler_free(s->chain);
    delete s;
}

static bool grammar_should_apply(const srv_sampler * s) {
    if (!s->grmr) {
        return false;
    }
    if (!s->rbudget) {
        return true;
    }
    if (s->params.grammar_lazy) {
        const auto state = common_reasoning_budget_get_state(s->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

void srv_sampler_accept(srv_sampler * s, llama_token token, bool is_generated) {
    if (!s) {
        return;
    }

    const auto accept_grammar = is_generated && grammar_should_apply(s);

    if (s->rbudget && is_generated) {
        llama_sampler_accept(s->rbudget, token);

        const bool is_done = common_reasoning_budget_get_state(s->rbudget) == REASONING_BUDGET_DONE;
        if (s->grmr && !accept_grammar && is_done) {
            const llama_tokens * end_seq = common_reasoning_budget_get_end_match(s->rbudget);
            if (end_seq) {
                for (const llama_token end_token : *end_seq) {
                    llama_sampler_accept(s->grmr, end_token);
                }
            }
        }
    }

    if (s->grmr && accept_grammar) {
        llama_sampler_accept(s->grmr, token);
    }

    llama_sampler_accept(s->chain, token);
}

void srv_sampler_accept_prompt(srv_sampler * s, const llama_tokens & prompt) {
    if (!s || s->prompt_n == 0 || prompt.empty()) {
        return;
    }
    const size_t n = s->prompt_n < 0 ? prompt.size() : std::min<size_t>(prompt.size(), (size_t) s->prompt_n);
    for (size_t i = prompt.size() - n; i < prompt.size(); ++i) {
        llama_sampler_accept(s->chain, prompt[i]);
    }
}

void srv_sampler_reset(srv_sampler * s) {
    if (s) {
        s->reset();
    }
}

llama_token srv_sampler_sample(srv_sampler * s, const float * logits, bool grammar_first) {
    llama_token id = LLAMA_TOKEN_NULL;

    auto & grmr    = s->grmr;
    auto & rbudget = s->rbudget;
    auto & chain   = s->chain;
    auto & cur_p   = s->cur_p;

    const bool prefilter = !grammar_should_apply(s) &&
        (!rbudget || common_reasoning_budget_get_state(rbudget) != REASONING_BUDGET_FORCING);

    s->set_logits(logits, prefilter);

    llama_sampler_apply(rbudget, &cur_p);

    if (grammar_first && grammar_should_apply(s)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !grammar_should_apply(s)) {
        return id;
    }

    // grammar-based rejection sampling: check the sampled token alone first
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resample with the grammar applied first
    s->set_logits(logits);

    llama_sampler_apply(rbudget, &cur_p);

    if (grammar_should_apply(s)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    id = cur_p.data[cur_p.selected].id;

    return id;
}

std::vector<llama_token> srv_sampler_sample_and_accept_n(srv_sampler * s, const float * rows, const llama_tokens & draft,
                                                          bool grammar_first) {
    std::vector<llama_token> result;
    result.reserve(draft.size() + 1);

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = srv_sampler_sample(s, rows + i * (size_t) s->n_vocab, grammar_first);

        srv_sampler_accept(s, id, true);

        result.push_back(id);

        if (draft[i] != id) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = srv_sampler_sample(s, rows + i * (size_t) s->n_vocab, grammar_first);

        srv_sampler_accept(s, id, true);

        result.push_back(id);
    }

    return result;
}

bool srv_sampler_is_greedy(const srv_sampler * s) {
    if (!s->greedy_chain) return false;
    if (grammar_should_apply(s)) return false;
    // a reasoning budget that is forcing its end sequence picks a token that need not be the maximum
    if (s->rbudget) {
        const auto st = common_reasoning_budget_get_state(s->rbudget);
        if (st == REASONING_BUDGET_FORCING || st == REASONING_BUDGET_WAITING_UTF8) return false;
    }
    return true;
}

bool srv_sampler_reasoning_budget_force(srv_sampler * s) {
    if (!s) {
        return false;
    }
    return common_reasoning_budget_force(s->rbudget);
}

llama_token_data_array * srv_sampler_get_candidates(srv_sampler * s, bool do_sort) {
    auto * res = &s->cur_p;

    if (do_sort && !res->sorted) {
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
        });

        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

std::string srv_sampler_print(const srv_sampler * s) {
    std::string result = "logits ";
    for (int i = 0; i < llama_sampler_chain_n(s->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(s->chain, i);
        result += std::string("-> ");
        result += std::string(llama_sampler_name(smpl)) + " ";
    }
    return result;
}

void srv_init_sampler_from_model(const llama_model * model, common_params_sampling & sparams) {
    const uint64_t config = sparams.user_sampling_config;

    auto get_int32 = [&](const char * key, int32_t & dst, uint64_t user_config) {
        if (config & user_config) {
            return;
        }

        char buf[64] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char * end = nullptr;
            int32_t v = strtol(buf, &end, 10);
            if (end && end != buf) {
                dst = v;
            }
        }
    };

    auto get_float = [&](const char * key, float & dst, uint64_t user_config) {
        if (config & user_config) {
            return;
        }

        char buf[128] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char * end = nullptr;
            float v = strtof(buf, &end);
            if (end && end != buf) {
                dst = v;
            }
        }
    };

    if (!(config & common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS)) {
        char buf[512] = {0};
        if (llama_model_meta_val_str(model, llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE), buf, sizeof(buf)) > 0) {
            const std::vector<std::string> sampler_names = string_split<std::string>(std::string(buf), ';');
            if (!sampler_names.empty()) {
                sparams.samplers = common_sampler_types_from_names(sampler_names);
            }
        }
    }

    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_K),           sparams.top_k,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_P),           sparams.top_p,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_P);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIN_P),           sparams.min_p,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIN_P);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY), sparams.xtc_probability, common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD),   sparams.xtc_threshold,   common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TEMP),            sparams.temp,            common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TEMP);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N),  sparams.penalty_last_n,  common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT),  sparams.penalty_repeat,  common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT),        sparams.mirostat,        common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU),    sparams.mirostat_tau,    common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA),    sparams.mirostat_eta,    common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA);
}
