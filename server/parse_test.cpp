// p100e-server --p100e-parse-test cases.json: parser goldens without a GPU.
//
// Each case is a request body (OAI chat, or Anthropic with "api": "messages") plus a raw model output (text, or token
// ids as the engine emitted them). The case runs through exactly what a completion runs through in the server, minus
// the engine: oaicompat_chat_params_parse (template, tools, grammar, parser selection) -> server_schema task params ->
// task_result_state::update_chat_msg (the fork's PEG tool-call / reasoning parser). Checked per case:
//   - final parse (is_partial = false) against the expected message (content, reasoning_content, tool call names and
//     arguments; argument values compared as JSON, or only the key sets with "keys_only")
//   - streaming consistency: the same raw output fed token by token (is_partial = true, then a final empty update)
//     must aggregate to the same message through the OAI diffs the server streams
#include "server-chat.h"
#include "server-common.h"
#include "server-schema.h"
#include "server-task.h"

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

struct agg_msg {
    std::string content, reasoning;
    std::map<size_t, std::pair<std::string, std::string>> tools;   // index -> (name, arguments)
};

void apply_diffs(agg_msg & a, const std::vector<common_chat_msg_diff> & diffs) {
    for (const auto & d : diffs) {
        a.content += d.content_delta;
        a.reasoning += d.reasoning_content_delta;
        if (d.tool_call_index != std::string::npos) {
            auto & t = a.tools[d.tool_call_index];
            t.first += d.tool_call_delta.name;
            t.second += d.tool_call_delta.arguments;
        }
    }
}

json parse_args(const std::string & s) {
    try {
        return json::parse(s);
    } catch (...) {
        return json(s);
    }
}

// JSON equality with object keys in any order (json is nlohmann::ordered_json, whose == compares key order too)
bool json_eq(const json & a, const json & b) {
    if (a.is_object() && b.is_object()) {
        if (a.size() != b.size()) return false;
        for (auto it = a.begin(); it != a.end(); ++it) {
            if (!b.contains(it.key()) || !json_eq(it.value(), b.at(it.key()))) return false;
        }
        return true;
    }
    if (a.is_array() && b.is_array()) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) if (!json_eq(a[i], b[i])) return false;
        return true;
    }
    return a == b;
}

std::vector<std::string> keys_of(const json & j) {
    std::vector<std::string> k;
    if (j.is_object()) for (auto it = j.begin(); it != j.end(); ++it) k.push_back(it.key());
    std::sort(k.begin(), k.end());
    return k;
}

} // namespace

int p100e_parse_test(common_params & params, const std::string & file) {
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "PARSE_TEST: cannot load the vocabulary of %s\n", params.model.path.c_str());
        return 2;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto tmpls = common_chat_templates_init(model, params.chat_template);
    const bool enable_thinking = params.use_jinja && params.enable_reasoning != 0 &&
                                 common_chat_templates_support_enable_thinking(tmpls.get());
    server_chat_params cp = {
        params.use_jinja, params.prefill_assistant, params.reasoning_format, params.default_template_kwargs,
        std::move(tmpls), false, false, false, enable_thinking, params.sampling.reasoning_budget_tokens,
        params.sampling.reasoning_budget_message, params.media_path, params.force_pure_content_parser,
    };
    std::vector<llama_logit_bias> eog;
    for (llama_token i = 0; i < llama_vocab_n_tokens(vocab); i++) {
        if (llama_vocab_is_eog(vocab, i)) eog.push_back({i, -INFINITY});
    }

    std::ifstream f(file, std::ios::binary);
    if (!f) {
        fprintf(stderr, "PARSE_TEST: cannot open %s\n", file.c_str());
        return 2;
    }
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const json cases = json::parse(text);
    int n_ok = 0, n = 0;
    for (const auto & c : cases) {
        ++n;
        const std::string name = c.value("name", "case");
        std::vector<std::string> errs;
        json got_msg;
        try {
            json body = c.at("request");
            if (c.value("api", "chat") == "messages") {
                body = server_chat_convert_anthropic_to_oai(body);
            }
            std::vector<raw_buffer> files;
            json parsed = oaicompat_chat_params_parse(body, cp, files);
            task_params tp = server_schema::eval_llama_cmpl_schema(vocab, params, eog, parsed);

            // the raw output and its tokens (as the engine would emit them)
            llama_tokens ids;
            std::string raw;
            if (c.contains("ids")) {
                ids = c.at("ids").get<llama_tokens>();
            } else {
                raw = c.at("raw").get<std::string>();
                ids = common_tokenize(vocab, raw, false, true);
            }
            // text pieces as the server builds them (special pieces only for the grammar's preserved tokens; an
            // end-of-generation token adds no text)
            std::vector<std::string> pieces;
            raw.clear();
            for (llama_token t : ids) {
                const bool special = tp.sampling.preserved_tokens.count(t) > 0;
                std::string p = llama_vocab_is_eog(vocab, t) ? std::string() : common_token_to_piece(vocab, t, special);
                pieces.push_back(p);
                raw += p;
            }

            task_result_state st(tp.chat_parser_params);
            std::vector<common_chat_msg_diff> diffs;
            const common_chat_msg msg = st.update_chat_msg(raw, false, diffs);
            got_msg = msg.to_json_oaicompat();

            // streaming: token by token, then the final update (server_task_result_cmpl_partial / _final::update)
            task_result_state ss(tp.chat_parser_params);
            agg_msg agg;
            for (const auto & p : pieces) {
                std::vector<common_chat_msg_diff> d;
                ss.update_chat_msg(p, true, d);
                apply_diffs(agg, d);
            }
            {
                std::vector<common_chat_msg_diff> d;
                ss.update_chat_msg("", false, d);
                apply_diffs(agg, d);
            }
            if (agg.content != msg.content) errs.push_back("stream content differs: " + json(agg.content).dump());
            if (agg.reasoning != msg.reasoning_content) errs.push_back("stream reasoning differs: " + json(agg.reasoning).dump());
            if (agg.tools.size() != msg.tool_calls.size()) {
                errs.push_back("stream tool calls " + std::to_string(agg.tools.size()) + " vs final " + std::to_string(msg.tool_calls.size()));
            } else {
                size_t i = 0;
                for (const auto & [idx, t] : agg.tools) {
                    const auto & tc = msg.tool_calls[i++];
                    if (t.first != tc.name || !json_eq(parse_args(t.second), parse_args(tc.arguments))) {
                        errs.push_back("stream tool call " + std::to_string(idx) + " differs: " + t.first + " " + t.second);
                    }
                }
            }

            // expected message
            if (c.contains("expect")) {
                const json & e = c.at("expect");
                const bool keys_only = c.value("keys_only", false);
                if (e.contains("content") && e.at("content").get<std::string>() != msg.content) {
                    errs.push_back("content " + json(msg.content).dump() + " != expected " + e.at("content").dump());
                }
                if (e.contains("reasoning_content") && e.at("reasoning_content").get<std::string>() != msg.reasoning_content) {
                    errs.push_back("reasoning " + json(msg.reasoning_content).dump() + " != expected " + e.at("reasoning_content").dump());
                }
                if (e.contains("tool_calls")) {
                    const json & et = e.at("tool_calls");
                    if (et.size() != msg.tool_calls.size()) {
                        errs.push_back("tool calls " + std::to_string(msg.tool_calls.size()) + " != expected " + std::to_string(et.size()));
                    } else {
                        for (size_t i = 0; i < et.size(); ++i) {
                            const auto & tc = msg.tool_calls[i];
                            const json args = parse_args(tc.arguments);
                            if (et[i].at("name").get<std::string>() != tc.name) {
                                errs.push_back("tool " + std::to_string(i) + " name " + tc.name);
                            }
                            const json & ea = et[i].at("arguments");
                            if (keys_only ? keys_of(ea) != keys_of(args) : !json_eq(ea, args)) {
                                errs.push_back("tool " + std::to_string(i) + " arguments " + tc.arguments + " != expected " + ea.dump());
                            }
                        }
                    }
                }
            }
        } catch (const std::exception & e) {
            errs.push_back(std::string("exception: ") + e.what());
        }
        const bool ok = errs.empty();
        n_ok += ok;
        printf("%s %s: %s\n", ok ? "PASS" : "FAIL", name.c_str(), got_msg.dump().c_str());
        for (const auto & e : errs) printf("     %s\n", e.c_str());
    }
    printf("PARSE_TEST %d/%d passed\n", n_ok, n);
    llama_model_free(model);
    return n_ok == n ? 0 : 1;
}
