// p100e-server self-test client (see selftest.h)
#include "selftest.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

namespace {

struct st_req {
    std::string name, method, path;
    json body;
    int expect_status = 200;
    int cancel_after_chunks = 0;   // > 0: close the connection after this many received chunks (cancel test)
    int group = 0;                 // > 0: run concurrently with the neighbouring requests of the same group
};

json chat(const std::string & user, int max_tokens, bool thinking, bool stream, double temperature = 0.0) {
    json b = {
        {"model", "model"},
        {"messages", json::array({ {{"role", "user"}, {"content", user}} })},
        {"temperature", temperature},
        {"max_tokens", max_tokens},
        {"chat_template_kwargs", {{"enable_thinking", thinking}}},
    };
    if (stream) {
        b["stream"] = true;
        b["stream_options"] = {{"include_usage", true}};
    }
    return b;
}

json weather_tool_oai() {
    return json::array({ {
        {"type", "function"},
        {"function", {
            {"name", "get_weather"},
            {"description", "Get the current weather for a city."},
            {"parameters", {
                {"type", "object"},
                {"properties", {{"city", {{"type", "string"}, {"description", "City name"}}}}},
                {"required", json::array({"city"})},
            }},
        }},
    } });
}

std::string long_document() {
    // ~3000 tokens of varied, deterministic text: prompts longer than a 2048-token ubatch run as chunks
    static const char * topics[] = {"rivers", "bridges", "clocks", "glaciers", "markets", "lighthouses", "orchards", "railways"};
    std::ostringstream ss;
    for (int i = 0; i < 96; ++i) {
        ss << "Section " << (i + 1) << ". The survey of " << topics[i % 8] << " in district " << (17 * i + 3) % 101
           << " recorded " << (i * 37) % 250 + 12 << " sites; inspectors noted that " << topics[(i + 3) % 8]
           << " nearby were older, and the committee asked for a second visit in year " << 1900 + (i * 13) % 120 << ".\n";
    }
    return ss.str();
}

std::vector<st_req> builtin() {
    std::vector<st_req> r;
    r.push_back({"health", "GET", "/health", nullptr});
    r.push_back({"props", "GET", "/props", nullptr});
    r.push_back({"models", "GET", "/v1/models", nullptr});
    r.push_back({"slots_idle", "GET", "/slots", nullptr});
    r.push_back({"tokenize", "POST", "/tokenize", {{"content", "Hello, world! <think>x</think> 1234"}, {"with_pieces", true}}});
    r.push_back({"detokenize", "POST", "/detokenize", {{"tokens", json::array({9419, 11, 1814, 0})}}});
    r.push_back({"apply_template", "POST", "/apply-template", {
        {"messages", json::array({ {{"role", "system"}, {"content", "You are helpful."}}, {{"role", "user"}, {"content", "Hi there."}} })},
        {"chat_template_kwargs", {{"enable_thinking", false}}}}});
    r.push_back({"count_tokens", "POST", "/v1/messages/count_tokens", {
        {"model", "model"}, {"messages", json::array({ {{"role", "user"}, {"content", "Hello there"}} })}}});
    // greedy generation (refcheck compares each with plain greedy decoding)
    r.push_back({"chat_greedy", "POST", "/v1/chat/completions",
                 chat("What is the capital of France? Answer in one sentence.", 48, false, false)});
    r.push_back({"chat_stream_think", "POST", "/v1/chat/completions",
                 chat("How many letters r are in the word strawberry? Think briefly.", 96, true, true)});
    {
        json b = chat("What is the weather in Paris right now? Use the get_weather tool.", 160, false, true);
        b["tools"] = weather_tool_oai();
        r.push_back({"chat_tools_stream", "POST", "/v1/chat/completions", b});
        b.erase("stream");
        b.erase("stream_options");
        r.push_back({"chat_tools", "POST", "/v1/chat/completions", b});
    }
    {
        // Claude Code shape through a proxy: system blocks, top-level reasoning_budget_tokens (honoured)
        json b = {
            {"model", "model"}, {"max_tokens", 160}, {"stream", true}, {"temperature", 0.0},
            {"system", json::array({ {{"type", "text"}, {"text", "You are a terse assistant."}} })},
            {"messages", json::array({ {{"role", "user"}, {"content", json::array({ {{"type", "text"}, {"text", "Explain in two sentences why the sky is blue."}} })}} })},
            {"reasoning_budget_tokens", 24},
            {"chat_template_kwargs", {{"enable_thinking", true}}},
        };
        r.push_back({"anthropic_stream_budget", "POST", "/v1/messages", b});
    }
    {
        json b = {
            {"model", "model"}, {"max_tokens", 160}, {"temperature", 0.0},
            {"messages", json::array({ {{"role", "user"}, {"content", "What is the weather in Tokyo? Use the tool."}} })},
            {"tools", json::array({ {{"name", "get_weather"}, {"description", "Get the current weather for a city."},
                                      {"input_schema", {{"type", "object"}, {"properties", {{"city", {{"type", "string"}}}}}, {"required", json::array({"city"})}}}} })},
            {"chat_template_kwargs", {{"enable_thinking", false}}},
        };
        r.push_back({"anthropic_tool", "POST", "/v1/messages", b});
    }
    {
        json b = chat("Write a haiku about graphics cards.", 64, false, false, 1.0);
        b["seed"] = 42;
        r.push_back({"chat_sampled", "POST", "/v1/chat/completions", b});
    }
    {
        // sampled with fixed seeds: --p100e-refcheck replays them without MTP and must get the same tokens
        json b = chat("Why do cats purr? Think it through first.", 160, true, false, 1.0);
        b["seed"] = 7;
        r.push_back({"chat_sampled_think", "POST", "/v1/chat/completions", b});
        b = chat("What is the weather in Berlin? Use the get_weather tool.", 160, false, false, 0.7);
        b["seed"] = 3;
        b["tools"] = weather_tool_oai();
        r.push_back({"chat_sampled_tool", "POST", "/v1/chat/completions", b});
        b = chat("Name three prime numbers and explain what makes them prime.", 160, true, true, 1.0);
        b["seed"] = 11;
        b["reasoning_budget_tokens"] = 32;
        r.push_back({"chat_sampled_budget", "POST", "/v1/chat/completions", b});
    }
    {
        json b = chat("What is the capital of France? Answer in one sentence.", 48, false, false);
        b["stop"] = json::array({"Paris"});
        r.push_back({"chat_stop", "POST", "/v1/chat/completions", b});
    }
    {
        st_req c{"cancel_stream", "POST", "/v1/chat/completions",
                 chat("Count from 1 to 400, separated by spaces.", 1200, false, true)};
        c.cancel_after_chunks = 12;
        r.push_back(c);
    }
    r.push_back({"chat_greedy_again", "POST", "/v1/chat/completions",
                 chat("What is the capital of France? Answer in one sentence.", 48, false, false)});
    r.push_back({"long_prompt", "POST", "/v1/chat/completions",
                 chat(long_document() + "\nWhich district had the most sites? Answer briefly.", 48, false, false)});
    r.push_back({"slots_after", "GET", "/slots", nullptr});
    return r;
}

// keep logs readable: the chat template (14 KB) is replaced by its size
std::string brief(const std::string & body) {
    try {
        json j = json::parse(body);
        if (j.is_object() && j.contains("chat_template") && j["chat_template"].is_string()) {
            j["chat_template"] = "<" + std::to_string(j["chat_template"].get<std::string>().size()) + " bytes>";
        }
        return j.dump();
    } catch (...) {
        return body;
    }
}

} // namespace

static bool load_spec(const std::string & spec, std::vector<st_req> & reqs) {
    if (spec == "builtin" || spec.empty()) {
        auto b = builtin();
        reqs.insert(reqs.end(), b.begin(), b.end());
        return true;
    }
    std::ifstream f(spec, std::ios::binary);
    if (!f) {
        fprintf(stderr, "SELFTEST: cannot open %s\n", spec.c_str());
        return false;
    }
    json j = json::parse(f);
    for (const auto & e : j) {
        st_req q;
        q.name = e.value("name", "req");
        q.method = e.value("method", "POST");
        q.path = e.at("path").get<std::string>();
        q.body = e.contains("body") ? e.at("body") : json();
        q.expect_status = e.value("expect_status", 200);
        q.cancel_after_chunks = e.value("cancel_after_chunks", 0);
        q.group = e.value("group", 0);
        reqs.push_back(q);
    }
    return true;
}

static std::string b64(const std::string & s) {
    static const char * T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    o.reserve((s.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < s.size(); i += 3) {
        const uint32_t v = ((uint8_t) s[i] << 16) | ((uint8_t) s[i + 1] << 8) | (uint8_t) s[i + 2];
        o += T[v >> 18]; o += T[(v >> 12) & 63]; o += T[(v >> 6) & 63]; o += T[v & 63];
    }
    if (i + 1 == s.size()) {
        const uint32_t v = (uint8_t) s[i] << 16;
        o += T[v >> 18]; o += T[(v >> 12) & 63]; o += "==";
    } else if (i + 2 == s.size()) {
        const uint32_t v = ((uint8_t) s[i] << 16) | ((uint8_t) s[i + 1] << 8);
        o += T[v >> 18]; o += T[(v >> 12) & 63]; o += T[(v >> 6) & 63]; o += '=';
    }
    return o;
}

// one line, one write call: the server's log (stderr, often the same file) can split lines between writes, never
// inside one, so a log parser finds every raw response intact
static void write_atomic(const std::string & line) {
    fflush(stdout);
#if defined(_WIN32)
    DWORD n = 0;
    WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), line.data(), (DWORD) line.size(), &n, nullptr);
#else
    fwrite(line.data(), 1, line.size(), stdout);
    fflush(stdout);
#endif
}

int p100e_selftest(const std::string & host, int port, const std::string & spec) {
    // spec: one or more of "builtin" / file.json separated by ';'
    std::vector<st_req> reqs;
    {
        std::stringstream ss(spec);
        std::string part;
        while (std::getline(ss, part, ';')) {
            if (!load_spec(part, reqs)) {
                return 2;
            }
        }
        if (reqs.empty() && !load_spec("builtin", reqs)) {
            return 2;
        }
    }

    // one request -> its printed record (consecutive requests with the same group > 0 run concurrently, one client
    // each; their records are printed in order afterwards)
    struct st_out { std::string text; std::string body, ctype; json r; bool ok = false; };
    auto run_one = [&](size_t i) {
        const st_req & q = reqs[i];
        st_out o;
        httplib::Client cli(host, port);
        cli.set_read_timeout(1800, 0);
        cli.set_write_timeout(60, 0);
        const auto t0 = std::chrono::steady_clock::now();
        std::string body;
        int chunks = 0;
        bool cancelled = false;
        httplib::Result res;
        if (q.method == "GET") {
            res = cli.Get(q.path);
            if (res) body = res->body;
        } else {
            const std::string payload = q.body.dump();
            res = cli.Post(q.path, httplib::Headers{}, payload, "application/json",
                           [&](const char * data, size_t len) {
                               body.append(data, len);
                               ++chunks;
                               if (q.cancel_after_chunks > 0 && chunks >= q.cancel_after_chunks) {
                                   cancelled = true;
                                   return false;   // close the connection: the server must cancel the task
                               }
                               return true;
                           });
            if (res && body.empty() && !res->body.empty()) {
                body = res->body;
            }
        }
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        int status = res ? res->status : -1;
        std::string ctype = res ? res->get_header_value("Content-Type") : "";
        char line[512];
        if (q.cancel_after_chunks > 0) {
            o.ok = cancelled;   // the transfer was cut by us after N chunks
            snprintf(line, sizeof line, "cancelled after %d chunks (%zu bytes)\n", chunks, body.size());
            o.text += line;
        } else {
            o.ok = status == q.expect_status;
            snprintf(line, sizeof line, "status: %d, content-type: %s, elapsed: %.3f s, %zu bytes\n", status, ctype.c_str(),
                     dt, body.size());
            o.text += line;
            if (!res) { snprintf(line, sizeof line, "httplib error %d\n", (int) res.error()); o.text += line; }
            o.text += "body: " + brief(body) + "\n";
        }
        o.r = {{"name", q.name}, {"method", q.method}, {"path", q.path}, {"status", status}, {"ok", o.ok},
               {"elapsed_s", dt}, {"bytes", body.size()}, {"chunks", chunks}, {"cancelled", cancelled},
               {"group", q.group}};
        o.body = body;
        o.ctype = ctype;
        return o;
    };
    int n_ok = 0;
    for (size_t i = 0; i < reqs.size();) {
        size_t j = i + 1;
        if (reqs[i].group > 0) while (j < reqs.size() && reqs[j].group == reqs[i].group) ++j;
        std::vector<st_out> outs(j - i);
        if (j - i == 1) {
            outs[0] = run_one(i);
        } else {
            std::vector<std::thread> th;
            for (size_t k = i; k < j; ++k) th.emplace_back([&, k] { outs[k - i] = run_one(k); });
            for (auto & t : th) t.join();
        }
        for (size_t k = i; k < j; ++k) {
            const st_req & q = reqs[k];
            const st_out & o = outs[k - i];
            printf("=== SELFTEST %zu %s: %s %s%s\n", k, q.name.c_str(), q.method.c_str(), q.path.c_str(),
                   q.group > 0 ? (" (concurrent group " + std::to_string(q.group) + ")").c_str() : "");
            printf("%s", o.text.c_str());
            fflush(stdout);
            write_atomic("SELFTEST_RESULT " + o.r.dump() + "\n");
            write_atomic("SELFTEST_RAW " + std::to_string(k) + " " + b64(o.ctype) + " " + b64(o.body) + "\n");
            n_ok += o.ok;
            if (q.cancel_after_chunks > 0) {
                // give the server loop a moment to see the disconnect before the next request
                std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            }
        }
        i = j;
    }
    printf("SELFTEST_SUMMARY %d/%zu ok\n", n_ok, reqs.size());
    fflush(stdout);
    return n_ok == (int) reqs.size() ? 0 : 1;
}
