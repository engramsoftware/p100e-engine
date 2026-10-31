// p100e-server live status: /slots and /metrics answer from a snapshot the task loop publishes, so the HTTP threads
// never wait for the loop (one prompt chunk can take seconds). Included by server_context.cpp only (after the fork's
// server headers: json, server_metrics, server_slot_stats, ggml_time_us).
//
// Publisher (task loop thread): p100e_live_publish() after every loop iteration (a prompt chunk and / or a decode step)
// and after every task it handles. Readers (HTTP threads): p100e_live_slots() / p100e_live_metrics(): a copy under the
// mutex, no engine call.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>

struct p100e_live_state {
    std::mutex mx;
    bool ready = false;
    json slots = json::array();   // server_slot::to_json(only_metrics = true) per slot
    int n_idle = 0;
    int n_processing = 0;
    int n_deferred = 0;
    server_metrics metrics;
    std::string extra;            // p100e:* lines (prompt cache counters + live per-slot gauges)
    int64_t t_us = 0;             // ggml_time_us() of the snapshot
    uint64_t n_published = 0;
    std::atomic<bool> reset_bucket{false};   // /metrics was read: the loop resets its rate buckets at the next publish
};

inline p100e_live_state & p100e_live() {
    static p100e_live_state s;
    return s;
}

// ggml_time_us() timestamp -> unix milliseconds (0 stays 0: not set)
inline int64_t p100e_unix_ms(int64_t t_us) {
    if (t_us <= 0) {
        return 0;
    }
    const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch()).count();
    return now_ms - (ggml_time_us() - t_us) / 1000;
}

// Prometheus gauge text
inline void p100e_prom(std::string & t, const std::string & name, const char * help, double v, const char * type = "gauge") {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.17g", v);
    t += "# HELP p100e:" + name + " " + help + "\n# TYPE p100e:" + name + " " + type + "\np100e:" + name + " " + buf + "\n";
}

// per-slot gauges from the slot JSON (one HELP / TYPE per family, a {slot="i"} sample per slot)
inline std::string p100e_live_slot_gauges(const json & slots, const server_metrics & m) {
    struct fam { const char * name; const char * help; };
    static const fam fams[] = {
        {"slot_busy",                    "1 while the slot has a task."},
        {"slot_prompt_tokens",           "Prompt rows of the slot's task (whole prompt, known at admission)."},
        {"slot_prompt_tokens_processed", "Prompt rows prefilled so far (cached rows excluded)."},
        {"slot_prompt_tokens_cached",    "Prompt rows taken from the prompt cache."},
        {"slot_tokens_decoded",          "Tokens generated so far by the slot's task."},
    };
    std::string t;
    double pp_live = 0, dec_live = 0, busy = 0;
    for (const auto & f : fams) {
        t += std::string("# HELP p100e:") + f.name + " " + f.help + "\n# TYPE p100e:" + f.name + " gauge\n";
        for (const auto & s : slots) {
            const bool on = s.value("is_processing", false);
            const json nt = s.contains("next_token") && !s.at("next_token").empty() ? s.at("next_token")[0] : json::object();
            double v = 0;
            const std::string n = f.name;
            if (n == "slot_busy")                         v = on ? 1 : 0;
            else if (n == "slot_prompt_tokens")           v = on ? s.value("n_prompt_tokens", 0) : 0;
            else if (n == "slot_prompt_tokens_processed") v = on ? s.value("n_prompt_tokens_processed", 0) : 0;
            else if (n == "slot_prompt_tokens_cached")    v = on ? s.value("n_prompt_tokens_cache", 0) : 0;
            else                                          v = on ? nt.value("n_decoded", 0) : 0;
            char buf[160];
            snprintf(buf, sizeof(buf), "p100e:%s{slot=\"%d\"} %.17g\n", f.name, s.value("id", 0), v);
            t += buf;
        }
    }
    for (const auto & s : slots) {
        if (!s.value("is_processing", false)) continue;
        busy += 1;
        if (s.value("state", std::string()) != "generating") pp_live += s.value("n_prompt_tokens_processed", 0);
        const json nt = s.contains("next_token") && !s.at("next_token").empty() ? s.at("next_token")[0] : json::object();
        dec_live += nt.value("n_decoded", 0);
    }
    p100e_prom(t, "slots_busy", "Slots with a task.", busy);
    p100e_prom(t, "prompt_tokens_processing", "Prompt rows prefilled so far by tasks still in prompt processing.", pp_live);
    p100e_prom(t, "tokens_decoding", "Tokens generated so far by the running tasks (not yet in llamacpp:tokens_predicted_total).", dec_live);
    p100e_prom(t, "tokens_predicted_live_total", "llamacpp:tokens_predicted_total plus the running tasks' tokens.",
               (double) m.predict.count + dec_live, "counter");
    p100e_prom(t, "prompt_tokens_live_total", "Prompt rows prefilled (= llamacpp:prompt_tokens_total, counted per chunk).",
               (double) m.prompt.count, "counter");
    return t;
}

// task loop: replace the snapshot (metrics: the loop's server_metrics; its rate buckets are reset here when a /metrics
// read asked for it, as llama-server resets them on every /metrics)
inline void p100e_live_publish(json slots, int n_idle, int n_processing, int n_deferred, server_metrics & metrics,
                               std::string cache_text) {
    auto & L = p100e_live();
    if (L.reset_bucket.exchange(false)) {
        metrics.reset_bucket();
    }
    std::lock_guard<std::mutex> lk(L.mx);
    L.slots        = std::move(slots);
    L.n_idle       = n_idle;
    L.n_processing = n_processing;
    L.n_deferred   = n_deferred;
    L.metrics      = metrics;
    L.extra        = std::move(cache_text);
    L.t_us         = ggml_time_us();
    L.n_published++;
    L.ready = true;
}

// HTTP: the slot array; false before the first publish
inline bool p100e_live_slots(json & slots, int & n_idle) {
    auto & L = p100e_live();
    std::lock_guard<std::mutex> lk(L.mx);
    if (!L.ready) return false;
    slots  = L.slots;
    n_idle = L.n_idle;
    return true;
}

// HTTP: a metrics result + the p100e:* text; resets the snapshot's rate buckets (and asks the loop to reset its own)
inline bool p100e_live_metrics(server_task_result_metrics & r, std::string & extra) {
    auto & L = p100e_live();
    std::lock_guard<std::mutex> lk(L.mx);
    if (!L.ready) return false;
    r.n_processing_slots = L.n_processing;
    r.n_tasks_deferred   = L.n_deferred;
    r.metrics            = L.metrics;
    extra = L.extra + p100e_live_slot_gauges(L.slots, L.metrics);
    p100e_prom(extra, "status_snapshot_age_seconds", "Age of the task loop's status snapshot this answer was built from.",
               (ggml_time_us() - L.t_us) / 1e6);
    L.metrics.reset_bucket();
    L.reset_bucket = true;
    return true;
}
