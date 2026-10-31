// p100e-server options that llama-server's argument parser does not know (--p100e-* flags, server_main.cpp)
#pragma once

#include <string>
#include <vector>

struct p100e_server_opts {
    std::vector<int> devices{0, 1};   // --p100e-devices 0,1
    std::vector<double> split;        // --p100e-split a,b,..: relative tensor-parallel share per GPU (empty: equal /
                                      //   P100E_SPLIT; tp_model::opts::split)
    int mtp_k = 3;                    // --p100e-mtp K (0: off; max drafts per step)
    float mtp_tau = 0.5f;             // --p100e-mtp-tau t (adaptive draft length; 0: always K)
    std::string kv_type;              // --p100e-kv f16|q8_0|q4_0 (default: -ctk when not f16, else q8_0)
    std::string engine;               // --p100e-engine tp|fn (default: fn for general.architecture qwen4exp, else tp)
    int fn_hot = 72;                  // --p100e-fn-hot N: Flash-Next GPU-resident experts per layer per GPU
    int fn_max_t = 512;               // --p100e-fn-max-t N: Flash-Next prefill chunk rows (0: -ub)
    std::vector<std::string> fn_pipe_nodes;   // --p100e-fn-pipe-nodes host:port,host:port (this server = node 1)
    int fn_pipe_layers = -1;                  // --p100e-fn-pipe-layers k|auto
    std::vector<int> fn_slots;                // --p100e-fn-slots a,b|auto: resident experts per layer per GPU (auto: fill)
    bool no_engine = false;           // --p100e-no-engine: text endpoints only (tokenizer / templates), no GPU
    bool print_ids = false;           // --p100e-print-ids: log the prompt + generated token ids of every task
    bool greedy_gpu = true;           // --p100e-no-greedy-gpu: always copy logits (host argmax) even for greedy
    bool refcheck = false;            // --p100e-refcheck: rerun every greedy task as plain greedy decoding, compare ids
                                      //   (debug only: runs synchronously in release() and blocks the loop)
    std::string selftest;             // --p100e-selftest [builtin|file.json]: serve a scripted request set, then exit
    std::string parse_test;           // --p100e-parse-test cases.json: parser goldens (no engine), then exit
    int vision_device = -1;           // --p100e-vision-device N: image encoder GPU (-1: the last engine device)
    bool no_remote_images = false;    // --p100e-no-remote-images: refuse http(s) image / media URLs (the server would
                                      //   fetch them: SSRF from the LAN); data: URIs still work. Default off (as before)
    int max_requests = 0;             // --p100e-max-requests N: exit after N completed completion tasks (0: never)
    int mtp_multi = -1;               // --p100e-mtp-multi K: max drafts per slot while several slots are busy (-1: no cap)
    int draft_rows = 4;               // --p100e-draft-rows N: several slots generating: rows per step max(slots, N), drafts
                                      //   shared (sched::draft_rows; 0: no budget, k per slot + 8-row cap as before)
    bool cache = true;                // --p100e-cache 0|1: prompt cache (idle sequences + checkpoints reused by prefix)
    int cache_resident = 4;           // --p100e-cache-resident N: idle cache entries kept on the GPU (state slots)
    size_t host_budget_mb = 20480;    // --p100e-host-budget-mb N: pinned host RAM for offloaded entries + checkpoints
    size_t ckpt_budget_mb = 8192;     // --p100e-ckpt-budget-mb N: of which checkpoints at most
    long long pool_tokens = 0;
    long long margin_mb = -1;         // --p100e-margin-mb N: VRAM per GPU the KV pool leaves free (-1: tp_model's 512)
    // --p100e-decode-share x (env P100E_DECODE_SHARE overrides): while a slot prefills and others generate, the
    //   generating slots get this share of the wall time (after a prompt chunk of t s, decode steps for t*x/(1-x) s
    //   before the next chunk; sched::decode_share). 0 = one decode step per chunk (the old loop). Clamped to [0, 0.95].
    double decode_share = 0.5;        // --p100e-pool-tokens N: KV pool positions (0: all free VRAM after the reserves)
};

// set by main() before server_context::load_model()
extern p100e_server_opts g_p100e_opts;
