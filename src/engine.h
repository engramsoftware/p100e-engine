// p100e engine: the object a server process holds - model + MTP draft head + KV pool + scheduler,
// built in the order that lets the KV pool take whatever VRAM is left. Engine only (no HTTP, no tokenizer).
//
//   engine_opts o;  o.model = "model.gguf";  o.devices = {0, 1};
//   engine e(o);                                   // prints "KV pool: N pages x 256 positions = T tokens ..."
//   sched::request r;  r.prompt = ids;  r.n_gen = 512;  r.mtp = true;  r.splits = message_starts;
//   int id = e.scheduler().submit(r);              // prompt cache: reuses the longest cached prefix (sched.h)
//   while (e.scheduler().step()) { ... e.scheduler().get(id).out grows ... }
//
// Construction: weights (tp_model, no KV yet) -> draft head (tp_mtp) -> rollback snapshots (sized for batched MTP
// verify) -> prefill pipeline buffers -> KV pool from the free VRAM minus reserve_mb per GPU (default 1024 MiB on the
// last GPU: the vision encoder) and margin_mb -> scheduler. Sequences: max_seqs delta-net state slots (resident sessions);
// more sessions than that are queued, or preempted to host RAM (sched::preempt / resume).
#pragma once

#include "p100e_sched.h"

#include <memory>
#include <string>
#include <vector>

namespace p100e {

struct engine_opts {
    std::string model;
    std::vector<int> devices{0, 1};
    uint32_t kv_type = fast::KV_Q8_0;      // KV_Q8_0 (default), KV_Q4_0, KV_F16
    int n_ctx = 262144;                     // per-sequence context cap
    int ubatch = 2048;                      // prefill chunk (a prompt's chunks are the same alone or with other sessions)
    int max_seqs = 3;                       // resident sequences (llama-server -np)
    bool mtp = true;                        // load the nextn block + draft head
    mtp_opts mo;                            // draft policy (default: adaptive k <= 3, tau 0.5)
    int64_t pool_tokens = -1;               // KV pool tokens; < 0: all free VRAM after the reserves
    std::vector<size_t> reserve_mb{};       // per GPU kept free; empty: 1024 MiB on the last GPU
    size_t margin_mb = 512;
    // pinned host RAM of the prompt cache, one budget for offloaded sequences (idle sessions / cache entries) and the
    // scheduler's delta-net checkpoints (at most so.ckpt_budget of it, default 8 GiB): ~12 GiB offload + 8 GiB
    // checkpoints by default; offloads may use checkpoint room that is free, and new checkpoints take it back by dropping
    // offloaded cache-only entries (least recently used first). P100E_HOST_BUDGET_MB overrides.
    size_t host_budget_mb = 20480;
    // scheduler options (max_active / mo are taken from max_seqs / mo above): prompt cache (cache, ckpt_budget,
    // ckpt_every, ckpt_tail, min_split, exact, draft_kv), draft_rows, decode_per_chunk
    sched_opts so;
};

class engine {
public:
    explicit engine(const engine_opts & o);
    tp_model & model() { return *m_; }
    tp_mtp * drafter() { return d_.get(); }
    sched & scheduler() { return *s_; }
    const engine_opts & opts() const { return o_; }

private:
    engine_opts o_;
    std::unique_ptr<tp_model> m_;
    std::unique_ptr<tp_mtp> d_;
    std::unique_ptr<sched> s_;
};

} // namespace p100e
