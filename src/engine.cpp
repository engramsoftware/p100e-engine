#include "engine.h"

#include <cstdlib>

namespace p100e {

engine::engine(const engine_opts & o) : o_(o) {
    tp_model::opts mo;
    mo.n_ctx = o.n_ctx;
    mo.max_batch = o.ubatch;
    mo.load_mtp = o.mtp;
    mo.kv_type = o.kv_type;
    mo.max_seqs = o.max_seqs;
    mo.pool_tokens = -1;          // deferred: after the drafter / snapshots / prefill buffers
    mo.default_seq = false;
    m_ = std::make_unique<tp_model>(o.model, o.devices, mo);
    if (o.mtp) d_ = std::make_unique<tp_mtp>(*m_);
    if (!getenv("P100E_HOST_BUDGET_MB")) m_->set_host_budget(o.host_budget_mb << 20);   // the env wins
    sched_opts so = o.so;
    so.max_active = o.max_seqs;
    so.mo = o.mo;
    s_ = std::make_unique<sched>(*m_, d_.get(), so);   // allocates the rollback snapshots
    m_->prefill_setup();
    m_->alloc_pool(o.pool_tokens, o.reserve_mb, o.margin_mb);
}

} // namespace p100e
