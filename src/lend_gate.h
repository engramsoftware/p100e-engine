// lend_gate: one buffer shared between its owner's work and ONE borrower at a time on another thread - tp_model's
// prefill ring (used by forward_prefill, idle between forwards) lent to the native vision encoder for its activations.
//   owner:    busy_scope around every use (nests on the owner's thread). Entering waits until the current lease is
//             returned (a borrow that started first finishes first; no new lease is handed out while an owner waits).
//   borrower: try_lend() -> a lease while the owner is neither busy nor waiting and nothing else is lent, else null (the
//             borrower then uses its own memory). The loan lasts until the last copy of the lease is destroyed.
// A thread must not enter a busy_scope while it holds a lease (it would wait for itself). The gate must outlive the
// leases it hands out.
#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>

namespace p100e {

class lend_gate {
public:
    class busy_scope {
    public:
        explicit busy_scope(lend_gate & g) : g_(g) {
            std::unique_lock<std::mutex> lk(g_.mu_);
            if (g_.busy_ == 0) {
                ++g_.waiting_;
                g_.cv_.wait(lk, [this] { return g_.leases_ == 0; });
                --g_.waiting_;
            }
            ++g_.busy_;
        }
        ~busy_scope() {
            std::lock_guard<std::mutex> lk(g_.mu_);
            --g_.busy_;
        }
        busy_scope(const busy_scope &) = delete;
        busy_scope & operator=(const busy_scope &) = delete;

    private:
        lend_gate & g_;
    };

    std::shared_ptr<void> try_lend() {
        std::lock_guard<std::mutex> lk(mu_);
        if (busy_ > 0 || waiting_ > 0 || leases_ > 0) return nullptr;
        ++leases_;
        return std::shared_ptr<void>(this, [](lend_gate * g) {
            std::lock_guard<std::mutex> lk(g->mu_);
            if (--g->leases_ == 0) g->cv_.notify_all();
        });
    }
    bool busy() const { std::lock_guard<std::mutex> lk(mu_); return busy_ > 0; }
    bool lent() const { std::lock_guard<std::mutex> lk(mu_); return leases_ > 0; }
    bool owner_waiting() const { std::lock_guard<std::mutex> lk(mu_); return waiting_ > 0; }   // (tests)

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    int busy_ = 0, waiting_ = 0, leases_ = 0;
};

} // namespace p100e
