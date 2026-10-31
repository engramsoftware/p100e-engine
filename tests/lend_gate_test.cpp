// lend_gate (src/lend_gate.h): the prefill ring's owner / borrower protocol, CPU only.
//  1. one lease at a time, none while busy, nested busy scopes
//  2. an owner entering a busy scope waits for the outstanding lease; no new lease while it waits
//  2b. no lease in the window between a returned lease and the waiting owner taking the gate (the waiting_ guard:
//     without it a borrower that asks again at once gets the ring back and the owner keeps waiting - starvation)
//  3. stress: owner and borrower threads never use the buffer at the same time (each checks the other's flag and
//     scribbles a pattern it then verifies)
#include "lend_gate.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using namespace p100e;
using clk = std::chrono::steady_clock;

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { ++fails; printf("FAIL: %s\n", msg); } } while (0)

int main() {
    {   // 1
        lend_gate g;
        auto a = g.try_lend();
        CHECK(a != nullptr, "first lease");
        CHECK(g.try_lend() == nullptr, "second lease while the first is out");
        auto a2 = a;   // copies extend the same loan
        a.reset();
        CHECK(g.lent() && g.try_lend() == nullptr, "a copy of the lease keeps the loan");
        a2.reset();
        CHECK(!g.lent(), "loan ends with the last copy");
        {
            lend_gate::busy_scope b(g);
            CHECK(g.busy() && g.try_lend() == nullptr, "no lease while busy");
            {
                lend_gate::busy_scope b2(g);   // nested (forward_mm -> forward_prefill): no wait, no deadlock
                CHECK(g.try_lend() == nullptr, "no lease while nested busy");
            }
            CHECK(g.busy(), "still busy after the inner scope");
        }
        CHECK(!g.busy() && g.try_lend() != nullptr, "lease after the busy scope");
    }
    {   // 2
        lend_gate g;
        std::atomic<bool> released{false}, entered{false};
        auto lease = g.try_lend();
        std::thread owner([&] {
            lend_gate::busy_scope b(g);
            entered = true;
            CHECK(released.load(), "owner entered before the lease was returned");
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK(!entered.load(), "owner did not wait for the lease");
        CHECK(g.try_lend() == nullptr, "new lease handed out while the owner waits");
        released = true;
        lease.reset();
        owner.join();
        CHECK(entered.load(), "owner entered after the lease was returned");
    }
    {   // 2b
        // the owner is waiting (owner_waiting) when the lease is returned, and this thread asks again at once. With the
        // waiting_ guard the answer is always null (the owner still counts as waiting, or is already inside its scope,
        // which it leaves only after the check); without it most rounds hand the ring straight back to the borrower
        // before the woken owner gets the mutex.
        int stolen = 0;
        const int rounds = 200;
        for (int k = 0; k < rounds; ++k) {
            lend_gate g;
            auto lease = g.try_lend();
            std::atomic<bool> done{false};
            std::thread owner([&] {   // stays inside its scope until this round is checked
                lend_gate::busy_scope b(g);
                while (!done.load()) std::this_thread::yield();
            });
            const auto t0 = clk::now();
            while (!g.owner_waiting() && clk::now() - t0 < std::chrono::seconds(5)) std::this_thread::yield();
            CHECK(g.owner_waiting(), "owner never started waiting");
            lease.reset();
            auto again = g.try_lend();
            if (again) ++stolen;
            again.reset();   // (else the owner would wait for it forever)
            done = true;
            owner.join();
        }
        printf("waiting guard: %d of %d immediate re-borrows got a lease while the owner waited\n", stolen, rounds);
        CHECK(stolen == 0, "a lease was handed out while the owner waited for the gate (waiting_ guard)");
    }
    {   // 3
        lend_gate g;
        std::vector<int> buf(1 << 16);
        std::atomic<int> in_owner{0}, in_borrower{0};
        std::atomic<long> owner_uses{0}, borrows{0}, refused{0}, overlaps{0}, corrupt{0};
        const auto end = clk::now() + std::chrono::milliseconds(1500);
        auto use = [&](int tag, std::atomic<int> & me, std::atomic<int> & other) {
            me = 1;
            if (other.load()) ++overlaps;
            for (auto & v : buf) v = tag;
            std::this_thread::yield();
            for (auto & v : buf) if (v != tag) { ++corrupt; break; }
            if (other.load()) ++overlaps;
            me = 0;
        };
        std::thread owner([&] {
            int k = 0;
            while (clk::now() < end) {
                lend_gate::busy_scope b(g);
                if (++k % 3 == 0) { lend_gate::busy_scope inner(g); use(1, in_owner, in_borrower); }
                else use(1, in_owner, in_borrower);
                ++owner_uses;
            }
        });
        std::thread borrower([&] {
            while (clk::now() < end) {
                auto lease = g.try_lend();
                if (!lease) { ++refused; std::this_thread::yield(); continue; }
                use(2, in_borrower, in_owner);
                ++borrows;
            }
        });
        owner.join();
        borrower.join();
        printf("stress: owner uses %ld, borrows %ld, refused %ld, overlaps %ld, corrupt %ld\n", owner_uses.load(),
               borrows.load(), refused.load(), overlaps.load(), corrupt.load());
        CHECK(overlaps == 0 && corrupt == 0, "owner and borrower used the buffer at the same time");
        CHECK(owner_uses > 0 && borrows > 0, "both sides got the buffer");
    }
    printf(fails ? "%d CHECK(S) FAILED\n" : "ALL CHECKS PASSED\n", fails);
    return fails ? 1 : 0;
}
