// pdl transport backend with net_link's interface (see pdl_link.h)
#include "pdl/pdl_link.h"
#include "pdl/pdl_util.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace p100e {

// ---- net_transport: allgather / agree on top of exchange (same messages and texts as net_link) ----------------------
std::vector<uint64_t> net_transport::allgather(uint64_t v, int timeout_ms) {
    const int N = n_nodes();
    std::vector<uint64_t> all((size_t) N, 0);
    all[(size_t) self()] = v;
    std::vector<std::vector<seg>> rs((size_t) N);
    for (int j = 0; j < N; ++j) if (j != self()) rs[(size_t) j] = {{&all[(size_t) j], 8}};
    exchange(net_link::T_AGREE, {{&v, 8}}, rs, timeout_ms);
    return all;
}

void net_transport::agree(const std::string & what, uint64_t v, int timeout_ms) {
    const std::vector<uint64_t> all = allgather(v, timeout_ms);
    bool same = true;
    for (uint64_t x : all) same = same && x == all[0];
    if (same) return;
    std::string m = std::string(kind()) + ": the nodes disagree on " + what + ":";
    for (size_t j = 0; j < all.size(); ++j) {
        char b[48];
        snprintf(b, sizeof b, " node %zu %llu%s", j, (unsigned long long) all[j], (int) j == self() ? " (this one)" : "");
        m += b;
    }
    throw std::runtime_error(m + " (run every node with the same model, arguments and P100E_* switches)");
}

// ---- pdl_link ------------------------------------------------------------------------------------------------------------
pdl_link::pdl_link(const std::vector<std::string> & nodes, int self, pdl::frame_port & port, int connect_ms, int io_ms,
                   const pdl::proto_opts & po)
    : nodes_(nodes), self_(self), io_ms_(io_ms > 0 ? io_ms : 60000), port_(&port), po_(po) {
    init(connect_ms);
}

void pdl_link::init(int connect_ms) {
    po_.io_timeout_ms = io_ms_;
    eng_ = std::make_unique<pdl::engine>(*port_, self_, (int) nodes_.size(), po_);
    printf("pdl: node %d of %zu on %s, waiting up to %d s for the other nodes\n", self_, nodes_.size(), port_->describe().c_str(),
           connect_ms / 1000);
    fflush(stdout);
    eng_->hello(connect_ms > 0 ? connect_ms : 600000);
    eng_->hello_wait = nullptr;
    printf("pdl: node %d connected to all %zu nodes, chunk payload %zu bytes, crc %s\n", self_, nodes_.size(), eng_->chunk_payload(),
           po_.crc ? (pdl::crc32c_hw() ? "on (sse4.2)" : "on (software)") : "off");
    fflush(stdout);
}

pdl_link::~pdl_link() {
    // Goodbye: this node has all its messages; a peer may still miss part of our last one and NACK it. Keep polling
    // (answering) and saying bye every ms until every peer said bye (it has everything too), at most 3 s. Not after
    // a failure, including one of pdl_gpu's spin thread (pump_dead_): the engine may be left half way through a frame.
    if (broken_ || pump_dead_ || !eng_) return;
    try {
        const uint64_t t0 = pdl::now_ns();
        uint64_t last = 0;
        while (pdl::now_ns() - t0 < 3000000000ull) {
            std::lock_guard<std::mutex> l(mu_);
            eng_->poll();
            const uint64_t now = pdl::now_ns();
            if (now - last > 1000000ull) { eng_->send_bye(); last = now; }
            if (eng_->all_bye() && !eng_->tx_busy() && port_->tx_idle()) break;
        }
    } catch (...) {}
}

void pdl_link::pump_failed(const std::string & why) {
    std::lock_guard<std::mutex> l(pump_err_mu_);
    pump_err_ = why;
    pump_dead_ = true;
}

void pdl_link::exchange(uint32_t tag, const std::vector<seg> & send, const std::vector<std::vector<seg>> & recv, int timeout_ms) {
    if (broken_) throw std::runtime_error("pdl: the link failed earlier (a peer died, lockstep lost or a timeout): restart every node");
    (void) timeout_ms;   // the engine's io timeout applies (per message, no progress)
    const int N = n_nodes();
    const uint64_t t0 = pdl::now_ns();
    const uint32_t seq = ++seq_;
    struct waiting {   // the spin thread (if any) steps off the lock while this thread takes it
        std::atomic<int> & w;
        explicit waiting(std::atomic<int> & x) : w(x) { ++w; }
        ~waiting() { --w; }
    };
    try {
        std::vector<pdl::cseg> s;
        uint64_t n_out = 0;
        for (const seg & x : send) { s.push_back({x.p, x.n}); n_out += x.n; }
        const bool pumped = pump_.load();
        std::vector<int> flag((size_t) N, 0);   // completion flags (pumped mode: the spin thread sets them)
        {
            waiting wg(host_waiting_);
            std::lock_guard<std::mutex> l(mu_);
            for (int j = 0; j < N; ++j) {
                if (j == self_) continue;
                std::vector<pdl::mseg> d;
                for (const seg & x : recv.at((size_t) j)) d.push_back({x.p, x.n});
                eng_->post_recv(seq, j, j, (uint8_t) tag, d, (volatile int *) &flag[(size_t) j], 1);
            }
            eng_->post_send(seq, self_, (uint8_t) tag, 0, 0, s, false);
        }
        if (pumped) {
            for (;;) {
                bool all = true;
                for (int j = 0; j < N && all; ++j) if (j != self_) all = ((volatile int *) flag.data())[j] != 0;
                if (all) break;
                if (pump_dead_) {
                    std::lock_guard<std::mutex> l(pump_err_mu_);
                    throw std::runtime_error(pump_err_);
                }
                pdl::cpu_relax();
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            waiting wg(host_waiting_);
            std::lock_guard<std::mutex> l(mu_);
            for (int j = 0; j < N; ++j) if (j != self_) eng_->forget(seq, j);
        }
        while (!pumped) {
            waiting wg(host_waiting_);
            std::lock_guard<std::mutex> l(mu_);
            eng_->poll();
            bool all = true;
            for (int j = 0; j < N && all; ++j) if (j != self_) all = eng_->recv_done(seq, j);
            if (all && !eng_->tx_busy()) {
                for (int j = 0; j < N; ++j) if (j != self_) eng_->forget(seq, j);
                break;
            }
        }
        st_.bytes_out += n_out * (uint64_t) (N - 1);
        for (int j = 0; j < N; ++j) if (j != self_) for (const seg & x : recv[(size_t) j]) st_.bytes_in += x.n;
        ++st_.calls;
        st_.sec += (double) (pdl::now_ns() - t0) * 1e-9;
    } catch (...) {
        broken_ = true;
        throw;
    }
}

std::unique_ptr<net_transport> net_transport_open(const std::vector<std::string> & nodes, int self, int connect_ms, int io_ms) {
    const char * e = getenv("P100E_NET");
    const std::string k = e && *e ? e : "tcp";
    if (k == "tcp") return std::make_unique<tcp_transport>(nodes, self, connect_ms, io_ms);
    throw std::runtime_error("P100E_NET=" + k + ": this build supports tcp only");
}

} // namespace p100e
