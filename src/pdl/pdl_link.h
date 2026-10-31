// pdl transport backend with net_link's interface (src/net_link.h), so tp_model's collectives do not depend on the link.
//
// net_transport  the interface: exchange / allgather / agree / stats, exactly net_link's public API and semantics
//                (SPMD lockstep, every node calls the same collectives in the same order; seg / tag types are
//                net_link's). allgather and agree are built on exchange here, as in net_link.
// tcp_transport  wraps net_link unchanged (the default).
// pdl_link       the same over the pdl engine (pdl_proto.h) on any frame_port (pdl_frame.h): exchange() = post every
//                peer's receive, post our message (copied into the send ring: the caller's buffers are free on return,
//                as with TCP), then poll until every peer's message is in. The calling thread busy-polls (no sleep, no
//                interrupt): that is the point.
//                One thread at a time drives the engine; pdl_gpu (pdl_gpu.cuh) may run a spin thread over the same
//                link, then exchange() takes the engine lock per poll step.
//
// Node naming: pdl only uses the node count and this node's index (MACs are learned by the hello). This repository
// ships no frame_port for real hardware, so net_transport_open offers TCP only.
#pragma once

#include "net_link.h"
#include "pdl/pdl_proto.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace p100e {

class net_transport {
public:
    using seg = net_link::seg;
    virtual ~net_transport() = default;
    virtual const char * kind() const = 0;   // "tcp" / "pdl"
    virtual int n_nodes() const = 0;
    virtual int self() const = 0;
    virtual const std::string & node_name(int i) const = 0;
    virtual void exchange(uint32_t tag, const std::vector<seg> & send, const std::vector<std::vector<seg>> & recv, int timeout_ms = 0) = 0;
    virtual const net_stats & stats() const = 0;
    virtual int io_timeout_ms() const = 0;
    // busy-poll window in us (net_link: P100E_NET_SPIN / _SPIN_US; > 0 = on). pdl always busy-polls.
    virtual int spin_us() const = 0;
    // every node's v, in node order (one exchange of 8 bytes, tag T_AGREE)
    virtual std::vector<uint64_t> allgather(uint64_t v, int timeout_ms = 0);
    // every node must hold the same v; otherwise throws "the nodes disagree on <what>: ..." (net_link's text)
    virtual void agree(const std::string & what, uint64_t v, int timeout_ms = 0);
};

class tcp_transport : public net_transport {
public:
    tcp_transport(const std::vector<std::string> & nodes, int self, int connect_ms, int io_ms) : l_(nodes, self, connect_ms, io_ms) {}
    const char * kind() const override { return "tcp"; }
    int n_nodes() const override { return l_.n_nodes(); }
    int self() const override { return l_.self(); }
    const std::string & node_name(int i) const override { return l_.node_name(i); }
    void exchange(uint32_t tag, const std::vector<seg> & s, const std::vector<std::vector<seg>> & r, int t = 0) override { l_.exchange(tag, s, r, t); }
    std::vector<uint64_t> allgather(uint64_t v, int t = 0) override { return l_.allgather(v, t); }
    void agree(const std::string & what, uint64_t v, int t = 0) override { l_.agree(what, v, t); }
    const net_stats & stats() const override { return l_.stats(); }
    int io_timeout_ms() const override { return l_.io_timeout_ms(); }
    int spin_us() const override { return l_.spin_us(); }
    net_link & link() { return l_; }

private:
    net_link l_;
};

class pdl_link : public net_transport {
public:
    // over an existing frame port; the port must outlive the link
    pdl_link(const std::vector<std::string> & nodes, int self, pdl::frame_port & port, int connect_ms, int io_ms,
             const pdl::proto_opts & po);
    ~pdl_link() override;

    const char * kind() const override { return "pdl"; }
    int n_nodes() const override { return (int) nodes_.size(); }
    int self() const override { return self_; }
    const std::string & node_name(int i) const override { return nodes_.at((size_t) i); }
    void exchange(uint32_t tag, const std::vector<seg> & send, const std::vector<std::vector<seg>> & recv, int timeout_ms = 0) override;
    const net_stats & stats() const override { return st_; }
    int io_timeout_ms() const override { return io_ms_; }
    int spin_us() const override { return 1; }   // > 0 = on: exchange() polls without sleeping for the whole wait

    // for pdl_gpu: the engine and its lock (whoever polls holds it)
    pdl::engine & eng() { return *eng_; }
    std::mutex & mu() { return mu_; }
    pdl::frame_port & port() { return *port_; }
    // sequence numbers: exchange() uses its own counter; pdl_gpu's GPU all-reduces use stream ids >= n_nodes() with
    // the GPU's counter, so the two never collide
    uint32_t seq() const { return seq_; }
    // a host collective wants the engine lock: the spin thread steps aside for that moment (std::mutex is not fair: a
    // thread that re-locks in a tight loop can starve the other one for a long time)
    bool host_waiting() const { return host_waiting_.load(std::memory_order_relaxed) != 0; }
    // pdl_gpu's spin thread polls the engine continuously (set_pump(true)): exchange() then only posts its messages
    // under the lock and waits on completion flags (the spin thread also sends the GPU's partials, which a host
    // collective must never hold up). pump_failed: the spin thread died; a waiting exchange() throws its message.
    void set_pump(bool on) { pump_.store(on); }
    void pump_failed(const std::string & why);

private:
    void init(int connect_ms);
    std::vector<std::string> nodes_;
    int self_ = 0, io_ms_ = 60000;
    pdl::frame_port * port_ = nullptr;
    pdl::proto_opts po_;
    std::unique_ptr<pdl::engine> eng_;
    std::mutex mu_;
    uint32_t seq_ = 0;
    std::atomic<int> host_waiting_{0};
    std::atomic<bool> pump_{false}, pump_dead_{false};
    std::string pump_err_;
    std::mutex pump_err_mu_;
    net_stats st_;
    bool broken_ = false;
};

// P100E_NET=tcp (the default and only choice here): the transport tp_model::net_init should construct
std::unique_ptr<net_transport> net_transport_open(const std::vector<std::string> & nodes, int self, int connect_ms, int io_ms);

} // namespace p100e
