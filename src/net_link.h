// Network link of a tensor-parallel run whose ranks sit on several machines. Plain TCP, one persistent
// full-duplex connection per node pair (TCP_NODELAY, 4 MiB socket buffers), Winsock on Windows, POSIX sockets elsewhere.
// No threads: every collective drives all of this node's sockets nonblocking through one poll loop, so a node sends and
// receives at once (two nodes sending each other 1 MB never wait on each other's socket buffers).
//
// SPMD lockstep (tp_model): every node runs the same program with the same arguments, so every node reaches the same
// collectives in the same order with the same sizes. The link only moves bytes: each message carries a header (magic,
// a tag naming the kind of collective, a per-link sequence number, the byte count) and the receiver checks all four, so a
// node that left the common control flow is caught at its next collective ("lockstep lost") instead of adding garbage.
//
// Setup: nodes = "host:port" per node in global rank order; node i listens on its own port when a later node exists
// (bound to its own host when that resolves to a local address, else to every interface) and connects to every earlier
// node, retrying until the connect timeout. A hello per connection carries the protocol version, the node count and
// the sender's index (a listener drops a mismatching or silent peer and keeps waiting for the right one). With
// P100E_NET_KEY_FILE set, every connection then runs the key handshake of net_auth.h (a peer without the key is dropped).
//
// Failure: a peer that closes or resets its connection (process killed, crashed) is reported at once; a peer that goes
// silent (machine off, cable) after the I/O timeout of no progress. Either throws std::runtime_error naming the node;
// the link is unusable afterwards (the caller exits). Nothing ever waits without a timeout.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace p100e {

struct net_stats {
    uint64_t bytes_out = 0, bytes_in = 0;   // payload bytes (headers not counted)
    uint64_t calls = 0;                     // collectives
    double sec = 0;                         // wall time inside the collectives (incl. waiting for a slower peer)
};

class net_link {
public:
    static constexpr uint32_t PROTO = 1;
    // message kinds (the header tag; a mismatch = the nodes are at different collectives)
    // T_PIPE: the hybrid pipeline's handoffs (residual rows node 0 -> 1, the result back)
    enum tag : uint32_t { T_AGREE = 1, T_ALLREDUCE = 2, T_PREFILL = 3, T_LOGITS = 4, T_ARGMAX = 5, T_GATHER = 6, T_PIPE = 7 };
    struct seg { void * p; size_t n; };

    // connect_ms: how long to wait for every peer to come up (listen / connect retries); io_ms: the default
    // no-progress timeout of a collective
    net_link(const std::vector<std::string> & nodes, int self, int connect_ms, int io_ms);
    ~net_link();
    net_link(const net_link &) = delete;
    net_link & operator=(const net_link &) = delete;

    int n_nodes() const { return (int) nodes_.size(); }
    int self() const { return self_; }
    const std::string & node_name(int i) const { return nodes_.at(i); }

    // the collective: send the concatenation of `send` to every other node and receive node j's message into the
    // segments recv[j] (recv[self] ignored; its total size is what node j must send). timeout_ms <= 0: the default.
    void exchange(uint32_t tag, const std::vector<seg> & send, const std::vector<std::vector<seg>> & recv, int timeout_ms = 0);
    // every node's v, in node order
    std::vector<uint64_t> allgather(uint64_t v, int timeout_ms = 0);
    // every node must hold the same v; otherwise throws "the nodes disagree on <what>: node 0 a, node 1 b ..."
    void agree(const std::string & what, uint64_t v, int timeout_ms = 0);

    const net_stats & stats() const { return st_; }
    int io_timeout_ms() const { return io_ms_; }
    int spin_us() const { return spin_us_; }

private:
    struct peer;
    std::vector<std::string> nodes_;
    int self_ = 0;
    int io_ms_ = 60000;
    int spin_us_ = 5000;          // busy-poll window after the last progress (P100E_NET_SPIN=0: off; P100E_NET_SPIN_US)
    std::vector<peer *> peers_;   // by node index (nullptr at self)
    uint64_t seq_ = 0;
    net_stats st_;
    bool broken_ = false;
};

// "host:port" -> host, port (throws on a malformed entry)
void net_parse_node(const std::string & s, std::string & host, int & port);

} // namespace p100e
