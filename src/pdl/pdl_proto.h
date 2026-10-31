// pdl frame protocol: messages (an all-reduce partial, a gather row block, an agreement word) as raw Ethernet frames of
// our own EtherType over a frame_port (pdl_frame.h). Single-threaded poll engine.
//
// Frame: [dst MAC 6][src MAC 6][EtherType 0x88B5 (IEEE 802 "local experimental 1")][pdl header 16][payload]
// pdl header (little endian):
//   0  u32 seq         collective sequence number (net_link's per-link seq / the GPU signal counter)
//   4  u16 chunk       chunk index in the message
//   6  u16 count       chunks in the message (>= 1; a message of 0 bytes is one empty chunk)
//   8  u16 len:14|ar:2 payload bytes of this frame; ar = all-reduce index within the layer (informational)
//   10 u8  layer       model layer (informational, shown in errors)
//   11 u8  type:2|tag:3|stream:3   type DATA 0 / NACK 1 / HELLO 2; tag = the kind of collective (net_link::tag,
//                      lockstep check); stream = the message's source: the sending node (exchange) or global rank (GPU)
//   12 u32 crc         CRC32C of the header (crc field 0) + payload, 0 when the link runs without (hello agrees)
// A message of B bytes is cut into count = ceil(B / cp) chunks of cp = the link's chunk payload (min over the nodes of
// max_frame - 30, rounded down to a multiple of 4: 1484 at MTU 1500, 8984 at 9000); chunk k carries bytes
// [k cp, min((k+1) cp, B)). The receiver scatters each chunk straight into its destination segments (any order,
// duplicates ignored by a per-message bitmap): the caller sees the message in exact byte order.
//
// Loss (a direct cable loses almost nothing, but USB / host hiccups and the fault injection of the tests do): the
// receiver NACKs the missing chunks of a started message when no new frame came for nack_gap_us, or of a message it
// has seen nothing of after nack_first_us (the peer may only be slower: a NACK for a message not sent yet is ignored by
// the sender), repeating at most every nack_repeat_us. The sender keeps the last send_ring messages per stream (copied
// unless the caller guarantees the memory stays put) and resends what is asked. No progress on an expected message
// for io_timeout_ms -> std::runtime_error naming node, seq, tag and the missing chunks (the run is over, as with a P2P
// all-reduce timeout). A frame of the wrong tag or size for a posted message = the nodes left lockstep -> error.
// Frames for messages not posted yet (a faster peer) are staged (copied) and applied at post_recv.
//
// Bye (teardown): a hello frame with the bye bit, see engine::send_bye.
// Hello: every node broadcasts {magic, protocol, node count, its index, its max payload, crc on/off, the set of nodes
// it has heard, a random session id} until it has heard everyone and everyone has heard it; this learns each peer's
// MAC and session and fixes cp. Once connected, only a known peer's MAC is listened to: DATA / NACK frames must come
// from it and be addressed to this adapter, and a hello counts only from that MAC with its learned session and the
// run's settings; anything else on the segment is counted as foreign, never fatal (no frame from a stranger can abort
// a running session). This is no authentication: a host on the same L2 segment that forges a peer's MAC can still
// inject frames (CRC32C is a checksum, not a key); run the adapters on their own cable or VLAN.
// (A 16-byte hello without the session id, from an older build, is still accepted; its session counts as 0.)
#pragma once

#include "pdl/pdl_frame.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace p100e::pdl {

constexpr uint16_t PDL_ETHERTYPE = 0x88b5;
constexpr uint16_t PDL_PROTO = 1;
constexpr size_t ETH_HDR = 14, PDL_HDR = 16;
constexpr int PDL_MAX_STREAMS = 8, PDL_MAX_NODES = 8;
enum pdl_ftype : uint8_t { F_DATA = 0, F_NACK = 1, F_HELLO = 2 };

struct proto_opts {
    int nack_gap_us = 400;        // started message, no new frame this long -> NACK the missing chunks
    int nack_first_us = 5000;     // expected message, nothing at all this long -> NACK all
    int nack_repeat_us = 1000;    // at most one NACK per message this often
    int send_ring = 8;            // messages per stream kept for resend
    bool crc = true;              // CRC32C per frame (Ethernet FCS already guards the wire; this guards USB + host)
    int io_timeout_ms = 60000;    // no progress on an expected message this long -> error
    // bytes of early frames kept for messages not posted yet (P100E_PDL_STAGING_MB). Lockstep bounds the legitimate
    // amount to about one message per stream and peer (a node sends message k + 1 only after it received the peer's k,
    // which the peer sent after posting its receive of k); the guard catches bugs, not flow control.
    size_t staging_max = 256u << 20;
};

struct proto_stats {
    uint64_t frames_tx = 0, frames_rx = 0, bytes_tx = 0, bytes_rx = 0;   // data frames / payload bytes
    uint64_t nacks_tx = 0, nacks_rx = 0, chunks_resent = 0, dup_chunks = 0, crc_errors = 0, staged_chunks = 0;
    uint64_t foreign_frames = 0, stale_frames = 0, hellos_tx = 0, hellos_rx = 0, tx_stalls = 0;
    uint64_t msgs_tx = 0, msgs_rx = 0;
    uint64_t nack_ignored = 0;  // NACKs not acted on: for chunks not sent yet, already queued again, or stale (see on_nack)
};

struct cseg { const void * p; size_t n; };
struct mseg { void * p; size_t n; };

class engine {
public:
    engine(frame_port & port, int self, int n_nodes, const proto_opts & o);

    // discover the peers (broadcast hello); throws after timeout_ms
    void hello(int timeout_ms);
    // called about every 5 s while hello() waits (ms waited so far): pdl_link prints a hint when the adapter receives
    // nothing at all
    std::function<void(uint64_t)> hello_wait;
    size_t chunk_payload() const { return cp_; }
    int self() const { return self_; }
    int n_nodes() const { return n_; }
    const uint8_t * peer_mac(int j) const { return mac_[j]; }

    // queue message (seq, stream) to every peer. stable = false: the bytes are copied into the send ring now (the
    // caller may reuse src at once); true: src must stay unchanged until send_ring later messages of this stream were
    // posted (zero copy from a ring the GPU writes). Frames go out from poll() as transmit room allows.
    void post_send(uint32_t seq, int stream, uint8_t tag, uint8_t layer, uint8_t ar, const std::vector<cseg> & src, bool stable);
    // expect message (seq, stream) from node `from`: exactly sum(dst.n) bytes, scattered into dst. On completion
    // *flag = flag_val (release order after the data; flag may be null). Applies staged early frames at once.
    void post_recv(uint32_t seq, int stream, int from, uint8_t tag, const std::vector<mseg> & dst, volatile int * flag = nullptr,
                   int flag_val = 0);
    bool recv_done(uint32_t seq, int stream) const;   // completed (and not yet forgotten)
    void forget(uint32_t seq, int stream);            // drop the completion record
    // streams >= min_stream keep no completion record (recv_done is false for them): pdl_gpu's GPU all-reduces, whose
    // completion is the flag word. Late duplicates of them are still dropped (stale by posted_hi_), and the record
    // list stays short for the per-frame scan of early frames.
    void skip_done_records(int min_stream) { done_from_ = min_stream; }
    // teardown of pdl_gpu: forget every message of streams >= min_stream (open receives, early frames, the send ring
    // and queued frames) and ignore their frames and NACKs from now on: their memory is about to be freed. A later
    // post on such a stream takes it up again.
    void drop_streams(int min_stream);
    size_t recvs_open() const { return rx_.size(); }
    size_t staged_bytes() const { return staged_bytes_; }   // early frames held for messages not posted yet
    bool tx_busy() const { return !jobs_.empty(); }   // frames not yet handed to the port

    // one step: receive + reassemble, NACK / resend timers, transmit queued frames. Never blocks. Throws on timeout,
    // lockstep loss, a peer's resend request that fell out of the send ring, queued frames the port took none of for
    // io_timeout_ms (a stuck transmit side).
    void poll();

    // shutdown: "bye" = this node has every message it will ever need. A leaving node keeps polling (answering NACKs)
    // and sending bye until every peer said bye too (or a bound passes): the last message is never stranded.
    void send_bye();
    bool all_bye() const;

    const proto_stats & stats() const { return st_; }
    void set_layer_hint(uint8_t layer, uint8_t ar) { hint_layer_ = layer; hint_ar_ = ar; }

private:
    struct tx_msg {
        uint32_t seq = 0;
        int stream = 0;
        uint8_t tag = 0, layer = 0, ar = 0;
        size_t total = 0;
        uint32_t count = 0;
        std::vector<uint8_t> copy;   // stable == false
        std::vector<cseg> src;
        uint32_t sent_upto[PDL_MAX_NODES]{};   // per peer: chunks [0, n) went out once (the first send is in order)
        uint64_t last_tx[PDL_MAX_NODES]{};     // per peer: when the last frame of this message went to it (ns)
    };
    struct tx_job {
        uint32_t seq;
        int stream, to;
        std::vector<std::pair<uint32_t, uint32_t>> ranges;   // chunk ranges [first, end)
        size_t ri = 0;
        uint32_t next = 0;
        bool resend = false;
    };
    struct rx_msg {
        uint32_t seq = 0;
        int stream = 0, from = 0;
        uint8_t tag = 0;
        std::vector<mseg> dst;
        size_t total = 0;
        uint32_t count = 0, got = 0;
        std::vector<uint8_t> have;
        volatile int * flag = nullptr;
        int flag_val = 0;
        uint64_t t_post = 0, t_last = 0, t_nack = 0;
        uint8_t layer = 0;
    };
    struct staged {
        uint8_t tag = 0;
        uint32_t count = 0;
        std::map<uint32_t, std::vector<uint8_t>> chunks;
        int from = -1;
    };
    static uint64_t key(uint32_t seq, int stream) { return (uint64_t) seq << 8 | (uint32_t) stream; }

    static void on_frame_cb(void * ctx, const uint8_t * f, size_t n) { ((engine *) ctx)->on_frame(f, n); }
    void on_frame(const uint8_t * f, size_t n);
    void on_data(int from, const uint8_t * h, const uint8_t * pl, size_t len);
    void on_nack(int from, const uint8_t * h, const uint8_t * pl, size_t len);
    bool nack_stale(int from, uint32_t seq, int stream) const;
    void on_hello(const uint8_t * f, const uint8_t * pl, size_t len);
    void apply_chunk(rx_msg & m, uint32_t idx, const uint8_t * pl, size_t len);
    void complete(rx_msg & m);
    void send_hello(bool bcast_all);
    void send_nack(rx_msg & m, uint64_t now);
    bool push_jobs();
    uint8_t * frame_begin(const uint8_t * dst, size_t payload, uint8_t type, uint32_t seq, uint32_t idx, uint32_t cnt,
                          uint8_t tag, int stream, uint8_t layer, uint8_t ar);
    void frame_end(uint8_t * hdr, size_t payload);
    size_t chunk_len(size_t total, uint32_t idx) const;
    int node_of(const uint8_t * mac) const;
    std::string who(int j) const;
    [[noreturn]] void fail(const std::string & m);

    frame_port & port_;
    int self_, n_;
    proto_opts o_;
    size_t cp_ = 0;
    uint8_t mac_[PDL_MAX_NODES][6]{};
    bool known_[PDL_MAX_NODES]{};
    uint16_t heard_ = 0, peer_heard_[PDL_MAX_NODES]{};
    size_t peer_payload_[PDL_MAX_NODES]{};
    uint32_t session_ = 0, peer_session_[PDL_MAX_NODES]{};   // random per engine; peers' as learned at hello (0 = none)
    bool connected_ = false, bye_ = false;
    uint16_t peer_bye_ = 0;
    uint64_t last_hello_reply_[PDL_MAX_NODES]{};

    std::deque<tx_msg> ring_[PDL_MAX_STREAMS];
    uint32_t newest_sent_[PDL_MAX_STREAMS]{};
    bool any_sent_[PDL_MAX_STREAMS]{};
    std::deque<tx_job> jobs_;
    std::map<uint64_t, rx_msg> rx_;
    std::deque<uint64_t> done_;            // completed keys, newest last (bounded)
    std::map<uint64_t, staged> staged_;
    uint32_t posted_hi_[PDL_MAX_STREAMS]{};   // per stream: the newest seq ever posted (receives are posted in seq order)
    // per peer and stream family (0: host exchanges, streams < n_nodes; 1: GPU all-reduces, streams >= n_nodes): the
    // newest seq of any data frame it sent us. Lockstep: a peer sends message k + 1 of a family only after it has every
    // message k of that family, so a NACK of ours for seq <= k - 1 ... k is stale once we saw its k + 1 (on_nack).
    uint32_t peer_hi_[PDL_MAX_NODES][2]{};
    bool peer_any_[PDL_MAX_NODES][2]{};
    struct held_nack { std::vector<uint8_t> f; uint64_t t = 0; int polls = 0; };
    std::vector<held_nack> nacks_;   // NACK frames held for a moment before acting on them (poll)
    bool any_posted_[PDL_MAX_STREAMS]{};
    bool dropped_[PDL_MAX_STREAMS]{};      // drop_streams
    int done_from_ = PDL_MAX_STREAMS;      // skip_done_records
    size_t staged_bytes_ = 0;
    uint64_t last_progress_ = 0;
    uint64_t tx_ok_ = 0;                   // queued frames: the last time one went to the port (transmit stall bound)
    uint8_t hint_layer_ = 0, hint_ar_ = 0;
    proto_stats st_;
};

} // namespace p100e::pdl
