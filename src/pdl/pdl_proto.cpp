// pdl frame protocol engine (see pdl_proto.h)
#include "pdl/pdl_proto.h"
#include "pdl/pdl_util.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>

namespace p100e::pdl {

namespace {
constexpr uint32_t HELLO_MAGIC = 0x484c4450u;   // "PDLH"
const uint8_t BCAST[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
bool newer(uint32_t a, uint32_t b) { return (int32_t) (a - b) > 0; }   // wrap-safe a > b
const char * tag_name(uint8_t t) {
    static const char * n[] = {"?", "agree", "allreduce", "prefill exchange", "logits gather", "argmax gather", "gather", "gpu allreduce"};
    return t < 8 ? n[t] : "?";
}
} // namespace

engine::engine(frame_port & port, int self, int n_nodes, const proto_opts & o) : port_(port), self_(self), n_(n_nodes), o_(o) {
    if (n_nodes < 2 || n_nodes > PDL_MAX_NODES) throw std::runtime_error("pdl: node count " + std::to_string(n_nodes) + " not in 2..8");
    if (self < 0 || self >= n_nodes) throw std::runtime_error("pdl: node index " + std::to_string(self) + " out of range");
    if (o_.send_ring < 2) o_.send_ring = 2;
    cp_ = ((port.max_frame() - ETH_HDR - PDL_HDR) / 4) * 4;
    if (cp_ > 0x3fff) cp_ = 0x3ffc;   // 14-bit length field
    memcpy(mac_[self], port.mac(), 6);
    known_[self] = true;
    heard_ = (uint16_t) (1u << self);
    peer_payload_[self] = cp_;
    last_progress_ = now_ns();
    // session id: tells this run's hellos from a restarted process's on the same adapter, and a peer's from a stranger's
    session_ = (uint32_t) last_progress_ ^ (uint32_t) (last_progress_ >> 32) ^ (uint32_t) mac_[self][5] << 24;
    try { std::random_device rd; session_ ^= (uint32_t) rd(); } catch (...) {}   // the clock alone if it has no source
    if (session_ == 0) session_ = 1;   // 0 = an older build without one
}

std::string engine::who(int j) const {
    char b[64];
    snprintf(b, sizeof b, "node %d (%02x:%02x:%02x:%02x:%02x:%02x)", j, mac_[j][0], mac_[j][1], mac_[j][2], mac_[j][3], mac_[j][4], mac_[j][5]);
    return b;
}

void engine::fail(const std::string & m) { throw std::runtime_error("pdl: " + m); }

int engine::node_of(const uint8_t * mac) const {
    for (int j = 0; j < n_; ++j)
        if (known_[j] && j != self_ && !memcmp(mac_[j], mac, 6)) return j;
    return -1;
}

size_t engine::chunk_len(size_t total, uint32_t idx) const {
    if (total == 0) return 0;
    const size_t off = (size_t) idx * cp_;
    return std::min(cp_, total - off);
}

// ---- framing ----------------------------------------------------------------------------------------------------------
uint8_t * engine::frame_begin(const uint8_t * dst, size_t payload, uint8_t type, uint32_t seq, uint32_t idx, uint32_t cnt,
                              uint8_t tag, int stream, uint8_t layer, uint8_t ar) {
    uint8_t * p = port_.tx_frame(ETH_HDR + PDL_HDR + payload);
    if (!p) return nullptr;
    memcpy(p, dst, 6);
    memcpy(p + 6, mac_[self_], 6);
    p[12] = (uint8_t) (PDL_ETHERTYPE >> 8);
    p[13] = (uint8_t) PDL_ETHERTYPE;
    uint8_t * h = p + ETH_HDR;
    st32(h, seq);
    st16(h + 4, (uint16_t) idx);
    st16(h + 6, (uint16_t) cnt);
    st16(h + 8, (uint16_t) ((payload & 0x3fff) | (size_t) (ar & 3) << 14));
    h[10] = layer;
    h[11] = (uint8_t) ((type & 3) << 6 | (tag & 7) << 3 | (stream & 7));
    st32(h + 12, 0);
    return h;
}

void engine::frame_end(uint8_t * h, size_t payload) {
    if (o_.crc) st32(h + 12, crc32c(h, PDL_HDR + payload));   // crc field is 0 while computing
}

void engine::send_hello(bool) {
    uint8_t * h = frame_begin(BCAST, 20, F_HELLO, 0, 0, 1, 0, 0, 0, 0);
    if (!h) return;
    uint8_t * pl = h + PDL_HDR;
    st32(pl, HELLO_MAGIC);
    st16(pl + 4, PDL_PROTO);
    pl[6] = (uint8_t) n_;
    pl[7] = (uint8_t) self_;
    st16(pl + 8, (uint16_t) ((port_.max_frame() - ETH_HDR - PDL_HDR) / 4 * 4));
    st16(pl + 10, heard_);
    pl[12] = o_.crc ? 1 : 0;
    pl[13] = connected_ ? 1 : 0;
    pl[14] = bye_ ? 1 : 0;
    pl[15] = 0;
    st32(pl + 16, session_);   // (older builds read the first 16 bytes only)
    frame_end(h, 20);
    port_.tx_flush();
    ++st_.hellos_tx;
}

void engine::hello(int timeout_ms) {
    const uint64_t t0 = now_ns();
    uint64_t last = 0;
    const uint16_t all = (uint16_t) ((1u << n_) - 1);
    uint64_t next_cb = t0 + 5000000000ull;
    for (;;) {
        const uint64_t now = now_ns();
        if (now - last > 20000000ull) { send_hello(true); last = now; }
        if (hello_wait && now > next_cb) { next_cb = now + 5000000000ull; hello_wait((now - t0) / 1000000); }
        port_.rx_poll(on_frame_cb, this);
        bool done = heard_ == all;
        for (int j = 0; j < n_ && done; ++j) if (j != self_) done = (peer_heard_[j] >> self_) & 1;
        if (done) break;
        if ((now - t0) / 1000000 > (uint64_t) timeout_ms) {
            std::string miss;
            for (int j = 0; j < n_; ++j)
                if (j != self_ && (!known_[j] || !((peer_heard_[j] >> self_) & 1)))
                    miss += (miss.empty() ? "" : ", ") + std::to_string(j) + (known_[j] ? " (does not hear us)" : " (silent)");
            fail("hello: node(s) " + miss + " not reached in " + std::to_string(timeout_ms / 1000) +
                 " s (other side running with the same node count? cable / link up? adapter in pdl mode on both ends?)");
        }
        cpu_relax();
    }
    connected_ = true;
    send_hello(true);   // the last word for peers that still wait to see themselves in our heard set
    cp_ = peer_payload_[self_];
    for (int j = 0; j < n_; ++j) cp_ = std::min(cp_, peer_payload_[j]);
    if (cp_ < 64 || cp_ % 4) fail("chunk payload " + std::to_string(cp_) + " bytes (want a multiple of 4, at least 64)");   // on_hello checks each
    last_progress_ = now_ns();
}

void engine::on_hello(const uint8_t * f, const uint8_t * pl, size_t len) {
    if (len < 16 || ld32(pl) != HELLO_MAGIC) { ++st_.foreign_frames; return; }
    const int pn = pl[6], j = pl[7];
    const uint32_t sess = len >= 20 ? ld32(pl + 16) : 0;   // 0: an older build without session ids
    if (connected_) {
        // running: only a known peer's own hellos count (its bye, "I do not hear you yet"). A hello from another
        // adapter, from another session (a restarted process) or with other settings is no peer of this run:
        // ignored, never fatal, so one broadcast frame on the segment cannot abort the session
        const int k = node_of(f + 6);
        if (k < 0 || k != j || pn != n_ || ld16(pl + 4) != PDL_PROTO || (pl[12] != 0) != o_.crc || sess != peer_session_[k]) {
            ++st_.foreign_frames;
            return;
        }
        ++st_.hellos_rx;
        peer_heard_[j] = ld16(pl + 10);
    } else {
        ++st_.hellos_rx;
        if (ld16(pl + 4) != PDL_PROTO) fail("a peer speaks pdl protocol " + std::to_string(ld16(pl + 4)) + ", this build " + std::to_string(PDL_PROTO));
        if (pn != n_) fail("a peer runs with " + std::to_string(pn) + " nodes, this node with " + std::to_string(n_));
        if (j == self_ || j >= n_) fail("a peer claims node index " + std::to_string(j) + " (this node is " + std::to_string(self_) + ")");
        if ((pl[12] != 0) != o_.crc) fail("crc setting differs between the nodes (P100E_PDL_CRC)");
        if (known_[j] && memcmp(mac_[j], f + 6, 6)) fail("two adapters claim node " + std::to_string(j));
        // the chunk payload divides every message into chunks and bounds the NACK ranges: 0 would divide by zero
        const size_t pp = ld16(pl + 8);
        if (pp < 64 || pp % 4) fail("node " + std::to_string(j) + " advertises a chunk payload of " + std::to_string(pp) + " bytes (want a multiple of 4, at least 64)");
        memcpy(mac_[j], f + 6, 6);
        known_[j] = true;
        heard_ |= (uint16_t) (1u << j);
        peer_heard_[j] = ld16(pl + 10);
        peer_payload_[j] = pp;
        peer_session_[j] = sess;   // the latest: a peer process restarted before the run began starts a new one
    }
    if (pl[14]) { peer_bye_ |= (uint16_t) (1u << j); return; }
    const uint64_t now = now_ns();
    const bool they_miss_us = !((peer_heard_[j] >> self_) & 1);
    if ((they_miss_us || pl[13] == 0) && now - last_hello_reply_[j] > 2000000ull) {
        last_hello_reply_[j] = now;
        send_hello(false);
    }
}

// ---- receive ------------------------------------------------------------------------------------------------------------
void engine::send_bye() {
    bye_ = true;
    send_hello(true);
}

bool engine::all_bye() const {
    for (int j = 0; j < n_; ++j) if (j != self_ && !((peer_bye_ >> j) & 1)) return false;
    return true;
}

void engine::on_frame(const uint8_t * f, size_t n) {
    if (n < ETH_HDR + PDL_HDR || f[12] != (uint8_t) (PDL_ETHERTYPE >> 8) || f[13] != (uint8_t) PDL_ETHERTYPE) {
        ++st_.foreign_frames;
        return;
    }
    const uint8_t * h = f + ETH_HDR;
    const size_t len = ld16(h + 8) & 0x3fff;
    if (ETH_HDR + PDL_HDR + len > n) { ++st_.foreign_frames; return; }
    if (o_.crc) {
        uint8_t hz[PDL_HDR];
        memcpy(hz, h, PDL_HDR);
        st32(hz + 12, 0);
        const uint32_t c = crc32c_update(crc32c(hz, PDL_HDR), h + PDL_HDR, len);
        if (c != ld32(h + 12)) { ++st_.crc_errors; return; }   // lost frame: the NACK path recovers it
    }
    const uint8_t type = h[11] >> 6;
    if (type == F_HELLO) { on_hello(f, h + PDL_HDR, len); return; }
    const int from = node_of(f + 6);
    // DATA / NACK go unicast to this adapter (frame_begin): a broadcast or someone else's never counts
    if (from < 0 || memcmp(f, mac_[self_], 6)) { ++st_.foreign_frames; return; }
    if (type == F_DATA) on_data(from, h, h + PDL_HDR, len);
    else if (type == F_NACK) nacks_.push_back(held_nack{std::vector<uint8_t>(f, f + ETH_HDR + PDL_HDR + len), now_ns(), 0});   // see poll()
    else ++st_.foreign_frames;
}

void engine::apply_chunk(rx_msg & m, uint32_t idx, const uint8_t * pl, size_t len) {
    size_t off = (size_t) idx * cp_, left = len;
    for (const mseg & s : m.dst) {   // scatter into the destination segments (few: one per rank)
        if (!left) break;
        if (off >= s.n) { off -= s.n; continue; }
        const size_t k = std::min(left, s.n - off);
        memcpy((uint8_t *) s.p + off, pl, k);
        pl += k;
        left -= k;
        off = 0;
    }
    m.have[idx] = 1;
    ++m.got;
    st_.bytes_rx += len;
}

void engine::complete(rx_msg & m) {
    if (m.flag) {
        std::atomic_thread_fence(std::memory_order_release);   // the payload stores before the flag store
        *m.flag = m.flag_val;
    }
    if (m.stream < done_from_) {   // (skip_done_records)
        done_.push_back(key(m.seq, m.stream));
        if (done_.size() > 512) done_.pop_front();
    }
    ++st_.msgs_rx;
}

void engine::on_data(int from, const uint8_t * h, const uint8_t * pl, size_t len) {
    const uint32_t seq = ld32(h), idx = ld16(h + 4), cnt = ld16(h + 6);
    const uint8_t tag = (h[11] >> 3) & 7, layer = h[10];
    const int stream = h[11] & 7;
    ++st_.frames_rx;
    const int fam = stream >= n_ ? 1 : 0;
    if (!peer_any_[from][fam] || newer(seq, peer_hi_[from][fam])) peer_hi_[from][fam] = seq;
    peer_any_[from][fam] = true;
    if (dropped_[stream]) { ++st_.stale_frames; return; }   // drop_streams: its memory is gone
    const uint64_t k = key(seq, stream);
    auto it = rx_.find(k);
    if (it != rx_.end()) {
        rx_msg & m = it->second;
        if (m.from != from) fail("message seq " + std::to_string(seq) + " stream " + std::to_string(stream) + " came from " + who(from) + ", expected " + who(m.from));
        if (m.tag != tag || cnt != m.count || idx >= cnt || len != chunk_len(m.total, idx))
            fail("lockstep lost with " + who(from) + ": this node expects " + tag_name(m.tag) + " #" + std::to_string(seq) + " of " +
                 std::to_string(m.total) + " bytes (" + std::to_string(m.count) + " chunks), the peer sent " + tag_name(tag) +
                 " chunk " + std::to_string(idx) + "/" + std::to_string(cnt) + " of " + std::to_string(len) + " bytes (layer " +
                 std::to_string(layer) + "; same arguments, environment and model on every node?)");
        if (m.have[idx]) { ++st_.dup_chunks; return; }
        apply_chunk(m, idx, pl, len);
        m.t_last = now_ns();
        m.layer = layer;
        last_progress_ = m.t_last;
        if (m.got == m.count) { complete(m); rx_.erase(it); }
        return;
    }
    for (auto d = done_.rbegin(); d != done_.rend(); ++d)
        if (*d == k) { ++st_.stale_frames; return; }   // a resend of something we already have
    // A frame of a message at or before the newest one posted on its stream that is not open any more: completed (and
    // maybe forgotten by pdl_link::exchange, so done_ no longer has it). Receives of a stream are posted in seq order,
    // so it can never become wanted again. Before this check such late duplicates were staged as "early" frames that
    // no post would ever claim: with a slow peer every exchange leaked its resent copies until staging_max (the
    // failure seen with a slower node 0).
    if (any_posted_[stream] && !newer(seq, posted_hi_[stream])) { ++st_.stale_frames; return; }
    staged & s = staged_[k];
    if (s.from < 0) { s.from = from; s.tag = tag; s.count = cnt; }
    if (s.from != from || s.tag != tag || s.count != cnt) fail("early frames of seq " + std::to_string(seq) + " disagree (tag / size / sender)");
    if (s.chunks.count(idx)) { ++st_.dup_chunks; return; }
    staged_bytes_ += len;
    if (staged_bytes_ > o_.staging_max) {
        std::string w;   // what is staged: per stream the oldest early seq vs the newest posted one
        int shown = 0;
        for (auto & kv : staged_) {
            if (shown++ == 6) { w += " ..."; break; }
            const int st = (int) (kv.first & 0xff);
            w += " seq " + std::to_string(kv.first >> 8) + "/stream " + std::to_string(st) + " (" + std::to_string(kv.second.chunks.size()) +
                 " chunks; newest posted " + (any_posted_[st] ? std::to_string(posted_hi_[st]) : std::string("none")) + ")";
        }
        fail("more than " + std::to_string(o_.staging_max >> 20) + " MiB of frames for messages this node has not posted (a peer is far "
             "ahead or the nodes left lockstep; P100E_PDL_STAGING_MB raises the bound): " + std::to_string(staged_.size()) +
             " messages staged:" + w);
    }
    s.chunks[idx].assign(pl, pl + len);
    ++st_.staged_chunks;
}

void engine::post_recv(uint32_t seq, int stream, int from, uint8_t tag, const std::vector<mseg> & dst, volatile int * flag, int flag_val) {
    if (stream < 0 || stream >= PDL_MAX_STREAMS) fail("stream out of range");
    if (from < 0 || from >= n_ || from == self_) fail("post_recv from node " + std::to_string(from));
    const uint64_t k = key(seq, stream);
    if (rx_.count(k)) fail("message seq " + std::to_string(seq) + " stream " + std::to_string(stream) + " posted twice");
    dropped_[stream] = false;
    rx_msg m;
    m.seq = seq; m.stream = stream; m.from = from; m.tag = tag; m.dst = dst; m.flag = flag; m.flag_val = flag_val;
    for (const mseg & s : dst) m.total += s.n;
    const size_t cnt = m.total == 0 ? 1 : (m.total + cp_ - 1) / cp_;
    if (cnt > 0xffff) fail("message of " + std::to_string(m.total) + " bytes needs more than 65535 chunks");
    m.count = (uint32_t) cnt;
    m.have.assign(cnt, 0);
    m.t_post = m.t_last = now_ns();
    if (!any_posted_[stream] || newer(seq, posted_hi_[stream])) posted_hi_[stream] = seq;
    any_posted_[stream] = true;
    auto st = staged_.find(k);
    if (st != staged_.end()) {
        staged & s = st->second;
        if (s.from != from || s.tag != tag || s.count != m.count)
            fail("lockstep lost with " + who(s.from) + ": early frames of " + tag_name(s.tag) + " #" + std::to_string(seq) + " (" +
                 std::to_string(s.count) + " chunks) do not match the posted " + tag_name(tag) + " of " + std::to_string(m.total) + " bytes");
        for (auto & c : s.chunks) {
            if (c.first >= m.count || c.second.size() != chunk_len(m.total, c.first)) fail("early frame of the wrong size");
            apply_chunk(m, c.first, c.second.data(), c.second.size());
            staged_bytes_ -= c.second.size();
        }
        staged_.erase(st);
    }
    if (m.got == m.count) { complete(m); return; }
    rx_.emplace(k, std::move(m));
}

bool engine::recv_done(uint32_t seq, int stream) const {
    const uint64_t k = key(seq, stream);
    if (rx_.count(k)) return false;
    for (auto d = done_.rbegin(); d != done_.rend(); ++d) if (*d == k) return true;
    return false;
}

void engine::forget(uint32_t seq, int stream) {
    const uint64_t k = key(seq, stream);
    for (auto d = done_.begin(); d != done_.end(); ++d) if (*d == k) { done_.erase(d); return; }
}

// ---- loss: NACK + resend -------------------------------------------------------------------------------------------------
void engine::send_nack(rx_msg & m, uint64_t now) {
    // payload: u16 ranges, then (u16 first, u16 n) pairs; (0, 0) = everything
    std::vector<std::pair<uint32_t, uint32_t>> r;
    if (m.got == 0) r.push_back({0, 0});
    else
        for (uint32_t i = 0; i < m.count;) {
            if (m.have[i]) { ++i; continue; }
            uint32_t e = i;
            while (e < m.count && !m.have[e] && e - i < 0xffff) ++e;
            r.push_back({i, e - i});
            i = e;
        }
    const size_t max_r = (cp_ - 2) / 4;
    if (r.size() > max_r) r.resize(max_r);
    const size_t pl = 2 + 4 * r.size();
    uint8_t * h = frame_begin(mac_[m.from], pl, F_NACK, m.seq, 0, 1, m.tag, m.stream, m.layer, 0);
    if (!h) return;   // no transmit room: next poll
    st16(h + PDL_HDR, (uint16_t) r.size());
    for (size_t i = 0; i < r.size(); ++i) {
        st16(h + PDL_HDR + 2 + 4 * i, (uint16_t) r[i].first);
        st16(h + PDL_HDR + 4 + 4 * i, (uint16_t) r[i].second);
    }
    frame_end(h, pl);
    m.t_nack = now;
    ++st_.nacks_tx;
}

// Stale by lockstep: the peer already sent a later message of this stream family, so it has this one. A peer sends
// message k + 1 of a family only after it has every node's message k of that family (host exchanges: exchange() returns
// only with every peer's message; GPU all-reduces: a rank publishes k + 1 after its wait for every partial of k).
bool engine::nack_stale(int from, uint32_t seq, int stream) const {
    const int fam = stream >= n_ ? 1 : 0;
    return from >= 0 && peer_any_[from][fam] && newer(peer_hi_[from][fam], seq);
}

void engine::on_nack(int from, const uint8_t * h, const uint8_t * pl, size_t len) {
    ++st_.nacks_rx;
    const uint32_t seq = ld32(h);
    const int stream = h[11] & 7;
    if (dropped_[stream]) { ++st_.nack_ignored; return; }   // drop_streams: nothing left to resend from
    const tx_msg * msg = nullptr;
    for (const tx_msg & t : ring_[stream]) if (t.seq == seq) { msg = &t; break; }
    if (!msg) {
        if (!any_sent_[stream] || newer(seq, newest_sent_[stream])) { ++st_.nack_ignored; return; }   // not sent yet: it will come
        fail(who(from) + " asks again for message seq " + std::to_string(seq) + " stream " + std::to_string(stream) +
             ", which left the send ring (" + std::to_string(o_.send_ring) + " messages): it fell too far behind");
    }
    if (len < 2) return;
    if (nack_stale(from, seq, stream)) { ++st_.nack_ignored; return; }   // (poll() checks before it calls; kept for safety)
    // Resend only what went out once already: a waiting receiver NACKs everything every nack_repeat_us while this node
    // is still computing, and those NACKs queue up in the adapter until this node polls again, which is right after
    // it posted the message. Resending the whole message once per queued NACK multiplied the traffic of a slow node
    // (and every copy after the first was a late duplicate at the receiver). Chunks not sent yet are on their way.
    // A resend job still queued for this message and peer covers the rest (the missing set only shrinks).
    // Stale NACKs: one the receiver sent just before our frames reached it arrives right after we sent them. Ignore
    // NACKs within nack_gap_us of the last frame of this message to that peer: the receiver itself waits nack_gap_us
    // after its last frame before it asks for a real loss (and repeats every nack_repeat_us), so a real NACK always
    // comes later than that and is honoured.
    if (now_ns() - msg->last_tx[from] < (uint64_t) o_.nack_gap_us * 1000) { ++st_.nack_ignored; return; }
    for (const tx_job & q : jobs_)
        if (q.resend && q.seq == seq && q.stream == stream && q.to == from) { ++st_.nack_ignored; return; }
    const uint32_t lim = msg->sent_upto[from];
    const size_t nr = std::min<size_t>(ld16(pl), (len - 2) / 4);
    tx_job j{seq, stream, from, {}, 0, 0, true};
    for (size_t i = 0; i < nr; ++i) {
        const uint32_t a = ld16(pl + 2 + 4 * i), n = ld16(pl + 4 + 4 * i);
        const bool all = a == 0 && n == 0;   // (0, 0) = everything
        const uint32_t e = std::min(lim, all ? msg->count : std::min(msg->count, a + n));
        if (a < e) j.ranges.push_back({a, e});
        if (all) break;
    }
    if (j.ranges.empty()) { ++st_.nack_ignored; return; }
    j.next = j.ranges[0].first;
    if (jobs_.empty()) tx_ok_ = now_ns();   // the transmit stall bound starts with the first queued frame
    jobs_.push_front(j);   // resends first: the peer is waiting on exactly these
}

// ---- send --------------------------------------------------------------------------------------------------------------
void engine::post_send(uint32_t seq, int stream, uint8_t tag, uint8_t layer, uint8_t ar, const std::vector<cseg> & src, bool stable) {
    if (!connected_) fail("post_send before hello");
    if (stream < 0 || stream >= PDL_MAX_STREAMS) fail("stream out of range");
    dropped_[stream] = false;
    if (jobs_.empty()) tx_ok_ = now_ns();   // the transmit stall bound starts with the first queued frame
    tx_msg m;
    m.seq = seq; m.stream = stream; m.tag = tag; m.layer = layer; m.ar = ar;
    for (const cseg & s : src) m.total += s.n;
    const size_t cnt = m.total == 0 ? 1 : (m.total + cp_ - 1) / cp_;
    if (cnt > 0xffff) fail("message of " + std::to_string(m.total) + " bytes needs more than 65535 chunks");
    m.count = (uint32_t) cnt;
    if (stable) m.src = src;
    else {
        m.copy.resize(m.total);
        size_t o = 0;
        for (const cseg & s : src) { if (s.n) memcpy(m.copy.data() + o, s.p, s.n); o += s.n; }
        m.src = {{m.copy.data(), m.total}};
    }
    auto & ring = ring_[stream];
    ring.push_back(std::move(m));
    while ((int) ring.size() > o_.send_ring) ring.pop_front();
    newest_sent_[stream] = seq;
    any_sent_[stream] = true;
    for (int j = 0; j < n_; ++j)
        if (j != self_) jobs_.push_back(tx_job{seq, stream, j, {{0, (uint32_t) cnt}}, 0, 0, false});
    ++st_.msgs_tx;
}

bool engine::push_jobs() {
    while (!jobs_.empty()) {
        tx_job & j = jobs_.front();
        tx_msg * msg = nullptr;
        for (tx_msg & t : ring_[j.stream]) if (t.seq == j.seq) { msg = &t; break; }
        if (!msg) fail("send ring too small: message seq " + std::to_string(j.seq) + " left it before all its frames went out");
        while (j.ri < j.ranges.size()) {
            if (j.next < j.ranges[j.ri].first) j.next = j.ranges[j.ri].first;
            if (j.next >= j.ranges[j.ri].second) { if (++j.ri < j.ranges.size()) j.next = j.ranges[j.ri].first; continue; }
            const uint32_t idx = j.next;
            const size_t L = chunk_len(msg->total, idx);
            uint8_t * h = frame_begin(mac_[j.to], L, F_DATA, j.seq, idx, msg->count, msg->tag, j.stream, msg->layer, msg->ar);
            if (!h) { ++st_.tx_stalls; port_.tx_flush(); return false; }
            // gather chunk bytes [idx cp, + L) from the source segments
            size_t off = (size_t) idx * cp_, left = L;
            uint8_t * d = h + PDL_HDR;
            for (const cseg & s : msg->src) {
                if (!left) break;
                if (off >= s.n) { off -= s.n; continue; }
                const size_t k = std::min(left, s.n - off);
                memcpy(d, (const uint8_t *) s.p + off, k);
                d += k;
                left -= k;
                off = 0;
            }
            frame_end(h, L);
            ++st_.frames_tx;
            st_.bytes_tx += L;
            if (j.resend) ++st_.chunks_resent;
            else if (idx + 1 > msg->sent_upto[j.to]) msg->sent_upto[j.to] = idx + 1;
            msg->last_tx[j.to] = now_ns();
            ++j.next;
        }
        jobs_.pop_front();
    }
    return true;
}

void engine::poll() {
    port_.rx_poll(on_frame_cb, this);
    // NACKs: held for at least one more rx batch and 50 us. A stale one (sent by the peer just before our frames reached
    // it, then waiting in our adapter while this node computed) is usually followed by the peer's next message, which
    // proves it stale (nack_stale); the next message may sit in the following USB buffer. A real loss costs 50 us more.
    {
        const uint64_t tn = now_ns();
        size_t keep = 0;
        for (size_t i = 0; i < nacks_.size(); ++i) {
            held_nack & n = nacks_[i];
            const uint8_t * h = n.f.data() + ETH_HDR;
            const int from = node_of(n.f.data() + 6);
            if (nack_stale(from, ld32(h), h[11] & 7)) { ++st_.nacks_rx; ++st_.nack_ignored; continue; }
            if (n.polls++ >= 1 && tn - n.t >= 50000) { on_nack(from, h, h + PDL_HDR, n.f.size() - ETH_HDR - PDL_HDR); continue; }
            if (keep != i) nacks_[keep] = std::move(n);
            ++keep;
        }
        nacks_.resize(keep);
    }
    const uint64_t now = now_ns();
    const uint64_t gap = (uint64_t) o_.nack_gap_us * 1000, first = (uint64_t) o_.nack_first_us * 1000,
                   rep = (uint64_t) o_.nack_repeat_us * 1000, to = (uint64_t) o_.io_timeout_ms * 1000000;
    for (auto & kv : rx_) {
        rx_msg & m = kv.second;
        const uint64_t idle = now - m.t_last;
        if (idle > to) {
            std::string miss;
            int shown = 0;
            for (uint32_t i = 0; i < m.count && shown < 8; ++i) if (!m.have[i]) { miss += (shown ? "," : "") + std::to_string(i); ++shown; }
            fail("no progress for " + std::to_string(o_.io_timeout_ms / 1000) + " s on " + tag_name(m.tag) + " #" + std::to_string(m.seq) +
                 " from " + who(m.from) + " (stream " + std::to_string(m.stream) + ", layer " + std::to_string(m.layer) + "): " +
                 std::to_string(m.got) + " of " + std::to_string(m.count) + " chunks, missing " + miss +
                 " (peer hung, crashed, cable or adapter gone; P100E_NET_TIMEOUT_S raises the limit)");
        }
        const bool due = m.got ? idle > gap : now - m.t_post > first;
        if (due && now - m.t_nack > rep) send_nack(m, now);
    }
    const uint64_t ftx = st_.frames_tx;
    push_jobs();
    port_.tx_flush();
    // Every receive may be complete while our own frames sit queued: the receive timeout above cannot see a transmit
    // side that stopped completing (all bulk-OUT transfers stuck in flight), so bound that wait as well
    if (jobs_.empty() || st_.frames_tx != ftx) tx_ok_ = now;
    else if (now > tx_ok_ && now - tx_ok_ > to)
        fail("transmit stalled: " + std::to_string(jobs_.size()) + " sends queued and the adapter took no frame for " +
             std::to_string(o_.io_timeout_ms / 1000) + " s (USB bulk-OUT stuck, adapter gone?; P100E_NET_TIMEOUT_S raises the limit)");
}

void engine::drop_streams(int min_stream) {
    if (min_stream < 0) min_stream = 0;
    auto gone = [&](uint64_t k) { return (int) (k & 0xff) >= min_stream; };
    for (int s = min_stream; s < PDL_MAX_STREAMS; ++s) { dropped_[s] = true; ring_[s].clear(); }
    for (auto it = rx_.begin(); it != rx_.end();) {
        if (gone(it->first)) it = rx_.erase(it);
        else ++it;
    }
    for (auto it = staged_.begin(); it != staged_.end();) {
        if (!gone(it->first)) { ++it; continue; }
        for (auto & c : it->second.chunks) staged_bytes_ -= c.second.size();
        it = staged_.erase(it);
    }
    jobs_.erase(std::remove_if(jobs_.begin(), jobs_.end(), [&](const tx_job & j) { return j.stream >= min_stream; }), jobs_.end());
    done_.erase(std::remove_if(done_.begin(), done_.end(), gone), done_.end());
}

} // namespace p100e::pdl
