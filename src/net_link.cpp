// TCP link between the nodes of a network tensor-parallel run: see net_link.h.
#include "net_link.h"
#include "net_auth.h"

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  if !defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0600
#    undef _WIN32_WINNT
#    define _WIN32_WINNT 0x0600   // WSAPoll
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#else
#  include <arpa/inet.h>
#  include <errno.h>
#  include <fcntl.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <poll.h>
#  include <sys/select.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace p100e {

namespace {

#ifdef _WIN32
using sock_t = SOCKET;
const sock_t BAD_SOCK = INVALID_SOCKET;
using pollfd_t = WSAPOLLFD;
int sock_err() { return WSAGetLastError(); }
bool would_block(int e) { return e == WSAEWOULDBLOCK || e == WSAEINPROGRESS || e == WSAEALREADY || e == WSAEINTR; }
void sock_close(sock_t s) { if (s != BAD_SOCK) closesocket(s); }
int do_poll(pollfd_t * f, size_t n, int ms) { return WSAPoll(f, (ULONG) n, ms); }
bool set_nonblock(sock_t s) { u_long m = 1; return ioctlsocket(s, FIONBIO, &m) == 0; }
const int SEND_FLAGS = 0;
std::string err_str(int e) {
    switch (e) {
        case WSAECONNRESET: return "connection reset by the peer (WSAECONNRESET)";
        case WSAECONNABORTED: return "connection aborted (WSAECONNABORTED)";
        case WSAECONNREFUSED: return "connection refused (WSAECONNREFUSED)";
        case WSAETIMEDOUT: return "timed out (WSAETIMEDOUT)";
        case WSAEADDRINUSE: return "address in use (WSAEADDRINUSE)";
        case WSAEHOSTUNREACH: return "host unreachable (WSAEHOSTUNREACH)";
        case WSAENETUNREACH: return "network unreachable (WSAENETUNREACH)";
        default: return "Winsock error " + std::to_string(e);
    }
}
void net_startup() {
    static std::once_flag once;
    std::call_once(once, [] {
        WSADATA d;
        if (WSAStartup(MAKEWORD(2, 2), &d) != 0) throw std::runtime_error("net: WSAStartup failed");
    });
}
#else
using sock_t = int;
const sock_t BAD_SOCK = -1;
using pollfd_t = struct pollfd;
int sock_err() { return errno; }
bool would_block(int e) { return e == EAGAIN || e == EWOULDBLOCK || e == EINPROGRESS || e == EALREADY || e == EINTR; }
void sock_close(sock_t s) { if (s >= 0) ::close(s); }
int do_poll(pollfd_t * f, size_t n, int ms) { return ::poll(f, (nfds_t) n, ms); }
bool set_nonblock(sock_t s) { const int fl = fcntl(s, F_GETFL, 0); return fl >= 0 && fcntl(s, F_SETFL, fl | O_NONBLOCK) == 0; }
const int SEND_FLAGS = MSG_NOSIGNAL;   // a closed peer is an error return, not SIGPIPE
std::string err_str(int e) { return std::string(strerror(e)) + " (errno " + std::to_string(e) + ")"; }
void net_startup() {}
#endif

using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

constexpr uint32_t HELLO_MAGIC = 0x484e3150u;   // "P1NH"
constexpr uint32_t MSG_MAGIC = 0x4d4e3150u;     // "P1NM"
constexpr uint32_t HELLO_AUTH = 0x100;          // hello.proto flag: this node runs the key handshake (net_auth.h)
struct hello { uint32_t magic, proto, n_nodes, node; };
struct msg_hdr { uint32_t magic, tag; uint64_t seq, bytes; };
static_assert(sizeof(msg_hdr) == 24, "msg_hdr layout");

const char * tag_name(uint32_t t) {
    switch (t) {
        case net_link::T_AGREE: return "agree";
        case net_link::T_ALLREDUCE: return "allreduce";
        case net_link::T_PREFILL: return "prefill exchange";
        case net_link::T_LOGITS: return "logits gather";
        case net_link::T_ARGMAX: return "argmax gather";
        case net_link::T_GATHER: return "gather";
        case net_link::T_PIPE: return "pipe handoff";
        default: return "?";
    }
}

void tune(sock_t s) {
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char *) &one, sizeof one);
    int buf = 4 << 20;
    setsockopt(s, SOL_SOCKET, SO_SNDBUF, (const char *) &buf, sizeof buf);
    setsockopt(s, SOL_SOCKET, SO_RCVBUF, (const char *) &buf, sizeof buf);
}

bool resolve(const std::string & host, int port, sockaddr_in & out) {
    memset(&out, 0, sizeof out);
    out.sin_family = AF_INET;
    out.sin_port = htons((uint16_t) port);
    if (host.empty() || host == "*" || host == "0.0.0.0") { out.sin_addr.s_addr = htonl(INADDR_ANY); return true; }
    addrinfo hints{}, * res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || !res) return false;
    out.sin_addr = ((sockaddr_in *) res->ai_addr)->sin_addr;
    freeaddrinfo(res);
    return true;
}

// whole buffer over a nonblocking socket, bounded by deadline (setup only)
void io_all(sock_t s, void * p, size_t n, bool out, clk::time_point deadline, const std::string & who) {
    char * c = (char *) p;
    while (n) {
        const int k = out ? (int) send(s, c, (int) std::min<size_t>(n, 1 << 20), SEND_FLAGS)
                          : (int) recv(s, c, (int) std::min<size_t>(n, 1 << 20), 0);
        if (k > 0) { c += k; n -= (size_t) k; continue; }
        if (k == 0 && !out) throw std::runtime_error("net: " + who + " closed the connection during the handshake");
        const int e = sock_err();
        if (k < 0 && !would_block(e)) throw std::runtime_error("net: " + who + ": " + err_str(e));
        const double left = std::chrono::duration<double, std::milli>(deadline - clk::now()).count();
        if (left <= 0) throw std::runtime_error("net: " + who + ": handshake timed out");
        pollfd_t f{};
        f.fd = s;
        f.events = out ? POLLOUT : POLLIN;
        do_poll(&f, 1, (int) std::min(left, 200.0) + 1);
    }
}

// the key handshake (net_auth.h) over a nonblocking setup socket: "" = the peer proved the key
std::string auth_peer(sock_t s, uint8_t role, clk::time_point deadline, const std::string & who) {
    return net_auth_handshake(net_auth_key(), role,
                              [&](const void * p, size_t n) { io_all(s, (void *) p, n, true, deadline, who); },
                              [&](void * p, size_t n) { io_all(s, p, n, false, deadline, who); });
}

std::string ip_of(const sockaddr_in & a) {
    char ip[64] = "?";
    inet_ntop(AF_INET, (void *) &a.sin_addr, ip, sizeof ip);
    return ip;
}

} // namespace

struct net_link::peer {
    sock_t s = BAD_SOCK;
};

void net_parse_node(const std::string & s, std::string & host, int & port) {
    const size_t c = s.rfind(':');
    char * end = nullptr;
    const long p = c == std::string::npos ? -1 : strtol(s.c_str() + c + 1, &end, 10);
    if (c == std::string::npos || !end || *end || p < 1 || p > 65535)
        throw std::runtime_error("net: node '" + s + "' is not host:port (e.g. 192.0.2.10:29500)");
    host = s.substr(0, c);
    port = (int) p;
}

net_link::net_link(const std::vector<std::string> & nodes, int self, int connect_ms, int io_ms)
    : nodes_(nodes), self_(self), io_ms_(io_ms > 0 ? io_ms : 60000) {
    if (const char * e = getenv("P100E_NET_SPIN"); e && *e && atoi(e) == 0) spin_us_ = 0;
    else if (const char * u = getenv("P100E_NET_SPIN_US"); u && *u) spin_us_ = atoi(u);
    net_startup();
    const net_key & key = net_auth_key();   // P100E_NET_KEY_FILE (unset: one warning, no handshake, as before)
    const uint32_t proto = PROTO | (key.on ? HELLO_AUTH : 0);
    const int N = (int) nodes.size();
    if (N < 2) throw std::runtime_error("net: --net-nodes needs at least 2 nodes");
    if (self < 0 || self >= N)
        throw std::runtime_error("net: --net-node " + std::to_string(self) + " is not in 0.." + std::to_string(N - 1));
    peers_.assign(N, nullptr);
    std::vector<std::string> host(N);
    std::vector<int> port(N);
    for (int i = 0; i < N; ++i) net_parse_node(nodes[i], host[i], port[i]);
    const auto deadline = clk::now() + std::chrono::milliseconds(std::max(1000, connect_ms));
    sock_t ls = BAD_SOCK;
    auto fail = [&](const std::string & m) {
        sock_close(ls);
        for (auto *& p : peers_) if (p) { sock_close(p->s); delete p; p = nullptr; }
        throw std::runtime_error(m);
    };
    try {
        // 1. listen first (a later node's connect then queues in the backlog while this node connects to earlier ones)
        if (self < N - 1) {
            sockaddr_in a{};
            if (!resolve(host[self], port[self], a)) {
                a.sin_addr.s_addr = htonl(INADDR_ANY);
                printf("net: warning: node %d's host '%s' does not resolve: listening on every interface\n", self, host[self].c_str());
            }
            ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            if (ls == BAD_SOCK) fail("net: socket(): " + err_str(sock_err()));
#ifndef _WIN32
            int one = 1;
            setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char *) &one, sizeof one);
#endif
            if (bind(ls, (sockaddr *) &a, sizeof a) != 0) {
                // the own host entry is not an address of this machine (e.g. a NAT name): every interface
                a.sin_addr.s_addr = htonl(INADDR_ANY);
                printf("net: warning: node %d cannot bind its host '%s': listening on every interface\n", self, host[self].c_str());
                if (bind(ls, (sockaddr *) &a, sizeof a) != 0)
                    fail("net: node " + std::to_string(self) + " cannot listen on port " + std::to_string(port[self]) + ": " +
                         err_str(sock_err()));
            }
            if (listen(ls, 16) != 0 || !set_nonblock(ls)) fail("net: listen(): " + err_str(sock_err()));
            char ip[64] = "?";
            inet_ntop(AF_INET, &a.sin_addr, ip, sizeof ip);
            printf("net: node %d listening on %s:%d for node(s) %d..%d\n", self, ip, port[self], self + 1, N - 1);
        }
        // 2. connect to every earlier node (retry until it listens)
        for (int j = 0; j < self; ++j) {
            sockaddr_in a{};
            if (!resolve(host[j], port[j], a)) fail("net: cannot resolve node " + std::to_string(j) + " host '" + host[j] + "'");
            int tries = 0;
            std::string last;
            for (;;) {
                sock_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
                if (s == BAD_SOCK) fail("net: socket(): " + err_str(sock_err()));
                set_nonblock(s);
                ++tries;
                bool ok = false;
                if (connect(s, (sockaddr *) &a, sizeof a) == 0) ok = true;
                else if (would_block(sock_err())) {
                    fd_set wf, ef;
                    FD_ZERO(&wf); FD_ZERO(&ef);
                    FD_SET(s, &wf); FD_SET(s, &ef);
                    timeval tv{2, 0};
                    if (select((int) s + 1, nullptr, &wf, &ef, &tv) > 0 && FD_ISSET(s, &wf)) {
                        int e = 0;
                        socklen_t el = sizeof e;
                        getsockopt(s, SOL_SOCKET, SO_ERROR, (char *) &e, &el);
                        ok = e == 0;
                        if (!ok) last = err_str(e);
                    } else {
                        last = "no answer";
                    }
                } else {
                    last = err_str(sock_err());
                }
                if (ok) {
                    tune(s);
                    peers_[j] = new peer{s};
                    break;
                }
                sock_close(s);
                if (clk::now() > deadline)
                    fail("net: node " + std::to_string(self) + " could not connect to node " + std::to_string(j) + " (" +
                         nodes[j] + ") after " + std::to_string(tries) + " tries: " + last +
                         " (is it running with the same --net-nodes? firewall / port?)");
                if (tries == 1) printf("net: node %d waiting for node %d (%s) ...\n", self, j, nodes[j].c_str());
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
            hello h{HELLO_MAGIC, proto, (uint32_t) N, (uint32_t) self}, r{};
            const std::string who = "node " + std::to_string(j) + " (" + nodes[j] + ")";
            io_all(peers_[j]->s, &h, sizeof h, true, deadline, who);
            io_all(peers_[j]->s, &r, sizeof r, false, deadline, who);
            if (r.magic != HELLO_MAGIC) fail("net: " + who + " is not a p100e node (bad hello)");
            if ((r.proto ^ proto) == HELLO_AUTH)
                fail("net: " + who + (key.on ? " runs without a key" : " expects a key") +
                     " (set the same P100E_NET_KEY_FILE on every node)");
            if (r.proto != proto)
                fail("net: " + who + " speaks protocol " + std::to_string(r.proto & ~HELLO_AUTH) + ", this build " + std::to_string(PROTO) +
                     " (rebuild both nodes from the same commit)");
            if ((int) r.n_nodes != N || (int) r.node != j)
                fail("net: " + who + " says it is node " + std::to_string(r.node) + " of " + std::to_string(r.n_nodes) +
                     ", expected node " + std::to_string(j) + " of " + std::to_string(N) + " (--net-nodes / --net-node differ)");
            if (key.on) {
                const std::string why = auth_peer(peers_[j]->s, NET_ROLE_CONNECT, deadline, who);
                if (!why.empty()) fail("net: " + who + ": key handshake failed: " + why);
            }
            printf("net: node %d connected to node %d (%s)\n", self, j, nodes[j].c_str());
        }
        // 3. accept every later node
        int need = N - 1 - self;
        while (need > 0) {
            const double left = std::chrono::duration<double, std::milli>(deadline - clk::now()).count();
            if (left <= 0) {
                std::string miss;
                for (int j = self + 1; j < N; ++j) if (!peers_[j]) miss += (miss.empty() ? "" : ", ") + std::to_string(j);
                fail("net: node " + std::to_string(self) + ": node(s) " + miss + " did not connect in time");
            }
            pollfd_t f{};
            f.fd = ls;
            f.events = POLLIN;
            if (do_poll(&f, 1, (int) std::min(left, 500.0) + 1) <= 0) continue;
            sockaddr_in ca{};
            socklen_t cl = sizeof ca;
            sock_t s = accept(ls, (sockaddr *) &ca, &cl);
            if (s == BAD_SOCK) continue;
            set_nonblock(s);
            tune(s);
            hello r{};
            try {
                io_all(s, &r, sizeof r, false, std::min(deadline, clk::now() + std::chrono::seconds(10)), "a connecting peer");
            } catch (const std::exception &) {
                sock_close(s);
                continue;
            }
            if (r.magic != HELLO_MAGIC) { sock_close(s); continue; }   // a stray connection
            const int j = (int) r.node;
            if (r.proto != proto || (int) r.n_nodes != N || j <= self || j >= N || peers_[j]) {
                // a misconfigured or stray peer: this node's hello tells it why (it fails there); keep waiting for the
                // right one instead of aborting the whole setup
                hello h{HELLO_MAGIC, proto, (uint32_t) N, (uint32_t) self};
                try { io_all(s, &h, sizeof h, true, clk::now() + std::chrono::seconds(2), "peer"); } catch (...) {}
                sock_close(s);
                printf("net: node %d: refused a peer from %s that says it is node %d of %u with protocol %u%s (this node: %d nodes, "
                       "protocol %u%s; --net-nodes / --net-node / build / P100E_NET_KEY_FILE differ?)\n",
                       self, ip_of(ca).c_str(), j, r.n_nodes, r.proto & ~HELLO_AUTH, (r.proto & HELLO_AUTH) ? " + key" : "",
                       N, PROTO, key.on ? " + key" : "");
                fflush(stdout);
                continue;
            }
            const std::string who = "node " + std::to_string(j) + " (" + ip_of(ca) + ")";
            hello h{HELLO_MAGIC, proto, (uint32_t) N, (uint32_t) self};
            std::string why;
            try {
                io_all(s, &h, sizeof h, true, std::min(deadline, clk::now() + std::chrono::seconds(10)), who);
                if (key.on) why = auth_peer(s, NET_ROLE_ACCEPT, std::min(deadline, clk::now() + std::chrono::seconds(10)), who);
            } catch (const std::exception & e) {
                why = e.what();
            }
            if (!why.empty()) {   // a peer without the key (or one that vanished): drop it, keep waiting
                sock_close(s);
                printf("net: node %d: refused %s: %s\n", self, who.c_str(), why.c_str());
                fflush(stdout);
                continue;
            }
            peers_[j] = new peer{s};
            --need;
            printf("net: node %d accepted node %d (%s)\n", self, j, nodes[j].c_str());
        }
        sock_close(ls);
        ls = BAD_SOCK;
    } catch (...) {
        sock_close(ls);
        for (auto *& p : peers_) if (p) { sock_close(p->s); delete p; p = nullptr; }
        throw;
    }
    fflush(stdout);
}

net_link::~net_link() {
    for (auto *& p : peers_) if (p) { sock_close(p->s); delete p; p = nullptr; }
}

void net_link::exchange(uint32_t tag, const std::vector<seg> & send_segs, const std::vector<std::vector<seg>> & recv_segs,
                        int timeout_ms) {
    if (broken_) throw std::runtime_error("net: the link failed earlier (a peer died or lockstep was lost): restart every node");
    const int N = n_nodes();
    const int to = timeout_ms > 0 ? timeout_ms : io_ms_;
    const auto t0 = clk::now();
    const uint64_t seq = ++seq_;
    uint64_t n_out = 0;
    for (const seg & s : send_segs) n_out += s.n;
    msg_hdr h{MSG_MAGIC, tag, seq, n_out};
    struct tx_t { size_t si = 0, off = 0; bool hdr_done = false; size_t hoff = 0; bool done = false; };
    struct rx_t { msg_hdr h{}; size_t hoff = 0; size_t si = 0, off = 0; uint64_t want = 0; bool done = false; };
    std::vector<tx_t> tx(N);
    std::vector<rx_t> rx(N);
    int pending = 0;
    for (int j = 0; j < N; ++j) {
        if (j == self_) { tx[j].done = rx[j].done = true; continue; }
        for (const seg & s : recv_segs.at(j)) rx[j].want += s.n;
        pending += 2;
    }
    auto die = [&](const std::string & m) {
        broken_ = true;
        throw std::runtime_error(m);
    };
    auto who = [&](int j) { return "node " + std::to_string(j) + " (" + nodes_[j] + ")"; };
    auto where = [&]() { return std::string(" at ") + tag_name(tag) + " #" + std::to_string(seq); };
    auto last = clk::now();
    std::vector<pollfd_t> fds;
    std::vector<int> fj;
    while (pending > 0) {
        bool progress = false;
        for (int j = 0; j < N; ++j) {
            if (j == self_) continue;
            const sock_t s = peers_[j]->s;
            // ---- send: header, then the segments ----
            tx_t & t = tx[j];
            while (!t.done) {
                const char * p;
                size_t n;
                if (!t.hdr_done) { p = (const char *) &h + t.hoff; n = sizeof h - t.hoff; }
                else if (t.si < send_segs.size()) { p = (const char *) send_segs[t.si].p + t.off; n = send_segs[t.si].n - t.off; }
                else { t.done = true; --pending; break; }
                if (n == 0) { if (t.hdr_done) { ++t.si; t.off = 0; } else t.hdr_done = true; continue; }
                const int k = (int) ::send(s, p, (int) std::min<size_t>(n, 1 << 22), SEND_FLAGS);
                if (k <= 0) {
                    const int e = sock_err();
                    if (k < 0 && would_block(e)) break;
                    die("net: sending to " + who(j) + " failed" + where() + ": " + err_str(e) + " (did that node exit? see its log)");
                }
                progress = true;
                if (!t.hdr_done) { t.hoff += (size_t) k; if (t.hoff == sizeof h) t.hdr_done = true; }
                else { t.off += (size_t) k; if (t.off == send_segs[t.si].n) { ++t.si; t.off = 0; } }
            }
            // ---- receive: header (checked), then into the segments ----
            rx_t & r = rx[j];
            while (!r.done) {
                char * p;
                size_t n;
                const bool in_hdr = r.hoff < sizeof(msg_hdr);
                if (in_hdr) { p = (char *) &r.h + r.hoff; n = sizeof(msg_hdr) - r.hoff; }
                else if (r.si < recv_segs[j].size()) {
                    const seg & sg = recv_segs[j][r.si];
                    if (r.off == sg.n) { ++r.si; r.off = 0; continue; }
                    p = (char *) sg.p + r.off; n = sg.n - r.off;
                } else { r.done = true; --pending; break; }
                const int k = (int) ::recv(s, p, (int) std::min<size_t>(n, 1 << 22), 0);
                if (k == 0) die("net: " + who(j) + " closed the connection" + where() + " (the process ended or crashed: see its log)");
                if (k < 0) {
                    const int e = sock_err();
                    if (would_block(e)) break;
                    die("net: receiving from " + who(j) + " failed" + where() + ": " + err_str(e) +
                        " (the process ended or crashed: see its log)");
                }
                progress = true;
                if (in_hdr) {
                    r.hoff += (size_t) k;
                    if (r.hoff == sizeof(msg_hdr)) {
                        if (r.h.magic != MSG_MAGIC) die("net: garbage from " + who(j) + where());
                        if (r.h.seq != seq || r.h.tag != tag || r.h.bytes != r.want)
                            die("net: lockstep lost with " + who(j) + ": this node is at " + tag_name(tag) + " #" +
                                std::to_string(seq) + " expecting " + std::to_string(r.want) + " bytes, the peer sent " +
                                tag_name(r.h.tag) + " #" + std::to_string(r.h.seq) + " with " + std::to_string(r.h.bytes) +
                                " bytes (the nodes ran different control flow: same arguments, environment and model on "
                                "every node?)");
                        if (r.want == 0) { r.done = true; --pending; break; }
                    }
                } else {
                    r.off += (size_t) k;
                }
            }
        }
        if (pending == 0) break;
        if (progress) { last = clk::now(); continue; }
        if (ms_since(last) > to) {
            std::string m;
            for (int j = 0; j < N; ++j) {
                if (j == self_ || (tx[j].done && rx[j].done)) continue;
                m += (m.empty() ? "" : ", ") + who(j) + (rx[j].done ? " (not reading)" : " (no data)");
            }
            die("net: no progress for " + std::to_string(to / 1000) + " s" + where() + ": " + m +
                " (peer hung, machine down or network cut; P100E_NET_TIMEOUT_S raises the limit)");
        }
        // busy-poll (P100E_NET_SPIN, default on): retry the nonblocking send / recv without sleeping for up to
        // spin_us_ after the last progress (a collective usually completes within one peer compute segment, well under
        // that); then sleep in poll(), which still wakes on data. The timeout check above runs either way.
        if (spin_us_ > 0 && ms_since(last) * 1000.0 < spin_us_) continue;
        fds.clear();
        fj.clear();
        for (int j = 0; j < N; ++j) {
            if (j == self_) continue;
            short ev = 0;
            if (!tx[j].done) ev |= POLLOUT;
            if (!rx[j].done) ev |= POLLIN;
            if (!ev) continue;
            pollfd_t f{};
            f.fd = peers_[j]->s;
            f.events = ev;
            fds.push_back(f);
            fj.push_back(j);
        }
        do_poll(fds.data(), fds.size(), 20);
    }
    st_.bytes_out += n_out * (uint64_t) (N - 1);
    for (int j = 0; j < N; ++j) if (j != self_) st_.bytes_in += rx[j].want;
    ++st_.calls;
    st_.sec += ms_since(t0) / 1000.0;
}

std::vector<uint64_t> net_link::allgather(uint64_t v, int timeout_ms) {
    const int N = n_nodes();
    std::vector<uint64_t> all(N, 0);
    all[self_] = v;
    std::vector<std::vector<seg>> rs(N);
    for (int j = 0; j < N; ++j) if (j != self_) rs[j] = {{&all[j], 8}};
    exchange(T_AGREE, {{&v, 8}}, rs, timeout_ms);
    return all;
}

void net_link::agree(const std::string & what, uint64_t v, int timeout_ms) {
    const std::vector<uint64_t> all = allgather(v, timeout_ms);
    bool same = true;
    for (uint64_t x : all) same = same && x == all[0];
    if (same) return;
    std::string m = "net: the nodes disagree on " + what + ":";
    for (size_t j = 0; j < all.size(); ++j) {
        char b[48];
        snprintf(b, sizeof b, " node %zu %llu%s", j, (unsigned long long) all[j], (int) j == self_ ? " (this one)" : "");
        m += b;
    }
    broken_ = true;   // every node throws here (each saw the same values)
    throw std::runtime_error(m + " (run every node with the same model, arguments and P100E_* switches)");
}

} // namespace p100e
