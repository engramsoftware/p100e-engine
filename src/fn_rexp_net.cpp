// Remote expert tier: the TCP connection, see fn_rexp.h.
#include "fn_rexp.h"
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
#    define _WIN32_WINNT 0x0600
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <mstcpip.h>   // SIO_KEEPALIVE_VALS
#else
#  include <arpa/inet.h>
#  include <errno.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <poll.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <unistd.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace p100e {

namespace {
#ifdef _WIN32
using sock_t = SOCKET;
const sock_t BAD = INVALID_SOCKET;
int last_err() { return WSAGetLastError(); }
void close_fd(sock_t s) { if (s != BAD) closesocket(s); }
int poll1(sock_t s, short ev, int ms) { WSAPOLLFD f{}; f.fd = s; f.events = ev; return WSAPoll(&f, 1, ms); }
void startup() {
    static std::once_flag once;
    std::call_once(once, [] { WSADATA d; if (WSAStartup(MAKEWORD(2, 2), &d) != 0) throw std::runtime_error("rexp: WSAStartup failed"); });
}
const int SFLAGS = 0;
#else
using sock_t = int;
const sock_t BAD = -1;
int last_err() { return errno; }
void close_fd(sock_t s) { if (s >= 0) ::close(s); }
int poll1(sock_t s, short ev, int ms) { struct pollfd f{}; f.fd = s; f.events = ev; return ::poll(&f, 1, ms); }
void startup() {}
const int SFLAGS = MSG_NOSIGNAL;
#endif
sock_t S(uint64_t v) { return (sock_t) v; }
// P100E_REXP_SPIN_US=N (default 0 = off): a receive first spins up to N us on non-blocking reads before it sleeps in
// poll(), so a reply that lands within the window is picked up without a scheduler wake-up (both ends read it)
int spin_us() { static const int v = getenv("P100E_REXP_SPIN_US") ? std::max(0, atoi(getenv("P100E_REXP_SPIN_US"))) : 0; return v; }
#ifdef _WIN32
bool would_block() { const int e = WSAGetLastError(); return e == WSAEWOULDBLOCK; }
int recv_nb(sock_t s, char * c, int n) { return (int) ::recv(s, c, n, 0); }   // the socket is non-blocking in spin mode
#else
bool would_block() { return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR; }
int recv_nb(sock_t s, char * c, int n) { return (int) ::recv(s, c, n, MSG_DONTWAIT); }
#endif

void set_spin_mode(sock_t s) {   // after connect / accept
#ifdef _WIN32
    if (spin_us() > 0) { u_long m = 1; ioctlsocket(s, FIONBIO, &m); }   // spin mode: non-blocking reads (sends poll first)
#else
    (void) s;
#endif
}
void tune(sock_t s) {
    int one = 1, buf = 4 << 20;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char *) &one, sizeof one);
    setsockopt(s, SOL_SOCKET, SO_SNDBUF, (const char *) &buf, sizeof buf);
    setsockopt(s, SOL_SOCKET, SO_RCVBUF, (const char *) &buf, sizeof buf);
}

// TCP keepalive: a peer that vanishes without FIN / RST (power, cable, hard crash) ends the connection after about a
// minute instead of leaving a receive waiting forever (node 1 waits between requests with no timeout). A live but idle
// peer answers the probes in its kernel, so idle sessions stay up.
void keepalive(sock_t s) {
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_KEEPALIVE, (const char *) &one, sizeof one);
#ifdef _WIN32
    tcp_keepalive ka{};
    ka.onoff = 1; ka.keepalivetime = 30000; ka.keepaliveinterval = 5000;   // ms; Windows sends 10 probes
    DWORD ret = 0;
    WSAIoctl(s, SIO_KEEPALIVE_VALS, &ka, sizeof ka, nullptr, 0, &ret, nullptr, nullptr);
#elif defined(TCP_KEEPIDLE)
    int idle = 30, intvl = 5, cnt = 6;   // s, s, probes
    setsockopt(s, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof idle);
    setsockopt(s, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof intvl);
    setsockopt(s, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof cnt);
#endif
}

// the key handshake of net_auth.h (P100E_NET_KEY_FILE) on a fresh connection: "" = the peer proved the key
std::string auth_sock(rexp_sock & k, uint8_t role, int timeout_ms) {
    return net_auth_handshake(net_auth_key(), role, [&](const void * p, size_t n) { k.send_all(p, n, timeout_ms); },
                              [&](void * p, size_t n) { k.recv_all(p, n, timeout_ms); });
}

std::string addr_str(const sockaddr * a) {
    char h[INET6_ADDRSTRLEN] = "?";
    int port = 0;
    if (a->sa_family == AF_INET) { auto * p = (const sockaddr_in *) a; inet_ntop(AF_INET, &p->sin_addr, h, sizeof h); port = ntohs(p->sin_port); }
    return std::string(h) + ":" + std::to_string(port);
}
}

void rexp_parse_addr(const std::string & s, std::string & host, int & port) {
    const size_t c = s.rfind(':');
    if (c == std::string::npos || c + 1 >= s.size()) throw std::runtime_error("rexp: address '" + s + "' is not host:port");
    host = s.substr(0, c);
    port = atoi(s.c_str() + c + 1);
    if (port <= 0 || port > 65535) throw std::runtime_error("rexp: bad port in '" + s + "'");
}

rexp_sock::~rexp_sock() { close(); }
rexp_sock::rexp_sock(rexp_sock && o) noexcept : fd_(o.fd_), peer_(std::move(o.peer_)) { o.fd_ = ~(uint64_t) 0; }
rexp_sock & rexp_sock::operator=(rexp_sock && o) noexcept {
    if (this != &o) { close(); fd_ = o.fd_; peer_ = std::move(o.peer_); o.fd_ = ~(uint64_t) 0; }
    return *this;
}
void rexp_sock::close() { if (ok()) { close_fd(S(fd_)); fd_ = ~(uint64_t) 0; } }

rexp_sock rexp_sock::accept_one(const std::string & host, int port, int timeout_ms) {
    startup();
    // P100E_NET_KEY_FILE set: a peer must pass the key handshake; one that fails (or a stray connect) is dropped and the
    // listener keeps waiting. Unset (one warning): the first connection is taken, as before.
    const net_key & key = net_auth_key();
    sock_t ls = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (ls == BAD) throw std::runtime_error("rexp: socket() failed");
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char *) &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t) port);
    if (host.empty() || host == "0.0.0.0" || host == "*") a.sin_addr.s_addr = htonl(INADDR_ANY);
    else if (inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) { close_fd(ls); throw std::runtime_error("rexp: listen host must be an IPv4 address: " + host); }
    if (::bind(ls, (sockaddr *) &a, sizeof a) != 0) { const int e = last_err(); close_fd(ls); throw std::runtime_error("rexp: bind port " + std::to_string(port) + " failed (error " + std::to_string(e) + ")"); }
    if (::listen(ls, key.on ? 4 : 1) != 0) { close_fd(ls); throw std::runtime_error("rexp: listen failed"); }
    const auto t0 = std::chrono::steady_clock::now();
    while (true) {
        int left = -1;
        if (timeout_ms >= 0) left = (int) std::max(0.0, timeout_ms - std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        const int r = poll1(ls, POLLIN, left);
        if (r <= 0) { close_fd(ls); throw std::runtime_error("rexp: no expert server connected to port " + std::to_string(port) + " within " + std::to_string(timeout_ms / 1000) + " s"); }
        sockaddr_in pa{};
#ifdef _WIN32
        int pl = sizeof pa;
#else
        socklen_t pl = sizeof pa;
#endif
        sock_t s = ::accept(ls, (sockaddr *) &pa, &pl);
        if (!key.on) close_fd(ls);
        if (s == BAD) {
            if (key.on) { std::this_thread::sleep_for(std::chrono::milliseconds(20)); continue; }   // EMFILE etc.: no hot spin
            throw std::runtime_error("rexp: accept failed");
        }
        tune(s);
        keepalive(s);
        set_spin_mode(s);
        rexp_sock k;
        k.fd_ = (uint64_t) s;
        k.peer_ = addr_str((sockaddr *) &pa);
        if (!key.on) return k;
        std::string why;
        try { why = auth_sock(k, NET_ROLE_ACCEPT, 10000); } catch (const std::exception & e) { why = e.what(); }
        if (why.empty()) { close_fd(ls); return k; }
        printf("rexp: refused a connection from %s: %s\n", k.peer_.c_str(), why.c_str());   // k closes; keep listening
        fflush(stdout);
    }
}

rexp_sock rexp_sock::connect_to(const std::string & host, int port, int timeout_ms) {
    startup();
    const net_key & key = net_auth_key();   // P100E_NET_KEY_FILE (unset: one warning, no handshake, as before)
    const auto t0 = std::chrono::steady_clock::now();
    while (true) {
        addrinfo hints{}, * res = nullptr;
        hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) == 0 && res) {
            sock_t s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            if (s != BAD) {
                tune(s);
                if (::connect(s, res->ai_addr, (int) res->ai_addrlen) == 0) {
                    keepalive(s);
                    set_spin_mode(s);
                    rexp_sock k;
                    k.fd_ = (uint64_t) s;
                    k.peer_ = host + ":" + std::to_string(port);
                    freeaddrinfo(res);
                    if (!key.on) return k;
                    // a wrong key shows in node 0's answer: stop. No answer (node 0 busy with another peer for up to
                    // 10 s, or it runs without a key): retry until the timeout.
                    std::string why;
                    try { why = auth_sock(k, NET_ROLE_CONNECT, 30000); }
                    catch (const std::exception & e) {
                        printf("rexp: key handshake with %s did not finish (%s); retrying (is P100E_NET_KEY_FILE set on node 0?)\n", k.peer_.c_str(), e.what());
                        fflush(stdout);
                        k.close();
                        res = nullptr;
                    }
                    if (k.ok()) {
                        if (!why.empty()) throw std::runtime_error("rexp: " + k.peer_ + ": key handshake failed: " + why);
                        return k;
                    }
                } else close_fd(s);
            }
            if (res) freeaddrinfo(res);
        }
        if (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() > timeout_ms)
            throw std::runtime_error("rexp: could not connect to " + host + ":" + std::to_string(port));
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void rexp_sock::send_all(const void * p, size_t n, int timeout_ms) {
    const char * c = (const char *) p;
    while (n > 0) {
        if (timeout_ms >= 0 && poll1(S(fd_), POLLOUT, timeout_ms) <= 0) throw std::runtime_error("rexp: send to " + peer_ + " timed out");
        const int chunk = (int) std::min<size_t>(n, (size_t) 1 << 30);
        const int k = (int) ::send(S(fd_), c, chunk, SFLAGS);
        if (k < 0 && spin_us() > 0 && would_block()) continue;   // non-blocking socket (spin mode): poll again
        if (k <= 0) throw std::runtime_error("rexp: connection to " + peer_ + " lost (send, error " + std::to_string(last_err()) + ")");
        c += k; n -= (size_t) k;
    }
}

void rexp_sock::recv_all(void * p, size_t n, int timeout_ms) {
    char * c = (char *) p;
    if (spin_us() > 0 && n > 0) {
        auto t0 = std::chrono::steady_clock::now();
        while (n > 0) {
            const int k = recv_nb(S(fd_), c, (int) std::min<size_t>(n, (size_t) 1 << 30));
            if (k > 0) { c += k; n -= (size_t) k; t0 = std::chrono::steady_clock::now(); continue; }
            if (k == 0) throw std::runtime_error("rexp: " + peer_ + " closed the connection");
            if (!would_block()) throw std::runtime_error("rexp: connection to " + peer_ + " lost (recv, error " + std::to_string(last_err()) + ")");
            if (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() > spin_us()) break;
        }
    }
    while (n > 0) {
        if (timeout_ms >= 0 || spin_us() > 0) {
            const int r = poll1(S(fd_), POLLIN, timeout_ms < 0 ? -1 : timeout_ms);
            if (r == 0) throw std::runtime_error("rexp: no data from " + peer_ + " for " + std::to_string(timeout_ms / 1000) + " s");
            if (r < 0) throw std::runtime_error("rexp: poll failed");
        }
        const int chunk = (int) std::min<size_t>(n, (size_t) 1 << 30);
        int k = (int) ::recv(S(fd_), c, chunk, 0);
        if (k < 0 && spin_us() > 0 && would_block()) continue;
        if (k == 0) throw std::runtime_error("rexp: " + peer_ + " closed the connection");
        if (k < 0) throw std::runtime_error("rexp: connection to " + peer_ + " lost (recv, error " + std::to_string(last_err()) + ")");
        c += k; n -= (size_t) k;
    }
}

void rexp_sock::send_msg(rexp_hdr h, const void * payload, size_t bytes, int timeout_ms) {
    h.magic = REXP_MAGIC;
    h.bytes = bytes;
    if (bytes <= (256u << 10)) {   // small: one send (one TCP push)
        thread_local std::vector<char> buf;
        buf.resize(sizeof h + bytes);
        memcpy(buf.data(), &h, sizeof h);
        if (bytes) memcpy(buf.data() + sizeof h, payload, bytes);
        send_all(buf.data(), buf.size(), timeout_ms);
        return;
    }
    send_all(&h, sizeof h, timeout_ms);
    send_all(payload, bytes, timeout_ms);
}

rexp_hdr rexp_sock::recv_hdr(uint32_t want, int timeout_ms) {
    rexp_hdr h;
    recv_all(&h, sizeof h, timeout_ms);
    if (h.magic != REXP_MAGIC) throw std::runtime_error("rexp: bad message from " + peer_ + " (magic)");
    if (want && h.op != want) throw std::runtime_error("rexp: unexpected message " + std::to_string(h.op) + " from " + peer_ + " (want " + std::to_string(want) + ")");
    return h;
}

} // namespace p100e
