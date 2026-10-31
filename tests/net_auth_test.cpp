// p100e-net-auth-test: the node-link key handshake (src/net_auth.h) - SHA-256 / HMAC-SHA256 against published test
// vectors, and the handshake between two threads over in-memory pipes (same key: both pass; other key, a peer without
// a key: both refuse). No network, no GPU.
#include "net_auth.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

using namespace p100e;

static int fails = 0;
#define EXPECT(c) do { if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static std::string hex(const uint8_t * d) {
    static const char * x = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < 32; ++i) { s += x[d[i] >> 4]; s += x[d[i] & 15]; }
    return s;
}

// one direction of a byte pipe; recv throws after 2 s without data (a stuck handshake fails instead of hanging)
struct pipe_ {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<uint8_t> q;
    void send(const void * p, size_t n) {
        std::lock_guard<std::mutex> lk(mu);
        q.insert(q.end(), (const uint8_t *) p, (const uint8_t *) p + n);
        cv.notify_all();
    }
    void recv(void * p, size_t n) {
        std::unique_lock<std::mutex> lk(mu);
        if (!cv.wait_for(lk, std::chrono::seconds(2), [&] { return q.size() >= n; })) throw std::runtime_error("pipe timeout");
        for (size_t i = 0; i < n; ++i) { ((uint8_t *) p)[i] = q.front(); q.pop_front(); }
    }
};

static net_key make_key(const char * s) {
    net_key k;
    k.on = true;
    k.k.assign((const uint8_t *) s, (const uint8_t *) s + strlen(s));
    return k;
}

// connect side on this thread, accept side on another; returns both results ("" = passed, "io: ..." = an exception)
static void run_pair(const net_key & kc, const net_key & ka, std::string & rc, std::string & ra) {
    pipe_ c2a, a2c;
    std::thread t([&] {
        try {
            ra = net_auth_handshake(ka, NET_ROLE_ACCEPT, [&](const void * p, size_t n) { a2c.send(p, n); },
                                    [&](void * p, size_t n) { c2a.recv(p, n); });
        } catch (const std::exception & e) { ra = std::string("io: ") + e.what(); }
    });
    try {
        rc = net_auth_handshake(kc, NET_ROLE_CONNECT, [&](const void * p, size_t n) { c2a.send(p, n); },
                                [&](void * p, size_t n) { a2c.recv(p, n); });
    } catch (const std::exception & e) { rc = std::string("io: ") + e.what(); }
    t.join();
}

int main() {
    EXPECT(net_auth_selftest());
    uint8_t d[32];
    sha256("", 0, d);
    EXPECT(hex(d) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    const char * m2 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";   // FIPS 180-4: two blocks
    sha256(m2, strlen(m2), d);
    EXPECT(hex(d) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    {   // RFC 4231 test case 2 (short key)
        const char * k = "Jefe", * m = "what do ya want for nothing?";
        hmac_sha256(k, strlen(k), m, strlen(m), d);
        EXPECT(hex(d) == "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    }
    {   // RFC 4231 test case 6 (a key longer than the block: hashed first)
        uint8_t k[131];
        memset(k, 0xaa, sizeof k);
        const char * m = "Test Using Larger Than Block-Size Key - Hash Key First";
        hmac_sha256(k, sizeof k, m, strlen(m), d);
        EXPECT(hex(d) == "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54");
    }
    {   // the RNG gives different nonces
        uint8_t a[32], b[32];
        net_random(a, sizeof a); net_random(b, sizeof b);
        EXPECT(!net_ct_equal(a, b, sizeof a));
    }
    const net_key k1 = make_key("0123456789abcdef-key-one"), k2 = make_key("0123456789abcdef-key-two");
    std::string rc, ra;
    run_pair(k1, k1, rc, ra);
    EXPECT(rc.empty() && ra.empty());
    run_pair(k1, k2, rc, ra);
    EXPECT(!rc.empty() && !ra.empty() && rc.find("io:") == std::string::npos && ra.find("io:") == std::string::npos);
    {   // a peer that sends something else (here: a 36-byte non-handshake message) is refused, not trusted
        pipe_ c2a, a2c;
        uint8_t junk[36] = {1, 2, 3, 4};
        c2a.send(junk, sizeof junk);
        c2a.send(junk, 32);
        std::string r = net_auth_handshake(k1, NET_ROLE_ACCEPT, [&](const void * p, size_t n) { a2c.send(p, n); },
                                           [&](void * p, size_t n) { c2a.recv(p, n); });
        EXPECT(!r.empty());
    }
    {   // reflection: a peer echoing this side's own messages back must not pass
        pipe_ loop;
        std::string r;
        try {
            r = net_auth_handshake(k1, NET_ROLE_ACCEPT, [&](const void * p, size_t n) { loop.send(p, n); },
                                   [&](void * p, size_t n) { loop.recv(p, n); });
        } catch (const std::exception & e) { r = e.what(); }
        EXPECT(!r.empty());
    }
    net_key off;
    EXPECT(net_auth_handshake(off, NET_ROLE_CONNECT, [](const void *, size_t) { throw std::runtime_error("no io"); },
                              [](void *, size_t) { throw std::runtime_error("no io"); }).empty());   // no key: no handshake
    printf("%s (%d failure(s))\n", fails ? "FAILED" : "all net_auth tests passed", fails);
    return fails ? 1 : 0;
}
