// Pre-shared key handshake of the node links: see net_auth.h.
#include "net_auth.h"

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <bcrypt.h>
#else
#  include <errno.h>
#  include <fcntl.h>
#  include <unistd.h>
#  if defined(__has_include)
#    if __has_include(<sys/random.h>)
#      include <sys/random.h>
#      define P100E_HAVE_GETRANDOM 1
#    endif
#  endif
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace p100e {

namespace {

// ---- SHA-256 (FIPS 180-4) ----
const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

struct sha256_ctx {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t buf[64] = {};
    size_t nbuf = 0;
    uint64_t total = 0;   // bytes

    void block(const uint8_t * p) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t) p[4 * i] << 24 | (uint32_t) p[4 * i + 1] << 16 | (uint32_t) p[4 * i + 2] << 8 | (uint32_t) p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + K256[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    void update(const void * data, size_t n) {
        const uint8_t * p = (const uint8_t *) data;
        total += n;
        while (n > 0) {
            if (nbuf == 0 && n >= 64) { block(p); p += 64; n -= 64; continue; }
            const size_t k = std::min<size_t>(64 - nbuf, n);
            memcpy(buf + nbuf, p, k);
            nbuf += k; p += k; n -= k;
            if (nbuf == 64) { block(buf); nbuf = 0; }
        }
    }
    void finish(uint8_t out[32]) {
        const uint64_t bits = total * 8;
        const uint8_t one = 0x80, zero = 0;
        update(&one, 1);
        while (nbuf != 56) update(&zero, 1);
        uint8_t len[8];
        for (int i = 0; i < 8; ++i) len[i] = (uint8_t) (bits >> (56 - 8 * i));
        update(len, 8);
        for (int i = 0; i < 8; ++i) {
            out[4 * i] = (uint8_t) (h[i] >> 24); out[4 * i + 1] = (uint8_t) (h[i] >> 16);
            out[4 * i + 2] = (uint8_t) (h[i] >> 8); out[4 * i + 3] = (uint8_t) h[i];
        }
    }
};

bool hex_eq(const uint8_t * d, const char * hex) {
    for (int i = 0; i < 32; ++i) {
        unsigned v = 0;
        if (sscanf(hex + 2 * i, "%2x", &v) != 1 || d[i] != (uint8_t) v) return false;
    }
    return true;
}

constexpr uint32_t AUTH_MAGIC = 0x414e3150u;   // "P1NA"
const char LABEL[] = "p100e-net-v1";          // 12 bytes, no terminator in the MAC

void mac_of(const net_key & key, uint8_t role, const uint8_t own[32], const uint8_t peer[32], uint8_t out[32]) {
    uint8_t m[sizeof LABEL - 1 + 1 + 64];
    memcpy(m, LABEL, sizeof LABEL - 1);
    m[sizeof LABEL - 1] = role;
    memcpy(m + sizeof LABEL, own, 32);
    memcpy(m + sizeof LABEL + 32, peer, 32);
    hmac_sha256(key.k.data(), key.k.size(), m, sizeof m, out);
}

} // namespace

void sha256(const void * p, size_t n, uint8_t out[32]) {
    sha256_ctx c;
    c.update(p, n);
    c.finish(out);
}

void hmac_sha256(const void * key, size_t kn, const void * msg, size_t mn, uint8_t out[32]) {
    uint8_t k0[64] = {};
    if (kn > 64) sha256(key, kn, k0);   // a key longer than the block is hashed first
    else if (kn) memcpy(k0, key, kn);
    uint8_t ip[64], op[64], inner[32];
    for (int i = 0; i < 64; ++i) { ip[i] = k0[i] ^ 0x36; op[i] = k0[i] ^ 0x5c; }
    sha256_ctx a;
    a.update(ip, 64); a.update(msg, mn); a.finish(inner);
    sha256_ctx b;
    b.update(op, 64); b.update(inner, 32); b.finish(out);
}

bool net_auth_selftest() {
    uint8_t d[32];
    sha256("abc", 3, d);
    if (!hex_eq(d, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")) return false;
    uint8_t k[20];
    memset(k, 0x0b, sizeof k);
    hmac_sha256(k, sizeof k, "Hi There", 8, d);   // RFC 4231 test case 1
    return hex_eq(d, "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
}

void net_random(void * p, size_t n) {
#ifdef _WIN32
    if (n > 0xffffffffu || BCryptGenRandom(nullptr, (PUCHAR) p, (ULONG) n, BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0)
        throw std::runtime_error("net: BCryptGenRandom failed");
#else
    uint8_t * c = (uint8_t *) p;
#  ifdef P100E_HAVE_GETRANDOM
    while (n > 0) {
        const ssize_t k = getrandom(c, n, 0);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) break;   // e.g. ENOSYS on an old kernel: /dev/urandom below
        c += k; n -= (size_t) k;
    }
#  endif
    if (n > 0) {
        const int fd = ::open("/dev/urandom", O_RDONLY);
        if (fd < 0) throw std::runtime_error("net: no random source (getrandom, /dev/urandom)");
        while (n > 0) {
            const ssize_t k = ::read(fd, c, n);
            if (k < 0 && errno == EINTR) continue;
            if (k <= 0) { ::close(fd); throw std::runtime_error("net: reading /dev/urandom failed"); }
            c += k; n -= (size_t) k;
        }
        ::close(fd);
    }
#endif
}

bool net_ct_equal(const void * a, const void * b, size_t n) {
    const volatile uint8_t * x = (const volatile uint8_t *) a;
    const volatile uint8_t * y = (const volatile uint8_t *) b;
    uint8_t acc = 0;
    for (size_t i = 0; i < n; ++i) acc |= (uint8_t) (x[i] ^ y[i]);
    return acc == 0;
}

const net_key & net_auth_key() {
    static const net_key key = [] {
        net_key r;
        const char * path = getenv("P100E_NET_KEY_FILE");
        if (!path || !*path) {
            printf("net: warning: P100E_NET_KEY_FILE is not set: node links are not authenticated (any host that reaches the port can join)\n");
            fflush(stdout);
            return r;
        }
        if (!net_auth_selftest()) throw std::runtime_error("net: SHA-256 / HMAC self-test failed (broken build)");
        FILE * f = fopen(path, "rb");
        if (!f) throw std::runtime_error(std::string("net: cannot read P100E_NET_KEY_FILE ") + path);
        uint8_t b[4096];
        size_t n;
        while ((n = fread(b, 1, sizeof b, f)) > 0 && r.k.size() <= 65536) r.k.insert(r.k.end(), b, b + n);
        fclose(f);
        if (r.k.size() > 65536)   // a wrong path (a model file) must not be read into memory as a key
            throw std::runtime_error(std::string("net: P100E_NET_KEY_FILE ") + path + " is larger than 64 KiB (not a key file?)");
        while (!r.k.empty() && (r.k.back() == '\n' || r.k.back() == '\r' || r.k.back() == ' ' || r.k.back() == '\t')) r.k.pop_back();
        if (r.k.size() < 16)
            throw std::runtime_error(std::string("net: P100E_NET_KEY_FILE ") + path + " holds fewer than 16 key bytes");
        r.on = true;
        printf("net: node links authenticated with the key in P100E_NET_KEY_FILE\n");
        fflush(stdout);
        return r;
    }();
    return key;
}

std::string net_auth_handshake(const net_key & key, uint8_t role, const std::function<void(const void *, size_t)> & send,
                               const std::function<void(void *, size_t)> & recv) {
    if (!key.on) return "";
    const uint8_t peer_role = role == NET_ROLE_CONNECT ? NET_ROLE_ACCEPT : NET_ROLE_CONNECT;
    uint8_t mine[36], theirs[36];
    const uint32_t mg = AUTH_MAGIC;
    memcpy(mine, &mg, 4);
    net_random(mine + 4, 32);
    send(mine, sizeof mine);
    recv(theirs, sizeof theirs);
    uint32_t pm = 0;
    memcpy(&pm, theirs, 4);
    if (pm != AUTH_MAGIC) return "the peer did not start the key handshake (is P100E_NET_KEY_FILE set on both nodes?)";
    if (net_ct_equal(mine + 4, theirs + 4, 32)) return "the peer echoed this node's nonce";
    uint8_t ans[32], got[32], want[32];
    mac_of(key, role, mine + 4, theirs + 4, ans);
    send(ans, sizeof ans);
    recv(got, sizeof got);
    mac_of(key, peer_role, theirs + 4, mine + 4, want);
    if (!net_ct_equal(got, want, 32)) return "wrong key (P100E_NET_KEY_FILE differs between the nodes)";
    return "";
}

} // namespace p100e
