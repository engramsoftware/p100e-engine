// pdl helpers (see pdl_util.h)
#include "pdl/pdl_util.h"

#include <chrono>
#include <cstring>
#include <thread>

#if defined(_M_X64) || defined(__x86_64__)
#  define PDL_X86 1
#  ifdef _MSC_VER
#    include <intrin.h>
#  else
#    include <cpuid.h>
#  endif
#  include <immintrin.h>
#  include <nmmintrin.h>
#endif

namespace p100e::pdl {

namespace {

uint32_t g_tab[8][256];

void crc_tables() {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = c & 1 ? (c >> 1) ^ 0x82f63b78u : c >> 1;
        g_tab[0][i] = c;
    }
    for (uint32_t i = 0; i < 256; ++i)
        for (int t = 1; t < 8; ++t) g_tab[t][i] = (g_tab[t - 1][i] >> 8) ^ g_tab[0][g_tab[t - 1][i] & 0xff];
}
const bool g_tab_ready = (crc_tables(), true);   // static init: no lazy init race between threads


uint32_t crc_sw(uint32_t c, const uint8_t * p, size_t n) {
    (void) g_tab_ready;
    while (n >= 8) {   // slicing by 8
        uint32_t a = c ^ ld32(p), b = ld32(p + 4);
        c = g_tab[7][a & 0xff] ^ g_tab[6][(a >> 8) & 0xff] ^ g_tab[5][(a >> 16) & 0xff] ^ g_tab[4][a >> 24] ^
            g_tab[3][b & 0xff] ^ g_tab[2][(b >> 8) & 0xff] ^ g_tab[1][(b >> 16) & 0xff] ^ g_tab[0][b >> 24];
        p += 8;
        n -= 8;
    }
    while (n--) c = (c >> 8) ^ g_tab[0][(c ^ *p++) & 0xff];
    return c;
}

#ifdef PDL_X86
#  ifndef _MSC_VER
__attribute__((target("sse4.2")))
#  endif
uint32_t crc_hw(uint32_t c, const uint8_t * p, size_t n) {
    uint64_t c64 = c;
    while (n >= 8) {
        uint64_t v;
        memcpy(&v, p, 8);
        c64 = _mm_crc32_u64(c64, v);
        p += 8;
        n -= 8;
    }
    uint32_t c32 = (uint32_t) c64;
    while (n--) c32 = _mm_crc32_u8(c32, *p++);
    return c32;
}

bool detect_sse42() {
#  ifdef _MSC_VER
    int r[4];
    __cpuid(r, 1);
    return (r[2] >> 20) & 1;
#  else
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return false;
    return (c >> 20) & 1;
#  endif
}
const bool g_hw = detect_sse42();
#else
const bool g_hw = false;
#endif

} // namespace

bool crc32c_hw() { return g_hw; }

uint32_t crc32c_update(uint32_t crc, const void * p, size_t n) {
    uint32_t c = ~crc;
#ifdef PDL_X86
    if (g_hw) return ~crc_hw(c, (const uint8_t *) p, n);
#endif
    return ~crc_sw(c, (const uint8_t *) p, n);
}

// FIPS 180-4 SHA-256
void sha256(const void * data, size_t n, uint8_t out[32]) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
        0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
        0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    auto rotr = [](uint32_t x, int k) { return (x >> k) | (x << (32 - k)); };
    auto block = [&](const uint8_t * b) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) w[i] = (uint32_t) b[4 * i] << 24 | (uint32_t) b[4 * i + 1] << 16 | (uint32_t) b[4 * i + 2] << 8 | b[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], bb = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25), ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22), mj = (a & bb) ^ (a & c) ^ (bb & c);
            const uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = bb; bb = a; a = t1 + t2;
        }
        h[0] += a; h[1] += bb; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    };
    const uint8_t * p = (const uint8_t *) data;
    size_t left = n;
    while (left >= 64) { block(p); p += 64; left -= 64; }
    uint8_t tail[128] = {};
    memcpy(tail, p, left);
    tail[left] = 0x80;
    const size_t tl = left + 9 <= 64 ? 64 : 128;
    const uint64_t bits = (uint64_t) n * 8;
    for (int i = 0; i < 8; ++i) tail[tl - 1 - i] = (uint8_t) (bits >> (8 * i));
    block(tail);
    if (tl == 128) block(tail + 64);
    for (int i = 0; i < 8; ++i) { out[4 * i] = (uint8_t) (h[i] >> 24); out[4 * i + 1] = (uint8_t) (h[i] >> 16); out[4 * i + 2] = (uint8_t) (h[i] >> 8); out[4 * i + 3] = (uint8_t) h[i]; }
}

uint64_t now_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void cpu_relax() {
#ifdef PDL_X86
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

} // namespace p100e::pdl
