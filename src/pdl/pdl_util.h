// pdl helpers: CRC32C (the frame check of pdl_proto, SSE4.2 instruction when the CPU has it), SHA-256, a
// monotonic ns clock and a spin-wait pause.
#pragma once

#include <cstddef>
#include <cstdint>

namespace p100e::pdl {

// CRC32C (Castagnoli), init ~0, final ~ (the iSCSI / ext4 one). crc32c_update continues a running value (pass the
// previous return); crc32c(p, n) = crc32c_update(0, p, n).
uint32_t crc32c_update(uint32_t crc, const void * p, size_t n);
inline uint32_t crc32c(const void * p, size_t n) { return crc32c_update(0, p, n); }
bool crc32c_hw();   // the SSE4.2 path is in use

void sha256(const void * p, size_t n, uint8_t out[32]);

uint64_t now_ns();   // steady clock
void cpu_relax();    // _mm_pause / yield instruction inside spin loops

inline uint16_t ld16(const uint8_t * p) { return (uint16_t) (p[0] | p[1] << 8); }
inline uint32_t ld32(const uint8_t * p) { return (uint32_t) p[0] | (uint32_t) p[1] << 8 | (uint32_t) p[2] << 16 | (uint32_t) p[3] << 24; }
inline void st16(uint8_t * p, uint16_t v) { p[0] = (uint8_t) v; p[1] = (uint8_t) (v >> 8); }
inline void st32(uint8_t * p, uint32_t v) { p[0] = (uint8_t) v; p[1] = (uint8_t) (v >> 8); p[2] = (uint8_t) (v >> 16); p[3] = (uint8_t) (v >> 24); }

} // namespace p100e::pdl
