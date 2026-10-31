// Optional pre-shared key for the node links (net_link's TCP node link, the remote expert tier's rexp_sock). No new
// dependencies: SHA-256 (FIPS 180-4) and HMAC-SHA256 (RFC 2104) live in net_auth.cpp.
//
// Key: the bytes of the file named by P100E_NET_KEY_FILE (trailing whitespace / newlines dropped, at least 16 bytes
// left). Never printed. Unset: the links are not authenticated (as before); one warning line per process.
//
// Handshake (right after the TCP connect, both sides at once, the same on both links):
//   each side sends   "P1NA" + a 32-byte random nonce
//   each side sends   HMAC-SHA256(key, "p100e-net-v1" | role | own nonce | peer nonce)   (role: 'C' connect, 'A' accept)
//   each side checks the peer's answer (with the peer's role, constant-time compare)
// The roles differ, so a peer cannot reflect our own answer back. It authenticates the peer; it does not encrypt.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace p100e {

void sha256(const void * p, size_t n, uint8_t out[32]);
void hmac_sha256(const void * key, size_t kn, const void * msg, size_t mn, uint8_t out[32]);
// FIPS 180-4 "abc" and RFC 4231 test case 1; false = this build's SHA-256 / HMAC is broken
bool net_auth_selftest();
// n bytes from the OS RNG (BCryptGenRandom / getrandom / /dev/urandom); throws when none is available
void net_random(void * p, size_t n);
bool net_ct_equal(const void * a, const void * b, size_t n);

struct net_key {
    bool on = false;
    std::vector<uint8_t> k;
};
// P100E_NET_KEY_FILE, read once per process (thread-safe). Throws when it is set but unreadable or too short, or when
// the self-test fails. Unset: on = false and one warning line on the first call.
const net_key & net_auth_key();

constexpr uint8_t NET_ROLE_CONNECT = 'C', NET_ROLE_ACCEPT = 'A';
// the handshake above over send(p, n) / recv(p, n), each moving exactly n bytes or throwing (timeouts are theirs).
// Returns "" when the peer proved the key, else why not (I/O errors propagate as exceptions).
std::string net_auth_handshake(const net_key & key, uint8_t role, const std::function<void(const void *, size_t)> & send,
                               const std::function<void(void *, size_t)> & recv);

} // namespace p100e
