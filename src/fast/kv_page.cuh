// Paged KV cache addressing, device helper shared by every kernel that reads or writes a
// fast::kv_cache (include from .cu files only). See kv_cache in fast/attn.cuh: position p of query row t lives at
// base + pt[(rtab ? rtab[t] : 0) + (p >> psh)] * pstride + (p & (2^psh - 1)) * rowb; pt == nullptr: base + p * rowb.
#pragma once
#include <cstddef>
#include <cstdint>
#include "fast/attn.cuh"

namespace p100e::fast {

struct kv_pg {
    const int32_t * pt;     // page table(s), nullptr = contiguous
    const int32_t * rtab;   // per-row offsets into pt (nullptr: 0 for every row)
    int psh;
    size_t pstride;
};

inline kv_pg kv_pg_of(const kv_cache & c) { return kv_pg{c.pt, c.rtab, c.psh, c.pstride}; }

// the page table of row t (nullptr when contiguous)
__device__ __forceinline__ const int32_t * kv_tab(const kv_pg & g, int t) {
    return g.pt ? (g.rtab ? g.pt + g.rtab[t] : g.pt) : nullptr;
}
// byte offset of position p with the row's table tab (kv_tab)
__device__ __forceinline__ size_t kv_off_tab(const kv_pg & g, const int32_t * tab, int p, size_t rowb) {
    if (!tab) return (size_t) p * rowb;
    return (size_t) tab[p >> g.psh] * g.pstride + (size_t) (p & ((1 << g.psh) - 1)) * rowb;
}
__device__ __forceinline__ size_t kv_off(const kv_pg & g, int t, int p, size_t rowb) {
    return kv_off_tab(g, kv_tab(g, t), p, rowb);
}

} // namespace p100e::fast
