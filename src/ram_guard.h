// Host-RAM reserve: keep the machine usable. Pinned memory cannot be paged
// out, so a load that pins more than the machine can spare makes the whole PC thrash. fn_model checks, BEFORE it
// pins or allocates host expert memory, that the RAM still available after the allocation stays >= the reserve, and
// refuses to load with a clear message (needed vs available vs reserve) instead of paging.
//   reserve: fn_model::opts::ram_reserve_mb (p100e-fn-run --p100e-ram-reserve-mb N) > P100E_RAM_RESERVE_MB > the
//   default 12288 MiB on Windows, 6144 MiB on Linux. 0 turns the check off.
//   available: Windows GlobalMemoryStatusEx ullAvailPhys, Linux /proc/meminfo MemAvailable.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace p100e {

inline int64_t ram_reserve_default_mb() {
#ifdef _WIN32
    return 12288;
#else
    return 6144;
#endif
}

// opt >= 0 wins, then P100E_RAM_RESERVE_MB, then the default
inline int64_t ram_reserve_mb(int64_t opt) {
    if (opt >= 0) return opt;
    if (const char * e = getenv("P100E_RAM_RESERVE_MB"); e && *e) return std::max<int64_t>(0, atoll(e));
    return ram_reserve_default_mb();
}

// bytes the OS can hand out now without paging (0 when unknown); src/ram_guard.cpp (keeps <windows.h> out of this header)
uint64_t ram_available_bytes();

// true when allocating `need` bytes leaves >= reserve_mb MiB of `avail`; else msg says why (what = what is allocated)
inline bool ram_check(uint64_t need, uint64_t avail, int64_t reserve_mb, const char * what, std::string & msg) {
    if (reserve_mb <= 0 || avail == 0) return true;   // off, or the OS did not say
    const uint64_t reserve = (uint64_t) reserve_mb << 20;
    if (avail >= need && avail - need >= reserve) return true;
    char b[512];
    snprintf(b, sizeof b, "host RAM reserve: %s needs %.0f MiB, %.0f MiB available, reserve %lld MiB -> %.0f MiB short "
             "(free RAM, use fewer layers / a smaller file / more resident experts, or lower P100E_RAM_RESERVE_MB)",
             what, need / 1048576.0, avail / 1048576.0, (long long) reserve_mb, (double) (need + reserve - std::min(avail, need + reserve)) / 1048576.0);
    msg = b;
    return false;
}

} // namespace p100e
