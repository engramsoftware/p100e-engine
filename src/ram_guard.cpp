// host-RAM availability for the reserve check (ram_guard.h)
#include "ram_guard.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace p100e {

uint64_t ram_available_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX m{}; m.dwLength = sizeof(m);
    return GlobalMemoryStatusEx(&m) ? (uint64_t) m.ullAvailPhys : 0;
#else
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char key[64]; unsigned long long v = 0; char unit[16];
    uint64_t r = 0;
    while (fscanf(f, "%63s %llu %15s", key, &v, unit) == 3) if (std::string(key) == "MemAvailable:") { r = (uint64_t) v * 1024; break; }
    fclose(f);
    return r;
#endif
}

} // namespace p100e
