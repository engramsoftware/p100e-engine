// p100e-ram-guard-test: the host-RAM reserve (src/ram_guard.h) - the check's arithmetic, the reserve's precedence
// (option > P100E_RAM_RESERVE_MB > platform default, 0 = off) and that the OS reports available RAM. No GPU needed.
#include "ram_guard.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace p100e;

static int fails = 0;
#define EXPECT(c) do { if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static void set_env(const char * v) {
#ifdef _WIN32
    _putenv_s("P100E_RAM_RESERVE_MB", v ? v : "");
#else
    if (v) setenv("P100E_RAM_RESERVE_MB", v, 1); else unsetenv("P100E_RAM_RESERVE_MB");
#endif
}

int main() {
    const uint64_t MB = 1ull << 20;
    std::string why;
    EXPECT(ram_check(10 * MB, 30 * MB, 20, "x", why));               // 30 - 10 = 20 >= 20: fits exactly
    EXPECT(!ram_check(11 * MB, 30 * MB, 20, "x", why));              // 19 left < 20
    EXPECT(why.find("needs 11 MiB") != std::string::npos && why.find("30 MiB available") != std::string::npos &&
           why.find("reserve 20 MiB") != std::string::npos && why.find("1 MiB short") != std::string::npos);
    EXPECT(!ram_check(40 * MB, 30 * MB, 20, "x", why));              // more than available
    EXPECT(ram_check(40 * MB, 30 * MB, 0, "x", why));                // reserve 0 = off
    EXPECT(ram_check(40 * MB, 0, 20, "x", why));                     // unknown availability: no refusal
    set_env(nullptr);
    EXPECT(ram_reserve_mb(-1) == ram_reserve_default_mb());
    EXPECT(ram_reserve_mb(5) == 5 && ram_reserve_mb(0) == 0);
    set_env("777");
    EXPECT(ram_reserve_mb(-1) == 777 && ram_reserve_mb(5) == 5);
    set_env(nullptr);
#ifdef _WIN32
    EXPECT(ram_reserve_default_mb() == 12288);
#else
    EXPECT(ram_reserve_default_mb() == 6144);
#endif
    const uint64_t av = ram_available_bytes();
    EXPECT(av > 0);
    printf("available RAM %.0f MiB, default reserve %lld MiB\n", av / 1048576.0, (long long) ram_reserve_default_mb());
    printf(fails ? "FAILED: %d check(s)\n" : "all checks passed\n", fails);
    return fails ? 1 : 0;
}
