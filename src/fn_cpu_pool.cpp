// CPU expert pool for fn_model (see fn_cpu_pool.h). Compiled with AVX2 + FMA + F16C (CMakeLists: this file only).
#include "fn_cpu_pool.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"   // iq2s_grid

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#else
#include <cpuid.h>
#include <pthread.h>
#include <sched.h>
#include <set>
#include <utility>
#endif

namespace p100e {

namespace {

inline float h2f(const uint8_t * p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h)));
}
inline float hsum8(__m256 v) {
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

using q8g = fn_cpu_q8;

// 32 f32 -> int16 codes, scale amax / 32767, code sum (scalar, round to nearest even). int16 rather than ggml's int8:
// the activation rounding (2^-15 of the group's max) stays well below the GPU's a16 (fp16) rounding, so a CPU-computed
// expert lands as close to the GPU-computed one as the GPU's own rounding allows (int8 put it ~10x further off)
inline void quant32(const float * x, q8g & o) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float d = amax / 32767.0f, id = amax > 0.0f ? 32767.0f / amax : 0.0f;
    int s = 0;
    for (int i = 0; i < 32; ++i) { const int q = (int) std::nearbyint(x[i] * id); o.q[i] = (int16_t) q; s += q; }
    o.d = d; o.s = s;
}

// 16 bytes: low nibble = value i, high nibble = value i + 16 (Q4_0 order) -> codes of values 0..15 / 16..31 as int16
inline void nib32(const uint8_t * p, __m256i & lo, __m256i & hi) {
    const __m128i raw = _mm_loadu_si128((const __m128i *) p);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    lo = _mm256_cvtepu8_epi16(_mm_and_si128(raw, m4));
    hi = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(raw, 4), m4));
}
// 16 code bytes of a Q2_0 block (value 4i + k at bits 2k of byte i) -> codes of values 0..63 as four int16 vectors
inline void unpack64(const uint8_t * codes, __m256i w[4]) {
    const __m128i b = _mm_loadu_si128((const __m128i *) codes);
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i c0 = _mm_and_si128(b, m3);
    const __m128i c1 = _mm_and_si128(_mm_srli_epi16(b, 2), m3);
    const __m128i c2 = _mm_and_si128(_mm_srli_epi16(b, 4), m3);
    const __m128i c3 = _mm_and_si128(_mm_srli_epi16(b, 6), m3);
    const __m128i a0 = _mm_unpacklo_epi8(c0, c1), a1 = _mm_unpacklo_epi8(c2, c3);
    const __m128i b0 = _mm_unpackhi_epi8(c0, c1), b1 = _mm_unpackhi_epi8(c2, c3);
    w[0] = _mm256_cvtepu8_epi16(_mm_unpacklo_epi16(a0, a1));   // values 0..15
    w[1] = _mm256_cvtepu8_epi16(_mm_unpackhi_epi16(a0, a1));   // 16..31
    w[2] = _mm256_cvtepu8_epi16(_mm_unpacklo_epi16(b0, b1));   // 32..47
    w[3] = _mm256_cvtepu8_epi16(_mm_unpackhi_epi16(b0, b1));   // 48..63
}
// sum over 32 values of w * a (w as two int16 vectors: values 0..15, 16..31): 8 int32 lanes
inline __m256i dot16(__m256i wlo, __m256i whi, const int16_t * a) {
    return _mm256_add_epi32(_mm256_madd_epi16(wlo, _mm256_loadu_si256((const __m256i *) a)),
                            _mm256_madd_epi16(whi, _mm256_loadu_si256((const __m256i *) (a + 16))));
}

// ---- row dots: one packed weight row against NT activation vectors a[t] (groups of 32 along n_in), out[t]
// q4r (Q4_0): [qs 16 B x nb][d 2 B x nb], value = (q - 8) * d
template <int NT>
void dot_q4r(const uint8_t * row, int n_in, const q8g * const * a, float * out) {
    const int nb = n_in / 32;
    const uint8_t * dp = row + 16 * (size_t) nb;
    __m256 acc[NT]; float corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); corr[t] = 0.0f; }
    for (int b = 0; b < nb; ++b) {
        __m256i lo, hi; nib32(row + 16 * (size_t) b, lo, hi);
        const float dw = h2f(dp + 2 * b);
        for (int t = 0; t < NT; ++t) {
            const q8g & q = a[t][b];
            const float s = dw * q.d;
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(s), _mm256_cvtepi32_ps(dot16(lo, hi, q.q)), acc[t]);
            corr[t] += s * (float) (8 * q.s);
        }
    }
    for (int t = 0; t < NT; ++t) out[t] = hsum8(acc[t]) - corr[t];
}

// q4kr (Q4_K): [qs plane: group t = 8 sb + g, 16 B in Q4_0 nibble order][header plane: 16 B per sb = d, dmin, scales[12]]
// value = d * sc[g] * q - dmin * m[g]
template <int NT>
void dot_q4kr(const uint8_t * row, int n_in, const q8g * const * a, float * out) {
    const int nb = n_in / 256;
    const uint8_t * hp = row + 128 * (size_t) nb;
    __m256 acc[NT]; float corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); corr[t] = 0.0f; }
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t * h = hp + 16 * (size_t) sb;
        const float d = h2f(h), dmin = h2f(h + 2);
        const uint8_t * q = h + 4;
        uint8_t sc[8], m[8];
        for (int j = 0; j < 4; ++j) { sc[j] = q[j] & 63; m[j] = q[j + 4] & 63; }
        for (int j = 4; j < 8; ++j) { sc[j] = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m[j] = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
        for (int g = 0; g < 8; ++g) {
            __m256i lo, hi; nib32(row + 16 * (size_t) (8 * sb + g), lo, hi);
            const float ds = d * sc[g], dm = dmin * m[g];
            for (int t = 0; t < NT; ++t) {
                const q8g & aq = a[t][8 * sb + g];
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(ds * aq.d), _mm256_cvtepi32_ps(dot16(lo, hi, aq.q)), acc[t]);
                corr[t] += dm * aq.d * (float) aq.s;
            }
        }
    }
    for (int t = 0; t < NT; ++t) out[t] = hsum8(acc[t]) - corr[t];
}

// iq2sr (IQ2_S): [group plane: 4 grid-low bytes, 4 sign bytes per group][qh 8 B per sb][scales 8 B per sb][d 2 B per sb]
// value = d * (2 * scale + 1) * 0.125 * grid * sign; chunks 0, 1 take the low scale nibble, 2, 3 the high one.
// The signed grid value times (2 * scale + 1) is an exact int16 (|43 * 31| < 2^15).
template <int NT>
void dot_iq2sr(const uint8_t * row, int n_in, const q8g * const * a, float * out) {
    const int nb = n_in / 256;
    const uint8_t * qhp = row + 64 * (size_t) nb, * scp = row + 72 * (size_t) nb, * dp = row + 80 * (size_t) nb;
    const __m256i mask1 = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    const __m256i mask2 = _mm256_set1_epi64x((long long) 0x8040201008040201ull);
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int sb = 0; sb < nb; ++sb) {
        const float d = 0.125f * h2f(dp + 2 * sb);
        for (int g = 0; g < 8; ++g) {
            const uint8_t * gp = row + 8 * (size_t) (8 * sb + g);
            const int qh = qhp[8 * sb + g], sc = scp[8 * sb + g];
            __m256i w = _mm256_set_epi64x((long long) iq2s_grid[gp[3] | ((qh << 2) & 0x300)], (long long) iq2s_grid[gp[2] | ((qh << 4) & 0x300)],
                                          (long long) iq2s_grid[gp[1] | ((qh << 6) & 0x300)], (long long) iq2s_grid[gp[0] | ((qh << 8) & 0x300)]);
            uint32_t sg; std::memcpy(&sg, gp + 4, 4);
            __m256i sm = _mm256_shuffle_epi8(_mm256_set1_epi32((int) sg), mask1);
            sm = _mm256_cmpeq_epi8(_mm256_and_si256(sm, mask2), mask2);
            w = _mm256_sub_epi8(_mm256_xor_si256(w, sm), sm);   // signed grid values (|v| <= 43)
            const __m256i wlo = _mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(w)), _mm256_set1_epi16((short) (2 * (sc & 15) + 1)));
            const __m256i whi = _mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1)), _mm256_set1_epi16((short) (2 * (sc >> 4) + 1)));
            for (int t = 0; t < NT; ++t) {
                const q8g & aq = a[t][8 * sb + g];
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * aq.d), _mm256_cvtepi32_ps(dot16(wlo, whi, aq.q)), acc[t]);
            }
        }
    }
    for (int t = 0; t < NT; ++t) out[t] = hsum8(acc[t]);
}

// q20 (Q2_0, raw GGUF): 18-byte blocks of 64 values, f16 d + 16 code bytes, value = (code - 1) * d
template <int NT>
void dot_q20(const uint8_t * row, int n_in, const q8g * const * a, float * out) {
    const int nb = n_in / 64;
    __m256 acc[NT]; float corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); corr[t] = 0.0f; }
    for (int b = 0; b < nb; ++b) {
        const uint8_t * blk = row + 18 * (size_t) b;
        const float dw = h2f(blk);
        __m256i w[4];
        unpack64(blk + 2, w);
        for (int t = 0; t < NT; ++t) {
            const q8g & q0 = a[t][2 * b], & q1 = a[t][2 * b + 1];
            const float s0 = dw * q0.d, s1 = dw * q1.d;
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(s0), _mm256_cvtepi32_ps(dot16(w[0], w[1], q0.q)), acc[t]);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(s1), _mm256_cvtepi32_ps(dot16(w[2], w[3], q1.q)), acc[t]);
            corr[t] += s0 * (float) q0.s + s1 * (float) q1.s;
        }
    }
    for (int t = 0; t < NT; ++t) out[t] = hsum8(acc[t]) - corr[t];
}

template <int NT>
inline void dot_any(int lay, const uint8_t * row, int n_in, const q8g * const * a, float * out) {
    switch (lay) {
        case FNCPU_Q4R:   dot_q4r<NT>(row, n_in, a, out); break;
        case FNCPU_Q4KR:  dot_q4kr<NT>(row, n_in, a, out); break;
        case FNCPU_IQ2SR: dot_iq2sr<NT>(row, n_in, a, out); break;
        default:          dot_q20<NT>(row, n_in, a, out); break;
    }
}
// rows [r0, r1) of a weight (row stride rb) against nt activations; out[t][r] = out_base[t] + r
void dot_rows(int lay, const uint8_t * w, size_t rb, int n_in, int r0, int r1, const q8g * const * a, int nt, float * const * outp) {
    float res[4];
    for (int r = r0; r < r1; ++r) {
        const uint8_t * row = w + (size_t) r * rb;
        for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = std::min(4, nt - t0);
            switch (k) {
                case 1: dot_any<1>(lay, row, n_in, a + t0, res); break;
                case 2: dot_any<2>(lay, row, n_in, a + t0, res); break;
                case 3: dot_any<3>(lay, row, n_in, a + t0, res); break;
                default: dot_any<4>(lay, row, n_in, a + t0, res); break;
            }
            for (int t = 0; t < k; ++t) outp[t0 + t][r] = res[t];
        }
    }
}

int env_int(const char * n, int def) { const char * e = getenv(n); return e && *e ? atoi(e) : def; }

// logical processors to pin the workers to: the first of each physical core, the first core left to the host thread
std::vector<int> worker_cpus(std::string & why) {
    std::vector<int> cpus;
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<char> buf(len);
    if (len && GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
        for (DWORD off = 0; off < len;) {
            auto * info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (buf.data() + off);
            if (info->Relationship == RelationProcessorCore && info->Processor.GroupMask[0].Group == 0) {
                const KAFFINITY m = info->Processor.GroupMask[0].Mask;
                for (int b = 0; b < 64; ++b) if (m & ((KAFFINITY) 1 << b)) { cpus.push_back(b); break; }
            }
            off += info->Size;
        }
        why = "GetLogicalProcessorInformationEx";
    }
#else
    cpu_set_t set; CPU_ZERO(&set);
    sched_getaffinity(0, sizeof(set), &set);
    std::set<std::pair<int, int>> seen;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &set)) continue;
        auto rd = [&](const char * f) { char p[128]; snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/topology/%s", c, f); FILE * fp = fopen(p, "r"); int v = -1; if (fp) { if (fscanf(fp, "%d", &v) != 1) v = -1; fclose(fp); } return v; };
        const int core = rd("core_id"), pkg = rd("physical_package_id");
        if (core < 0) { cpus.push_back(c); continue; }
        if (seen.insert({pkg, core}).second) cpus.push_back(c);
    }
    why = "/sys topology";
#endif
    if (cpus.empty()) { const int n = (int) std::max(1u, std::thread::hardware_concurrency()); for (int i = 0; i < n; ++i) cpus.push_back(i); why = "no topology: every logical processor"; }
    std::sort(cpus.begin(), cpus.end());
    cpus.erase(cpus.begin());   // the first core: the host thread's
    if (cpus.size() > 32) cpus.resize(32);
    return cpus;
}

void pin_thread(std::thread & t, int cpu) {
#ifdef _WIN32
    GROUP_AFFINITY ga{};
    ga.Group = 0; ga.Mask = (KAFFINITY) 1 << cpu;
    SetThreadGroupAffinity((HANDLE) t.native_handle(), &ga, nullptr);
#else
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    pthread_setaffinity_np(t.native_handle(), sizeof(s), &s);
#endif
}

}  // namespace

bool fn_cpu_pool::cpu_ok() {
    int r[4];
#ifdef _WIN32
    __cpuid(r, 1);
    const bool fma = (r[2] >> 12) & 1, f16c = (r[2] >> 29) & 1, osx = (r[2] >> 27) & 1;
    __cpuidex(r, 7, 0);
    const bool avx2 = (r[1] >> 5) & 1;
#else
    unsigned a, b, c, d;
    __cpuid(1, a, b, c, d);
    const bool fma = (c >> 12) & 1, f16c = (c >> 29) & 1, osx = (c >> 27) & 1;
    __cpuid_count(7, 0, a, b, c, d);
    const bool avx2 = (b >> 5) & 1;
    (void) r;
#endif
    return fma && f16c && osx && avx2;
}

bool fn_cpu_pool::supports(int lay_gu, int lay_d) {
    return (lay_gu == FNCPU_Q4KR || lay_gu == FNCPU_IQ2SR || lay_gu == FNCPU_Q4R) && (lay_d == FNCPU_Q4R || lay_d == FNCPU_Q20);
}

// state word: generation << 32 | participants inside. Odd generation = a job is open.
static inline uint64_t pack_st(uint32_t g, uint32_t n) { return ((uint64_t) g << 32) | n; }

fn_cpu_pool::fn_cpu_pool(int workers) {
    std::string why;
    std::vector<int> cpus = worker_cpus(why);
    int n = env_int("P100E_FN_CPU_WORKERS", workers);
    const bool auto_n = n < 0;
    if (auto_n) n = (int) cpus.size();
    const bool pin = env_int("P100E_FN_CPU_PIN", 1) != 0;
    spin_us_ = std::max(0, env_int("P100E_FN_CPU_SPIN_US", 2000));
    for (int k = 0; k < NPH; ++k) { next_[k] = 0; done_[k] = 0; }
    th_.reserve(n);
    std::string where;
    for (int i = 0; i < n; ++i) {
        th_.emplace_back([this, i] { worker(i); });
        if (pin && i < (int) cpus.size()) { pin_thread(th_.back(), cpus[i]); where += (where.empty() ? "" : ",") + std::to_string(cpus[i]); }
    }
    placement_ = std::to_string(n) + " workers" + (auto_n ? " (one per physical core but the first, " + why + ")" : " (P100E_FN_CPU_WORKERS)") +
                 (where.empty() ? std::string(", not pinned") : ", pinned to logical processors " + where);
}

fn_cpu_pool::~fn_cpu_pool() {
    stop_ = true;
    { std::lock_guard<std::mutex> lk(mu_); }
    cv_.notify_all();
    for (auto & t : th_) t.join();
}

void fn_cpu_pool::submit(const float * x, int T, int E, int F, const fn_cpu_job * jobs, int n_jobs) {
    if (active_) wait();
    x_ = x; T_ = T; E_ = E; F_ = F; jobs_ = jobs; nj_ = n_jobs;
    const int ge = E / 32, gf = F / 32;
    if ((int) qx_.size() < T * ge) qx_.resize((size_t) T * ge);
    row0_.resize(n_jobs + 1);
    row0_[0] = 0;
    for (int j = 0; j < n_jobs; ++j) row0_[j + 1] = row0_[j] + jobs[j].n;
    const int rows = row0_[n_jobs];
    if ((int) (g_.size() / std::max(F, 1)) < rows) { g_.resize((size_t) rows * F); u_.resize((size_t) rows * F); }
    if ((int) (qh_.size() / std::max(gf, 1)) < rows) qh_.resize((size_t) rows * gf);
    // tasks: phase 1 = (job, CH1 gate|up rows), phase 3 = (job, CH3 down rows)
    t1_.clear(); t3_.clear();
    for (int j = 0; j < n_jobs; ++j) {
        for (int r = 0; r < F; r += CH1) t1_.push_back({j, r, std::min(F, r + CH1)});
        for (int r = 0; r < E; r += CH3) t3_.push_back({j, r, std::min(E, r + CH3)});
    }
    ntask_[0] = T; ntask_[1] = (int) t1_.size(); ntask_[2] = rows; ntask_[3] = (int) t3_.size();
    for (int k = 0; k < NPH; ++k) { next_[k].store(0, std::memory_order_relaxed); done_[k].store(0, std::memory_order_relaxed); }
    active_ = true;
    const uint64_t s = st_.load();
    // open (odd generation). seq_cst, not release: with the worker's ++sleepers_ then st_.load(), both seq_cst,
    // one side must see the other (a release store may pass the sleepers_ load and lose the wakeup)
    st_.store(pack_st((uint32_t) (s >> 32) + 1, 0), std::memory_order_seq_cst);
    if (sleepers_.load() > 0) { std::lock_guard<std::mutex> lk(mu_); cv_.notify_all(); }
}

void fn_cpu_pool::wait() {
    if (!active_) return;
    run_phases();
    // close the job once no worker is inside: CAS (g, 0) -> (g + 1, 0); a worker joining later sees an even generation
    for (;;) {
        uint64_t s = st_.load(std::memory_order_acquire);
        if ((uint32_t) s == 0 && st_.compare_exchange_weak(s, pack_st((uint32_t) (s >> 32) + 1, 0))) break;
        _mm_pause();
    }
    active_ = false;
}

void fn_cpu_pool::run_phases() {
    for (int p = 0; p < NPH; ++p) {
        for (;;) {
            const int i = next_[p].fetch_add(1, std::memory_order_relaxed);
            if (i >= ntask_[p]) break;
            task(p, i);
            done_[p].fetch_add(1, std::memory_order_release);
        }
        while (done_[p].load(std::memory_order_acquire) < ntask_[p]) _mm_pause();
    }
}

void fn_cpu_pool::worker(int) {
    uint32_t last = 0;   // generation of the last job this worker took part in
    for (;;) {
        uint64_t s = st_.load(std::memory_order_acquire);
        auto t0 = std::chrono::steady_clock::now();
        int spins = 0;
        while (!stop_ && !((s >> 32) & 1 && (uint32_t) (s >> 32) != last)) {
            _mm_pause();
            if (++spins == 256) {
                spins = 0;
                if (std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(spin_us_)) {
                    std::unique_lock<std::mutex> lk(mu_);
                    ++sleepers_;
                    cv_.wait_for(lk, std::chrono::milliseconds(20), [&] {
                        const uint64_t v = st_.load();
                        return stop_.load() || (((v >> 32) & 1) && (uint32_t) (v >> 32) != last);
                    });
                    --sleepers_;
                    t0 = std::chrono::steady_clock::now();
                }
            }
            s = st_.load(std::memory_order_acquire);
        }
        if (stop_) return;
        // join: (g, n) -> (g, n + 1) while g is still the open generation
        if (!st_.compare_exchange_strong(s, s + 1, std::memory_order_acq_rel)) continue;
        last = (uint32_t) (s >> 32);
        run_phases();
        st_.fetch_sub(1, std::memory_order_acq_rel);
    }
}

void fn_cpu_pool::task(int p, int i) {
    const int ge = E_ / 32, gf = F_ / 32;
    if (p == 0) {   // quantize x row i
        for (int g = 0; g < ge; ++g) quant32(x_ + (size_t) i * E_ + 32 * g, qx_[(size_t) i * ge + g]);
        return;
    }
    if (p == 2) {   // routed row i: h = silu(gate) * up, quantized for the down projection
        float * g = g_.data() + (size_t) i * F_;
        const float * u = u_.data() + (size_t) i * F_;
        for (int k = 0; k < F_; ++k) g[k] = g[k] / (1.0f + expf(-g[k])) * u[k];
        for (int k = 0; k < gf; ++k) quant32(g + 32 * k, qh_[(size_t) i * gf + k]);
        return;
    }
    const tk & t = p == 1 ? t1_[i] : t3_[i];
    const fn_cpu_job & J = jobs_[t.j];
    const int base = row0_[t.j];
    for (int k0 = 0; k0 < J.n; k0 += MAXN) {
        const int nt = std::min(MAXN, J.n - k0);
        const q8g * a[MAXN];
        float * o[MAXN];
        if (p == 1) {
            for (int k = 0; k < nt; ++k) a[k] = qx_.data() + (size_t) J.tok[k0 + k] * ge;
            for (int k = 0; k < nt; ++k) o[k] = g_.data() + (size_t) (base + k0 + k) * F_;
            dot_rows(J.lay_gu, J.gate, J.rb_gu, E_, t.r0, t.r1, a, nt, o);
            for (int k = 0; k < nt; ++k) o[k] = u_.data() + (size_t) (base + k0 + k) * F_;
            dot_rows(J.lay_gu, J.up, J.rb_gu, E_, t.r0, t.r1, a, nt, o);
        } else {
            for (int k = 0; k < nt; ++k) a[k] = qh_.data() + (size_t) (base + k0 + k) * gf;
            for (int k = 0; k < nt; ++k) o[k] = J.y + (size_t) (k0 + k) * E_;
            dot_rows(J.lay_d, J.down, J.rb_d, F_, t.r0, t.r1, a, nt, o);
        }
    }
}

} // namespace p100e
