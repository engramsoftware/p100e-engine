// CPU expert pool for fn_model (P100E_FN_CPU=1): the CPU computes the
// experts that are not on the GPU in place, at the same time as the GPU works on the cached ones.
//
// The pool computes gate / up / SwiGLU / down of the routed rows of non-resident experts directly on the host copy of
// the expert (the pinned gate|up|down record fn_model already keeps for PCIe staging), in that record's fast layout:
//   gate / up: q4kr (Q4_K), iq2sr (IQ2_S), q4r (Q4_0);  down: q4r (Q4_0), q20 (Q2_0, raw GGUF blocks)
// The layouts are per-row byte permutations of the GGUF rows (src/fast/matvec_pack_body.inc, matvec_iq2r.cu), so no
// second copy of the experts is needed and the GPU fallback keeps reading the same bytes.
//
// Numerics: activations are quantized to int16 per 32 values (f32 scale; int8 as in ggml was ~10x further from the GPU's
// result), the weight codes are exact integers, the dot products are exact int32, the scales are applied in f32. This is not the GPU's a16 / fp16
// HFMA2 order, so a CPU-computed expert differs from the GPU-computed one in the last bits (reported by the gate runs).
// Each output value depends only on its own (row, token) inputs: the result does not depend on the worker count or
// on how the work was cut into tasks.
//
// Threads: one worker per physical core except the first (the host thread's), pinned to the first logical processor of
// its core (Windows GetLogicalProcessorInformationEx, Linux /sys topology), at most 32; P100E_FN_CPU_WORKERS=N
// overrides. Workers spin briefly between jobs, then sleep. The host thread works on the job too while it waits.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace p100e {

// fast:: layout ids this pool reads (values of fast::LAYOUT_*; fn_model.cu static_asserts them)
enum { FNCPU_Q4R = 1, FNCPU_Q4KR = 2, FNCPU_IQ2SR = 14, FNCPU_Q20 = 18 };

struct fn_cpu_q8 { int16_t q[32]; float d; int32_t s; };   // 32 activations: int16 codes, scale, code sum

struct fn_cpu_job {               // one expert's routed rows
    const uint8_t * gate = nullptr, * up = nullptr, * down = nullptr;   // packed rows of the expert (host)
    size_t rb_gu = 0, rb_d = 0;   // bytes per row: gate / up rows (n_in = E), down rows (n_in = F)
    int lay_gu = 0, lay_d = 0;    // FNCPU_* layouts
    int n = 0;                    // routed rows
    const int32_t * tok = nullptr; // [n] token row of each routed row (into x)
    float * y = nullptr;          // [n][E] outputs (unweighted expert outputs, as the GPU's S.gout rows)
};

class fn_cpu_pool {
public:
    explicit fn_cpu_pool(int workers = -1);   // -1: one per physical core minus the host's (P100E_FN_CPU_WORKERS wins)
    ~fn_cpu_pool();
    static bool cpu_ok();                      // AVX2 + FMA + F16C at run time
    static bool supports(int lay_gu, int lay_d);
    int workers() const { return (int) th_.size(); }
    const char * placement() const { return placement_.c_str(); }
    // start the job (returns at once): x [T][E] f32 host rows, jobs[i].tok index them; F = expert ffn width
    void submit(const float * x, int T, int E, int F, const fn_cpu_job * jobs, int n_jobs);
    void wait();                               // the calling thread helps, returns when every output is written
    bool busy() const { return active_; }

private:
    using q8g = fn_cpu_q8;
    void worker(int i);
    void run_phases();                          // claim and run tasks of the current job until every phase is done
    void task(int phase, int i);
    // job state
    const float * x_ = nullptr; int T_ = 0, E_ = 0, F_ = 0;
    const fn_cpu_job * jobs_ = nullptr; int nj_ = 0;
    std::vector<q8g> qx_;                        // [T][E/32]
    std::vector<float> g_, u_;                   // [rows][F] gate / up outputs (rows = all routed rows of the job)
    std::vector<q8g> qh_;                        // [rows][F/32]
    std::vector<int> row0_;                      // first row of job j
    struct tk { int j, r0, r1; };
    std::vector<tk> t1_, t3_;                    // phase 1 / 3 tasks: (job, weight rows r0..r1)
    static constexpr int CH1 = 64, CH3 = 256, MAXN = 64;   // gate|up rows / down rows per task, tokens per pass
    static constexpr int NPH = 4;                // 0 quantize x, 1 gate|up, 2 silu*up + quantize, 3 down
    int ntask_[NPH] = {};
    std::atomic<int> next_[NPH], done_[NPH];
    std::atomic<uint64_t> st_{0};                // generation << 32 | workers inside the job; odd generation = open
    std::atomic<bool> stop_{false};
    bool active_ = false;
    int spin_us_ = 2000;                         // P100E_FN_CPU_SPIN_US: idle spin before a worker sleeps
    std::vector<std::thread> th_;
    std::mutex mu_; std::condition_variable cv_;
    std::atomic<int> sleepers_{0};
    std::string placement_;
};

} // namespace p100e
