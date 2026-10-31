// pdl GPU rings + spin thread (see pdl_gpu.cuh)
#include "pdl/pdl_gpu.cuh"
#include "pdl/pdl_util.h"

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <pthread.h>
#  include <sched.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace p100e::pdl {

namespace {

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("pdl gpu: ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ unsigned long long g_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Copy the partial into the mapped tx slot, then (last block) seq = ++*ctr, bytes[slot] = size, bell = seq, and the
// node-local flag word of this rank. The last-block pattern: every block fences its stores system-wide before it
// counts itself done, so the block that finishes last publishes after all bytes are visible to the CPU.
__global__ void publish_k(const float4 * __restrict__ src, float4 * __restrict__ dst_base, size_t n4, size_t slot_f, int depth,
                          int * ctr, int * bell, int * flags, int grank, int * done) {
    const int seq_next = *((volatile int *) ctr) + 1;   // every block reads the same value (ctr changes at the end)
    float4 * dst = dst_base + (size_t) (((unsigned) seq_next) % (unsigned) depth) * (slot_f / 4);
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (size_t) gridDim.x * blockDim.x) dst[i] = src[i];
    __threadfence_system();
    __syncthreads();
    __shared__ bool last;
    if (threadIdx.x == 0) last = atomicAdd(done, 1) == (int) gridDim.x - 1;
    __syncthreads();
    if (!last || threadIdx.x != 0) return;
    *done = 0;
    *ctr = seq_next;
    ((volatile int *) bell)[8 + ((unsigned) seq_next) % (unsigned) depth] = (int) (n4 * 16);
    __threadfence_system();
    ((volatile int *) bell)[0] = seq_next;
    ((volatile int *) flags)[fast::P2P_FLAG_PEER0 + grank] = seq_next;
}

// p2p_wait_word of fused.cu with the flag block in mapped host memory (same bound, timeout count, status line)
__device__ void wait_word(const int * flag, const int * word, int seq) {
    if ((int) ((unsigned) *((volatile const int *) word) - (unsigned) seq) >= 0) return;
    if (*((volatile const int *) (flag + 1)) != 0) return;
    const int ms = flag[fast::P2P_FLAG_TIMEOUT_MS];
    const unsigned long long lim = (unsigned long long) (ms > 0 ? ms : 2000) * 1000000ull, t0 = g_ns();
    const long long c0 = clock64();
    while ((int) ((unsigned) *((volatile const int *) word) - (unsigned) seq) < 0) {
        const unsigned long long dt = g_ns() - t0;
        if (dt > lim || clock64() - c0 > (long long) (2 * lim)) {
            atomicAdd(const_cast<int *>(flag) + 1, 1);
            return;
        }
    }
}

__global__ void wait_add_k(float * __restrict__ x, const gpu_parts P, int R, int self, int depth, const int * flags, const int * ctr,
                           size_t n) {
    const int seq = *((volatile const int *) ctr);
    if (flags && threadIdx.x == 0)
        for (int q = 0; q < R; ++q)
            if (q != self) wait_word(flags, flags + fast::P2P_FLAG_PEER0 + q, seq);
    __syncthreads();
    const size_t slot = ((unsigned) seq) % (unsigned) depth;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        float a = x[i];
        for (int q = 0; q < R; ++q) a += __ldcv(P.base[q] + (size_t) P.stride[q] * slot + i);   // rank order, no FMA
        x[i] = a;
    }
}

__global__ void tick_k(int * ctr) { *ctr = *((volatile int *) ctr) + 1; }

struct copy_list {   // recv_copy: remote rank q[k]'s rx ring base -> dst[k]
    const float * src[GPU_MAX_RANKS];
    float * dst[GPU_MAX_RANKS];
    int q[GPU_MAX_RANKS];
    int n;
};

// thread 0 of every block waits for each listed rank's flag word (>= this rank's seq), then the block copies its
// share. The source is mapped host memory written by the spin thread: volatile loads (__ldcv), never a cached line.
__global__ void recv_copy_k(copy_list C, size_t slot_f, int depth, const int * flags, const int * ctr, size_t n4) {
    const int seq = *((volatile const int *) ctr);
    if (threadIdx.x == 0)
        for (int k = 0; k < C.n; ++k) wait_word(flags, flags + fast::P2P_FLAG_PEER0 + C.q[k], seq);
    __syncthreads();
    const size_t off = (size_t) (((unsigned) seq) % (unsigned) depth) * slot_f / 4;
    for (int k = 0; k < C.n; ++k) {
        const float4 * src = (const float4 *) C.src[k] + off;
        float4 * dst = (float4 *) C.dst[k];
        for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (size_t) gridDim.x * blockDim.x) dst[i] = __ldcv(src + i);
    }
}

} // namespace

void wait_add(float * x, const gpu_parts & parts, int R, int self, int depth, const int * flags, const int * ctr, size_t n, cudaStream_t s) {
    const int threads = 256;
    const int blocks = (int) std::min<size_t>((n + threads - 1) / threads, 1024);
    wait_add_k<<<blocks, threads, 0, s>>>(x, parts, R, self, depth, flags, ctr, n);
    ck(cudaGetLastError(), "wait_add launch");
}

gpu_link::gpu_link(pdl_link & link, const gpu_link_opts & o) : L_(link) {
    node_of_ = o.node_of_rank;
    R_ = (int) node_of_.size();
    N_ = link.n_nodes();
    depth_ = o.depth < 2 ? 2 : o.depth;
    if (R_ < 2 || R_ > GPU_MAX_RANKS || R_ > fast::P2P_MAX_RANKS * 2) throw std::runtime_error("pdl gpu: rank count out of range");
    if (N_ + R_ > PDL_MAX_STREAMS) throw std::runtime_error("pdl gpu: nodes + ranks > 8 pdl streams");
    r0_ = -1;
    for (int q = 0; q < R_; ++q) {
        if (node_of_[q] < 0 || node_of_[q] >= N_) throw std::runtime_error("pdl gpu: node_of_rank out of range");
        if (node_of_[q] == link.self()) { if (r0_ < 0) r0_ = q; ++nl_; }
        else if (r0_ >= 0 && q < r0_ + nl_ && node_of_[q] != link.self()) throw std::runtime_error("pdl gpu: a node's ranks must be adjacent");
    }
    if (nl_ == 0) throw std::runtime_error("pdl gpu: this node holds no rank");
    slot_f_ = (o.slot_bytes + 15) / 16 * 4;   // floats, 16-byte multiple
    if (slot_f_ == 0) throw std::runtime_error("pdl gpu: slot_bytes is 0");
    if (link.eng().chunk_payload() == 0) throw std::runtime_error("pdl gpu: link not connected");
    cpu_ = o.cpu;
    if (cpu_ < 0) { const char * e = getenv("P100E_PDL_CPU"); if (e && *e) cpu_ = atoi(e); }
    sends_ = o.sends.empty() ? std::vector<char>((size_t) R_, 1) : o.sends;
    if ((int) sends_.size() != R_) throw std::runtime_error("pdl gpu: sends[] must have one entry per global rank");
    if (!o.devices.empty() && (int) o.devices.size() != nl_) throw std::runtime_error("pdl gpu: devices[] must have one entry per local rank");
    int cur = 0;
    ck(cudaGetDevice(&cur), "cudaGetDevice");
    dev_ = o.devices.empty() ? std::vector<int>((size_t) nl_, cur) : o.devices;

    const unsigned fl = cudaHostAllocMapped | cudaHostAllocPortable;
    auto mapped = [&](size_t bytes, void ** h, void ** d) {
        ck(cudaHostAlloc(h, bytes, fl), "cudaHostAlloc(mapped)");
        memset(*h, 0, bytes);
        ck(cudaHostGetDevicePointer(d, *h, 0), "cudaHostGetDevicePointer");
    };
    mapped((size_t) nl_ * depth_ * slot_f_ * 4, (void **) &tx_h_, (void **) &tx_d_);
    mapped((size_t) R_ * depth_ * slot_f_ * 4, (void **) &rx_h_, (void **) &rx_d_);
    mapped((size_t) nl_ * 64 * 4, (void **) &bell_h_, (void **) &bell_d_);
    mapped(64 * 4, (void **) &flags_h_, (void **) &flags_d_);
    flags_h_[fast::P2P_FLAG_TIMEOUT_MS] = o.timeout_ms;
    ctr_d_.assign((size_t) nl_, nullptr);
    done_d_.assign((size_t) nl_, nullptr);
    for (int lr = 0; lr < nl_; ++lr) {
        ck(cudaSetDevice(dev_[(size_t) lr]), "cudaSetDevice");
        ck(cudaMalloc((void **) &ctr_d_[(size_t) lr], 64 * 4), "cudaMalloc ctr");
        ck(cudaMemset(ctr_d_[(size_t) lr], 0, 64 * 4), "cudaMemset ctr");
        ck(cudaMalloc((void **) &done_d_[(size_t) lr], 4), "cudaMalloc done");
        ck(cudaMemset(done_d_[(size_t) lr], 0, 4), "cudaMemset done");
        ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
    ck(cudaSetDevice(cur), "cudaSetDevice");
    sent_.assign((size_t) nl_, 0);
    // GPU receives complete through their flag words: no completion records for the GPU streams (see engine)
    std::lock_guard<std::mutex> l(link.mu());
    link.eng().skip_done_records(N_);
}

gpu_link::~gpu_link() {
    try { stop(); } catch (...) {}
    // The engine still points into the mapped rings: send-ring entries posted stable (tx_h_), open receives (rx_h_)
    // and their flag words (flags_h_). Send what is still queued (bounded), then drop the GPU streams from the engine
    // before the rings are freed, so pdl_link's goodbye polls never read or write freed memory (a peer's NACK for a
    // GPU message is ignored from here on: the GPU waits are over at teardown).
    {
        std::lock_guard<std::mutex> l(L_.mu());
        engine & E = L_.eng();
        try {
            const uint64_t t0 = now_ns();
            while (E.tx_busy() && now_ns() - t0 < 200000000ull) E.poll();
        } catch (...) {}
        E.drop_streams(N_);
    }
    cudaFreeHost(tx_h_);
    cudaFreeHost(rx_h_);
    cudaFreeHost(bell_h_);
    cudaFreeHost(flags_h_);
    int cur = 0;
    cudaGetDevice(&cur);
    for (size_t lr = 0; lr < ctr_d_.size(); ++lr) {
        cudaSetDevice(dev_[lr]);
        cudaFree(ctr_d_[lr]);
        cudaFree(done_d_[lr]);
    }
    cudaSetDevice(cur);
}

void gpu_link::publish(int lr, const float * part, size_t n, cudaStream_t s) {
    if (n * 4 > slot_f_ * 4) throw std::runtime_error("pdl gpu: partial larger than a ring slot");
    if (n % 4 || ((uintptr_t) part & 15)) throw std::runtime_error("pdl gpu: partial must be 16-byte aligned, n % 4 == 0");
    const int threads = 256;
    const int blocks = (int) std::max<size_t>(1, std::min<size_t>((n / 4 + threads - 1) / threads, 64));
    publish_k<<<blocks, threads, 0, s>>>((const float4 *) part, (float4 *) (tx_d_ + (size_t) lr * depth_ * slot_f_), n / 4, slot_f_,
                                         depth_, ctr(lr), bell_d_ + lr * 64, flags_d_, r0_ + lr, done_d_[(size_t) lr]);
    ck(cudaGetLastError(), "publish launch");
    // WDDM (Windows, GeForce) batches launches on the host until something flushes the queue: without this the publish
    // can sit in the batch while a peer (or this node's own wait kernel) spins for it. cudaStreamQuery flushes; it
    // costs a few us and is harmless under TCC / Linux. Not while capturing (a query would invalidate the capture; the
    // caller flushes after launching the graph).
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(s, &cs) == cudaSuccess && cs == cudaStreamCaptureStatusNone) (void) cudaStreamQuery(s);
}

void gpu_link::tick(int lr, cudaStream_t s) {
    tick_k<<<1, 1, 0, s>>>(ctr(lr));
    ck(cudaGetLastError(), "tick launch");
}

void gpu_link::recv_copy(int lr, float * const * dst, size_t n, cudaStream_t s) {
    if (n * 4 > slot_f_ * 4) throw std::runtime_error("pdl gpu: partial larger than a ring slot");
    if (n % 4) throw std::runtime_error("pdl gpu: recv_copy needs n % 4 == 0");
    copy_list C{};
    for (int q = 0; q < R_; ++q) {
        if (!dst[q]) continue;
        if (node_of_[q] == L_.self() || !sends_[(size_t) q]) throw std::runtime_error("pdl gpu: recv_copy of a rank that sends nothing here");
        if ((uintptr_t) dst[q] & 15) throw std::runtime_error("pdl gpu: recv_copy destination must be 16-byte aligned");
        C.src[C.n] = rx_d_ + (size_t) q * depth_ * slot_f_;
        C.dst[C.n] = dst[q];
        C.q[C.n] = q;
        ++C.n;
    }
    if (C.n == 0) return;
    // few blocks: the waiting block occupies an SM while it spins (other local ranks may share the GPU); the copy
    // itself is PCIe-bound (n x 4 bytes per rank, 20 KB for a 27B decode row)
    const int threads = 256;
    const int blocks = (int) std::max<size_t>(1, std::min<size_t>((n / 4 + threads - 1) / threads, 8));
    recv_copy_k<<<blocks, threads, 0, s>>>(C, slot_f_, depth_, flags_d_, ctr(lr), n / 4);
    ck(cudaGetLastError(), "recv_copy launch");
}

void gpu_link::wait_add_local(int lr, float * x, const float * const * local, size_t n, cudaStream_t s) {
    gpu_parts P{};
    for (int q = 0; q < R_; ++q) {
        if (q >= r0_ && q < r0_ + nl_) { P.base[q] = local[q - r0_]; P.stride[q] = 0; }
        else { P.base[q] = rx_d_ + (size_t) q * depth_ * slot_f_; P.stride[q] = (long long) slot_f_; }
    }
    wait_add(x, P, R_, r0_ + lr, depth_, flags_d_, ctr(lr), n, s);
}

fast::p2p_ptrs gpu_link::ptrs(uint32_t seq, const float * const * local) const {
    fast::p2p_ptrs p{};
    for (int q = 0; q < R_ && q < fast::P2P_MAX_RANKS; ++q)
        p.p[q] = (q >= r0_ && q < r0_ + nl_) ? local[q - r0_] : rx(q, seq);
    return p;
}

int gpu_link::timeouts() const { return ((volatile int *) flags_h_)[1]; }

bool gpu_link::host_ready(uint32_t seq) const {
    for (int q = 0; q < R_; ++q)
        if (node_of_[q] != L_.self() && sends_[(size_t) q] && (int32_t) ((uint32_t) ((volatile int *) flags_h_)[fast::P2P_FLAG_PEER0 + q] - seq) < 0) return false;
    return true;
}

bool gpu_link::host_wait(uint32_t seq, int timeout_ms) const {
    const uint64_t t0 = now_ns();
    while (!host_ready(seq)) {
        if (!run_ || (now_ns() - t0) / 1000000 > (uint64_t) timeout_ms) return false;
        cpu_relax();
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    return true;
}

std::string gpu_link::error() const {
    std::lock_guard<std::mutex> l(err_mu_);
    return err_;
}

void gpu_link::start() {
    if (run_) return;
    run_ = true;
    L_.set_pump(true);
    th_ = std::thread([this] { run(); });
}

void gpu_link::stop() {
    if (!run_ && !th_.joinable()) return;
    run_ = false;
    if (th_.joinable()) th_.join();
    L_.set_pump(false);
    const std::string e = error();
    if (!e.empty()) throw std::runtime_error(e);
}

// The spin thread: never sleeps. Each loop: bells -> receives + sends; one engine poll.
void gpu_link::run() {
    if (cpu_ >= 0) {
#ifdef _WIN32
        SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << cpu_);
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
#else
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(cpu_, &cs);
        pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
#endif
    }
    try {
        engine & E = L_.eng();
        while (run_) {
            if (L_.host_waiting()) { cpu_relax(); continue; }   // a host collective polls the engine meanwhile
            std::lock_guard<std::mutex> l(L_.mu());
            // newest seq any local rank published: post the receives of every all-reduce up to it
            uint32_t top = posted_;
            for (int lr = 0; lr < nl_; ++lr) {
                const uint32_t b = (uint32_t) ((volatile int *) bell_h_)[lr * 64];
                if ((int32_t) (b - top) > 0) top = b;
            }
            while ((int32_t) (top - posted_) > 0) {
                const uint32_t s = ++posted_;
                std::atomic_thread_fence(std::memory_order_acquire);
                size_t bytes = 0;   // the size this all-reduce has, from a local rank that already published it (all equal)
                for (int lr = 0; lr < nl_; ++lr) {
                    volatile int * bell = (volatile int *) (bell_h_ + lr * 64);
                    if ((int32_t) ((uint32_t) bell[0] - s) >= 0) { bytes = (size_t) (uint32_t) bell[8 + s % depth_]; break; }
                }
                for (int q = 0; q < R_; ++q) {
                    if (node_of_[q] == L_.self() || !sends_[(size_t) q]) continue;
                    E.post_recv(s, N_ + q, node_of_[q], TAG_GPU_AR, {{rx_h_ + ((size_t) q * depth_ + s % depth_) * slot_f_, bytes}},
                                (volatile int *) (flags_h_ + fast::P2P_FLAG_PEER0 + q), (int) s);
                }
            }
            for (int lr = 0; lr < nl_; ++lr) {
                volatile int * bell = (volatile int *) (bell_h_ + lr * 64);
                const uint32_t b = (uint32_t) bell[0];
                while ((int32_t) (b - sent_[lr]) > 0) {
                    const uint32_t s = ++sent_[lr];
                    std::atomic_thread_fence(std::memory_order_acquire);
                    const size_t bytes = (size_t) (uint32_t) bell[8 + s % depth_];
                    E.post_send(s, N_ + r0_ + lr, TAG_GPU_AR, 0, 0, {{tx_h_ + ((size_t) lr * depth_ + s % depth_) * slot_f_, bytes}}, true);
                    ++st_.published;
                }
            }
            E.poll();
            // completed receives already set their flag words (and keep no completion record: skip_done_records)
            ++st_.loops;
        }
    } catch (const std::exception & e) {
        std::lock_guard<std::mutex> l(err_mu_);
        err_ = e.what();
        run_ = false;
        L_.pump_failed(e.what());
        // make every GPU wait fail fast instead of spinning to its timeout
        ((volatile int *) flags_h_)[1] = 1;
    }
}

} // namespace p100e::pdl
