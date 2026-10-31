// pdl GPU side: all-reduce partials go GPU -> host-mapped ring -> pdl frames -> peer's host-mapped ring -> peer GPU,
// with no cudaMemcpy, no host staging buffer and no host round trip in the GPU stream. The same flag protocol as
// p100e's P2P all-reduce (src/fast/fused.cuh: p2p_signal_n / addn_rmsnorm_q_wait_ctr), with the flag block in
// host-mapped memory and a CPU spin thread standing in for the peer GPU's P2P store.
//
//   publish (GPU, one launch per local rank and all-reduce): copy the partial into tx slot seq % depth (mapped host
//     memory, PCIe writes), seq = ++*ctr (this rank's own counter, device memory), bytes[slot] = size,
//     __threadfence_system(), bell = seq, and flags[P2P_FLAG_PEER0 + global rank] = seq (so this node's other ranks'
//     waits see it, exactly as p2p_signal_n stores into every peer's flag block).
//   spin thread (CPU, pinned with P100E_PDL_CPU=<core>): sees a bell move -> posts the receives of that all-reduce
//     (every remote rank's partial into rx slot seq % depth) and the send of the tx slot (stable memory: the frames are
//     built straight from the mapped ring); polls the adapter; a remote rank's message complete -> flags[P2P_FLAG_PEER0
//     + q] = seq (release order after the payload).
//   wait (GPU): thread 0 of every block spins on the mapped flag words until every other rank's word reached *ctr
//     (bounded by flags[P2P_FLAG_TIMEOUT_MS], counted in flags[1]: the fused kernels' convention), then
//     x += p_0 + p_1 + ... in global rank order (the same f32 adds on every node: bit-identical replicas).
//     Either pdl::wait_add (below; the remote slot pointer is computed on the device from *ctr, so it can live in a
//     CUDA graph) or the existing fast::addn_rmsnorm_q_wait_ctr(x, ptrs(...), R, self, flags(), ctr(lr), ...) with
//     ptrs() of this all-reduce (host-known seq: eager launches; graphs need ptrs that do not move, see tp_model.cu).
//
// Copies per all-reduce and direction: the CPU builds frames from the mapped tx slot into the port's transfer buffer
// (one memcpy, fused with the CRC) and scatters received payload into the mapped rx slot (one memcpy). The GPU reads
// and writes host memory directly over PCIe.
//
// Slot reuse is safe with depth >= 2 (4 by default): a node publishes seq + depth only after its wait for
// seq + depth - 1, which needs every peer's partial of that all-reduce, which a peer sends only after it finished seq.
#pragma once

#include "pdl/pdl_link.h"
#include "fast/fused.cuh"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace p100e::pdl {

constexpr uint8_t TAG_GPU_AR = 7;   // pdl header tag of a GPU all-reduce message (net_link tags are 1..6)
constexpr int GPU_MAX_RANKS = 8;

struct gpu_parts {   // per global rank: base pointer and slot stride in floats (0 = a local partial, no slots)
    const float * base[GPU_MAX_RANKS];
    long long stride[GPU_MAX_RANKS];
};

// x[i] += p_0[i] + p_1[i] + ... + p_{R-1}[i] (global rank order) after waiting for every other rank's flag word
// (flags + P2P_FLAG_PEER0 + q >= *ctr); slot of a remote rank = (*ctr) % depth. flags = nullptr: no wait.
void wait_add(float * x, const gpu_parts & parts, int R, int self, int depth, const int * flags, const int * ctr, size_t n,
              cudaStream_t s);

struct gpu_link_opts {
    int depth = 4;                    // ring slots per rank
    size_t slot_bytes = 0;            // largest partial (max_batch x E x 4)
    std::vector<int> node_of_rank;    // global rank -> node index (tp_model node_r0_ / node_nr_)
    int timeout_ms = 10000;           // GPU spin bound (P2P_FLAG_TIMEOUT_MS of the flag block)
    int cpu = -1;                     // pin the spin thread to this core (-1: P100E_PDL_CPU or no pinning)
    // per global rank: 1 = sends its partial (default: every rank). tp_model's pre-add (P100E_NET_PREADD) folds node
    // 0's ranks into one message from rank 0: the others send nothing and nobody posts a receive for them
    std::vector<char> sends;
    // per local rank: the CUDA device whose kernels publish / wait for it (its counter lives there; empty = the
    // current device for every local rank). Several GPUs per node (2x P100) need it: a counter in another GPU's
    // memory would need peer access.
    std::vector<int> devices;
};

struct gpu_link_stats {
    uint64_t published = 0, received = 0, loops = 0;
};

class gpu_link {
public:
    gpu_link(pdl_link & link, const gpu_link_opts & o);
    ~gpu_link();
    gpu_link(const gpu_link &) = delete;
    gpu_link & operator=(const gpu_link &) = delete;

    void start();   // the spin thread
    void stop();    // joins it; rethrows its error (if any)

    int n_ranks() const { return R_; }
    int r0() const { return r0_; }       // this node's first global rank
    int n_local() const { return nl_; }

    // GPU side (device pointers)
    float * tx(int lr, uint32_t seq) const { return tx_d_ + ((size_t) lr * depth_ + seq % depth_) * slot_f_; }
    float * rx(int q, uint32_t seq) const { return rx_d_ + ((size_t) q * depth_ + seq % depth_) * slot_f_; }
    int * flags() const { return flags_d_; }      // P2P-format flag block (mapped), shared by this node's ranks
    int * ctr(int lr) const { return ctr_d_[(size_t) lr]; }   // own signal counter of local rank lr (its device's memory)
    uint32_t next_seq() const { return host_seq_ + 1; }    // the seq the next publish will use (host count)

    // launches (stream order): publish local rank lr's partial (n floats, 16-byte aligned) for the next all-reduce;
    // wait_add for local rank lr: x += every rank's partial, rank order (local[] = this node's partial pointers)
    // (the launch is followed by a cudaStreamQuery to flush the WDDM batch, except while s is being captured)
    void publish(int lr, const float * part, size_t n, cudaStream_t s);
    // the counter step of a local rank that sends nothing this all-reduce (sends[] = 0): its waits need the same seq
    void tick(int lr, cudaStream_t s);
    // tp_model (P100E_NET_GPULINK): wait until every remote rank q with dst[q] != nullptr has its partial of this
    // rank's current seq (*ctr(lr)) in the rx ring, then copy each (n floats) to dst[q] (device memory). Graph safe:
    // the rx slot comes from the device counter. Same bound / timeout count as the other waits (flags[1]).
    void recv_copy(int lr, float * const * dst, size_t n, cudaStream_t s);
    void wait_add_local(int lr, float * x, const float * const * local, size_t n, cudaStream_t s);
    // the pointers of all-reduce `seq` for fast::addn_rmsnorm_q_wait_ctr (local[] = this node's partials)
    fast::p2p_ptrs ptrs(uint32_t seq, const float * const * local) const;
    void advance() { ++host_seq_; }   // call once per all-reduce after every local rank published (host count)

    // host-side wait: every remote rank's partial of all-reduce seq has landed (flag words); host_wait spins on it
    // (false after timeout_ms or a spin-thread failure). For GPUs shared by several processes (tests), where a
    // spinning wait kernel would hold the GPU against the peer process's kernels.
    bool host_ready(uint32_t seq) const;
    bool host_wait(uint32_t seq, int timeout_ms) const;
    int timeouts() const;              // flags[1]: GPU waits that timed out (results invalid)
    const gpu_link_stats & stats() const { return st_; }
    std::string error() const;         // the spin thread's failure ("" while fine)

private:
    void run();
    pdl_link & L_;
    int R_ = 0, r0_ = 0, nl_ = 0, depth_ = 4, N_ = 0;
    size_t slot_f_ = 0;
    std::vector<int> node_of_;
    int cpu_ = -1;
    // mapped host memory (h = host pointer, d = device pointer of the same bytes)
    float * tx_h_ = nullptr, * tx_d_ = nullptr, * rx_h_ = nullptr, * rx_d_ = nullptr;
    int * bell_h_ = nullptr, * bell_d_ = nullptr;     // per local rank 64 ints: [0] seq, [8 + slot] bytes of that slot
    int * flags_h_ = nullptr, * flags_d_ = nullptr;   // 64 ints, P2P layout
    std::vector<int *> ctr_d_;                        // per local rank, on its device: 64 ints ([0] = the counter)
    std::vector<int *> done_d_;                       // per local rank, on its device: publish's last-block counter
    std::vector<int> dev_;                            // per local rank: its device
    std::vector<char> sends_;                         // per global rank (opts.sends)
    uint32_t host_seq_ = 0;
    std::vector<uint32_t> sent_;
    uint32_t posted_ = 0;
    std::thread th_;
    std::atomic<bool> run_{false};
    std::string err_;
    mutable std::mutex err_mu_;
    gpu_link_stats st_;
};

} // namespace p100e::pdl
