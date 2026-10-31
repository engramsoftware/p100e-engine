#include "allreduce.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "convert.cuh"
#include "ggml-impl.h"
#include "pf-xchg.cuh"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>

// ---------------------------------------------------------------------------
// CUDA AllReduce for tensor-parallel inference across two GPUs.
//
// Provides an in-place sum reduction over matching tensors on two CUDA
// devices in the same process.  Used by the tensor-split path alongside
// NCCL; targets setups without NVLink, where data is exchanged between the
// GPUs by staging it through pinned host memory over PCIe.
//
// Two reduction strategies are selected per call by tensor size:
//
//   * Chunked kernel path (small reductions): a single CUDA kernel both
//     stages data through pinned host memory and performs the local sum.
//     Cross-GPU synchronization happens *inside the kernel* (busy-wait on
//     a host-memory flag), which keeps launch overhead low for the
//     latency-sensitive token-generation case.
//
//   * Copy-engine path (large reductions): the transfer is split into
//     D2H + H2D cudaMemcpyAsync chunks driven by the GPU's copy engine,
//     followed by a small device-side add kernel.  Cross-GPU
//     synchronization happens *outside the kernel*, via CUDA events
//     between streams.  This keeps the compute engine free while large
//     transfers are in flight, which matters for prefill-sized tensors.
//     Reductions larger than the per-call inner cap are processed by an
//     outer chunker that issues sequential inner calls.
//
// Pascal (sm_60/sm_61) opt-in, GGML_CUDA_AR_PASCAL=1: a pair that includes a
// pre-Volta device (e.g. sm_86 + sm_60) uses a separate F32-only chunked
// kernel (ggml_cuda_ar_kernel_f32_pascal) instead of the template below:
//   * one fixed 16-byte float4 stripe unit on every arch, so both GPUs agree
//     on which elements each block's arrival flag covers;
//   * every peer-written value (flag and data) is read with ld.global.cv, so
//     a stale L1/L2 line from the slot's previous use is never consumed;
//   * every cross-GPU spin is bounded by %globaltimer (plus a clock64
//     backstop); on timeout the block records the failure in mapped pinned
//     host memory and exits, and the host aborts with a clear message
//     (ggml_cuda_ar_pascal_check_status) instead of hanging or tripping a TDR;
//   * the wire is always exact F32 (BF16 is forced off, F16/BF16 inputs go to
//     the meta-backend butterfly per call).
// With GGML_CUDA_AR_PASCAL unset, pre-Volta devices are rejected exactly as
// before and the Volta+ path is unchanged.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cross-GPU signal mechanism
//
// One int per (slot, rank) pair in pinned host memory.  Each AR call writes a
// strictly increasing token (= the AR call number) into its own arrival int.
// The peer spins until its read of the other's arrival int equals the token
// it expects for this call -- a mismatch means the peer hasn't arrived yet.
// Tokens never repeat over realistic call rates (32-bit int wraps in tens of
// days at thousands of ARs/sec), so arrival ints don't need to be reset
// between calls; we initialize once at pipeline init and let the values
// accumulate.
//
// There is exactly one writer (the owning GPU) and one reader (the peer), so
// we don't need atomics.  A volatile store paired with __threadfence_system()
// provides the release ordering that makes the D2H writes visible system-wide
// before the arrival token is observed.
//
// atomicAdd_system() requires hostNativeAtomicSupported, which is unavailable
// on PCIe-attached consumer GPUs without NVLink, so the volatile path is the
// portable choice.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void ggml_cuda_ar_signal_set(int * p, int token) {
    *(volatile int *)p = token;
}
static __device__ __forceinline__ int ggml_cuda_ar_signal_get(const int * p) {
    return *(const volatile int *)p;
}

// Byte spacing between adjacent arrival ints.  64 bytes (one cache line)
// ensures each GPU/block's arrival slot lives on its own line, preventing
// false-sharing stalls on the polling GPU.
static constexpr size_t GGML_CUDA_AR_ARRIVAL_STRIDE = 64;

// Number of blocks the chunked kernel launches with.  Each block stripes a
// disjoint slice of the data and synchronizes through its own arrival-token
// slot so multiple SMs can pump PCIe stores in parallel.
static constexpr int GGML_CUDA_AR_KERNEL_BLOCKS = 8;

// GGML_CUDA_AR_PASCAL=1 only.  One 64-byte status line per rank in mapped
// pinned host memory: [0] timed-out flag, [1] token, [2] block, [3] last peer
// token seen, [4] elapsed ms, [5] device id.
static constexpr int GGML_CUDA_AR_STATUS_INTS = 16;

// GGML_CUDA_AR_PASCAL=1 only.  In-kernel spin limits (ms).  A device with
// cudaDevAttrKernelExecTimeout=1 (WDDM, e.g. the display GPU) is clamped well
// under the default 2 s TDR delay; a TCC device has no watchdog.
// Override with GGML_CUDA_AR_SPIN_TIMEOUT_MS (clamped to [MIN, *_MAX]).
static constexpr uint64_t GGML_CUDA_AR_SPIN_MS_MIN      = 10;
static constexpr uint64_t GGML_CUDA_AR_SPIN_MS_WDDM     = 500;
static constexpr uint64_t GGML_CUDA_AR_SPIN_MS_WDDM_MAX = 750;
static constexpr uint64_t GGML_CUDA_AR_SPIN_MS_TCC      = 2000;
static constexpr uint64_t GGML_CUDA_AR_SPIN_MS_TCC_MAX  = 60000;

// ---------------------------------------------------------------------------
// Chunked kernel AllReduce -- 2 GPUs, supports float, half, and bfloat16.
//
// Both GPUs run this kernel simultaneously on independent streams.  sendbuf
// and recvbuf live in T_dst (the caller's tensor type); host_mine / host_other
// carry data in T_wire (the on-wire type, possibly narrower than T_dst -- e.g.
// T_dst=F32 with T_wire=BF16 halves the bytes pushed across PCIe).  When
// T_dst == T_wire the casts below are no-ops.
//
// Each GPU runs three phases:
//
//   Phase 1 (all threads): cast sendbuf (T_dst) -> T_wire and store as
//                          single-instruction-width vectors into host_mine.
//                          __threadfence_system() commits these writes to host
//                          memory.
//   Phase 2 (thread 0):    write token to arrival_mine; spin until
//                          arrival_other == token.
//   Phase 3 (all threads): read T_wire vectors from host_other, cast
//                          each element to T_dst, and sum with the local
//                          sendbuf value (also rounded through T_wire so that
//                          both GPUs truncate identically -- this guarantees
//                          bit-equivalent results across the two devices).
//
// Multi-block: blocks stripe vectors across (gridDim.x * blockDim.x) global
// threads to keep multiple SMs issuing PCIe stores in parallel.  Each block
// has its own arrival-token slot (offset by blockIdx.x * ARRIVAL_STRIDE);
// thread 0 of each block signals/spins on that slot independently of other
// blocks.  Tail elements (the leftover < ELEMS_PER_VEC at the end) are
// handled only by block 0 to avoid cross-block writes to the same slots.
// ---------------------------------------------------------------------------
template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_ar_kernel(
        const T_dst  *              sendbuf,
        T_dst        *              recvbuf,
        T_wire       * __restrict__ host_mine,
        const T_wire * __restrict__ host_other,
        int                         count,
        int *                       arrival_mine,
        int *                       arrival_other,
        int                         token) {

    // Vector unit for the wire type, sized to the arch's widest single-instruction
    // copy (16 B on Volta+).  Each phase-1 iter writes one vector to host memory;
    // each phase-3 iter reads one and produces ELEMS_PER_VEC sums.
    constexpr int ELEMS_PER_VEC = ggml_cuda_get_max_cpy_bytes() / sizeof(T_wire);
    constexpr int ARRIVAL_INTS  = (int)(GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));

    const int tid       = threadIdx.x;
    const int nt        = blockDim.x;
    const int bid       = blockIdx.x;
    const int gtid      = bid * nt + tid;
    const int gnt       = gridDim.x * nt;
    const int count_vec = count / ELEMS_PER_VEC;
    const int tail      = count_vec * ELEMS_PER_VEC;

    // Phase 1: cast sendbuf (T_dst) -> host_mine (T_wire) and store as vectors.
    {
        for (int i = gtid; i < count_vec; i += gnt) {
            const int off = i * ELEMS_PER_VEC;
            T_wire wire[ELEMS_PER_VEC];
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                wire[k] = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
            }
            ggml_cuda_memcpy_1<sizeof(wire)>(&host_mine[off], wire);
        }
        if (bid == 0 && tid < count - tail) {
            host_mine[tail + tid] = ggml_cuda_cast<T_wire>(sendbuf[tail + tid]);
        }
    }

    // Commit this block's host writes before signalling.
    __threadfence_system();
    __syncthreads();

    // Phase 2: thread 0 of each block signals on its own arrival slot, then
    // spins for the matching slot from peer.  Per-block tokens mean blocks
    // proceed independently -- no inter-block barrier needed.
    if (tid == 0) {
        int       * my_slot    = arrival_mine  + bid * ARRIVAL_INTS;
        const int * other_slot = arrival_other + bid * ARRIVAL_INTS;

        ggml_cuda_ar_signal_set(my_slot, token);
        __threadfence_system(); // make our signal visible system-wide

        while (ggml_cuda_ar_signal_get(other_slot) != token) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(100);
#else
            NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
        }
    }

    __syncthreads();

    // Acquire peer's host_other writes (this block's stripe of them).
    __threadfence_system();

    // Phase 3: read peer's T_wire vector, cast both sides through T_wire for
    // bit-equivalence, sum in T_dst precision, and write back to recvbuf.
    {
        for (int i = gtid; i < count_vec; i += gnt) {
            const int off = i * ELEMS_PER_VEC;
            T_wire wire[ELEMS_PER_VEC];
            ggml_cuda_memcpy_1<sizeof(wire)>(wire, &host_other[off]);
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                const T_wire d_low = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
                recvbuf[off + k] = ggml_cuda_cast<T_dst>(
                    ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(wire[k]));
            }
        }
        if (bid == 0 && tid < count - tail) {
            const T_wire d_low = ggml_cuda_cast<T_wire>(sendbuf[tail + tid]);
            recvbuf[tail + tid] = ggml_cuda_cast<T_dst>(
                ggml_cuda_cast<float>(d_low) +
                ggml_cuda_cast<float>(host_other[tail + tid]));
        }
    }
}

// ---------------------------------------------------------------------------
// Pascal-safe chunked kernel (GGML_CUDA_AR_PASCAL=1) -- 2 GPUs, F32 only.
//
// Same three phases and the same per-block arrival-token protocol as
// ggml_cuda_ar_kernel, with three differences that make it correct on a mixed
// pre-Volta / Volta+ pair:
//
//   * Fixed 16-byte float4 stripe unit on every arch.  ggml_cuda_ar_kernel
//     sizes its vector from ggml_cuda_get_max_cpy_bytes(), which is 8 B on
//     sm_6x and 16 B on sm_70+; with different widths a block's flag would
//     cover different elements on the two GPUs.  Grid and block size are the
//     same host constants on both devices, so vector i is owned by the same
//     (block, thread) on both sides.
//   * All peer-written memory (arrival token and data) is read with
//     ld.global.cv, which re-fetches system memory instead of hitting a line
//     cached in L1/L2 from this slot's previous use (2 ARs ago).  Pre-Volta has
//     no fence-based acquire for plain loads.
//   * The spin is bounded (wall clock via %globaltimer, clock64 backstop).  On
//     timeout thread 0 records the failure in the mapped status line and the
//     block exits without summing; the host aborts at its next status check.
//     Our own token is always written before we spin, so a timeout on one side
//     never strands the peer.
//
// Sum is own + peer in one fp32 add: the same IEEE operation the meta-backend
// butterfly's GGML_OP_ADD performs, and commutative, so both GPUs and both
// paths produce identical bits.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ uint64_t ggml_cuda_ar_globaltimer_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

static __device__ __forceinline__ int ggml_cuda_ar_load_cv(const int * p) {
    int v;
    asm volatile("ld.global.cv.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

static __global__ void ggml_cuda_ar_kernel_f32_pascal(
        float       *              data,
        float       * __restrict__ host_mine,
        const float * __restrict__ host_other,
        int                        count,
        int         *              arrival_mine,
        const int   *              arrival_other,
        int                        token,
        unsigned long long         timeout_ns,
        int         *              status,
        const int   *              status_peer,
        int                        dev_id,
        int                        skip_signal) {

    constexpr int V            = 4; // float4 on EVERY arch: sm_6x and sm_70+ must agree
    constexpr int ARRIVAL_INTS = (int)(GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));

    const int tid       = threadIdx.x;
    const int bid       = blockIdx.x;
    const int gtid      = bid * blockDim.x + tid;
    const int gnt       = gridDim.x * blockDim.x;
    const int count_vec = count / V;
    const int tail      = count_vec * V;

    // Phase 1: stage our partial into host_mine (16-B stores).
    for (int i = gtid; i < count_vec; i += gnt) {
        reinterpret_cast<float4 *>(host_mine)[i] = reinterpret_cast<const float4 *>(data)[i];
    }
    if (bid == 0 && tid < count - tail) {
        host_mine[tail + tid] = data[tail + tid];
    }

    // MEMBAR.SYS: this thread's host writes are performed at system scope
    // before the barrier, hence before thread 0's flag store.
    __threadfence_system();
    __syncthreads();

    // Phase 2: thread 0 signals, then waits for the peer with a bounded spin.
    __shared__ int s_ok;
    if (tid == 0) {
        const int * other_slot = arrival_other + bid * ARRIVAL_INTS;
        if (!skip_signal) {
            *(volatile int *)(arrival_mine + bid * ARRIVAL_INTS) = token;
        }
        __threadfence_system();

        const uint64_t t0 = ggml_cuda_ar_globaltimer_ns();
        long long      c0 = clock64();
        int seen = ggml_cuda_ar_load_cv(other_slot);
        // Fail fast: once either rank has recorded a timeout the host is going
        // to abort, so a later AR must not spin out a full timeout again.
        bool failed = ggml_cuda_ar_load_cv(status) != 0 || ggml_cuda_ar_load_cv(status_peer) != 0;
        unsigned int polls = 0;
        while (seen != token && !failed) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(100);
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            // clock64 is a per-SM counter: if it reads behind c0 (e.g. the block
            // resumed on another SM after preemption), re-base instead of
            // wrapping to a huge unsigned delta.
            long long dc = clock64() - c0;
            if (dc < 0) {
                c0 = clock64();
                dc = 0;
            }
            if (ggml_cuda_ar_globaltimer_ns() - t0 > timeout_ns ||
                (unsigned long long) dc > 2ull * timeout_ns) {
                // One last look: we may have been time-sliced out while the
                // peer arrived.
                seen = ggml_cuda_ar_load_cv(other_slot);
                break;
            }
            seen = ggml_cuda_ar_load_cv(other_slot);
            if ((++polls & 1023u) == 0) {
                failed = ggml_cuda_ar_load_cv(status_peer) != 0;
            }
        }
        s_ok = seen == token;
        if (!s_ok && !failed) {
            volatile int * st = status;
            st[1] = token;
            st[2] = bid;
            st[3] = seen;
            st[4] = (int) ((ggml_cuda_ar_globaltimer_ns() - t0) / 1000000ull);
            st[5] = dev_id;
            __threadfence_system();   // fields before the flag
            st[0] = 1;
            __threadfence_system();
        }
    }
    __syncthreads();

    if (!s_ok) {
        return; // data keeps the local partial; the host aborts at its next status check
    }

    __threadfence_system();

    // Phase 3: own + peer, peer read with ld.global.cv (never a stale line).
    for (int i = gtid; i < count_vec; i += gnt) {
        const float4 w = __ldcv(reinterpret_cast<const float4 *>(host_other) + i);
        float4       d = reinterpret_cast<const float4 *>(data)[i];
        d.x += w.x;
        d.y += w.y;
        d.z += w.z;
        d.w += w.w;
        reinterpret_cast<float4 *>(data)[i] = d;
    }
    if (bid == 0 && tid < count - tail) {
        data[tail + tid] += __ldcv(host_other + tail + tid);
    }
}

// Combined load-convert-add kernel.  The peer's contribution arrives as T_src
// (which may be a lower-precision type than T_dst when the BF16 round-trip is
// active).  For bit-equivalence between the two GPUs, dst is first rounded
// through T_src's precision via ggml_cuda_cast -- peer already truncated its
// own value the same way before sending -- so both sides perform identical
// arithmetic.  When T_dst == T_src the round-trip cast is a no-op.
template <typename T_dst, typename T_src>
static __global__ void ggml_cuda_ar_add_kernel(
        T_dst       * __restrict__ dst,
        const T_src * __restrict__ src,
        int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        const T_src d_low = ggml_cuda_cast<T_src>(dst[i]);
        dst[i] = ggml_cuda_cast<T_dst>(
            ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(src[i]));
    }
}

// 8-bit wire (GGML_CUDA_AR_Q8=1, copy-engine path only, lossy): 32 F32 values -> fp32 scale + 32 int8
// (36 B instead of 128 B).  Each GPU quantizes its own contribution once; the add kernel then computes
// deq(own) + deq(peer) on both GPUs, so both get bit-identical sums (float + is commutative) and no
// GPU sees its own exact value mixed with the peer's rounded one.
struct ggml_cuda_ar_q8_32 {
    float  d;
    int8_t q[32];
};
static_assert(sizeof(ggml_cuda_ar_q8_32) == 36, "q8 block size");

template <typename T> struct ggml_cuda_ar_elems           { static constexpr int n = 1;  };
template <>           struct ggml_cuda_ar_elems<ggml_cuda_ar_q8_32> { static constexpr int n = 32; };

// one warp per block of 32 values
static __global__ void ggml_cuda_ar_q8_quantize(const float * __restrict__ x, ggml_cuda_ar_q8_32 * __restrict__ y, int nblk) {
    const int b    = blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (b >= nblk) {
        return;
    }
    const float v = x[(int64_t) b*32 + lane];
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    }
    const float d  = amax/127.0f;
    const float id = d > 0.0f ? 1.0f/d : 0.0f;
    y[b].q[lane] = (int8_t) __float2int_rn(v*id);
    if (lane == 0) {
        y[b].d = d;
    }
}

static __global__ void ggml_cuda_ar_q8_add_kernel(
        float * __restrict__ dst, const ggml_cuda_ar_q8_32 * __restrict__ peer, const ggml_cuda_ar_q8_32 * __restrict__ own,
        int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        const int b = i >> 5, l = i & 31;
        dst[i] = own[b].d*(float) own[b].q[l] + peer[b].d*(float) peer[b].q[l];
    }
}

// add dispatch used by copy_impl: ne counts T_src elements
template <typename T_dst, typename T_src>
static void ggml_cuda_ar_launch_add(T_dst * dst, const T_src * peer, const T_src * own, int64_t ne, cudaStream_t stream) {
    GGML_UNUSED(own);
    const int block_size = 256;
    int n_blocks = (int) ((ne + block_size - 1) / block_size);
    if (n_blocks > 1024) {
        n_blocks = 1024;
    }
    ggml_cuda_ar_add_kernel<T_dst, T_src><<<n_blocks, block_size, 0, stream>>>(dst, peer, (int) ne);
}
template <>
void ggml_cuda_ar_launch_add<float, ggml_cuda_ar_q8_32>(float * dst, const ggml_cuda_ar_q8_32 * peer, const ggml_cuda_ar_q8_32 * own,
        int64_t ne, cudaStream_t stream) {
    const int64_t nf = ne*32;
    const int block_size = 256;
    int n_blocks = (int) ((nf + block_size - 1) / block_size);
    if (n_blocks > 1024) {
        n_blocks = 1024;
    }
    ggml_cuda_ar_q8_add_kernel<<<n_blocks, block_size, 0, stream>>>(dst, peer, own, (int) nf);
}

// ---------------------------------------------------------------------------
// Pipeline structure
// ---------------------------------------------------------------------------

// Number of slots in the event / arrival ring.  Two slots is sufficient:
// lockstep guarantees the two GPUs are at most one AR (or chunk) apart, so
// slot[N%2] is always safe to reuse -- peer has already consumed slot[N%2]
// from AR N-2 by the time we get to AR N.  acquire_slot's
// cudaEventSynchronize on ev.ker for both devices makes that consumption
// explicit before we overwrite host_buf[slot] for the new AR.
static constexpr int GGML_CUDA_AR_POOL_SIZE = 2;

// Maximum chunk size (bytes per GPU) handled by one chunked kernel launch.
// Larger tensors are reduced by issuing multiple chunked launches.
static constexpr size_t GGML_CUDA_AR_MAX_BYTES = 1024 * 1024; // 1 MB

// Copy-engine path: largest tensor accepted on this path; sets host_large /
// dev_tmp allocation size.
static constexpr size_t GGML_CUDA_AR_COPY_MAX_BYTES = 32 * 1024 * 1024; // 32 MB

// AR wire size at which the copy-engine path takes over from the chunked-
// kernel path.  Override via GGML_CUDA_AR_COPY_THRESHOLD.
static constexpr size_t GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT = 1024 * 1024; // 1 MB
// Per-call CE chunk-size heuristic: chunk_bytes = clamp(nbytes / 4, MIN, MAX).
// The /4 keeps ~4 chunks in flight at any moment (good D2H/H2D overlap with
// the peer); the clamps cover the cases where nbytes/4 is too small (per-
// memcpy fixed cost dominates) or too large (chunk-level pipelining stalls).
// Env var GGML_CUDA_AR_COPY_CHUNK_BYTES can override with a fixed value.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN = 512 * 1024;       // 512 KB
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX = 2 * 1024 * 1024;  // 2 MB
// Absolute floor that an env-var override is allowed to set; this caps the
// per-slot copy-event array.  256 KB -> up to 128 chunks per 32 MB tensor.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN = 256 * 1024;
static constexpr int GGML_CUDA_AR_COPY_MAX_CHUNKS =
    static_cast<int>((GGML_CUDA_AR_COPY_MAX_BYTES + GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN - 1) /
                    GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);

struct ggml_cuda_ar_event_slot {
    cudaEvent_t app = nullptr;  // upstream computation complete
    cudaEvent_t cpy[GGML_CUDA_AR_COPY_MAX_CHUNKS] = {};  // copy-engine D2H chunks complete
    cudaEvent_t h2d = nullptr;  // copy-engine H2Ds complete (handoff AR stream -> compute stream)
    cudaEvent_t ker = nullptr;  // AllReduce kernel complete
};

// Mapped pinned host allocation: cudaHostAlloc + cudaHostGetDevicePointer
// in one place, with the host handle preserved for cudaFreeHost.  Used where
// the CPU never touches the buffer -- only the device reads/writes via the
// mapped device pointer.  Required on systems where cudaDevAttrCanUseHost-
// PointerForRegisteredMem is 0 and the host pointer can't be used as a
// device pointer.
struct ggml_cuda_ar_host_mapping {
    uint8_t * host = nullptr;   // cudaFreeHost handle; also the H-side ptr for cudaMemcpyAsync
    uint8_t * dev  = nullptr;   // device-side pointer for kernels / cudaMemset

    cudaError_t alloc(size_t bytes) {
        cudaError_t rc = cudaHostAlloc(reinterpret_cast<void **>(&host), bytes,
                                       cudaHostAllocPortable | cudaHostAllocMapped);
        if (rc != cudaSuccess) {
            host = nullptr;
            return rc;
        }
        rc = cudaHostGetDevicePointer(reinterpret_cast<void **>(&dev), host, 0);
        if (rc != cudaSuccess) {
            cudaFreeHost(host);
            host = nullptr;
            dev  = nullptr;
        }
        return rc;
    }

    void free() {
        if (host) {
            cudaFreeHost(host);
            host = nullptr;
            dev  = nullptr;
        }
    }
};

struct ggml_cuda_ar_pipeline {
    int      n_devices;
    int      devices[GGML_CUDA_MAX_DEVICES];
    size_t   buf_bytes;    // bytes per device in host_buf[]
    size_t   copy_bytes;   // bytes per device in host_large[] / dev_tmp[]
    size_t   copy_threshold;
    size_t   copy_chunk_bytes;
    size_t   bf16_threshold; // tensors >= this size (bytes) are reduced via FP32->BF16 round-trip; 0 disables
    bool     q8_wire = false; // GGML_CUDA_AR_Q8=1: copy-engine ARs (F32, ne % 32 == 0) go over PCIe as 8-bit blocks
    uint64_t call_count;

    // GGML_CUDA_AR_PASCAL=1 only (all false/zero on the Volta+ path).
    bool     pascal    = false;  // use ggml_cuda_ar_kernel_f32_pascal, F32 wire only
    bool     pascal_ce = false;  // copy-engine path allowed (always true when !pascal)
    uint64_t spin_timeout_ns[GGML_CUDA_MAX_DEVICES] = {};  // per rank
    bool     wddm[GGML_CUDA_MAX_DEVICES] = {};  // cudaDevAttrKernelExecTimeout set (WDDM, may batch launches)
    int      launch_order[2] = { 0, 1 };        // no-watchdog (TCC) rank first, WDDM rank second
    uint64_t test_drop_token = 0;  // fault injection: rank test_drop_rank skips its signal at this token
    int      test_drop_rank  = 0;

    // Per-device resources.
    ggml_cuda_ar_host_mapping host_buf[GGML_CUDA_MAX_DEVICES];   // pinned staging (chunked kernel)
    ggml_cuda_ar_host_mapping host_large[GGML_CUDA_MAX_DEVICES]; // pinned staging (copy-engine)
    char *                    dev_tmp[GGML_CUDA_MAX_DEVICES];    // device scratch for copy-engine path
    cudaStream_t             streams[GGML_CUDA_MAX_DEVICES];   // non-blocking
    ggml_cuda_ar_event_slot  ev_pool[GGML_CUDA_MAX_DEVICES][GGML_CUDA_AR_POOL_SIZE];

    // Copy-engine: per-device "I finished reading my peer's host_large"
    // event.  Indexed by RECORDER device.  Recorded same-device on streams[i]
    // after stage 2's last H2D from host_large[peer].  Waited cross-device
    // by peer's stage-1 stream before the next AR overwrites host_large[peer].
    cudaEvent_t              host_large_read_done[GGML_CUDA_MAX_DEVICES];
    bool                     host_large_read_done_valid;

    // Copy-engine: per-device "my add_kernel is done with dev_tmp" event.
    // Recorded on the compute stream after each add_kernel; the AR stream
    // waits on it before the next copy_impl's H2D overwrites dev_tmp.  Lets us
    // single-buffer dev_tmp despite add_kernel running on a separate stream.
    cudaEvent_t              dev_tmp_kernel_done[GGML_CUDA_MAX_DEVICES];
    bool                     dev_tmp_kernel_done_valid;

    // Arrival ring: ARRIVAL_STRIDE bytes between adjacent ints.  Mapped pinned
    // memory; CPU never reads/writes -- only the kernel and cudaMemset.
    // Use ggml_cuda_ar_arrival_ptr() to index.
    ggml_cuda_ar_host_mapping arrival;

    // GGML_CUDA_AR_P2P=1 (Pascal path, peer access available): each device owns an inbox
    // (staging ring + arrival ring) in its own memory.  A rank writes its partial and token
    // into the peer's inbox over PCIe, then spins on and reads from its own inbox.
    bool   p2p = false;
    char * p2p_buf[GGML_CUDA_MAX_DEVICES]     = {};
    char * p2p_arrival[GGML_CUDA_MAX_DEVICES] = {};
};

// Base pointer for the (slot, rank) per-block token block.  The kernel adds
// blockIdx.x * (ARRIVAL_STRIDE/sizeof(int)) internally to land on its own slot.
static int * ggml_cuda_ar_arrival_ptr(const ggml_cuda_ar_pipeline * p, int slot, int rank) {
    const size_t offset = ((size_t)slot * p->n_devices + rank) *
                          GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    return reinterpret_cast<int *>(p->arrival.dev + offset);
}

// P2P: the same (slot, rank) token block, inside device dev's own arrival ring.
static int * ggml_cuda_ar_p2p_arrival_ptr(const ggml_cuda_ar_pipeline * p, int dev, int slot, int rank) {
    const size_t offset = ((size_t)slot * p->n_devices + rank) *
                          GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    return reinterpret_cast<int *>(p->p2p_arrival[dev] + offset);
}

// GGML_CUDA_AR_PASCAL=1 only: process-global status lines shared by every
// Pascal pipeline (e.g. target + MTP draft contexts), so that
// ggml_backend_cuda_synchronize can check them without knowing the pipelines.
// Allocated once (1 KB mapped pinned), never freed.  g_ar_pascal_status_host
// stays nullptr while no Pascal pipeline exists, which makes the check a no-op.
static ggml_cuda_ar_host_mapping g_ar_pascal_status;
static std::atomic<const int *>  g_ar_pascal_status_host{nullptr};
static std::mutex                g_ar_pascal_status_mutex;

static int * ggml_cuda_ar_status_ptr(int rank) {
    return reinterpret_cast<int *>(g_ar_pascal_status.dev) + rank * GGML_CUDA_AR_STATUS_INTS;
}

static bool ggml_cuda_ar_pascal_status_init() {
    std::lock_guard<std::mutex> lock(g_ar_pascal_status_mutex);
    if (g_ar_pascal_status.host != nullptr) {
        return true;
    }
    const size_t bytes = (size_t) GGML_CUDA_MAX_DEVICES * GGML_CUDA_AR_STATUS_INTS * sizeof(int);
    if (g_ar_pascal_status.alloc(bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    memset(g_ar_pascal_status.host, 0, bytes);
    g_ar_pascal_status_host.store(reinterpret_cast<const int *>(g_ar_pascal_status.host), std::memory_order_release);
    return true;
}

void ggml_cuda_ar_pascal_check_status() {
    const int * st = g_ar_pascal_status_host.load(std::memory_order_acquire);
    if (st == nullptr) {
        return;
    }
    for (int r = 0; r < 2; ++r) {
        const volatile int * s = st + r * GGML_CUDA_AR_STATUS_INTS;
        if (s[0] != 0) {
            const int token = s[1], block = s[2], seen = s[3], ms = s[4], dev = s[5];
            GGML_LOG_ERROR("internal AllReduce (GGML_CUDA_AR_PASCAL=1): device %d (rank %d) gave up after %d ms "
                           "waiting for its peer at AllReduce #%d (block %d, peer token seen %d); this step's "
                           "tensor-parallel result is invalid, stopping instead of hanging the GPUs or tripping a TDR. "
                           "Unset GGML_CUDA_AR_PASCAL to use the meta-backend butterfly, or raise "
                           "GGML_CUDA_AR_SPIN_TIMEOUT_MS\n", dev, r, ms, token, block, seen);
            GGML_ABORT("internal AllReduce (GGML_CUDA_AR_PASCAL=1) spin timeout: device %d rank %d, %d ms, AR #%d, block %d",
                       dev, r, ms, token, block);
        }
    }
}

bool ggml_cuda_ar_pipeline_is_pascal(const ggml_cuda_ar_pipeline * p) {
    return p != nullptr && p->pascal;
}

static uint64_t ggml_cuda_ar_env_u64(const char * name, uint64_t default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const unsigned long long parsed = strtoull(value, &end, 10);
    return end != value ? (uint64_t) parsed : default_value;
}

struct ggml_cuda_ar_slot_info {
    int slot;
    int token;
};

static ggml_cuda_ar_slot_info ggml_cuda_ar_acquire_slot(ggml_cuda_ar_pipeline * p) {
    const int  slot        = static_cast<int>(p->call_count % GGML_CUDA_AR_POOL_SIZE);
    const bool pool_lapped = p->call_count >= GGML_CUDA_AR_POOL_SIZE;
    p->call_count++;

    if (pool_lapped) {
        for (int i = 0; i < p->n_devices; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventSynchronize(p->ev_pool[i][slot].ker));
        }
    }

    // Pascal path: AR N-2's kernels have finished on both devices (synced
    // above), so a timeout there is visible now.
    if (p->pascal) {
        ggml_cuda_ar_pascal_check_status();
    }

    return { slot, (int) p->call_count };
}

// Per-AR copy-engine chunk size: env-var override if set, else heuristic
// (clamp(nbytes/4, HEURISTIC_MIN, HEURISTIC_MAX)).
static size_t ggml_cuda_ar_chunk_bytes(const ggml_cuda_ar_pipeline * p, size_t nbytes) {
    if (p->copy_chunk_bytes > 0) {
        return p->copy_chunk_bytes;
    }
    return std::min(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX,
                    std::max(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN, nbytes / 4));
}

static void ggml_cuda_ar_wait_for_compute(
        ggml_cuda_ar_pipeline * p, ggml_backend_cuda_context * cuda_ctx, int rank, int slot) {
    ggml_cuda_ar_event_slot & ev = p->ev_pool[rank][slot];
    CUDA_CHECK(cudaEventRecord(ev.app, cuda_ctx->stream()));
    CUDA_CHECK(cudaStreamWaitEvent(p->streams[rank], ev.app));
}

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------

// GGML_CUDA_AR_PASCAL=1 only: device checks before any allocation.  On any
// failure warn and return false (the caller falls back to the butterfly).
// kexec_timeout[i] receives cudaDevAttrKernelExecTimeout for rank i.
static bool ggml_cuda_ar_pascal_precheck(const int * devices, size_t n_devices, int * kexec_timeout) {
    const ggml_cuda_device_info & info = ggml_cuda_info();

    // Two kernels spinning on each other must run on two physical GPUs.
    if (info.devices[devices[0]].physical_device == info.devices[devices[1]].physical_device) {
        GGML_LOG_WARN("%s: GGML_CUDA_AR_PASCAL=1: devices %d and %d share physical GPU %d; "
                      "the Pascal AllReduce needs two distinct GPUs; falling back\n",
                      __func__, devices[0], devices[1], info.devices[devices[0]].physical_device);
        return false;
    }

    for (size_t i = 0; i < n_devices; ++i) {
        const int phys = info.devices[devices[i]].physical_device;
        int can_map = 0;
        int uva     = 0;
        int kto     = 0;
        if (cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory,   phys) != cudaSuccess ||
            cudaDeviceGetAttribute(&uva,     cudaDevAttrUnifiedAddressing,  phys) != cudaSuccess ||
            cudaDeviceGetAttribute(&kto,     cudaDevAttrKernelExecTimeout,  phys) != cudaSuccess) {
            (void) cudaGetLastError();
            GGML_LOG_WARN("%s: GGML_CUDA_AR_PASCAL=1: cudaDeviceGetAttribute failed for device %d; falling back\n",
                          __func__, devices[i]);
            return false;
        }
        if (can_map != 1 || uva != 1) {
            GGML_LOG_WARN("%s: GGML_CUDA_AR_PASCAL=1: device %d lacks mapped host memory (%d) or unified "
                          "addressing (%d); falling back\n", __func__, devices[i], can_map, uva);
            return false;
        }
        kexec_timeout[i] = kto;

        // Force the (lazy) module load now, not between the two AR launches of
        // the first AllReduce, and prove an image for this arch exists.
        ggml_cuda_set_device(devices[i]);
        cudaFuncAttributes fa;
        const cudaError_t err = cudaFuncGetAttributes(&fa, ggml_cuda_ar_kernel_f32_pascal);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            GGML_LOG_WARN("%s: GGML_CUDA_AR_PASCAL=1: no usable ggml_cuda_ar_kernel_f32_pascal image for device %d "
                          "(%s); falling back\n", __func__, devices[i], cudaGetErrorString(err));
            return false;
        }
    }
    return true;
}

ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int * devices, size_t n_devices) {

    if (n_devices != 2) {
        GGML_LOG_DEBUG("%s: internal AllReduce only supports n_devices=2 (got %zu); "
                       "falling back\n", __func__, n_devices);
        return nullptr;
    }

    // GGML_CUDA_AR_PASCAL=1 lets a pair that includes a Pascal (sm_6x) device
    // use ggml_cuda_ar_kernel_f32_pascal.  Unset: unchanged behaviour.
    const char * pascal_env = getenv("GGML_CUDA_AR_PASCAL");
    const bool   pascal_req = pascal_env != nullptr && atoi(pascal_env) != 0;
    bool         pascal     = false;

    // The chunked kernel uses __nanosleep, which is sm70+ (Volta+).
    for (size_t i = 0; i < n_devices; ++i) {
        const int cc = ggml_cuda_info().devices[devices[i]].cc;
        if (cc < GGML_CUDA_CC_VOLTA) {
            if (pascal_req && cc >= GGML_CUDA_CC_PASCAL) {
                pascal = true;
                continue;
            }
            GGML_LOG_DEBUG("%s: internal AllReduce requires compute capability >= %d "
                           "(device %d has cc=%d); falling back\n",
                           __func__, GGML_CUDA_CC_VOLTA, devices[i], cc);
            return nullptr;
        }
    }

    int kexec_timeout[GGML_CUDA_MAX_DEVICES] = {};
    if (pascal && !ggml_cuda_ar_pascal_precheck(devices, n_devices, kexec_timeout)) {
        return nullptr;
    }

    auto * p = new ggml_cuda_ar_pipeline{};
    p->n_devices        = n_devices;
    p->copy_bytes       = GGML_CUDA_AR_COPY_MAX_BYTES;
    p->copy_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_THRESHOLD", GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT);
    // 0 = use the per-call heuristic (default).  Non-zero env value forces a
    // fixed chunk size for diagnostics, with a floor at COPY_CHUNK_BYTES_MIN.
    p->copy_chunk_bytes = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_CHUNK_BYTES", 0);
    if (p->copy_chunk_bytes > 0 && p->copy_chunk_bytes < GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN) {
        GGML_LOG_WARN("%s: GGML_CUDA_AR_COPY_CHUNK_BYTES=%zu below minimum %zu; clamping\n",
                      __func__, p->copy_chunk_bytes, GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);
        p->copy_chunk_bytes = GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN;
    }
    // Default 1: BF16 round-trip is always on for F32 inputs (any non-zero
    // ne).  Set GGML_CUDA_AR_BF16_THRESHOLD=0 to disable, or to a larger
    // byte threshold to opt out for small tensors.
    p->bf16_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", 1);
    p->q8_wire          = ggml_cuda_ar_env_u64("GGML_CUDA_AR_Q8", 0) != 0;
    for (size_t i = 0; i < n_devices; ++i) {
        p->devices[i] = devices[i];
    }

    p->pascal    = pascal;
    p->pascal_ce = !pascal || ggml_cuda_ar_env_u64("GGML_CUDA_AR_PASCAL_CE", 1) != 0;
    if (pascal && p->pascal_ce && ggml_cuda_pf_xchg_engaged(devices[0], devices[1])) {
        // pf-xchg (GGML_PF_XCHG=1, peer access both ways): copy-engine sized ARs (prefill) go to the meta backend's
        // butterfly, whose peer copies then run on the duplex copy stream (ggml-cuda.cu); small ARs (decode) stay on
        // this pipeline's kernel.  Same effect as GGML_CUDA_AR_PASCAL_CE=0, and the copy-engine staging (32 MiB
        // pinned + 32 MiB device per GPU) is not allocated.
        p->pascal_ce = false;
        GGML_LOG_INFO("%s: GGML_PF_XCHG=1: large AllReduces use the meta-backend butterfly over the duplex peer copy stream\n",
                      __func__);
    }
    if (pascal) {
        // Exact F32 wire only: the BF16 round-trip is lossy and sm_6x has no
        // native bf16.  This keeps results bit-identical to the butterfly.
        if (p->bf16_threshold != 0 && getenv("GGML_CUDA_AR_BF16_THRESHOLD") != nullptr) {
            GGML_LOG_WARN("%s: GGML_CUDA_AR_BF16_THRESHOLD ignored: the Pascal path is F32-only "
                          "(exact, bit-identical to the butterfly)\n", __func__);
        }
        p->bf16_threshold = 0;

        for (size_t i = 0; i < n_devices; ++i) {
            const bool     kto    = kexec_timeout[i] != 0;
            const uint64_t def_ms = kto ? GGML_CUDA_AR_SPIN_MS_WDDM     : GGML_CUDA_AR_SPIN_MS_TCC;
            const uint64_t max_ms = kto ? GGML_CUDA_AR_SPIN_MS_WDDM_MAX : GGML_CUDA_AR_SPIN_MS_TCC_MAX;
            const uint64_t req_ms = ggml_cuda_ar_env_u64("GGML_CUDA_AR_SPIN_TIMEOUT_MS", def_ms);
            const uint64_t ms     = std::clamp<uint64_t>(req_ms, GGML_CUDA_AR_SPIN_MS_MIN, max_ms);
            if (ms != req_ms) {
                GGML_LOG_WARN("%s: GGML_CUDA_AR_SPIN_TIMEOUT_MS=%llu out of range for device %d "
                              "(kernel exec timeout %s); using %llu ms\n", __func__,
                              (unsigned long long) req_ms, p->devices[i], kto ? "on" : "off",
                              (unsigned long long) ms);
            }
            p->spin_timeout_ns[i] = ms * 1000000ull;
            p->wddm[i]            = kto;
        }
        // Launch the no-watchdog (TCC) rank first: a host stall between the two
        // launches then costs the TCC rank's longer budget, not the WDDM one.
        if (p->wddm[0] && !p->wddm[1]) {
            p->launch_order[0] = 1;
            p->launch_order[1] = 0;
        }

        p->test_drop_token = ggml_cuda_ar_env_u64("GGML_CUDA_AR_PASCAL_TEST_DROP", 0);
        p->test_drop_rank  = (int) ggml_cuda_ar_env_u64("GGML_CUDA_AR_PASCAL_TEST_DROP_RANK", 1);
        if (p->test_drop_token != 0) {
            GGML_LOG_WARN("%s: GGML_CUDA_AR_PASCAL_TEST_DROP=%llu: fault injection armed, rank %d will skip "
                          "its arrival signal at that AllReduce and the peer must time out\n", __func__,
                          (unsigned long long) p->test_drop_token, p->test_drop_rank);
        }
    }

    // Per-device streams and event pools.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);

        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaStreamCreateWithFlags failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        p->streams[i] = stream;

        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            bool ok =
                cudaEventCreateWithFlags(&p->ev_pool[i][s].app, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->ev_pool[i][s].h2d, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->ev_pool[i][s].ker, cudaEventDisableTiming) == cudaSuccess;
            for (int c = 0; ok && c < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++c) {
                ok = cudaEventCreateWithFlags(&p->ev_pool[i][s].cpy[c], cudaEventDisableTiming) == cudaSuccess;
            }
            if (!ok) {
                GGML_LOG_ERROR("%s: cudaEventCreate failed for device %d slot %d\n",
                               __func__, p->devices[i], s);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
        }

        if (cudaEventCreateWithFlags(&p->host_large_read_done[i], cudaEventDisableTiming) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaEventCreate for host_large_read_done failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        if (cudaEventCreateWithFlags(&p->dev_tmp_kernel_done[i], cudaEventDisableTiming) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaEventCreate for dev_tmp_kernel_done failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // Arrival ring: cache-line padded so each GPU's int is on its own line.
    const size_t arrival_bytes =
        (size_t)GGML_CUDA_AR_POOL_SIZE * n_devices *
        GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    if (p->arrival.alloc(arrival_bytes) != cudaSuccess) {
        GGML_LOG_ERROR("%s: alloc for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }
    ggml_cuda_set_device(p->devices[0]);
    if (cudaMemset(p->arrival.dev, 0, arrival_bytes) != cudaSuccess) {
        GGML_LOG_ERROR("%s: cudaMemset for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }

    // Per-device pinned staging buffers -- POOL_SIZE-deep ring so the chunked-
    // kernel can write the next slot's data while the peer is still reading
    // the previous slot's. Indexed by (slot * buf_bytes) at the call site.
    p->buf_bytes = GGML_CUDA_AR_MAX_BYTES;
    const size_t host_buf_total = (size_t) GGML_CUDA_AR_POOL_SIZE * p->buf_bytes;
    for (size_t i = 0; i < n_devices; ++i) {
        if (p->host_buf[i].alloc(host_buf_total) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for staging failed (%zu bytes)\n",
                           __func__, host_buf_total);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    if (p->pascal) {
        if (!ggml_cuda_ar_pascal_status_init()) {
            GGML_LOG_WARN("%s: GGML_CUDA_AR_PASCAL=1: alloc for the status line failed; falling back\n", __func__);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        // The kernel on device i dereferences mapped pointers that were
        // obtained while another device was current; that is only valid when
        // every mapped device pointer is the (UVA) host pointer.
        bool uva_ok = p->arrival.dev == p->arrival.host &&
                      g_ar_pascal_status.dev == g_ar_pascal_status.host;
        for (size_t i = 0; i < n_devices; ++i) {
            uva_ok = uva_ok && p->host_buf[i].dev == p->host_buf[i].host;
        }
        if (!uva_ok) {
            GGML_LOG_WARN("%s: GGML_CUDA_AR_PASCAL=1: mapped device pointers differ from host pointers "
                          "(no UVA mapping); falling back\n", __func__);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }

        // GGML_CUDA_AR_P2P=1: inboxes in device memory, written over PCIe peer access.
        // Any failure keeps the host-staged exchange (p->p2p stays false).
        const char * p2p_env = getenv("GGML_CUDA_AR_P2P");
        if (p2p_env != nullptr && atoi(p2p_env) != 0 && n_devices == 2) {
            bool ok = true;
            for (int i = 0; i < 2 && ok; ++i) {
                int can = 0;
                ok = cudaDeviceCanAccessPeer(&can, p->devices[i], p->devices[1 - i]) == cudaSuccess && can;
                if (ok) {
                    ggml_cuda_set_device(p->devices[i]);
                    const cudaError_t e = cudaDeviceEnablePeerAccess(p->devices[1 - i], 0);
                    ok = e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled;
                }
            }
            for (int i = 0; i < 2 && ok; ++i) {
                ggml_cuda_set_device(p->devices[i]);
                ok = cudaMalloc(reinterpret_cast<void **>(&p->p2p_buf[i]), host_buf_total) == cudaSuccess &&
                     cudaMalloc(reinterpret_cast<void **>(&p->p2p_arrival[i]), arrival_bytes) == cudaSuccess &&
                     cudaMemset(p->p2p_arrival[i], 0, arrival_bytes) == cudaSuccess &&
                     cudaDeviceSynchronize() == cudaSuccess;
            }
            (void) cudaGetLastError();
            if (ok) {
                p->p2p = true;
                GGML_LOG_INFO("%s: GGML_CUDA_AR_P2P=1: Pascal AllReduce exchanges through peer device memory "
                              "(devices %d <-> %d)\n", __func__, p->devices[0], p->devices[1]);
            } else {
                GGML_LOG_WARN("%s: GGML_CUDA_AR_P2P=1: peer access or inbox allocation failed; "
                              "using the host-staged exchange\n", __func__);
            }
        }
    }

    // Copy-engine path: pinned host staging + device scratch, sized for the
    // largest tensor we accept on this path (GGML_CUDA_AR_COPY_MAX_BYTES).
    // dev_tmp is single-buffered; cross-AR safety is enforced by an explicit
    // cross-stream wait in copy_impl on the prior AR's add_kernel-done event.
    // pascal_ce is always true on the Volta+ path; GGML_CUDA_AR_PASCAL_CE=0
    // skips these buffers and sends large ARs to the butterfly instead.
    if (p->pascal_ce) {
        for (size_t i = 0; i < n_devices; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            if (p->host_large[i].alloc(p->copy_bytes) != cudaSuccess) {
                GGML_LOG_ERROR("%s: alloc for large staging failed (%zu bytes)\n",
                               __func__, p->copy_bytes);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
            if (cudaMalloc(reinterpret_cast<void **>(&p->dev_tmp[i]), p->copy_bytes) != cudaSuccess) {
                GGML_LOG_ERROR("%s: cudaMalloc for copy scratch failed (%zu bytes) on device %d\n",
                               __func__, p->copy_bytes, p->devices[i]);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
        }
    }

    if (p->pascal) {
        GGML_LOG_INFO("%s: GGML_CUDA_AR_PASCAL=1: internal AllReduce on device %d (cc %d) + device %d (cc %d): "
                      "F32 wire, 16-byte stripe, spin timeout %llu/%llu ms, copy-engine path %s\n",
                      __func__,
                      p->devices[0], ggml_cuda_info().devices[p->devices[0]].cc,
                      p->devices[1], ggml_cuda_info().devices[p->devices[1]].cc,
                      (unsigned long long) (p->spin_timeout_ns[0] / 1000000ull),
                      (unsigned long long) (p->spin_timeout_ns[1] / 1000000ull),
                      p->pascal_ce ? "on (>= threshold)" : "off (large ARs use the butterfly)");
    } else {
        GGML_LOG_INFO("%s: initialized AllReduce pipeline: %zu GPUs, "
                      "%zu KB chunked kernel staging + %zu MB copy-engine staging per GPU\n",
                      __func__, n_devices, p->buf_bytes >> 10, p->copy_bytes >> 20);
    }

    return p;
}

void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * p) {
    if (!p) {
        return;
    }

    // Drain all in-flight kernels before tearing down resources.
    for (int i = 0; i < p->n_devices; ++i) {
        if (p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaStreamSynchronize(p->streams[i]);
        }
    }

    // Pascal path: its AR kernels run on the backends' compute streams, not on
    // p->streams, so wait for the last ones (each bounded by the spin timeout)
    // before their mapped staging/arrival memory is freed.  ev.ker is recorded
    // after every AR; a never-recorded event completes immediately.
    if (p->pascal) {
        for (int i = 0; i < p->n_devices; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
                if (p->ev_pool[i][s].ker) {
                    (void) cudaEventSynchronize(p->ev_pool[i][s].ker);
                }
            }
        }
        (void) cudaGetLastError();
    }

    for (int i = 0; i < p->n_devices; ++i) {
        if (p->p2p_buf[i] || p->p2p_arrival[i]) {
            ggml_cuda_set_device(p->devices[i]);
            if (p->p2p_buf[i])     { cudaFree(p->p2p_buf[i]); }
            if (p->p2p_arrival[i]) { cudaFree(p->p2p_arrival[i]); }
        }
        p->host_buf[i].free();
        p->host_large[i].free();
        if (p->dev_tmp[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaFree(p->dev_tmp[i]);
        }
        ggml_cuda_set_device(p->devices[i]);
        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            if (p->ev_pool[i][s].app) { cudaEventDestroy(p->ev_pool[i][s].app); }
            for (int c = 0; c < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++c) {
                if (p->ev_pool[i][s].cpy[c]) { cudaEventDestroy(p->ev_pool[i][s].cpy[c]); }
            }
            if (p->ev_pool[i][s].h2d) { cudaEventDestroy(p->ev_pool[i][s].h2d); }
            if (p->ev_pool[i][s].ker) { cudaEventDestroy(p->ev_pool[i][s].ker); }
        }
        if (p->host_large_read_done[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaEventDestroy(p->host_large_read_done[i]);
        }
        if (p->dev_tmp_kernel_done[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaEventDestroy(p->dev_tmp_kernel_done[i]);
        }
        if (p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaStreamDestroy(p->streams[i]);
        }
    }
    p->arrival.free();
    delete p;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

// Asymmetric copy_impl: data sent over PCIe in T_src precision (one element of
// nbytes per ne element); accumulated locally into a T_dst buffer.  When
// T_src == T_dst this is the original homogeneous reduction.  When they differ
// (e.g. BF16 wire / F32 accumulator) the add kernel rounds dst through T_src
// for bit-equivalence between GPUs and we skip the otherwise-needed
// post-conversion entirely.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_impl(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne,
        size_t                  nbytes) {
    GGML_ASSERT(p->n_devices == 2);
    GGML_ASSERT(nbytes <= p->copy_bytes);
    GGML_ASSERT(ne <= std::numeric_limits<int>::max());

    const size_t chunk_bytes = ggml_cuda_ar_chunk_bytes(p, nbytes);
    GGML_ASSERT(chunk_bytes > 0);

    const int slot = ggml_cuda_ar_acquire_slot(p).slot;
    const size_t copy_chunks = (nbytes + chunk_bytes - 1) / chunk_bytes;
    GGML_ASSERT(copy_chunks <= GGML_CUDA_AR_COPY_MAX_CHUNKS);

    ggml_backend_cuda_context * cuda_ctx[2] = {};

    // Stage 1: both GPUs copy their local contribution to pinned host memory.
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        cuda_ctx[i] = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx[i]->device == p->devices[i]);

        ggml_cuda_ar_wait_for_compute(p, cuda_ctx[i], i, slot);

        // Wait for peer's H2D from our host_large[i] (recorded in the
        // previous AR's stage 2) to complete before we overwrite host_large[i].
        // host_large_read_done[peer] = peer finished reading host_large[i].
        // No-op on the first AR -- no prior record exists.
        if (p->host_large_read_done_valid) {
            const int peer = 1 - i;
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->host_large_read_done[peer]));
        }

        if (!compute[i]) {
            CUDA_CHECK(cudaMemsetAsync(src_buf[i], 0, nbytes, p->streams[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = (nbytes - offset) < chunk_bytes ?
                (nbytes - offset) : chunk_bytes;

            CUDA_CHECK(cudaMemcpyAsync(
                p->host_large[i].host + offset, reinterpret_cast<char *>(src_buf[i]) + offset, this_bytes,
                cudaMemcpyDeviceToHost, p->streams[i]));
            CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].cpy[c], p->streams[i]));
        }
    }

    // Stage 2: each GPU waits for each peer D2H chunk, pulls that chunk back to
    // local device scratch (dev_tmp), then performs one device-local add over
    // the assembled peer tensor.  The H2Ds run on the AR stream (copy engine)
    // and the add_kernel runs on the caller's compute stream, so the AR stream
    // stays pure-copy and avoids an in-stream copy->compute engine switch every
    // AR.  dev_tmp is single-buffered: the AR stream waits cross-stream on the
    // prior AR's add_kernel-done event before overwriting it.
    for (int i = 0; i < 2; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);

        // Wait for the previous AR's add_kernel (on the compute stream) to
        // finish reading dev_tmp before our H2D overwrites it.  No-op on the
        // first copy_impl call.
        if (p->dev_tmp_kernel_done_valid) {
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->dev_tmp_kernel_done[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = (nbytes - offset) < chunk_bytes ?
                (nbytes - offset) : chunk_bytes;

            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->ev_pool[peer][slot].cpy[c]));
            CUDA_CHECK(cudaMemcpyAsync(
                p->dev_tmp[i] + offset, p->host_large[peer].host + offset, this_bytes,
                cudaMemcpyHostToDevice, p->streams[i]));
        }

        // Mark our reads of host_large[peer] complete so peer's next AR can
        // safely overwrite it.
        CUDA_CHECK(cudaEventRecord(p->host_large_read_done[i], p->streams[i]));

        // Hand off from AR stream (copy engine) to compute stream: compute
        // stream waits for all H2Ds to finish, then runs the add_kernel.
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].h2d, p->streams[i]));
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx[i]->stream(), p->ev_pool[i][slot].h2d));

        ggml_cuda_ar_launch_add<T_dst, T_src>(
            dst_buf[i], reinterpret_cast<const T_src *>(p->dev_tmp[i]), src_buf[i], ne, cuda_ctx[i]->stream());
        CUDA_CHECK(cudaGetLastError());

        // Record dev_tmp-released on the compute stream so the next copy_impl
        // can wait for the kernel to finish before overwriting dev_tmp.  Also
        // record AR-done as ev.ker for acquire_slot's pool-wraparound sync.
        CUDA_CHECK(cudaEventRecord(p->dev_tmp_kernel_done[i], cuda_ctx[i]->stream()));
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, cuda_ctx[i]->stream()));
    }
    p->host_large_read_done_valid = true;
    p->dev_tmp_kernel_done_valid = true;

    return true;
}

// Outer-level chunker: copy_impl handles up to copy_bytes per call (limited by
// the host_large / dev_tmp allocation size).  When the full AR exceeds that,
// slice the tensor into copy_bytes-sized pieces and call copy_impl repeatedly.
// Each slice goes through its own stage 1 -> stage 2 cycle and acquires its own
// slot, so cross-AR fences and pool wraparound work the same way as for any
// other sequence of small ARs.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_outer(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne) {
    const int64_t outer_max_elems = (int64_t) (p->copy_bytes / sizeof(T_src));
    GGML_ASSERT(outer_max_elems > 0);

    bool ok = true;
    for (int64_t outer_start = 0; outer_start < ne && ok; outer_start += outer_max_elems) {
        const int64_t outer_ne     = std::min(outer_max_elems, ne - outer_start);
        const size_t  outer_nbytes = (size_t) outer_ne * sizeof(T_src);

        T_src * src[GGML_CUDA_MAX_DEVICES] = {};
        T_dst * dst[GGML_CUDA_MAX_DEVICES] = {};
        for (int i = 0; i < p->n_devices; ++i) {
            src[i] = src_buf[i] + outer_start;
            dst[i] = dst_buf[i] + outer_start*ggml_cuda_ar_elems<T_src>::n;
        }
        ok = ggml_cuda_ar_allreduce_copy_impl<T_src, T_dst>(
            p, backends, src, dst, compute, outer_ne, outer_nbytes);
    }
    return ok;
}

bool ggml_cuda_ar_pipeline_copy_engine_for(const ggml_cuda_ar_pipeline * p, const size_t nbytes) {
    return p != nullptr && p->copy_threshold > 0 && nbytes >= p->copy_threshold && (!p->pascal || p->pascal_ce);
}

bool ggml_cuda_ar_allreduce(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors) {
    return ggml_cuda_ar_allreduce_ex(p, backends, tensors, /*f16_wire =*/ false);
}

bool ggml_cuda_ar_allreduce_ex(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors,
        const bool              f16_wire) {
    GGML_ASSERT(p != nullptr);

    const int n = p->n_devices;
    GGML_ASSERT(n == 2);

    const ggml_type input_type = tensors[0]->type;
    GGML_ASSERT(input_type == GGML_TYPE_F32 || input_type == GGML_TYPE_F16 || input_type == GGML_TYPE_BF16);

    // Pascal path is F32-only.  Nothing is enqueued yet, so returning false
    // lets the meta backend run its butterfly for this call.
    if (p->pascal && input_type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t ne = ggml_nelements(tensors[0]);
    GGML_ASSERT(ne > 0);

    const size_t   input_nbytes = ggml_nbytes(tensors[0]);

    // BF16 round-trip: F32 inputs >= bf16_threshold are converted to BF16 for
    // the reduction (chunked or copy-engine), halving on-wire bytes. Matches
    // NCCL's behaviour. The pre-conversion zeroes inactive shards so the
    // inner paths see them as already-prepared compute tensors.
    const bool use_bf16 =
        input_type == GGML_TYPE_F32 &&
        p->bf16_threshold > 0 &&
        input_nbytes >= p->bf16_threshold;

    const ggml_type kernel_type = use_bf16 ? GGML_TYPE_BF16 : input_type;
    const size_t    type_size   = ggml_type_size(kernel_type);
    GGML_ASSERT(p->buf_bytes >= type_size);
    const size_t    nbytes      = (size_t) ne * type_size;

    bool compute_flag[GGML_CUDA_MAX_DEVICES] = {};
    for (int i = 0; i < n; ++i) {
        compute_flag[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    }

    // Decide between copy-engine and chunked kernel paths based on the working
    // type's actual byte count.  No upper bound: copy_outer slices reductions
    // larger than copy_bytes into copy_bytes-sized pieces.
    const bool use_copy_engine =
        p->copy_threshold > 0 &&
        nbytes >= p->copy_threshold;

    // GGML_CUDA_AR_PASCAL_CE=0: no copy-engine buffers were allocated; hand
    // large ARs to the butterfly (still before any side effect).
    if (p->pascal && use_copy_engine && !p->pascal_ce) {
        return false;
    }

    if (p->pascal) {
        // One chunk per AR on the Pascal chunked path: between chunks nothing
        // on the host gates ring-slot reuse, and a timed-out side could run
        // ahead into a slot its late peer is still reading.  Only reachable
        // with a non-default GGML_CUDA_AR_COPY_THRESHOLD.
        if (!use_copy_engine && nbytes > p->buf_bytes) {
            return false;
        }
        // Never inside a CUDA graph capture: the host-side token would be baked
        // into every replay and the spin would pass on a stale flag.
        for (int i = 0; i < n; ++i) {
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
            CUDA_CHECK(cudaStreamIsCapturing(cuda_ctx->stream(), &cs));
            if (cs != cudaStreamCaptureStatusNone) {
                return false;
            }
        }
    }

    // BF16 inactive-shard zeroing: when use_bf16 is on, the combined kernel
    // (chunked kernel path) and the combined add kernel (copy_engine path)
    // both accumulate into the F32 tensor data directly, so an inactive
    // shard's accumulator must start at zero.
    if (use_bf16) {
        for (int i = 0; i < n; ++i) {
            if (!compute_flag[i]) {
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                ggml_cuda_set_device(p->devices[i]);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, (size_t) ne * sizeof(float), cuda_ctx->stream()));
            }
        }
    }

    // Pre-convert F32 -> BF16 into bf16_tmp ONLY for the copy_engine + use_bf16
    // path; the chunked kernel path's combined kernel does the conversion
    // inline as it writes to host_buf.
    ggml_cuda_pool_alloc<nv_bfloat16> bf16_tmp[GGML_CUDA_MAX_DEVICES];
    void * copy_src_ptr[GGML_CUDA_MAX_DEVICES] = {};

    if (use_copy_engine && use_bf16) {
        to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
        for (int i = 0; i < n; ++i) {
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            bf16_tmp[i].pool = &cuda_ctx->pool();
            bf16_tmp[i].alloc(ne);
            ggml_cuda_set_device(p->devices[i]);
            if (compute_flag[i]) {
                to_bf16(tensors[i]->data, bf16_tmp[i].get(), ne, cuda_ctx->stream());
                CUDA_CHECK(cudaGetLastError());
            } else {
                CUDA_CHECK(cudaMemsetAsync(bf16_tmp[i].get(), 0, nbytes, cuda_ctx->stream()));
            }
            copy_src_ptr[i] = bf16_tmp[i].get();
        }
    }

    // 8-bit wire (GGML_CUDA_AR_Q8=1): copy-engine ARs only (prefill-sized; decode ARs stay exact on the
    // chunked path).  Quantize each GPU's contribution (zeros for an inactive shard) into a pool buffer,
    // zero inactive accumulators, then reduce with deq(own) + deq(peer) on both sides.
    if (use_copy_engine && p->q8_wire && input_type == GGML_TYPE_F32 && ne % 32 == 0) {
        const int64_t nblk = ne/32;
        ggml_cuda_pool_alloc<ggml_cuda_ar_q8_32> q8_tmp[GGML_CUDA_MAX_DEVICES];
        ggml_cuda_ar_q8_32 * src[GGML_CUDA_MAX_DEVICES] = {};
        float              * dst[GGML_CUDA_MAX_DEVICES] = {};
        bool inner_compute[GGML_CUDA_MAX_DEVICES];
        for (int i = 0; i < n; ++i) {
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            ggml_cuda_set_device(p->devices[i]);
            q8_tmp[i].pool = &cuda_ctx->pool();
            q8_tmp[i].alloc(nblk);
            if (compute_flag[i]) {
                const int warps = 8;
                ggml_cuda_ar_q8_quantize<<<(unsigned) ((nblk + warps - 1)/warps), warps*WARP_SIZE, 0, cuda_ctx->stream()>>>(
                    static_cast<const float *>(tensors[i]->data), q8_tmp[i].get(), (int) nblk);
                CUDA_CHECK(cudaGetLastError());
            } else {
                CUDA_CHECK(cudaMemsetAsync(q8_tmp[i].get(), 0, (size_t) nblk*sizeof(ggml_cuda_ar_q8_32), cuda_ctx->stream()));
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, (size_t) ne*sizeof(float), cuda_ctx->stream()));
            }
            src[i] = q8_tmp[i].get();
            dst[i] = static_cast<float *>(tensors[i]->data);
            inner_compute[i] = true;
        }
        return ggml_cuda_ar_allreduce_copy_outer<ggml_cuda_ar_q8_32, float>(p, backends, src, dst, inner_compute, nblk);
    }

    // pf-xchg, GGML_PF_AR_F16=1 (f16 wire of Kmic-68/llama.cpp e5c264b71, MIT, moved to this host-staged path): the
    // caller found both partials f16-exact (cuBLAS COMPUTE_16F output on sm_6x), so each GPU narrows its partial to
    // f16 (zeros for an inactive shard), the copy engines move half the bytes through host_large/dev_tmp, and the add
    // kernel computes (float)(half) own + (float) peer = own + peer exactly -- the same bits as the F32 wire, on both
    // GPUs.  The copy-engine / kernel split is decided on the F32 size as before, so only the wire changes.
    // GGML_CUDA_AR_Q8=1 keeps precedence (handled above) when both are set.
    if (f16_wire && use_copy_engine && input_type == GGML_TYPE_F32 && !use_bf16) {
        ggml_cuda_pool_alloc<half> f16_tmp[GGML_CUDA_MAX_DEVICES];
        half  * src[GGML_CUDA_MAX_DEVICES] = {};
        float * dst[GGML_CUDA_MAX_DEVICES] = {};
        bool inner_compute[GGML_CUDA_MAX_DEVICES];
        for (int i = 0; i < n; ++i) {
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            ggml_cuda_set_device(p->devices[i]);
            f16_tmp[i].pool = &cuda_ctx->pool();
            f16_tmp[i].alloc(ne);
            if (compute_flag[i]) {
                ggml_cuda_pf_f32_to_f16(static_cast<const float *>(tensors[i]->data), f16_tmp[i].get(), ne, nullptr, cuda_ctx->stream());
                CUDA_CHECK(cudaGetLastError());
            } else {
                // inactive shard: contributes zeros, and its accumulator must start at zero (as on the BF16 path)
                CUDA_CHECK(cudaMemsetAsync(f16_tmp[i].get(), 0, (size_t) ne*sizeof(half), cuda_ctx->stream()));
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, (size_t) ne*sizeof(float), cuda_ctx->stream()));
            }
            src[i] = f16_tmp[i].get();
            dst[i] = static_cast<float *>(tensors[i]->data);
            inner_compute[i] = true;
        }
        return ggml_cuda_ar_allreduce_copy_outer<half, float>(p, backends, src, dst, inner_compute, ne);
    }

    bool ok = true;
    if (use_copy_engine) {
        // After up-front BF16 conversion, the tmp buffers already hold the
        // (possibly zeroed-for-inactive) data, so the inner path can treat
        // every shard as compute.
        bool inner_compute[GGML_CUDA_MAX_DEVICES];
        for (int i = 0; i < n; ++i) {
            inner_compute[i] = use_bf16 ? true : compute_flag[i];
        }

        // Dispatch into copy_impl with explicit src/dst types.  When use_bf16
        // is on, the wire type is BF16 (src = bf16_tmp) and the accumulator
        // is F32 (dst = tensors[i]->data); the combined add kernel rounds dst
        // through BF16 for bit-equivalence and writes F32 directly, so no
        // post-conversion is needed.  Otherwise src == dst (same native type).
        if (use_bf16) {
            GGML_ASSERT(kernel_type == GGML_TYPE_BF16);
            nv_bfloat16 * src[GGML_CUDA_MAX_DEVICES] = {};
            float       * dst[GGML_CUDA_MAX_DEVICES] = {};
            for (int i = 0; i < n; ++i) {
                src[i] = static_cast<nv_bfloat16 *>(copy_src_ptr[i]);
                dst[i] = static_cast<float *>(tensors[i]->data);
            }
            ok = ggml_cuda_ar_allreduce_copy_outer<nv_bfloat16, float>(
                p, backends, src, dst, inner_compute, ne);
        } else {
            switch (kernel_type) {
                case GGML_TYPE_F32: {
                    float * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<float *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<float, float>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                case GGML_TYPE_BF16: {
                    nv_bfloat16 * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<nv_bfloat16 *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<nv_bfloat16, nv_bfloat16>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                case GGML_TYPE_F16: {
                    half * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<half *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<half, half>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                default:
                    GGML_ASSERT(false);
            }
        }
    } else {
        // host_buf carries T_wire-typed data; max_chunk_elems is the count that
        // fits in one host_buf at the wire size.
        const size_t max_chunk_elems = p->buf_bytes / type_size;
        const size_t input_type_size = ggml_type_size(input_type);

        // Chunked kernel path runs entirely on the caller's compute stream:
        // since AR is a barrier here, same-stream ordering subsumes any
        // cross-stream event handshake that the copy-engine path needs, and
        // skips the cross-stream scheduling overhead that was hurting the
        // small-tensor (tg) latency on the AR-stream variant.  Only ev.ker is
        // still recorded at end-of-AR for acquire_slot's pool-wraparound check.
        for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += (int64_t) max_chunk_elems) {
            const size_t remaining_elems = (size_t) (ne - chunk_start);
            const size_t chunk_elems = remaining_elems < max_chunk_elems ? remaining_elems : max_chunk_elems;
            const size_t chunk_dst_bytes  = chunk_elems * input_type_size;

            const auto [slot, token] = ggml_cuda_ar_acquire_slot(p);
            const bool last_chunk = chunk_start + (int64_t) chunk_elems == ne;

            for (int k = 0; k < n; ++k) {
                const int i    = p->pascal ? p->launch_order[k] : k;
                const int peer = 1 - i;  // valid for n == 2 only
                ggml_cuda_set_device(p->devices[i]);
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                cudaStream_t stream = cuda_ctx->stream();

                char * data = static_cast<char *>(tensors[i]->data) + chunk_start * (int64_t) input_type_size;

                // Match NCCL/meta-backend semantics: inactive shards contribute
                // zeros.  On the BF16 path the F32 tensor data was already
                // zeroed up-front (above), so per-chunk zeroing isn't needed.
                if (!compute_flag[i] && !use_bf16) {
                    CUDA_CHECK(cudaMemsetAsync(data, 0, chunk_dst_bytes, stream));
                }

                if (p->pascal) {
                    // F32 only (other types returned false above); same grid
                    // and block size on both devices -> identical stripe and
                    // tail ownership.  No blocking CUDA call may sit between
                    // the i=0 and i=1 launches.
                    GGML_ASSERT(input_type == GGML_TYPE_F32 && !use_bf16);
                    const int skip = p->test_drop_token != 0 &&
                                     (uint64_t) token == p->test_drop_token &&
                                     i == p->test_drop_rank;
                    // P2P: write our partial and token into the peer's inbox, read the peer's from ours.
                    float * stage_out = p->p2p
                        ? reinterpret_cast<float *>(p->p2p_buf[peer] + (size_t) slot * p->buf_bytes)
                        : reinterpret_cast<float *>(p->host_buf[i].dev + (size_t) slot * p->buf_bytes);
                    const float * stage_in = p->p2p
                        ? reinterpret_cast<const float *>(p->p2p_buf[i] + (size_t) slot * p->buf_bytes)
                        : reinterpret_cast<const float *>(p->host_buf[peer].dev + (size_t) slot * p->buf_bytes);
                    int * arrive_out = p->p2p ? ggml_cuda_ar_p2p_arrival_ptr(p, peer, slot, i)
                                              : ggml_cuda_ar_arrival_ptr(p, slot, i);
                    const int * arrive_in = p->p2p ? ggml_cuda_ar_p2p_arrival_ptr(p, i, slot, peer)
                                                   : ggml_cuda_ar_arrival_ptr(p, slot, peer);
                    ggml_cuda_ar_kernel_f32_pascal<<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>(
                        reinterpret_cast<float *>(data),
                        stage_out,
                        stage_in,
                        static_cast<int>(chunk_elems),
                        arrive_out,
                        arrive_in,
                        token,
                        (unsigned long long) p->spin_timeout_ns[i],
                        ggml_cuda_ar_status_ptr(i),
                        ggml_cuda_ar_status_ptr(peer),
                        p->devices[i],
                        skip);
                } else {
#define LAUNCH_AR_KERNEL(T_dst, T_wire) \
                ggml_cuda_ar_kernel<T_dst, T_wire><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>( \
                    reinterpret_cast<const T_dst *>(data), \
                    reinterpret_cast<T_dst *>(data), \
                    reinterpret_cast<T_wire *>(p->host_buf[i].dev + (size_t) slot * p->buf_bytes), \
                    reinterpret_cast<const T_wire *>(p->host_buf[peer].dev + (size_t) slot * p->buf_bytes), \
                    static_cast<int>(chunk_elems), \
                    ggml_cuda_ar_arrival_ptr(p, slot, i), \
                    ggml_cuda_ar_arrival_ptr(p, slot, peer), \
                    token)

                if (use_bf16) {
                    GGML_ASSERT(input_type == GGML_TYPE_F32);
                    LAUNCH_AR_KERNEL(float, nv_bfloat16);
                } else {
                    switch (input_type) {
                        case GGML_TYPE_F32:  LAUNCH_AR_KERNEL(float,       float);       break;
                        case GGML_TYPE_F16:  LAUNCH_AR_KERNEL(half,        half);        break;
                        case GGML_TYPE_BF16: LAUNCH_AR_KERNEL(nv_bfloat16, nv_bfloat16); break;
                        default: GGML_ASSERT(false);
                    }
                }

#undef LAUNCH_AR_KERNEL
                }
                CUDA_CHECK(cudaGetLastError());

                if (last_chunk) {
                    CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, stream));
                }

                if (p->pascal && p->wddm[i]) {
                    // WDDM may keep this launch in the driver's software queue
                    // until something flushes it.  The peer's kernel waits on a
                    // memory flag the driver cannot see, so submit it now; a
                    // stream query flushes the queue without blocking.
                    const cudaError_t q = cudaStreamQuery(stream);
                    if (q == cudaErrorNotReady) {
                        (void) cudaGetLastError();
                    } else {
                        CUDA_CHECK(q);
                    }
                }
            }
        }
    }

    return ok;
}

#else // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

// HIP and MUSA lack the host-mapped pinned-memory APIs (cudaHostAllocPortable
// / cudaHostAllocMapped / cudaHostGetDevicePointer) and __nanosleep that this
// implementation relies on, so the internal AllReduce is a CUDA-only feature.
// The dispatcher in ggml-cuda.cu treats a nullptr pipeline as "init failed"
// and silently falls back to the meta backend's generic AllReduce.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int *, size_t) {
    return nullptr;
}
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline *) {
}
bool ggml_cuda_ar_allreduce(ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_tensor **) {
    return false;
}
bool ggml_cuda_ar_allreduce_ex(ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_tensor **, bool) {
    return false;
}
bool ggml_cuda_ar_pipeline_copy_engine_for(const ggml_cuda_ar_pipeline *, size_t) {
    return false;
}
void ggml_cuda_ar_pascal_check_status() {
}
bool ggml_cuda_ar_pipeline_is_pascal(const ggml_cuda_ar_pipeline *) {
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
