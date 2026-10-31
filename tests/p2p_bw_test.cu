// p100e-p2p-bw-test [MB] [reps]: GPU0 <-> GPU1 transfer options for the tensor-parallel exchange.
// Measures (wall time per transfer of MB megabytes, event timed):
//   p2p 0->1, p2p 1->0, p2p both directions at once (cudaMemcpyPeerAsync, what tp_model uses),
//   host-staged both directions (D2H to pinned host, then H2D on the other GPU, chunked + pipelined),
//   kernel P2P read (each GPU pulls the peer's buffer with loads), kernel P2P write (each pushes with stores),
//   D2H alone / H2D alone (both GPUs at once).
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

__global__ void copy_k(const float4 * __restrict__ src, float4 * __restrict__ dst, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) dst[i] = src[i];
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const double mb = argc > 1 ? atof(argv[1]) : 10.0;
    const int reps = argc > 2 ? atoi(argv[2]) : 20;
    const size_t bytes = (size_t) (mb * 1048576.0) / 16 * 16;
    int n = 0;
    CK(cudaGetDeviceCount(&n));
    if (n < 2) { printf("need 2 GPUs\n"); return 1; }
    float * src[2], * dst[2];
    cudaStream_t s[2], s2[2];
    cudaEvent_t e0[2], e1[2];
    for (int d = 0; d < 2; ++d) {
        CK(cudaSetDevice(d));
        int can = 0;
        CK(cudaDeviceCanAccessPeer(&can, d, 1 - d));
        if (can) { cudaError_t e = cudaDeviceEnablePeerAccess(1 - d, 0); if (e != cudaErrorPeerAccessAlreadyEnabled) CK(e); }
        printf("GPU %d can access peer: %d\n", d, can);
        CK(cudaMalloc(&src[d], bytes)); CK(cudaMalloc(&dst[d], bytes));
        CK(cudaMemset(src[d], 1, bytes)); CK(cudaMemset(dst[d], 0, bytes));
        CK(cudaStreamCreateWithFlags(&s[d], cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&s2[d], cudaStreamNonBlocking));
        CK(cudaEventCreate(&e0[d])); CK(cudaEventCreate(&e1[d]));
    }
    char * host[2];
    for (int d = 0; d < 2; ++d) CK(cudaMallocHost(&host[d], bytes));
    const size_t CH = 1 << 20;   // host-staging chunk
    std::vector<cudaEvent_t> hev[2];
    for (int d = 0; d < 2; ++d) {
        CK(cudaSetDevice(d));
        hev[d].resize((bytes + CH - 1) / CH);
        for (auto & e : hev[d]) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    }
    auto sync = [&]() { for (int d = 0; d < 2; ++d) { CK(cudaSetDevice(d)); CK(cudaDeviceSynchronize()); } };
    auto run = [&](const char * name, auto && body) {
        body(); sync();   // warm-up
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) body();
        sync();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        printf("%-34s %9.1f us  %6.2f GB/s per direction\n", name, us, bytes / us / 1e3);
    };
    run("p2p 0->1", [&]() { CK(cudaSetDevice(0)); CK(cudaMemcpyPeerAsync(dst[1], 1, src[0], 0, bytes, s[0])); });
    run("p2p 1->0", [&]() { CK(cudaSetDevice(1)); CK(cudaMemcpyPeerAsync(dst[0], 0, src[1], 1, bytes, s[1])); });
    run("p2p both (each pushes)", [&]() {
        for (int d = 0; d < 2; ++d) { CK(cudaSetDevice(d)); CK(cudaMemcpyPeerAsync(dst[1 - d], 1 - d, src[d], d, bytes, s[d])); }
    });
    run("p2p both (each pulls)", [&]() {
        for (int d = 0; d < 2; ++d) { CK(cudaSetDevice(d)); CK(cudaMemcpyPeerAsync(dst[d], d, src[1 - d], 1 - d, bytes, s[d])); }
    });
    run("p2p both, 4 chunks each", [&]() {
        for (int c = 0; c < 4; ++c)
            for (int d = 0; d < 2; ++d) {
                CK(cudaSetDevice(d));
                const size_t o = bytes / 4 * c;
                CK(cudaMemcpyPeerAsync((char *) dst[1 - d] + o, 1 - d, (char *) src[d] + o, d, bytes / 4, s[d]));
            }
    });
    run("D2H both GPUs", [&]() { for (int d = 0; d < 2; ++d) { CK(cudaSetDevice(d)); CK(cudaMemcpyAsync(host[d], src[d], bytes, cudaMemcpyDeviceToHost, s[d])); } });
    run("H2D both GPUs", [&]() { for (int d = 0; d < 2; ++d) { CK(cudaSetDevice(d)); CK(cudaMemcpyAsync(dst[d], host[1 - d], bytes, cudaMemcpyHostToDevice, s[d])); } });
    run("D2H+H2D same time both GPUs", [&]() {
        for (int d = 0; d < 2; ++d) {
            CK(cudaSetDevice(d));
            CK(cudaMemcpyAsync(host[d], src[d], bytes, cudaMemcpyDeviceToHost, s[d]));
            CK(cudaMemcpyAsync(dst[d], host[1 - d], bytes, cudaMemcpyHostToDevice, s2[d]));
        }
    });
    run("host-staged both (1MB pipelined)", [&]() {
        // GPU d: D2H chunks on s[d] with an event per chunk; GPU 1-d: H2D chunk after that event on s2[1-d]
        const int nc = (int) hev[0].size();
        for (int c = 0; c < nc; ++c)
            for (int d = 0; d < 2; ++d) {
                CK(cudaSetDevice(d));
                const size_t o = (size_t) c * CH, l = std::min(CH, bytes - o);
                CK(cudaMemcpyAsync(host[d] + o, (char *) src[d] + o, l, cudaMemcpyDeviceToHost, s[d]));
                CK(cudaEventRecord(hev[d][c], s[d]));
            }
        for (int c = 0; c < nc; ++c)
            for (int d = 0; d < 2; ++d) {
                const int o_ = 1 - d;
                CK(cudaSetDevice(o_));
                const size_t o = (size_t) c * CH, l = std::min(CH, bytes - o);
                CK(cudaStreamWaitEvent(s2[o_], hev[d][c], 0));
                CK(cudaMemcpyAsync((char *) dst[o_] + o, host[d] + o, l, cudaMemcpyHostToDevice, s2[o_]));
            }
    });
    for (int blocks : {28, 56, 112}) {
        char nm[64];
        snprintf(nm, sizeof nm, "kernel pull both (%d blocks)", blocks);
        run(nm, [&]() {
            for (int d = 0; d < 2; ++d) {
                CK(cudaSetDevice(d));
                copy_k<<<blocks, 256, 0, s[d]>>>((const float4 *) src[1 - d], (float4 *) dst[d], bytes / 16);
            }
        });
        snprintf(nm, sizeof nm, "kernel push both (%d blocks)", blocks);
        run(nm, [&]() {
            for (int d = 0; d < 2; ++d) {
                CK(cudaSetDevice(d));
                copy_k<<<blocks, 256, 0, s[d]>>>((const float4 *) src[d], (float4 *) dst[1 - d], bytes / 16);
            }
        });
    }
    run("kernel push 0->1 only (56 blocks)", [&]() {
        CK(cudaSetDevice(0));
        copy_k<<<56, 256, 0, s[0]>>>((const float4 *) src[0], (float4 *) dst[1], bytes / 16);
    });
    printf("done\n");
    return 0;
}
