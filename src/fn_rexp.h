// Remote expert tier for fn_model: node 0 (for example 2 x P100) runs the whole
// Flash-Next model as before; node 1 (for example RTX 3070 + GTX 1660) holds a set of Q4 routed experts per
// layer that node 0 does not have resident and computes them on request. One round trip per layer per forward: node 0
// routes, sends the FFN input rows that need node-1 experts + (expert, rows) jobs in ONE message, runs its own experts
// meanwhile, and adds node 1's outputs into the same combine as its own rows (fn_model.cu, rexp_*).
//
// Wire format: plain TCP (TCP_NODELAY, 4 MiB socket buffers, keepalive ~1 min), one connection, node 0 listens and
// node 1 connects. With P100E_NET_KEY_FILE set the key handshake of net_auth.h runs first (node 0 drops a peer that
// fails it and keeps listening).
// Every message = rexp_hdr + payload (little endian, both ends are x86-64). Setup:
//   node 1 -> HELLO  (rexp_hello: model identity, GPUs and their free VRAM)
//   node 0 -> SERVE  (rexp_serve + int32 file layer per served layer)
//   node 1 -> CAPS   a = n GPUs; int32 slots per layer per GPU (after its byte plan), then int32 fixed flag per GPU
//                    (1 = a fixed set loaded once, never swapped: a GPU on a narrow bus, e.g. the 1660 at x4)
//   node 0 -> SET    per served layer, per GPU: int32 n, then n expert ids (node 0 gives the fixed GPUs the best-ranked
//                    experts of its list, the swappable ones the rest)
//   node 1 -> READY  (a = 0 ok; after every expert is loaded and packed)
// Per layer:
//   node 0 -> FFN    a = served layer index, b = nx token rows, c = jobs, d = pairs; payload f32 x[nx][E],
//                    int32 jobs[c][2] = (expert, rows), int32 xrow[d] (index into x of each pair, job by job),
//                    f32 w[d] (router weight of each pair; used by the sum mode only)
//   node 1 -> FFN_R  a = mode; payload f32 [d][E] (exact: each pair's unweighted expert output, the rows node 0's own
//                    grouped path would have produced) or [nx][E] (sum mode: per row sum_k w_k * y_k over its pairs),
//                    then c int32 pairs (served layer, expert) of swaps that landed since the last reply (see SWAP)
// Between forwards (node 0's adapt step, adaptive files only):
//   node 0 -> SWAP   a = n; payload int32 [n][3] = (served layer, expert to drop, expert to load). Node 0 stops sending
//                    the dropped expert at once; node 1 reads the new one from its file on a loader thread into the
//                    dropped one's slot and lists it in the next FFN_R once its copy + pack are queued ahead of any later
//                    request on that GPU, so node 0 only ever asks for experts node 1 has. No reply, no extra round trip.
//   node 0 -> QUIT   (node 1 frees and waits for the next connection, or exits with --once)
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace p100e {

constexpr uint32_t REXP_MAGIC = 0x50584552u;   // "REXP"
constexpr uint32_t REXP_PROTO = 1;
constexpr int REXP_MAX_GPU = 4;

enum rexp_op : uint32_t { REXP_HELLO = 1, REXP_SERVE = 2, REXP_CAPS = 3, REXP_SET = 4, REXP_READY = 5, REXP_FFN = 6, REXP_FFN_R = 7, REXP_QUIT = 8, REXP_SWAP = 9 };

struct rexp_hdr {
    uint32_t magic = REXP_MAGIC, op = 0;
    uint64_t seq = 0;
    int32_t a = 0, b = 0, c = 0, d = 0;
    uint64_t bytes = 0;   // payload bytes after the header
};
static_assert(sizeof(rexp_hdr) == 40, "rexp_hdr layout");

struct rexp_hello {
    uint32_t proto = REXP_PROTO;
    int32_t n_gpu = 0;
    uint64_t file_bytes = 0, table_hash = 0;   // model identity: file size + gguf_file::tensor_table_hash
    int32_t E = 0, FFE = 0, NEXP = 0, n_blk = 0;
    uint64_t gpu_free[REXP_MAX_GPU] = {};       // bytes free on each GPU before node 1 allocates anything
    char gpu_name[REXP_MAX_GPU][64] = {};
};

struct rexp_serve {
    int32_t n_layers = 0;    // served layers (followed by int32 file layer index per layer)
    int32_t max_x = 0;       // most token rows per FFN request
    int32_t max_pairs = 0;   // most (row, expert) pairs per FFN request
    int32_t fold_min = 16, fold_max = 48;   // node 0's P100E_FN_FOLD_MIN / _MAX (the same per-expert GEMM choice)
    int32_t sum_mode = 0;    // 1: replies are per-row weighted sums
    int32_t reply_f16 = 0;   // sum mode only: reply rows as f16 (half the bytes; P100E_FN_REXP_HALF)
    int32_t pad = 0;
};

// one blocking TCP connection (Winsock / POSIX). Every call throws std::runtime_error on a closed / reset peer or a
// timeout of no progress (timeout_ms < 0 = wait forever, for a server between requests).
class rexp_sock {
public:
    rexp_sock() = default;
    ~rexp_sock();
    rexp_sock(const rexp_sock &) = delete;
    rexp_sock & operator=(const rexp_sock &) = delete;
    rexp_sock(rexp_sock && o) noexcept;
    rexp_sock & operator=(rexp_sock && o) noexcept;

    // listen on host:port (host "" or "0.0.0.0" = every interface) and accept one connection within timeout_ms
    static rexp_sock accept_one(const std::string & host, int port, int timeout_ms);
    // connect to host:port, retrying until timeout_ms
    static rexp_sock connect_to(const std::string & host, int port, int timeout_ms);

    void send_all(const void * p, size_t n, int timeout_ms);
    void recv_all(void * p, size_t n, int timeout_ms);
    // header + payload in one send (the payload pointer may be null when bytes == 0)
    void send_msg(rexp_hdr h, const void * payload, size_t bytes, int timeout_ms);
    // the next header (magic checked; op checked when want != 0)
    rexp_hdr recv_hdr(uint32_t want, int timeout_ms);
    bool ok() const { return fd_ != ~(uint64_t) 0; }
    void close();
    std::string peer() const { return peer_; }

private:
    uint64_t fd_ = ~(uint64_t) 0;   // SOCKET / int
    std::string peer_;
};

// "host:port" -> host, port
void rexp_parse_addr(const std::string & s, std::string & host, int & port);

} // namespace p100e
