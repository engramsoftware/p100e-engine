#include "gguf.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <strings.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
extern char ** environ;
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace p100e {

namespace {

struct reader {
    const uint8_t * p;
    const uint8_t * end;

    void need(size_t n) const {
        if ((size_t)(end - p) < n) throw std::runtime_error("gguf: unexpected end of file");
    }
    template <typename T> T get() {
        need(sizeof(T));
        T v;
        memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        const uint64_t n = get<uint64_t>();
        need(n);
        std::string s((const char *) p, (size_t) n);
        p += n;
        return s;
    }
    void read_scalar(gguf_vtype t, gguf_value & v) {
        switch (t) {
            case gguf_vtype::U8:   v.u = get<uint8_t>();  v.i = (int64_t) v.u; break;
            case gguf_vtype::I8:   v.i = get<int8_t>();   v.u = (uint64_t) v.i; break;
            case gguf_vtype::U16:  v.u = get<uint16_t>(); v.i = (int64_t) v.u; break;
            case gguf_vtype::I16:  v.i = get<int16_t>();  v.u = (uint64_t) v.i; break;
            case gguf_vtype::U32:  v.u = get<uint32_t>(); v.i = (int64_t) v.u; break;
            case gguf_vtype::I32:  v.i = get<int32_t>();  v.u = (uint64_t) v.i; break;
            case gguf_vtype::U64:  v.u = get<uint64_t>(); v.i = (int64_t) v.u; break;
            case gguf_vtype::I64:  v.i = get<int64_t>();  v.u = (uint64_t) v.i; break;
            case gguf_vtype::F32:  v.f = get<float>();  break;
            case gguf_vtype::F64:  v.f = get<double>(); break;
            case gguf_vtype::BOOL: v.u = get<uint8_t>() != 0; v.i = (int64_t) v.u; break;
            case gguf_vtype::STRING: v.s = str(); break;
            default: throw std::runtime_error("gguf: bad scalar type");
        }
        if (t != gguf_vtype::F32 && t != gguf_vtype::F64 && t != gguf_vtype::STRING) v.f = (double) v.i;
    }
};

} // namespace

#ifndef _WIN32
// POSIX: file_ / rfile_ hold fd + 1 (so nullptr = none); mapping_ is unused, munmap needs only base_ + file_size
static int fd_of(void * h) { return (int) (intptr_t) h - 1; }
static void * fd_handle(int fd) { return fd < 0 ? nullptr : (void *) (intptr_t) (fd + 1); }
#endif

gguf_file::gguf_file(const std::string & path) {
#ifdef _WIN32
    HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (f == INVALID_HANDLE_VALUE) throw std::runtime_error("gguf: cannot open " + path);
    LARGE_INTEGER sz;
    GetFileSizeEx(f, &sz);
    file_size = (uint64_t) sz.QuadPart;
    HANDLE m = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) { CloseHandle(f); throw std::runtime_error("gguf: cannot map " + path); }
    base_ = (const uint8_t *) MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (!base_) { CloseHandle(m); CloseHandle(f); throw std::runtime_error("gguf: cannot view " + path); }
    file_ = f;
    mapping_ = m;
    path_ = path;
    rfile_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING, nullptr);
    if (rfile_ == INVALID_HANDLE_VALUE) rfile_ = nullptr;
#else
    const int f = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (f < 0) throw std::runtime_error("gguf: cannot open " + path);
    struct stat st;
    if (fstat(f, &st) != 0) { close(f); throw std::runtime_error("gguf: cannot stat " + path); }
    file_size = (uint64_t) st.st_size;
    void * b = mmap(nullptr, (size_t) file_size, PROT_READ, MAP_SHARED, f, 0);
    if (b == MAP_FAILED) { close(f); throw std::runtime_error("gguf: cannot map " + path); }
    base_ = (const uint8_t *) b;
    posix_fadvise(f, 0, 0, POSIX_FADV_SEQUENTIAL);
    file_ = fd_handle(f);
    path_ = path;
    rfile_ = fd_handle(open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT));   // may fail (tmpfs etc.): buffered pread then
#endif

    reader r{base_, base_ + file_size};
    if (r.get<uint32_t>() != 0x46554747u) throw std::runtime_error("gguf: bad magic");
    version = r.get<uint32_t>();
    if (version < 2 || version > 3) throw std::runtime_error("gguf: unsupported version " + std::to_string(version));
    const uint64_t n_tensors = r.get<uint64_t>();
    const uint64_t n_kv      = r.get<uint64_t>();
    // every kv takes >= 13 bytes (key length, type, 1-byte value) and every tensor info >= 32 (name length, n_dims,
    // one dim, type, offset): bigger counts are a corrupt header, not a reason to reserve terabytes
    if (n_kv > file_size / 13 || n_tensors > file_size / 32) throw std::runtime_error("gguf: bad kv / tensor count");

    for (uint64_t k = 0; k < n_kv; ++k) {
        std::string key = r.str();
        gguf_value v;
        v.type = (gguf_vtype) r.get<uint32_t>();
        if (v.type == gguf_vtype::ARRAY) {
            v.arr_type = (gguf_vtype) r.get<uint32_t>();
            const uint64_t n = r.get<uint64_t>();
            if (n > (uint64_t) (r.end - r.p)) throw std::runtime_error("gguf: array longer than the file: " + key);   // >= 1 byte each
            if (v.arr_type == gguf_vtype::STRING) {
                v.arr_str.reserve((size_t) n);
                for (uint64_t j = 0; j < n; ++j) v.arr_str.push_back(r.str());
            } else {
                v.arr_num.reserve((size_t) n);
                for (uint64_t j = 0; j < n; ++j) {
                    gguf_value e;
                    r.read_scalar(v.arr_type, e);
                    v.arr_num.push_back(e.f);
                }
            }
            v.u = n;
        } else {
            r.read_scalar(v.type, v);
        }
        kv.emplace(std::move(key), std::move(v));
    }
    // the spec (and ggml) require a non-zero power of two; 0 or a non-integer value would divide by zero below
    const gguf_value * al = get("general.alignment");
    if (al && (al->type == gguf_vtype::STRING || al->type == gguf_vtype::ARRAY || al->type == gguf_vtype::F32 ||
               al->type == gguf_vtype::F64 || al->type == gguf_vtype::BOOL))
        throw std::runtime_error("gguf: general.alignment is not an integer");
    alignment = (uint64_t) get_int("general.alignment", 32);
    if (alignment == 0 || (alignment & (alignment - 1)) || alignment > (1u << 30))
        throw std::runtime_error("gguf: bad general.alignment " + std::to_string(alignment));

    tensors.resize((size_t) n_tensors);
    for (auto & t : tensors) {
        t.name = r.str();
        t.n_dims = (int) r.get<uint32_t>();
        if (t.n_dims < 1 || t.n_dims > 4) throw std::runtime_error("gguf: bad n_dims for " + t.name);
        for (int d = 0; d < t.n_dims; ++d) {
            const uint64_t ne = r.get<uint64_t>();
            // >= 1: the loaders divide by ne[1..3]; < 2^62 so the products below cannot wrap (tensor_nbytes checks them)
            if (ne < 1 || ne >= (1ull << 62)) throw std::runtime_error("gguf: bad dimension for " + t.name);
            t.ne[d] = (int64_t) ne;
        }
        t.type = r.get<uint32_t>();
        t.offset = r.get<uint64_t>();
        if (t.offset % alignment) throw std::runtime_error("gguf: misaligned tensor offset for " + t.name);
        t.nbytes = tensor_nbytes(t.type, t.ne);
    }

    const uint64_t header = (uint64_t)(r.p - base_);
    const uint64_t data_start = (header + alignment - 1) / alignment * alignment;
    if (data_start > file_size) throw std::runtime_error("gguf: data section past EOF");
    const uint64_t data_size = file_size - data_start;
    for (size_t i = 0; i < tensors.size(); ++i) {
        auto & t = tensors[i];
        // written so nothing can wrap: offset and size are each checked against what is left before adding
        if (t.offset > data_size || t.nbytes > data_size - t.offset) throw std::runtime_error("gguf: tensor past EOF: " + t.name);
        t.data = base_ + data_start + t.offset;
        index_[t.name] = i;
    }
}

gguf_file::~gguf_file() {
#ifdef _WIN32
    if (base_) UnmapViewOfFile(base_);
    if (mapping_) CloseHandle((HANDLE) mapping_);
    if (file_) CloseHandle((HANDLE) file_);
    if (rfile_) CloseHandle((HANDLE) rfile_);
#else
    if (base_) munmap((void *) base_, (size_t) file_size);
    if (file_) close(fd_of(file_));
    if (rfile_) close(fd_of(rfile_));
#endif
}

#ifdef _WIN32
static void read_handle(HANDLE h, void * dst, uint64_t off, size_t n) {
    char * d = (char *) dst;
    while (n > 0) {
        const DWORD chunk = (DWORD) std::min<size_t>(n, 1u << 30);
        OVERLAPPED ov{};
        ov.Offset = (DWORD) (off & 0xffffffffu); ov.OffsetHigh = (DWORD) (off >> 32);
        DWORD got = 0;
        if (!ReadFile(h, d, chunk, &got, &ov) || got == 0) throw std::runtime_error("gguf: ReadFile failed");
        d += got; off += got; n -= got;
    }
}

#else
// pread until n bytes or EOF; returns the bytes read (short only at EOF)
static size_t read_fd(int fd, void * dst, uint64_t off, size_t n) {
    char * d = (char *) dst;
    size_t done = 0;
    while (done < n) {
        const ssize_t got = pread(fd, d + done, std::min<size_t>(n - done, 1u << 30), (off_t) (off + done));
        if (got < 0) throw std::runtime_error("gguf: pread failed");
        if (got == 0) break;
        done += (size_t) got;
    }
    return done;
}
static void read_handle(void * h, void * dst, uint64_t off, size_t n) {
    if (read_fd(fd_of(h), dst, off, n) != n) throw std::runtime_error("gguf: pread hit EOF");
}
#endif

void gguf_file::read_at(void * dst, const uint8_t * src, size_t n) const {
    if (src < base_ || src + n > base_ + file_size) throw std::runtime_error("gguf: read_at outside the file");
    const uint64_t off = (uint64_t) (src - base_);
    if (!rfile_ || n == 0) { if (n) read_handle(file_, dst, off, n); return; }
    // unbuffered: sector-aligned offset / size / buffer; read the covering range into an aligned scratch and copy the slice,
    // in windows of W bytes, so the per-thread scratch stays <= W + A instead of growing to the largest tensor ever read
    constexpr uint64_t A = 4096, W = 64ull << 20;
    const uint64_t o0 = off & ~(A - 1), o1 = std::min<uint64_t>((off + n + A - 1) & ~(A - 1), (file_size + A - 1) & ~(A - 1));
    const size_t need = (size_t) std::min<uint64_t>(o1 - o0, W);
    thread_local std::vector<char> scratch;
    thread_local char * aligned = nullptr;
    if (scratch.size() < need + A) { scratch.resize(need + A); aligned = (char *) (((uintptr_t) scratch.data() + A - 1) & ~(uintptr_t) (A - 1)); }
    for (uint64_t w0 = o0; w0 < o1; w0 += W) {
        const uint64_t w1 = std::min<uint64_t>(o1, w0 + W);
        const uint64_t avail = std::min<uint64_t>(w1, file_size) - w0;   // the last sector may run past EOF: ReadFile returns the short tail
        char * d = aligned;
        uint64_t pos = w0, left = avail;
#ifndef _WIN32
        // O_DIRECT: aligned offset / buffer / size; the read covering EOF returns the short tail
        if (read_fd(fd_of(rfile_), d, pos, (size_t) (w1 - w0)) < left) throw std::runtime_error("gguf: unbuffered pread short");
        (void) pos; (void) left;
#else
        while (left > 0) {
            const DWORD chunk = (DWORD) std::min<uint64_t>(left, 64u << 20);
            OVERLAPPED ov{};
            ov.Offset = (DWORD) (pos & 0xffffffffu); ov.OffsetHigh = (DWORD) (pos >> 32);
            DWORD got = 0;
            if (!ReadFile((HANDLE) rfile_, d, (DWORD) ((chunk + A - 1) & ~(A - 1)), &got, &ov) || got == 0) throw std::runtime_error("gguf: unbuffered ReadFile failed");
            if (got > left) got = (DWORD) left;
            d += got; pos += got; left -= got;
        }
#endif
        // the part of [off, off + n) inside this window
        const uint64_t s0 = std::max<uint64_t>(off, w0), s1 = std::min<uint64_t>(off + n, w1);
        memcpy((char *) dst + (s0 - off), aligned + (s0 - w0), (size_t) (s1 - s0));
    }
}

const gguf_tensor * gguf_file::find(const std::string & name) const {
    auto it = index_.find(name);
    return it == index_.end() ? nullptr : &tensors[it->second];
}

const gguf_value * gguf_file::get(const std::string & key) const {
    auto it = kv.find(key);
    return it == kv.end() ? nullptr : &it->second;
}

int64_t gguf_file::get_int(const std::string & key, int64_t def) const {
    const gguf_value * v = get(key);
    return v ? v->i : def;
}

double gguf_file::get_float(const std::string & key, double def) const {
    const gguf_value * v = get(key);
    return v ? v->f : def;
}

std::string gguf_file::get_str(const std::string & key, const std::string & def) const {
    const gguf_value * v = get(key);
    return v && v->type == gguf_vtype::STRING ? v->s : def;
}

uint64_t hash_bytes(uint64_t h, const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        memcpy(&w, b + i, 8);
        h ^= w;
        h *= 1099511628211ull;
    }
    for (; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

template <typename T> static uint64_t hash_pod(uint64_t h, const T & v) { return hash_bytes(h, &v, sizeof v); }
static uint64_t hash_str(uint64_t h, const std::string & s) { return hash_bytes(hash_pod(h, (uint64_t) s.size()), s.data(), s.size()); }

uint64_t gguf_file::meta_hash() const {
    uint64_t h = kHashSeed;
    for (const auto & [k, v] : kv) {   // std::map: key order
        h = hash_str(h, k);
        h = hash_pod(h, (uint32_t) v.type);
        h = hash_pod(h, (uint32_t) v.arr_type);
        h = hash_pod(h, v.u);
        h = hash_pod(h, v.i);
        h = hash_pod(h, v.f);
        h = hash_str(h, v.s);
        h = hash_pod(h, (uint64_t) v.arr_str.size());
        for (const std::string & x : v.arr_str) h = hash_str(h, x);
        h = hash_pod(h, (uint64_t) v.arr_num.size());
        if (!v.arr_num.empty()) h = hash_bytes(h, v.arr_num.data(), v.arr_num.size() * sizeof(double));
    }
    return h;
}

uint64_t gguf_file::tensor_table_hash() const {
    uint64_t h = kHashSeed;
    for (const gguf_tensor & t : tensors) {
        h = hash_str(h, t.name);
        h = hash_pod(h, t.type);
        h = hash_bytes(h, t.ne, sizeof t.ne);
        h = hash_pod(h, t.offset);
        h = hash_pod(h, t.nbytes);
    }
    return h;
}

uint64_t gguf_file::tensor_sample_hash() const {
    uint64_t h = kHashSeed;
    const uint64_t S = 4096;
    for (const gguf_tensor & t : tensors) {
        if (!t.data || t.nbytes == 0) continue;
        if (t.nbytes <= 3 * S) { h = hash_bytes(h, t.data, t.nbytes); continue; }
        h = hash_bytes(h, t.data, S);
        h = hash_bytes(h, t.data + (t.nbytes / 2 - S / 2), S);
        h = hash_bytes(h, t.data + (t.nbytes - S), S);
    }
    return h;
}

uint64_t env_hash(const std::string & prefix, const std::vector<std::string> & ignore, std::string * used) {
    std::vector<std::string> ent;
#ifdef _WIN32
    if (char * env = GetEnvironmentStringsA()) {
        for (const char * e = env; *e; e += strlen(e) + 1) {
#else
#define _stricmp strcasecmp
    if (char ** env = environ) {
        for (; *env; ++env) {
            const char * e = *env;
#endif
            const std::string s(e);
            if (s.compare(0, prefix.size(), prefix) != 0) continue;
            const std::string name = s.substr(0, s.find('='));
            bool skip = false;
            for (const std::string & ig : ignore) skip = skip || _stricmp(ig.c_str(), name.c_str()) == 0;
            if (!skip) ent.push_back(s);
        }
#ifdef _WIN32
        FreeEnvironmentStringsA(env);
#endif
    }
    std::sort(ent.begin(), ent.end());
    uint64_t h = kHashSeed;
    if (used) used->clear();
    for (const std::string & s : ent) {
        h = hash_str(h, s);
        if (used) *used += (used->empty() ? "" : " ") + s;
    }
    return h;
}

uint64_t tensor_nbytes(uint32_t type, const int64_t ne[4]) {
    const qtype_info & ti = get_type_info(type);
    if (ti.blck == 0 || ne[0] % ti.blck != 0) throw std::runtime_error(std::string("gguf: bad row size for type ") + ti.name);
    // checked products: a crafted shape must not wrap to a small size that then passes the EOF check
    uint64_t n = (uint64_t) (ne[0] / ti.blck);
    const uint64_t f[4] = {(uint64_t) ti.size, (uint64_t) ne[1], (uint64_t) ne[2], (uint64_t) ne[3]};
    for (uint64_t x : f) {
        if (x != 0 && n > UINT64_MAX / x) throw std::runtime_error(std::string("gguf: tensor size overflows for type ") + ti.name);
        n *= x;
    }
    return n;
}

} // namespace p100e
