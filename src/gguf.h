// GGUF v2/v3 reader: memory-maps the file, parses metadata and tensor infos.
// Tensor data stays in the mapping; callers copy what they need to the GPU.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace p100e {

enum class gguf_vtype : uint32_t {
    U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6, BOOL = 7,
    STRING = 8, ARRAY = 9, U64 = 10, I64 = 11, F64 = 12,
};

struct gguf_value {
    gguf_vtype type = gguf_vtype::U8;
    gguf_vtype arr_type = gguf_vtype::U8;   // element type when type == ARRAY
    uint64_t   u = 0;                       // any unsigned / bool
    int64_t    i = 0;                       // any signed
    double     f = 0.0;                     // f32 / f64
    std::string s;                          // string
    std::vector<std::string> arr_str;       // string arrays (vocab)
    std::vector<double>      arr_num;       // numeric arrays (scores, token types, ...)
};

struct gguf_tensor {
    std::string name;
    uint32_t    type = 0;                   // ggml_type
    int64_t     ne[4] = {1, 1, 1, 1};
    int         n_dims = 0;
    uint64_t    offset = 0;                 // relative to data section
    uint64_t    nbytes = 0;
    const uint8_t * data = nullptr;         // pointer into the mapping
    int64_t nelements() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
};

class gguf_file {
public:
    explicit gguf_file(const std::string & path);
    ~gguf_file();
    gguf_file(const gguf_file &) = delete;
    gguf_file & operator=(const gguf_file &) = delete;

    uint32_t version = 0;
    uint64_t alignment = 32;
    uint64_t file_size = 0;
    std::map<std::string, gguf_value> kv;
    std::vector<gguf_tensor> tensors;

    const gguf_tensor * find(const std::string & name) const;
    // read n bytes of the file behind mapping pointer src into dst with ReadFile (no page faults on the mapping: the
    // fault path of a huge mapping ran at ~30 MB/s on a SATA disk after its cache was evicted; ReadFile 1.6 GB/s)
    void read_at(void * dst, const uint8_t * src, size_t n) const;
    bool mapped(const void * p) const { return p >= base_ && p < base_ + file_size; }   // p points into the file mapping
    const gguf_value  * get(const std::string & key) const;
    int64_t     get_int(const std::string & key, int64_t def) const;
    double      get_float(const std::string & key, double def) const;
    std::string get_str(const std::string & key, const std::string & def) const;

    // identity hashes (session files): every metadata key / value; the tensor table (name, type, shape,
    // offset, size); a content sample of every tensor (its first, middle and last 4 KiB) - a re-quant or finetune with
    // the same sizes and shapes changes the sample. Cheap right after loading (the pages are cached).
    uint64_t meta_hash() const;
    uint64_t tensor_table_hash() const;
    uint64_t tensor_sample_hash() const;

private:
    void * file_ = nullptr;
    void * mapping_ = nullptr;
    void * rfile_ = nullptr;        // unbuffered handle for read_at (FILE_FLAG_NO_BUFFERING: sector-aligned, no cache read-ahead)
    std::string path_;
    const uint8_t * base_ = nullptr;
    std::map<std::string, size_t> index_;
};

// ggml type info (block size in elements, bytes per block, name)
struct qtype_info {
    const char * name;
    int64_t blck;
    size_t  size;
    bool    has_gpu_dequant;
};
const qtype_info & get_type_info(uint32_t type);

// 64-bit FNV-1a over bytes (8-byte words where possible), continuing from h
uint64_t hash_bytes(uint64_t h, const void * p, size_t n);
constexpr uint64_t kHashSeed = 1469598103934665603ull;
// hash of this process's environment variables whose names start with `prefix` (NAME=value, sorted), except the names
// in `ignore`; *used = the hashed entries (space separated)
uint64_t env_hash(const std::string & prefix, const std::vector<std::string> & ignore, std::string * used = nullptr);
uint64_t tensor_nbytes(uint32_t type, const int64_t ne[4]);

} // namespace p100e
