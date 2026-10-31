// Reference encoder for nvit parity tests: the llama.cpp fork's mtmd / clip (same commit and DLLs as the server),
// run in-process through delay-loaded DLLs. Plain C++ interface (no p100e or ggml headers).
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

struct nvit_ref_tensor {
    int rows = 0, cols = 0;           // [rows][cols] row-major (ggml ne1 x ne0)
    std::vector<float> data;
};

struct nvit_ref_result {
    int nx = 0, ny = 0;               // preprocessed image size (px)
    std::vector<float> inp_raw;       // planar [3][ny][nx] (clip's graph input)
    int tx = 0, ty = 0;               // token grid
    std::vector<float> embd;          // [ty*tx][n_embd]
    int n_embd = 0;
    std::vector<double> encode_ms;    // mtmd_encode_chunk wall time per run (callback-free runs)
    std::map<std::string, nvit_ref_tensor> t;   // captured graph tensors by ggml name ("ln1-0", ...)
};

class nvit_ref {
public:
    // dll_dir empty: P100E_VISION_DLLS, the exe directory, the fork build
    nvit_ref(const std::string & mmproj, const std::string & vocab_model, int device, int min_tok, int max_tok,
             const std::string & dll_dir, bool verbose);
    ~nvit_ref();
    // encode the file bytes `runs` times without callbacks (timing + final embeddings), then once with the eval
    // callback capturing the input image and the named tensors (names: exact ggml names)
    void encode(const std::string & bytes, int runs, const std::vector<std::string> & names, nvit_ref_result & out);
    std::string dll_dir() const;
    double vram_mib() const;          // device memory taken by the reference encoder (load + warmup)

private:
    struct impl;
    std::unique_ptr<impl> p_;
};
