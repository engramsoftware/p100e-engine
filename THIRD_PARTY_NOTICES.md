# Third-party notices

Parts of this project are ports of, or copied from, other open-source projects.

## llama.cpp / ggml (MIT)

Copyright (c) 2023-2026 The ggml authors

Used in: `third_party/llama.cpp-fork/` (the parts of a modified llama.cpp that the server and
image input build from), `third_party/ggml/` (kernels and headers copied as-is),
`server/server_context.cpp`, `server/srv_sampler.cpp`/`.h` (ported from the llama.cpp server and
sampling code), `src/kld.cpp`, `src/nvit/nvit_pre.cpp`, `src/vision/`, and the kernels in `src/fast/` that follow
llama.cpp designs, including `matvec*`. The full MIT licence
text is in `third_party/ggml/LICENSE` and `third_party/llama.cpp-fork/LICENSE`. Other projects
bundled inside the fork keep their own notices in `third_party/llama.cpp-fork/vendor/` and
`third_party/llama.cpp-fork/licenses/`.

## Kmic-68/llama.cpp (MIT)

Copyright (c) 2023-2026 The ggml authors

Used in: `src/fast/gemm.cu` and `gemm.cuh`, `src/fast/attn_fold.cu` and `attn_fold.cuh`, `src/fast/gdn_chunked.cu`,
`gdn_chunked.cuh` and `gdn_chunked_kernels.cuh`, and `tools/sass/fold_bankfix.py`. Those files name the source and
the commit they were ported from. The fork in `third_party/llama.cpp-fork` also contains prefill work from this
project. Its licence is the same MIT text as llama.cpp.

## stb_image (public domain / MIT)

`third_party/stb/stb_image.h` by Sean Barrett. The licence notice is at the end of that header.
