# p100e

p100e is an inference engine for Qwen 3.8 models (the `qwen35` hybrid architecture, tested with the 27B model) on NVIDIA Tesla P100 GPUs. It is written in CUDA from scratch. It does not use ggml at run time.

It was built because the P100 (Pascal, `sm_60`) is old, cheap and still has fast HBM2 memory, but common inference stacks no longer tune for it. p100e uses custom kernels, CUDA graphs and tensor parallelism to get a lot more speed out of two or four of these cards.

It ships two programs you will use:

| program | what it is |
|---|---|
| `p100e-server` | An HTTP server that speaks the llama-server API (OpenAI chat, OpenAI responses, Anthropic messages, tools, reasoning, image input). It runs on the p100e engine. |
| `p100e-run` | A command-line tool to load a model, run a prompt, benchmark, test MTP drafting, and run quality checks. |

There are also small helpers: `p100e-info` (read a GGUF file) and `p100e-dequant` (test the weight decoders).

## The easy way

You need Python 3, a `qwen35` GGUF model, and the tools that `check` lists. On Linux, type `python3` instead of `python`.

```
git clone https://github.com/engramsoftware/p100e-engine.git
cd p100e-engine
python p100e.py check
python p100e.py setup
python p100e.py start path/to/model.gguf
```

1. `check` says what is missing and how to fix it.
2. `setup` builds the server. Add `--vision` for image input.
3. `start` finds your GPUs and starts the server at `http://127.0.0.1:8080/v1`.

Then point any OpenAI-compatible app at that address. To let other machines connect, add `--share`, which also makes a key in `keys.txt`.

The full walkthrough is in [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## What you get

- **Tensor parallel over 1 to 4 GPUs.** Weights stay resident in VRAM. Two or four GPUs get equal shares. Three GPUs, or GPUs of different sizes, get uneven shares (see Multi-GPU below).
- **Hand-written kernels** for decode, prefill, attention and the delta-net (the linear-attention part of the model). One CUDA graph per decode step.
- **MTP speculative decoding.** The model's own extra "next token" block drafts up to 3 tokens per step. The engine checks them in one pass. Output is identical to normal greedy decoding.
- **Long context.** Paged KV cache shared by all chats, with `f16`, `q8_0` (default) or `q4_0` storage.
- **Several chats at once** (`-np` slots), with a prompt cache that keeps idle chats on the GPU and then in pinned host RAM.
- **Image input** through a native vision encoder (`nvit`) that runs over two GPUs and uses very little VRAM.
- **Drop-in server.** Existing llama-server command lines keep working. Flags that only make sense for llama.cpp backends are accepted and ignored.

## Requirements

- Windows 10 or 11, 64-bit, or Linux (tested to build on Ubuntu 24.04; 22.04 should work too).
- One to four NVIDIA GPUs. Tested on Tesla P100 16 GB. The code also builds for `sm_75` so you can test on a GTX 1660.
- NVIDIA driver with CUDA 12 support.
- **CUDA Toolkit 12.9.** CUDA 13 dropped Pascal, so do not use it.
- Windows: Visual Studio 2022 Build Tools (C++ workload). Linux: GCC 11 to 14 (`build-essential`).
- CMake 3.24 or newer, and Ninja.
- Python 3 on the PATH (the build runs a small check script; without Python the build still works but turns one optional kernel optimisation off by default).
- A GGUF model of the `qwen35` family, for example Qwen3.8-27B in a Q4-class quantisation. For image input you also need the matching `mmproj` GGUF.
- Enough pinned host RAM for the prompt cache (default budget 20 GB). Lower it with `--p100e-host-budget-mb`.

## Build

### Linux (Ubuntu)

```
sudo apt install build-essential cmake ninja-build git python3
# CUDA Toolkit 12.9 from NVIDIA's apt repo (package cuda-toolkit-12-9), not 13
./build.sh                       # architectures 60, 75 and 86
./build.sh 60                    # P100 only
# for image input, build the included fork first (see "The llama.cpp fork" below)
```

`build.sh` finds CUDA in `/usr/local/cuda-12.9` (or `/usr/local/cuda`; set `CUDA_HOME` to change it), configures CMake with Ninja in `build/` and builds everything. `JOBS` sets the number of parallel compile jobs (default 8). On Linux the programs have no `.exe` suffix: `build/p100e-server`, `build/p100e-run`.

What is different on Linux for now:

- **Image input works** when `P100E_FORK_BUILD` points at a shared-library build of the fork. The fork's `.so` files in `<build>/bin` are linked directly (found through RPATH, no DLL loading), and the server gets `build/libp100e-vision.so`. Without that fork build, `P100E_VISION` is skipped, `p100e-server` is text only and `--mmproj` is refused.
- **Windows-only test helpers** in `p100e-run` (the batch mode with `++`, `--seq`, and the new-process step of `--cache-test --disk`) print "Windows only" and stop. `p100e-nvit --seq` works.

### Windows

The simple way:

```
build.cmd            # architectures 60, 75 and 86
build.cmd 60         # P100 only
```

`build.cmd` loads the Visual Studio environment, points at CUDA 12.9, configures CMake with Ninja in `build\`, and builds everything. Open `build.cmd` and check that the two paths at the top match your install (Visual Studio Build Tools and the CUDA toolkit). Set `JOBS` to change the number of parallel compile jobs (default 8; use 2 if memory is tight).

The same thing by hand, from a "x64 Native Tools" prompt:

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_CUDA_ARCHITECTURES=60 ^
      -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" ^
      -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler
cmake --build build -j 8
```

Useful CMake options:

| option | default | meaning |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | `60;75;86` | GPU architectures to compile for. `60` is the P100, `75` the GTX 1660, `86` the RTX 3070. |
| `P100E_SERVER` | `ON` | Build `p100e-server`. Turned off automatically (with a warning) if the fork sources are not found. |
| `P100E_FORK_DIR` | `third_party/llama.cpp-fork` | Source tree of the llama.cpp fork used for the server's text layer (tokenizer, chat templates, parsers, sampler). Set it with `-DP100E_FORK_DIR=<path>`. |
| `P100E_FORK_COMMIT` | `98ab7e5` | The fork commit the server was verified against. |
| `P100E_FORK_CHECK` | `OFF` | Turn on when using an outside fork checkout: the build then fails unless that tree is at this commit with no local changes. |
| `P100E_VISION` | `ON` | Build the image encoder used by `p100e-run`. Needs the fork build (`P100E_FORK_SRC`, `P100E_FORK_BUILD`). |
| `P100E_FORK_SRC`, `P100E_FORK_BUILD` | `third_party/llama.cpp-fork`, `third_party/llama.cpp-fork/build` | Fork source and shared-library build used for image input. |
| `P100E_NVIT_FORK_SRC`, `P100E_NVIT_FORK_BUILD` | the same | The same two paths for the reference mode of `p100e-nvit`. |
| `P100E_SERVER_FAKE` | `ON` | Also build `p100e-server-fake`, a CPU-only test double. Never deploy it. |

Output files land in `build\`. For image input in the server, copy `p100e-vision.dll` next to `p100e-server.exe` (Windows; on Linux `libp100e-vision.so` is found through RPATH).

### The llama.cpp fork

The parts of the fork that `p100e-server` and image input need are included in `third_party/llama.cpp-fork`, so the server builds with no extra steps. They provide the tokenizer, chat templates, parsers and sampler, and the `mtmd` and `clip` code.

Image input also needs a shared-library build of that fork. Run this once, before the main build, from the repository root (on Windows, in a "x64 Native Tools" prompt, and add `-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler`):

```
cmake -S third_party/llama.cpp-fork -B third_party/llama.cpp-fork/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_MTMD=ON
cmake --build third_party/llama.cpp-fork/build -j 8
```

Without that step, `P100E_VISION` is skipped, `p100e-server` is text only and `--mmproj` is refused.

## Run

Short versions. More copy-paste lines are in [docs/QUICKSTART.md](docs/QUICKSTART.md).

Start the server on two GPUs:

```
p100e-server -m model.gguf --host 127.0.0.1 --port 8080 -c 131072 -np 1 -ub 2048 --jinja --p100e-devices 0,1
```

Ask it something:

```
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":64}"
```

Run a quick speed test with the command-line tool:

```
p100e-run model.gguf --devices 0,1 --bench 512 64
```

Every option is listed in [docs/FLAGS.md](docs/FLAGS.md).

## Multi-GPU

Each layer is split across the GPUs (tensor parallel). Give the GPUs with `--p100e-devices` (server) or `--devices` (`p100e-run`), 1 to 4 of them.

- **1, 2 or 4 GPUs:** every GPU gets the same share. For Qwen3.8-27B with 4 GPUs: 1 KV head, 4 delta-net heads, 4352 FFN rows and 62080 vocab rows each.
- **3 GPUs:** the model does not divide by 3, so the shares are uneven. Each GPU always holds whole pieces: whole KV heads (with their query heads), whole delta-net head groups, and FFN / vocab rows in blocks of 256. For Qwen3.8-27B the default split is KV heads 2/1/1 and delta-net heads 4/8/4 (they move in fours because of the quant block size and kernel alignment). The FFN and vocab blocks then even out the rest: FFN 23/20/25 blocks and vocab 400/235/335 blocks, so each GPU holds the same number of weights (8.54 billion). The KV cache follows the KV heads, so the first GPU holds half of it. The split is printed at load.
- **Unequal GPUs** (for example an RTX 3070, a P100 and a GTX 1660): give each GPU a relative weight with `--p100e-split 2,1,1` (server), `--split 2,1,1` (`p100e-run`) or the environment variable `P100E_SPLIT=2,1,1`. `2,1,1` gives the first GPU about half of every part. A weight list that would leave a GPU with no KV head, no head group or no block is refused at load with a message that names the part.

The MTP draft block splits its small input projection evenly in 256-column blocks and ignores the weights. Saved sessions remember the split: a session file from another split is refused. Two or four GPUs without `--p100e-split` give exactly the same results as before this option existed.

### Across machines (network tensor parallel)

The same per-layer split can span several machines over the LAN (`p100e-run` only for now; the server comes later). Every machine runs the same command with the same model, prompt and options, plus `--net-nodes host:port,host:port` (the machines in rank order), `--net-node i` (which one this is) and `--devices` = its own GPUs. `--split` lists a weight for every GPU of every machine, in that order. Example, two P100s on a Linux machine plus an RTX 3070 on Windows, the 3070 getting about a fifth:

```
# Linux, node 0 (ranks 0 and 1)
./build/p100e-run /models/Qwen3.8-27B-Q6_K.gguf --tokens-file prompt.txt --gen 128 --devices 0,1 --split 2,2,1 \
    --net-nodes 192.0.2.10:29500,192.0.2.11:29501 --net-node 0
# Windows, node 1 (rank 2)
build\p100e-run.exe models\Qwen3.8-27B-Q6_K.gguf --tokens-file prompt.txt --gen 128 --devices 0 --split 2,2,1 --net-nodes 192.0.2.10:29500,192.0.2.11:29501 --net-node 1
```

Each layer's two allreduces then cross the network (one TCP message per machine), so decode speed is bound by the network round trips, not the GPUs. The results are bit-identical to one process running the same split. The machines check at connect that they run the same model, arguments and switches and refuse otherwise; if one machine dies the other stops with an error naming it. Not supported over the network yet: MTP, the server, sessions / prompt cache files, vision, KLD scoring, `--ab`. CUDA graphs and the P2P allreduce are off in this mode.

### Across two machines, whole layers (hybrid pipeline)

`--pipe-layers k` (or `--pipe-split a,b`) with two `--net-nodes` keeps the per-layer split inside each machine and hands whole layers between them. Node 0 runs layers 0..k-1 split over its own GPUs, exactly as one process would (P2P allreduce, CUDA graphs, prefill pipeline). Node 1 runs the rest plus the LM head on its GPUs. Per token only one hidden-state row crosses the network (E x 4 bytes: 20 KB for Qwen3.8-27B), and the greedy token id (4 bytes) comes back. The prompt sends T rows once. Everything stays f32. `--split` here weighs only that machine's GPUs, so node 1 can leave it out. At load both machines print a byte plan (weights, head, KV, state, buffers per GPU against free VRAM) and refuse to start if either does not fit. `--pipe-layers auto` gives node 1 as many layers as it holds. Example: two P100s for layers 0..48, an RTX 3070 for 49..63 + head:

```
# Linux, node 0
P100E_PF_WIRE16=0 ./build/p100e-run /models/Qwen3.8-27B-Q6_K.gguf \
    --ctx 4096 --kv q8_0 --devices 0,1 --split 1,1 --pipe-split 49,15 \
    --net-nodes 192.0.2.10:29500,192.0.2.11:29501 --net-node 0 --tokens-file prompt.txt --gen 128
# Windows, node 1 (the same arguments in the same order, apart from the node-local --devices / --split / --net-node)
set P100E_PF_WIRE16=0& build\p100e-run.exe models\Qwen3.8-27B-Q6_K.gguf --ctx 4096 --kv q8_0 --devices 0 --pipe-split 49,15 --net-nodes 192.0.2.10:29500,192.0.2.11:29501 --net-node 1 --tokens-file prompt.txt --gen 128
```

With node 0 on one GPU the results are bit-identical to a one-GPU run. With node 0 split over two GPUs they match a run whose first k layers use that split. This mode is about memory, not speed: a layer on the 3070 is slower than on the P100 pair, and every token pays one network round trip. Not supported yet: MTP, the server, sessions, vision, KLD.

## Documentation

| file | content |
|---|---|
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Step-by-step guide: build, get a model, run the server, connect apps, share it safely. |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | Copy-paste command lines for common jobs. |
| [docs/KUBERNETES.md](docs/KUBERNETES.md) | Run the server in a container on Kubernetes. |
| [docs/FLAGS.md](docs/FLAGS.md) | Every command-line flag and environment variable of `p100e-server` and `p100e-run`. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the engine, the server, tensor parallelism, vision and drafting fit together. |
| [server/README.md](server/README.md) | Server internals: build, engine interface, prompt cache, failure handling. |

## Repository layout

```
src/            the engine (kernels, tensor-parallel model, MTP, scheduler, vision)
src/fast/       hand-tuned kernels (matvec, attention, GEMM, fused decode)
src/nvit/       native vision encoder
server/         p100e-server (HTTP layer on top of the engine)
tools/          p100e-run, p100e-info, p100e-dequant and helper scripts
tests/          unit and integration tests
third_party/    vendored weight decoders and the llama.cpp fork code (MIT)
cmake/          CMake modules
```

## Limits

- Linux has been run on one machine (Ubuntu, two P100s) with text and image input. Windows-only test helpers are not ported (see Build, Linux).
- Built and tuned for the Qwen 3.8 `qwen35` architecture. Other architectures are not supported by the main engine.
- The server does not support audio or video input, embeddings, rerank, LoRA, slot save/restore, or `n_cmpl > 1`. It returns explicit errors for these.
- Four-GPU tensor parallel is implemented and tested with virtual ranks on one GPU. It has not been measured on a real four-GPU machine.
- Three-GPU and weighted splits are tested with virtual ranks on one GPU and a small random model only. They have not run on three real GPUs or on the 27B model yet.

## Licence

MIT. See [LICENSE](LICENSE). Code from other projects is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
