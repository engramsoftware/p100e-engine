# Flags and environment variables

This page lists every command-line flag and environment variable of `p100e-server` and `p100e-run`. It was written from the source code.

How to read it:

- "Default" is the value used when you do not set the option.
- Switches written as `0|1` are read as numbers: `0` means off, anything else means on.
- Environment variables are read at start-up unless the table says otherwise. In PowerShell use `$env:NAME = "value"`. In `cmd` use `set NAME=value`.
- Most `P100E_*` variables are tuning or debug switches. You normally do not need them. Part 3 groups them so you can find the ones that matter.

Contents:

1. [p100e-server flags](#1-p100e-server-flags)
2. [p100e-run flags](#2-p100e-run-flags)
3. [Environment variables](#3-environment-variables)
4. [Exit codes](#4-exit-codes)
5. [Not confirmed](#5-not-confirmed)

---

## 1. p100e-server flags

`p100e-server` first removes every flag that starts with `--p100e-`. The rest of the command line goes to the llama-server argument parser. So all usual llama-server flags keep working. An unknown `--p100e-*` flag stops the program with `error: unknown p100e flag <name>`. A flag that needs a value but has none stops with `error: <flag> needs a value`.

### 1.1 Standard llama-server flags that matter

| flag | default | meaning |
|---|---|---|
| `-m <file>` | none, required | Model GGUF file. |
| `--host <addr>` | llama-server default (`127.0.0.1`) | Address to listen on. Use `0.0.0.0` to accept other machines. |
| `--port <n>` | llama-server default (`8080`) | Port to listen on. |
| `-c <n>` | llama-server default | Total context in tokens, shared by all slots. Per-slot limit is `-c / -np`. |
| `-np <n>`, `--parallel <n>` | **3** | Number of slots (chats at once). If you do not give `-np`, p100e uses 3 (not the llama.cpp default). |
| `-ub <n>`, `-b <n>` | llama-server default | Largest chunk of prompt tokens in one forward pass. A smaller value leaves more VRAM for the KV pool. |
| `--jinja` | off | Use the model's Jinja chat template. Needed for tools and reasoning. |
| `--chat-template-file <file>` | model's template | Use this chat template file. |
| `--alias <name>` | model file name | Model name reported by the API. |
| `-ctk <type>` | none | KV type. If `--p100e-kv` is not given, `q8_0` or `q4_0` here is used. Anything else gives `q8_0`. |
| `--mmproj <file>` | none | Vision projector GGUF. Turns on image input. |
| `--image-min-tokens <n>` | 1024 | Fewest tokens per image. Also the size used for `detail: "low"`. |
| `--image-max-tokens <n>` | 2048 | Largest tokens per image for `detail` `high`, `auto` or unset. |
| `--no-mmproj-offload` | off | Run the image encoder on the CPU. |
| `--log-file <file>`, `--log-timestamps`, `--log-prefix` | off | Server log options. |

Flags for llama.cpp backends (`--split-mode`, `-ngl`, `--flash-attn`, `--spec-*`, `--cache-type-v`, and similar) are accepted and ignored. The engine is set up by the `--p100e-*` flags below. The server also forces `sleep_idle_seconds` off, because the weights stay in VRAM.

### 1.2 `--p100e-*` flags

| flag | default | meaning | example |
|---|---|---|---|
| `--p100e-devices a,b,..` | `0,1` | GPU indices to use for tensor parallel. Give 1 to 4 indices. 2 and 4 split the 27B evenly; 3 gets uneven shares of whole heads and 256-row blocks (printed at load). With 3 or 4 GPUs the native vision encoder uses the first two. | `--p100e-devices 0` |
| `--p100e-split a,b,..` | equal | Relative share of each `--p100e-devices` GPU, for GPUs of different size or speed. One positive weight per GPU. A split that leaves a GPU with no KV head, head group or block is refused at load. Overrides `P100E_SPLIT`. | `--p100e-split 2,1,1` |
| `--p100e-kv f16\|q8_0\|q4_0` | `q8_0` (or `-ctk` when that is `q8_0`/`q4_0`) | KV cache type. `q8_0` costs about 17 KiB per token per GPU. `q4_0` is smaller. `f16` is the most exact. | `--p100e-kv q4_0` |
| `--p100e-mtp K` | `3` | Maximum drafts per step for MTP speculative decoding. `0` turns MTP off. Max 3. Needs a model with the MTP ("nextn") block, else the server stops with a clear message. | `--p100e-mtp 2` |
| `--p100e-mtp-tau t` | `0.5` | Adaptive draft length. A draft is checked only while the product of the draft head's probabilities is at least `t`. `0` always drafts the full `K`. | `--p100e-mtp-tau 0.7` |
| `--p100e-no-remote-images` | off | Refuse http(s) `image_url` / `input_audio` / `input_video` URLs with a 400 error (no server-side downloads); `data:` URIs still work. For servers reachable from untrusted pages or hosts. | `--p100e-no-remote-images` |
| `--p100e-mtp-multi K` | `-1` (no cap) | Maximum drafts per slot while several slots are busy. `0` stops drafting while more than one chat runs. | `--p100e-mtp-multi 0` |
| `--p100e-draft-rows N` | `4` | When several slots generate, the rows per step are `max(slots, N)` and the drafts are shared. `0` removes the row budget. | `--p100e-draft-rows 6` |
| `--p100e-decode-share x` | `0.5` | While one slot reads a prompt and others generate, the generating slots get this share of the wall time. `0` gives one decode step per prompt chunk. Clamped to 0..0.95. Env `P100E_DECODE_SHARE` overrides it. | `--p100e-decode-share 0.7` |
| `--p100e-cache 0\|1` | `1` | Prompt cache. Idle chats are kept and reused by prefix. Per request, `cache_prompt` also applies. Env `P100E_CACHE` is read by the scheduler. | `--p100e-cache 0` |
| `--p100e-cache-resident N` | `4` | Idle cache entries kept on the GPU (extra state slots). | `--p100e-cache-resident 2` |
| `--p100e-host-budget-mb N` | `20480` | Pinned host RAM for offloaded cache entries and checkpoints, in MiB. Env `P100E_HOST_BUDGET_MB` wins over this flag. | `--p100e-host-budget-mb 8192` |
| `--p100e-ckpt-budget-mb N` | `8192` | Part of the host budget that checkpoints may use, in MiB. | `--p100e-ckpt-budget-mb 4096` |
| `--p100e-pool-tokens N` | `0` (all free VRAM after reserves) | Size of the shared KV pool in tokens. | `--p100e-pool-tokens 200000` |
| `--p100e-margin-mb N` | `-1` (engine default, 512) | VRAM per GPU that the KV pool leaves free for temporary buffers. | `--p100e-margin-mb 768` |
| `--p100e-vision-device N` | last engine GPU | GPU for the image encoder when `--mmproj` is given. | `--p100e-vision-device 0` |
| `--p100e-engine tp\|fn` | `tp`; `fn` if the model's `general.architecture` is `qwen4exp` | Pick the engine. `tp` is the 27B tensor-parallel engine. `fn` is the experimental Flash-Next engine (mixture of experts, one slot). | `--p100e-engine tp` |
| `--p100e-fn-hot N` | `72` | `fn` only. GPU-resident experts per layer per GPU. | `--p100e-fn-hot 64` |
| `--p100e-fn-max-t N` | `512` | `fn` only. Prefill chunk rows. `0` uses `-ub`. | `--p100e-fn-max-t 256` |
| `--p100e-max-requests N` | `0` (never) | Exit after `N` finished completion requests. Useful for scripted tests. | `--p100e-max-requests 5` |

### 1.3 Debug and test flags (not for production)

| flag | default | meaning |
|---|---|---|
| `--p100e-no-engine` | off | No GPU. Only the text endpoints work (tokenize, templates, token counts, props). With `--mmproj` the image encoder runs on the CPU. |
| `--p100e-print-ids` | off | Log the prompt and generated token ids of every task. Puts prompt content in the log, so do not use it in production. |
| `--p100e-refcheck` | off | Re-run every greedy task as plain greedy decoding and log `IDENTICAL` or `MISMATCH`. The replay blocks the whole server loop. Debug only. |
| `--p100e-no-greedy-gpu` | greedy argmax on the GPU | Copy the logits and take the argmax on the host, even for greedy requests. For A/B checks. |
| `--p100e-selftest [builtin\|file.json]` | off | Serve a scripted request set to itself, print the answers, and exit. Without a value, `builtin` is used. The exit code is the test result. |
| `--p100e-parse-test cases.json` | off | Run the tool-call and reasoning parser tests (request plus raw model output to parsed message). Needs `-m`, `--jinja` and a template. No engine is loaded. Exits when done. |

---

## 2. p100e-run flags

Usage: `p100e-run <model.gguf> [options]`. The model file is always the first argument. Options are matched by name.

Three special modes are picked from the arguments before the normal parsing:

- `p100e-run --seq <args> ::: [VAR=v ...] <args> ::: ...` runs this program once per group, one after the other, with optional environment variables for each group.
- Groups separated by `++`, each starting with optional `--env A=1,B=2`, run as a batch on Windows. Each group is a separate run and the summary lists the exit codes.
- If `--mmproj` appears anywhere after the model, the image mode (section 2.9) is used.

### 2.1 Prompt and generation

| flag | default | meaning | example |
|---|---|---|---|
| `--tokens a,b,c` | none | Prompt as token ids. Several prompts are separated by `;`. | `--tokens 1,2,3;4,5,6` |
| `--tokens-file f` | none | Prompt ids from a file (comma or space separated). | `--tokens-file doc.txt` |
| `--tokens-max N` | all | Cut the ids from `--tokens-file` to `N`. | `--tokens-max 4096` |
| `--tokens-files f1,f2` | none | Several prompts, one file each. | `--tokens-files a.txt,b.txt` |
| `--prompts-file f` | none | Several prompts, one per line. | `--prompts-file prompts.txt` |
| `--gen N` | `0` | Number of tokens to generate greedily. (Test modes use 32 when this is 0.) | `--gen 128` |
| `--force a,b,c` | none | Teacher forcing. Feed these ids as the generated tokens, instead of the argmax. `--gen` is their count. | `--force 10,20,30` |
| `--ctx N` | `4096` (image mode: `8192`) | Context size to allocate. | `--ctx 32768` |
| `--debug` | off | Print per-layer sums, to compare with a reference. | `--debug` |
| `--logits file` | none | Write the logits to a file of 32-bit floats. | `--logits out.f32` |
| `--devices a,b,..` | none (single-GPU reference path) | GPUs for tensor parallel with resident weights. 1 to 4 indices (3: uneven shares). Needed for MTP, benches, caches and sessions. `0,0` runs two virtual ranks on one GPU, for tests. | `--devices 0,1` |
| `--split a,b,..` | equal | Relative tensor-parallel share per `--devices` GPU (sets `P100E_SPLIT` for the run). | `--split 2,1,1` |
| `--net-nodes h:p,h:p,..` | none | Network tensor parallel: the machines of the run in global rank order (node 0's `--devices` are ranks 0.., then node 1's ...). Every node runs the same command; node i listens on its own port for the later nodes and connects to the earlier ones. With it `--split` lists every node's GPUs. Plain prompt + `--gen` and `--bench pp tg` only. Env `P100E_NET_NODES`. | `--net-nodes 192.0.2.10:29500,192.0.2.11:29501` |
| `--net-node i` | `0` | Which entry of `--net-nodes` this machine is. Env `P100E_NET_NODE`. | `--net-node 1` |
| `--net-logits` | off | Network mode: every decode step gathers the full logits on every node (for `P100E_PRINT_LOGITS` hashes). Default: only the greedy id's argmax pairs cross the network. | `--net-logits` |
| `--pipe-layers k\|auto` | off | Hybrid pipeline over the 2 machines of `--net-nodes` (instead of tensor parallel over them): node 0 runs layers 0..k-1 split over ITS `--devices` (its own `--split`, local P2P allreduce, graphs, prefill pipeline), node 1 runs layers k..L-1 + the LM head over its `--devices`. Per step one residual row (E x 4 bytes, 20 KB on the 27B) goes 0 -> 1 and the greedy id (or the logits: `--net-logits`, the prompt) comes back. `auto` = the most layers node 1 holds. Both nodes print the byte plan at load; a node that does not fit is refused. Both nodes pass the same value. | `--pipe-layers 49` |
| `--pipe-split a,b` | off | Same as `--pipe-layers a`, and checks a + b = the model's layer count. | `--pipe-split 49,15` |
| `--kv f16\|q8_0\|q4_0` | `q8_0` | KV cache type. A comma list is allowed with `--kld-base` (one run per type). | `--kv q4_0` |
| `--ubatch U` | whole prompt | Largest tokens per forward pass. Longer prompts run as chunks. | `--ubatch 2048` |

### 2.2 MTP drafting

| flag | default | meaning | example |
|---|---|---|---|
| `--mtp K[,K2..]` | off | MTP speculative greedy decoding. Runs the plain greedy baseline, then each `K`. Prints speed, acceptance and `IDENTICAL` or `MISMATCH` against the baseline. Needs `--devices`. | `--mtp 1,2,3` |
| `--mtp-adaptive tau[,tau2..]` | `0` (fixed `K`) | Draft up to `K`. Check draft `j` only while the product of draft probabilities is at least `tau`. A list runs one sweep per value. | `--mtp 3 --mtp-adaptive 0.5` |
| `--mtp-kmin m` | `1` | Smallest number of drafts per step in adaptive mode. `0` allows a plain one-token step when the first draft is weak. | `--mtp-kmin 0` |
| `--mtp-tau1 t` | `0` | Sets `kmin` to 0 and requires the first draft to have probability at least `t`. | `--mtp-tau1 0.8` |
| `--draft-vocab N[,N2..]` | built-in 32768 | Draft head scores only the `N` most frequent tokens. `0` is the full head. A list runs one pass per value. | `--draft-vocab 0,32768` |
| `--load-mtp` | off | Load the MTP block and draft buffers, but do not use MTP. For memory checks. | `--load-mtp` |

The library default mode (what the server uses) is `--mtp 3 --mtp-adaptive 0.5`. In `p100e-run`, `--mtp K` is fixed `K` unless you add `--mtp-adaptive`.

### 2.3 Benchmarks

| flag | default | meaning | example |
|---|---|---|---|
| `--bench PP TG` | off | Time a `PP`-token prefill and `TG` decode steps (synthetic tokens, or the first prompt). | `--bench 2048 64` |
| `--bench-depth D1,D2,..` | `0` | With `--bench`: for each depth, fill the context first (not timed), then time prefill and decode. | `--bench-depth 0,8192,32768` |
| `--bench-real-max N` | all | Fill depths with real forward passes up to `N` tokens. Deeper rows are synthetic. | `--bench-real-max 32768` |
| `--ab "VAR=v,VAR=v/VAR=v"` | off | In-process A/B of environment switches. Arms are separated by `/`, pairs by `,`. An empty arm means defaults. Works with `--bench`, or with a prompt plus `--gen`. | `--ab "P100E_PF_WIRE16=0/P100E_PF_WIRE16=1"` |
| `--ab-reps N` | `3` | Repetitions per arm. | `--ab-reps 5` |

### 2.4 Quality check (KL divergence)

| flag | default | meaning |
|---|---|---|
| `--kld-base file` | off | Compare against a llama.cpp `--kl-divergence-base` file. Reproduces the perplexity tool's evaluation. Needs `--devices`. |
| `--kld-chunks N` | all | Number of chunks to score. |
| `--kld-dump file.bin` | off | Write per-token (KLD, p_diff) float pairs. With a `--kv` list, one file per type (`file.q8_0.bin`). |
| `--kld-threads N` | auto | Host threads for the statistics. |
| `--kl-pair` | off | Paired KL mode over the given prompts. |
| `--wire-scan` | off | Scan prompts for the largest prefill exchange value and its margin to the f16 limit. Use with `--prompts-file` or `--tokens`. |

### 2.5 Sessions (several chats in one engine)

| flag | default | meaning |
|---|---|---|
| `--sessions n1[,n2..]` | off | Run each prompt alone, then `n` concurrent sessions per count. Prints per-session and total speed and `IDENTICAL`/`MISMATCH`. |
| `--sessions-sync` | off | Synchronise session steps. |
| `--sessions-pool N\|auto` | `0` (default pool) | KV pool size in tokens, or `auto`. |
| `--reserve-mb a,b` | `0,1024` | VRAM reserved per GPU, in MiB. |
| `--max-active N` | `0` | Maximum active sessions. |
| `--offload-test N` | off | Offload and restore test with `N` sessions. |
| `--draft-rows N` | `4` | Row budget for shared drafts. |
| `--decode-per-chunk N` | `1` | Decode steps per prompt chunk. |
| `--decode-share x` | `0.5` | Share of time for decoding while a prompt is read. |

### 2.6 Prompt cache tests

| flag | default | meaning |
|---|---|---|
| `--cache-test` | off | Run the prompt cache tests on `--tokens-file`: A, A+B, multi-turn replay and a disk round trip. Compares each cached run with its cold run. |
| `--cache-a N`, `--cache-b N` | test defaults | Sizes of the two prompts. |
| `--cache-turn0 N`, `--cache-step N`, `--cache-turns K` | test defaults | Multi-turn replay start, step and number of turns. |
| `--cache-gen N` | test default | Tokens generated per turn. |
| `--cache-dturns K`, `--cache-dgen N` | test defaults | Decoded-token reuse replay. |
| `--cache-no-disk` | disk on | Skip the disk round trip. |
| `--cache-no-variant` | variant on | Turn off the split variant. |
| `--cache-no-offload` | on | Skip offloading turns. |
| `--cache-save-ckpts N` | test default | Checkpoints to save to disk. |
| `--cache-dir dir` | temp | Folder for session files. |
| `--cache-only name` | all | Run only one named test. |
| `--cache-inexact` | off | Allow reuse at checkpoints with different chunk boundaries. |
| `--ckpt-every N` | test default | Checkpoint every `N` rows. |
| `--min-split N` | test default | Smallest split for a checkpoint. |
| `--cache-child sess expect` | internal | Used by the test to start the second process of the disk round trip. |
| `--sched-test` | off | Scheduler regression tests. |
| `--sched-pool N` | `0` | Pool size in tokens for `--sched-test`. |
| `--fair-test N` | off | Decode speed of one session while an `N`-token prompt is read. |
| `--fair-shares 0,0.5` | test default | Decode shares to compare in `--fair-test`. |

### 2.7 Image mode

Starts when `--mmproj` is present.

```
p100e-run model.gguf --devices 0,1 --mmproj mmproj.gguf --image photo.png --chat "What is in this picture?" --gen 128
```

| flag | default | meaning |
|---|---|---|
| `--mmproj f` | none | Vision projector GGUF. |
| `--image f` | none | Image file. Repeat for several images. |
| `--chat text` | none | One user turn, with thinking off. Images are placed before the text. |
| `--chat-file f` | none | Same as `--chat`, text from a file. |
| `--prompt-file f` | none | A fully rendered prompt with one `<__media__>` marker per image. |
| `--gen N` | `64` | Tokens to generate. |
| `--ctx N` | `8192` | Context size. |
| `--ubatch U` | `2048` | Chunk size. |
| `--kv type` | `q8_0` | KV type. |
| `--vision-dev N` | last of `--devices` | GPU for the encoder (mtmd path). |
| `--vision-dlls dir` | auto | Folder with the llama.cpp fork DLLs (also env `P100E_VISION_DLLS`). |
| `--vision-model f` | the model | Vocab-only GGUF used for tokenizing. |
| `--image-min-tokens N` | `1024` | Fewest tokens per image. |
| `--image-max-tokens N` | `2048` | Most tokens per image. |
| `--detail low\|high` | high | Per-image cap. `low` uses the minimum. |
| `--image-cap N` | `0` (off) | Hard cap on image tokens. |
| `--repeat-penalty r` | `1.0` | Repeat penalty. |
| `--repeat-last-n n` | `64` | Window for the repeat penalty. |
| `--top K` | `0` | Print the top `K` log-probabilities for each generated step. |
| `--runs N` | `1` | Repeat the whole request `N` times (images come from the cache after run 1). |
| `--vision-cache-mb N` | `2048` | Encoder output cache in MiB. `0` turns it off. |
| `--encode-only` | off | Run only the encoder. |
| `--encode-bench N` | `0` | Encode every image `N` more times with the cache cleared. |
| `--overlap-bench n` | `0` | Decode `n` text tokens while the encoder runs on another thread. |
| `--vision-verbose` | off | More encoder logging. |
| `--no-fold` | fold GEMM on | Use cuBLAS fp16 matmuls for the mtmd encoder instead of the fold GEMM. |
| `--model-first` | off | Load the model before the encoder, like the server does. |
| `--mtp K[,K..]`, `--mtp-adaptive tau` | off | MTP greedy run after the plain run, ids compared. |
| `--turn2 text` | none | A second text-only user turn on the same context. |
| `--depth D` | `0` | Timing at context depth `D` (output is meaningless). |

### 2.8 Other programs

| program | usage |
|---|---|
| `p100e-info model.gguf [--tensors\|--vocab]` | Print metadata and the quant histogram. `--tensors` lists every tensor. `--vocab` dumps the tokenizer in hex. |
| `p100e-dequant model.gguf outdir [rows]` | GPU-dequantise one tensor per quant type (default 64 rows) for comparison with a reference. |
| `p100e-reinit-test model.gguf --devices 0,1 [--loads 3] [--prompt 2048] [--gen 8]` | Load the model several times in one process and check the results are identical. Exit 0 means `REINIT PASS`. |
| `p100e-tp-ntp-test model --configs "0\|0,0\|0,0,0\|0,0,0@2,1,1\|0,0,0,0"` | Check tensor parallel over 1 to 4 ranks; `@a,b,..` sets the split weights (virtual ranks need `P100E_P2P_SAME_DEV=1`). |

---

## 3. Environment variables

All names start with `P100E_`. The "who reads it" column says where the variable has an effect: **server**, **run** (`p100e-run`), or **engine** (both, because the engine is shared).

### 3.1 Server, memory and cache

| variable | default | who | meaning |
|---|---|---|---|
| `P100E_DECODE_SHARE` | `0.5` | server, run | Decode share while a prompt is read. Overrides `--p100e-decode-share`. |
| `P100E_SPLIT` | equal | engine | Tensor-parallel weight per GPU, e.g. `2,1,1`. Used when `--p100e-split` / `--split` is not given. |
| `P100E_NET_CONNECT_S` | `600` | engine | Network tensor parallel: seconds to wait for every node to come up at start. |
| `P100E_NET_TIMEOUT_S` | `60` | engine | Network tensor parallel: a collective with no progress this long fails (peer hung or unreachable). A peer process that exits is reported at once. |
| `P100E_NET_PREADD` | `1` | engine | Network mode: node 0 sends one pre-added block `(x + p0) + p1 ..` instead of one per rank (decode; prefill on the f32 wire only). Bit-exact. Must be the same on every node (checked). |
| `P100E_NET_GRAPH` | `1` | engine | Network mode: T = 1 decode steps run as CUDA graphs, one per segment between network exchanges. Bit-exact; per node. |
| `P100E_NET_SPIN` | `1` | engine | Network mode: busy-poll the sockets while waiting for a peer (`P100E_NET_SPIN_US`, default 5000 µs after the last progress, then sleep in poll). Per node. |
| `P100E_NET_KEY_FILE` | unset | engine, fn | Path of a file holding a pre-shared key (at least 16 bytes; trailing newlines ignored; never printed). Set: every TCP node link (`net_link`) and the remote expert tier (`--rexp-listen` / `p100e-fn-expert-server`) run a mutual HMAC-SHA256 challenge after connecting (src/net_auth.h); a peer without the key is dropped and the listener keeps waiting. Authenticates the peer, does not encrypt. Must be the same file on every node. Unset: no handshake, as before, plus one warning line. |
| `P100E_NET` | `tcp` | engine | Network mode transport. This build supports `tcp` only. Must be the same on every node (checked). |
| `P100E_PROMPT_ORDER` | `srf` | server | `srf`: the slot with the fewest prompt rows left runs next. `first`: first come, first served. |
| `P100E_CACHE` | `1` | engine | `0` turns the prompt cache off in the scheduler. |
| `P100E_CACHE_EXACT` | scheduler default | engine | `1` requires exact reuse; `0` allows reuse with different chunk boundaries. |
| `P100E_HOST_BUDGET_MB` | `20480` | engine | Pinned host RAM for offloaded entries, in MiB. Wins over `--p100e-host-budget-mb`. |
| `P100E_CKPT_BUDGET_MB` | `8192` | engine | Checkpoint budget in MiB. |
| `P100E_LCP_CKPT` | `1024` | server | Make a checkpoint at the longest prefix the prompt shares with any cache entry, when it is at least this many rows past the reuse point. `0` is off. |
| `P100E_CKPT_EVERY` | `8192` | server | Checkpoint every `N` prefilled rows. `0` is off. |
| `P100E_CKPT_ROOM` | `1` | server | A checkpoint that finds no room recycles the least recently used one instead of being skipped. |
| `P100E_TAKEOVER_TAIL` | `8192` | server | Cut an idle entry in place when the rows past the shared prefix are at most this many. `0` always copies. |
| `P100E_TAKEOVER_ANY` | `1` | server | `1` (default, the earlier rule): any entry within the tail can be cut in place. `0`: only an entry the prompt still agrees with past its last message start (the same conversation), so two chats that only share a long system / tools prefix are copied and stop cutting each other's tails (more host RAM for the copies). |
| `P100E_PF_CKPT_SPARE` | `2` | server | Spare checkpoint slots kept during prefill. |
| `P100E_PF_CKPT_ASYNC` | `1` | server | Take prefill checkpoints asynchronously. |
| `P100E_PF_CKPT_CHECK` | `0` | server | Verify checkpoints (debug). |
| `P100E_DRAFT_BATCH` | `1` | engine | Draft for all finished slots in one batched pass. `0` drafts per slot. |
| `P100E_DRAFT_TRIM` | unset | server | `conf`: when too many drafts are queued, cut the one with the lowest draft probability first. Default cuts the longest list first. |
| `P100E_METRICS_LOG` | unset | server | Path of a CSV file. One row at start, one per minute while a slot is busy, and one when the last slot finishes. Keeps counters across restarts. |
| `P100E_NGRAM_DRAFT` | `0` (off) | server | Experimental n-gram drafting from the chat history. A number sets the minimum suffix-match length (8, 12 or 24). |
| `P100E_NGRAM_SHADOW` | `1` | server | Telemetry only: measure what an n-gram drafter would have proposed. Never changes output. `0` turns it off. |
| `P100E_PAGED` | `1` | engine | `0` uses contiguous KV caches (one sequence only, no batching). |
| `P100E_PAGE_SHIFT` | `8` | engine | KV page size is `2^N` positions. Allowed 5..12. |
| `P100E_PF_RING_MB` | `640` | engine | Size of the f16 weight ring used by the prefill pipeline, in MiB. |

### 3.2 Vision

| variable | default | who | meaning |
|---|---|---|---|
| `P100E_NVIT` | native on | server, run | `0` uses the llama.cpp mtmd encoder instead of the native encoder. Unset or `1` uses the native encoder when the projector is supported. |
| `P100E_NVIT_DEVICES` | `0,1` | server, run | GPUs for the native encoder. |
| `P100E_NVIT_BORROW` | `1` | server, run | `0` stops the encoder from borrowing scratch memory from the prefill ring. |
| `P100E_NVIT_SCRATCH_WAIT_S` | `600` | engine | Seconds to wait for the prefill ring when VRAM is short. |
| `P100E_NVIT_PACE` | `1` | engine | Pace the encoder so it does not stall decoding. |
| `P100E_VISION_DLLS` | auto | server, run | Folder with the llama.cpp fork DLLs for the mtmd path. |
| `P100E_VISION_GGML_ENV` | clean | engine | `keep` keeps inherited `GGML_*` variables (except `GGML_CUDA_P2P`). |
| `P100E_MM_PACK` | `1` | run | `0` runs one forward per piece instead of one forward over the whole text-and-image prompt. |
| `P100E_MROPE_SEQ` | `0` | engine | `1` puts image rows at 1-D positions (debug). |
| `P100E_MEM_POLL_MS` | off | run | Poll free VRAM every `N` ms in image mode and print `MEM` lines. |

### 3.3 MTP and drafting

| variable | default | who | meaning |
|---|---|---|---|
| `P100E_DRAFT_VOCAB` | `32768` | engine | Draft head scores this many top tokens. `0` is the full head. |
| `P100E_DRAFT_FUSE` | `1` | engine | `0` runs the draft block per operation instead of fused. |
| `P100E_DRAFT_GRAPH` | `1` | engine | `0` runs the draft block without CUDA graphs. |
| `P100E_MTP_KV_ONLY` | `1` | engine | `0` uses the old full prompt catch-up for the draft head. |
| `P100E_MTP_CARRY_ZERO` | `0` | engine | `1` zeroes the draft carry at each turn (old behaviour). |
| `P100E_RB_KERNEL` | `1` | engine | `0` rolls back with plain copies instead of one gather kernel. |
| `P100E_RB_ASYNC` | `1` | engine | `0` makes the rollback run in the main stream. |
| `P100E_FUSE_VERIFY` | `1` | engine | `0` verifies drafts through the per-operation path. |
| `P100E_GDN_BATCH` | `1` | engine | `0` runs the delta net one token at a time with snapshot copies. |
| `P100E_MTP_CHECK` | off | run | Compare the logits of every verify row with plain greedy decoding. Prints mismatch counts. |
| `P100E_MTP_PROF` | off | run | Print prompt and catch-up times. |
| `P100E_MTP_CALIB` | off | run | Print the draft probability calibration tables. |
| `P100E_SESS_ORACLE` | `0` | run | Test mode: drafts are the expected tokens. |

### 3.4 Engine switches (tuning and debug)

These select between kernels or code paths. Defaults are the tested, fastest choice. Change them only to compare speed or to find a bug. Several can be compared in one process with `p100e-run --ab`.

| variable | default | meaning |
|---|---|---|
| `P100E_FAST` | `1` | `0` turns off the fast matvec path (reference kernels). |
| `P100E_FUSE` | `1` | `0` runs one-token decode through the per-operation path. |
| `P100E_CONCAT` | `1` | `0` uses one launch per weight instead of concatenated weights. |
| `P100E_CONCAT_T` | `1` | `0` uses one in-projection launch per weight for batched (T above 1) steps. |
| `P100E_VERIFY_CAT` | `1` | Same effect as `P100E_CONCAT_T=0` when `0`. |
| `P100E_MERGE_INPROJ` | `0` | `1` merges the mixed-type one-token in-projections into one launch. |
| `P100E_GDN_FUSE` | `0` | `1` fuses the one-token delta-net prep and scan. |
| `P100E_Q8_FORK` | `0` | `1` runs the small q8 in-projection on a side stream. |
| `P100E_GRAPH` | `8` | Steps with at most this many rows replay a CUDA graph. `0` is eager. |
| `P100E_GRAPH_QUIET` | off | Set to hide CUDA-graph log lines. |
| `P100E_STREAM_PRIO` | `1` | `0` turns off high priority for the main streams. |
| `P100E_P2P_AR` | `1` | `0` uses an event-based allreduce in the fused decode. |
| `P100E_AR_SAFE` | `0` | `1` uses the older, slower allreduce handshake. |
| `P100E_AR_SIG_FUSE` | `0` | `1` fuses the allreduce signal (2 GPUs only). |
| `P100E_AR_FENCE` | `0` | `1` adds a system-wide fence before the signal (costs about 1.3 percent decode). |
| `P100E_AR_SPIN_TIMEOUT_MS` | `2000` (`500` on a GPU with a kernel watchdog) | Wait limit for the GPU allreduce. On timeout the server exits with code 3. |
| `P100E_AR_TEST_DROP` | `0` | Fault injection: stop one GPU's signals from its N-th on. |
| `P100E_AR_TEST_DROP_RANK` | `1` | The rank used by `P100E_AR_TEST_DROP`. |
| `P100E_P2P_SAME_DEV` | `0` | Allow virtual ranks on one physical GPU (tests). |
| `P100E_FAST_MAXT` | `8` | Largest row count that uses the decode (matvec) kernels. Larger uses the prefill GEMM path. |
| `P100E_GEMM` | `1` | `0` disables the GEMM path. |
| `P100E_PF` | `1` | `0` turns off the prefill pipeline. |
| `P100E_PF_ATTN` | `1` | `0` turns off the prefill attention path. |
| `P100E_PF_CHUNKS` | `4` | Number of prefill pipeline chunks. |
| `P100E_PF_PRIO` | `1` | `0` uses default priority for the prefill streams. |
| `P100E_PF_FOLD` | `1` | `0` uses fp32 prefill attention only. |
| `P100E_PF_FOLD_MIN` | `0` | Fold-GEMM attention starts when a chunk sees at least this many keys. |
| `P100E_PF_FOLD_TMIN` | `64` | Short chunks at depth use the simple attention below this size (64 means off). |
| `P100E_PF_FOLD_TMIN_DEPTH` | `0` | Depth from which `P100E_PF_FOLD_TMIN` applies. |
| `P100E_PF_FUSEDQ` | `2` | Fused dequant in the fold GEMM: `0` off, `1` always, `2` only for short prompts. |
| `P100E_PF_FUSEDQ_TMAX` | `32` | Longest prompt for `P100E_PF_FUSEDQ=2`. |
| `P100E_PF_TINY_ATTN` | `1` | `0` turns off the small-prompt attention shortcut. |
| `P100E_PF_RASTER` | `4` | Block order of the fold GEMM. |
| `P100E_PF_PRESCALE_V` | `1` | One-read prescale kernel for V. |
| `P100E_UNPACK` | direct | Weight unpack mode for prefill. |
| `P100E_PF_WIRE16` | `1` | `0` sends f32 partials between GPUs during prefill (exact, slower). |
| `P100E_PF_HOST_XCHG` | `0` | `1` stages the prefill exchange through host memory. |
| `P100E_PF_SASS` | `1` when built with Python | Use the hand-patched register-allocation kernel. `0` off, `3` unpatched. |
| `P100E_PF_SASS_DEFAULT` | build time | Build-time default of the previous switch. |
| `P100E_PF_TEST_SAT` | `0` | Inject N f16 saturation events (tests the redo path). |
| `P100E_PF_WIRE_STATS` | `0` | Print the largest exchanged value per prefill. |
| `P100E_PF_DEBUG` | off | Print prefill pipeline debug lines. |
| `P100E_PREFILL_PROF` | `0` | Profile prefill phases. |
| `P100E_GDN_CHUNKED` | `0` | `1` uses the chunked delta-net kernel for prefill. |
| `P100E_ATTN_DEC` | `2` | Decode attention: `1` selects the older split kernels. |
| `P100E_ATTN_PF` | `4` (auto) | Prefill attention row tiling variant. |
| `P100E_ATTN_OPT`, `P100E_ATTN_2S` | `1` | `0` turns off the optimised or two-stage fold attention. |
| `P100E_DEC2_LEAN`, `P100E_DEC2_SPLIT` | auto | Decode attention v2 tuning. |
| `P100E_EW_THR` | `128` | Element-wise kernel thread count. |
| `P100E_MV_FLOAT` | `1` | `0` sends F32, F16 and BF16 weights to the generic matvec. |
| `P100E_MV_Q41` | `1` | `0` sends Q4_1 weights to the generic matvec. |
| `P100E_MV_Q2KR`, `P100E_MV_IQ1SR`, `P100E_MV_IQ1MR`, `P100E_MV_IQ2XXSR`, `P100E_MV_IQ2XSR`, `P100E_MV_IQ2SR`, `P100E_MV_IQ3XXSR`, `P100E_MV_IQ3SR`, `P100E_MV_Q10`, `P100E_MV_Q20` | `1` | `0` sends that quant type to the generic matvec. |
| `P100E_MV_WRM` | `1` | `0` uses the older multi-column matvec. |
| `P100E_MV_WRM_GEO`, `P100E_MV_WRM_KPG`, `P100E_MV_WRM_PH`, `P100E_MV_WRM_NC`, `P100E_MV_WRM_WGEO`, `P100E_MV_WRM_WROWS`, `P100E_MV_Q8_GEO`, `P100E_MV_VARIANT`, `P100E_MV_WRC` | auto | Geometry and variant overrides for the matvec kernels. Same results, different speed. |
| `P100E_MV_Q8_BIG` | `215` | q8_0 GEMV at T=1 on shards with more than 2048 rows and at most 5120 inputs (ssm_out on Q6_K): `215` = q8wr nw4 rw2 staged (default), `0` = the old legacy kernel, `211`..`218` = other q8wr shapes. Same results, different speed. |
| `P100E_MV_OLD_LAYOUTS` | `0` | `1` uses older weight layouts. |
| `P100E_Q6_PAD` | `1` | Q6_K weights whose rows are 4- but not 8-byte aligned (the 2-GPU `ffn_down` shard of Qwen3.8-27B: n_in 8704, 7140 bytes per row) get each row padded to an 8-byte multiple (7144 bytes, +4 bytes per row, about 20 KB per layer per GPU) so they use the faster q6r3 layout and the warp-rows matvec kernels instead of the legacy q6r2 kernel. The pad bytes are zero and never read as weights. `0` keeps the old layout (output bit-identical to builds before this switch). `16` pads to 16 bytes instead (same output bits as `1`). With the padding on, the output is not bit-identical to `0` (a different summation order, as when the warp-rows kernels were adopted). Ignored with `P100E_MV_OLD_LAYOUTS=1`. A value that is not a number is ignored with a warning (the default stays). Read once. |
| `P100E_MV_Q4R_TINV` | `0` | Q4_0 (q4r) warp count. `0`: 4 warps at T = 1, 2 for T >= 2. `1`: 4 warps up to T = 4, so MTP verify columns equal the T = 1 result bit for bit (same speed). `2`: 4 warps at every T (T-invariant batched decode; 12-25% slower at T = 8). Read once. |
| `P100E_MV_Q6_H16` | `2` | Q6_K (q6r3) HFMA2 dot tail (`src/fast/matvec_dec_q6h.cuh`). The Q6_K path multiplies in fp16 (HFMA2 chains of 8 products per half lane); `0` changes to fp32 per 16 weights (the dec_q6r3 tail; the dec_q6r3 bits). `1`: the two half lanes are added in fp16 first (2 conversions instead of 4). `2` (default): the int8 sub-block scales are applied in fp16 too (sc/4, exact), one conversion per 32 weights, no int-to-float conversions. `1` and `2` change the numerics vs `0` and run every Q6_K matvec, T = 1 included, on the multi-column kernel, so MTP verify columns stay bit-identical to T = 1. P100 gate for `2`: MTP 73.53 -> 76.36 tok/s (+3.85%), plain decode flat, decode-path KLD vs the Q8_0 reference 0.006516 (`0`: 0.006757), top-1 about 97.9%. Prefill (T > 8, the GEMM) is not affected. No fp16 overflow below |activation| of about 130000 (bound in the header; the largest seen live is 467). While `1` or `2` is on, `P100E_MV_WRM=0`, positive `P100E_MV_VARIANT` ids and `P100E_MVWRP_Q6` have no effect on Q6_K (q6r3); `P100E_MV_VARIANT=-1` still selects the legacy kernels with the `0` tail. Any other value is ignored with a warning (the default stays). Read once. |
| `P100E_Q8_TN` | `1` | `0` turns off an alternative q8 matvec kernel variant (name guessed from the source; effect not confirmed). |
| `P100E_IQ1_BLOCKS_PER_SM`, `P100E_Q10_BLOCKS_PER_SM`, `P100E_Q20_BLOCKS_PER_SM` | `2` | Blocks per SM for those kernels. |
| `P100E_IQ1_R`, `P100E_Q10_R`, `P100E_Q20_R` | `0` (auto) | Rows per warp (2, 4 or 8). |
| `P100E_WPAD` | `0` | Extra bytes of weight padding (tests). |


### 3.5 Diagnostics and test hooks

| variable | default | who | meaning |
|---|---|---|---|
| `P100E_TOPK` | `0` | run | Print the top `N` logits of the prompt and each generated step. |
| `P100E_PRINT_LOGITS` | `0` | run | Print logits. |
| `P100E_LOGITS_DUMP` | unset | run | File: every step's logits (prompt, then each `--gen` step) appended as raw f32, for max \|diff\| / top-1 comparisons between runs. |
| `P100E_LOGITS_HASH` | off | engine | Print a hash of the logits of each step (for bit-exact comparisons). |
| `P100E_DEBUG_NAN` | off | engine | Check for NaN values after steps. |
| `P100E_FAST_CHECK` | `0` | engine | Check fast-path results against the reference. |
| `P100E_WHASH` | `0` | engine | Print a hash of the loaded weights. |
| `P100E_BENCH_WARMUP` | `3` | run | Seconds of untimed warm-up before the first timed bench. `0` is off. |
| `P100E_KLD_HEAD` | decode GEMV | run | `gemm` scores KLD rows through the prefill GEMM. |
| `P100E_NVIT_TEST_OOM` | `0` | engine | Test hook: pretend the encoder is out of memory. |
| `P100E_FAKE_SEQS`, `P100E_FAKE_FATAL_AT` | unset | fake server | Used only by `p100e-server-fake` tests. |

### 3.6 Flash-Next engine (`--p100e-engine fn`, experimental)

| variable | default | meaning |
|---|---|---|
| `P100E_FN_KV` | `f16`, or `q8` when the context is over 32768 | KV type: `f32`, `f16` or `q8`. |
| `P100E_FN_TP` | `1` | `0` replicates dense attention and delta-net blocks instead of splitting by heads. |
| `P100E_FN_MTP` | `1` | `0` does not load the draft block. |
| `P100E_FN_QSA` | `1` | `0` forces dense attention instead of the sparse (indexer) path. |
| `P100E_FN_QSA_FORCE` | `0` | `1` forces the sparse path. |
| `P100E_FN_ADAPT` | `1` | `0` turns off adaptive expert swapping. `P100E_FN_ADAPT_EVERY`, `_MAX`, `_MIN` tune it. |
| `P100E_FN_PIN` | `1` | `0` uses pageable host memory (for PCs short on RAM). |
| `P100E_FN_STAGE_MB` | engine default | Staging buffer size in MiB. |
| `P100E_FN_PREFETCH_T` | engine default | Prefetch threshold in rows. `0` is off. |
| `P100E_FN_FOLD_MIN`, `P100E_FN_FOLD_MAX` | `16`, `48` | Row range for the fold kernels. |
| `P100E_FN_ATTN_T` | `5` | Row threshold for attention kernels. |
| `P100E_FN_ATTN_FAST` | `0` | `1` runs dense decode / MTP-verify attention (fewer rows than `P100E_FN_ATTN_T`) on an online-softmax kernel with 16 warps over the keys instead of the serial reference kernel. Not bit-identical to `0` (fp32 rounding: max abs diff under 5e-7 in `tests/ref_attn_fast_test.cu`). About 10-40x faster per call at 0.5K-8K keys on a 3070. |
| `P100E_FN_GEMM_T` | `0` (engine picks 64) | Minimum rows for GEMM. |
| `P100E_FN_GROUPED`, `P100E_FN_HOST_ROUTE`, `P100E_FN_GDN_REF`, `P100E_FN_LOAD_PREFETCH`, `P100E_FN_PLE_PREFETCH` | see source | Kernel and loading options. |
| `P100E_FN_TIME`, `P100E_FN_LOAD_DEBUG`, `P100E_FN_ADAPT_DEBUG` | off | Timing and debug output. |
| `P100E_RAM_RESERVE_MB` | `12288` Windows, `6144` Linux | RAM the byte plan leaves to the OS and other programs. A load whose pinned host memory would cut into it is refused (needed vs available vs reserve). `p100e-fn-run --ram-reserve-mb N` sets it per run. |
| `P100E_FN_HOST_FREE_MB`, `P100E_FN_GPU_FREE_MB` | measured | Test overrides for the byte plan: available RAM, and a cap on free VRAM per rank (comma separated MiB). |
| `P100E_FN_MTP_FORCE_ACC` | unset | Test hook: MTP accept counts cycle through this list (the output is then not greedy). |
| `P100E_FN_RUN_HASH` | `0` | `p100e-fn-run`: print a hash of every returned logit row (bitwise identity checks). |

**Flash-Next hybrid pipeline (`p100e-fn-run`).** `--net-nodes host0:port,host1:port --net-node 0\|1 --pipe-layers k\|auto` (or `--pipe-split a,b`): node 0 loads layers 0..k-1 + the embedding and serves them; node 1 loads layers k..47 + head + MTP draft block and runs the prompt and generation (`--mtp` works across the pipe). Per forward one command goes 0 <- 1 and the residual rows [T][4][2560] f32 come back. `--slots a,b\|auto` sets resident experts per layer per local GPU (`auto`: fill each GPU). `--pipe-plan "F0a,F0b:H0/F1a,F1b:H1"` prints the byte plan offline (MiB of free VRAM per GPU and available RAM per node). The server takes `--p100e-fn-pipe-nodes a,b`, `--p100e-fn-pipe-layers k\|auto`, `--p100e-fn-slots a,b\|auto` and is then node 1; node 0 is `p100e-fn-run --net-node 0`. Images are not supported over the pipe.

**Flash-Next remote expert tier.** Node 0 (`p100e-fn-run`) runs the whole model; node 1 (`p100e-fn-expert-server`) holds extra Q4 experts on its GPUs and computes them, one TCP round trip per layer per forward. Off unless `--rexp-listen` is given; off = bit-identical to the CPU expert path.

| Flag / variable | Default | What it does |
|---|---|---|
| `p100e-fn-run --rexp-listen host:port` | off | Node 0 listens there (host `0.0.0.0` = every interface) and waits for the expert server. |
| `--rexp-max-t N`, `P100E_FN_REXP_MAX_T` | `4` | Forwards of at most N rows use node 1 (decode, MTP drafts, MTP verify). Bigger prefill chunks stay local. |
| `--rexp-sum`, `P100E_FN_REXP_SUM=1` | off | Node 1 sends one weighted sum per row instead of one row per (row, expert) pair: ~4x fewer reply bytes on a verify step, output no longer bit-equal (summation order). |
| `--rexp-connect-s S` | `1800` | How long node 0 waits for node 1 to connect / finish loading. |
| `P100E_FN_REXP_ADAPT=0` | on with `P100E_FN_ADAPT` | Node 1's set follows node 0's routing counts (SWAP between forwards, landed swaps reported in replies). |
| `P100E_FN_REXP_ADAPT_MAX`, `_CAP` | `4`, `96` | Swaps per layer / per adapt step. |
| `p100e-fn-expert-server <gguf> --connect host:port` | required | Node 1: the same GGUF file (size + tensor table checked). |
| `--devices 0,1`, `--slots a,b\|auto` | `0`, auto | GPUs and experts per layer per GPU (auto: fill after buffers + margin). |
| `--swap-gpus 0\|0,1\|none` | `0` | Positions in `--devices` that take swaps; the others hold a fixed set loaded once (the 1660 sits on PCIe x4). |
| `--margin-mb N` | `384` | VRAM left free per GPU. |
| `--no-keepalive` | keep-alive on | Without the keep-alive kernel a GeForce card idles in P8 between requests (3070: 210 MHz core, 405 MHz memory; 2.1 ms instead of 0.24 ms of GPU time per request). |
| `--once`, `--verbose`, `--connect-s S` | | Exit after one session; per-GPU timing; connect retry window. |
| `p100e-fn-run --plan "F0,F1:H"` | | One-process byte plan offline for F0, F1 MiB free VRAM and H MiB available RAM (with `--slots`, `--ctx`, `--max-t`, `--ram-reserve-mb`). |
| `P100E_FN_LOAD_DEBUG=2` | off | VRAM after every dense upload (byte plan checks). |
| `P100E_FN_PLE_ASYNC=1` | off | PLE rows: Linux asks for every row's page at once (madvise WILLNEED) instead of one page fault at a time, the prompt prefetch thread does the same, and `generate_mtp` asks for each verify row's PLE rows as soon as its draft is known. Same output bits. |
| `P100E_FN_REXP_EARLY_SHARED=0` | on with the tier | Node 0 runs the shared expert's matvecs while it waits for node 1 (its add stays after the combine; same bits); 0 = after the reply as before. |
| `p100e-fn-expert-server --no-threads` | threads on | Without it each extra GPU is issued and drained by its own spinning worker thread, so both GPUs work at once (localhost: 0.82 -> 0.41 ms per request). |
| `P100E_FN_TIME=1` | off | Now also prints `fn time mtp T=..` per draft-block pass (mtp_in, the layer phases, mtp_head). |
| `P100E_FN_TIME=2` | off | The same phase report without the device-wide sync per phase: events on rank 0's stream, phases = gaps on its GPU timeline (a host wait lands in the phase whose work waited). |
| `P100E_FN_ADAPT_ASYNC=1` | off | Promotions copy + pack on a lowest-priority stream; the old expert leaves the maps at once, the new one joins when its copy is done. Removes the stall in the first pass after an adapt step but not the PCIe bytes (on the x4 1660 the stall moved into the next FFN's staging; untimed 27.8 vs 26.1 t/s). |
| `P100E_REXP_SPIN_US=N` | `0` | Both remote-tier ends spin up to N us on non-blocking reads before sleeping in poll (localhost: node 0 waited 0.61 -> 0.24 ms per request, node 1 0.62 -> 0.37 ms). Costs a busy core on each side while waiting. |
| `P100E_FN_PROF_HC=1` | off | GPU time of each hc_mix kernel on rank 0 (events between them), averages printed at exit. |
| `P100E_FN_ADAPT_CAP=N` | `0` | Promotions per rank per adapt step, the biggest gains of all layers first (0 = only `P100E_FN_ADAPT_MAX` per layer). Bounds the PCIe bytes of each adapt step. |
| `P100E_FN_REXP_HALF=1` | off | Sum mode only: node 1 sends its per-row sums as f16 (half the reply bytes; localhost max logit diff 0.275 vs 0.277 for f32 sums, same ids). |
| `p100e-fn-expert-server --no-one-copy` | one copy on | Without it each GPU's request inputs (x rows, gather index, tables) go up in one copy instead of three (localhost 0.48 -> 0.38 ms per request). |
| `p100e-fn-hc-bench <gguf> --dev N` | | The two hc_mix matvecs alone, T = 1..4 (3070: 19-30 us each; 1660: 30-53 us). |
| `P100E_FN_HC_TILE=1` | off | hc_mix's Q8_0 down matvec (10240 -> 320) as 8-row tiles x 8 K slices with the activations in shared memory, then one fixed-order reduce (src/fn_hc_kernels.cuh). Other summation order. 3070 at T = 4: 23.2 vs 29.9 us; 1660: slower (49 vs 43 us); P100 unmeasured (`p100e-fn-hc-bench --dev N` prints both). |
| `p100e-fn-expert-server --merge-gu` | off | Gate and up groups in one grouped launch per column width (2 launches instead of 4; same bits). |
| `p100e-fn-expert-server --spin-sync` | off | cudaDeviceScheduleSpin on node 1's GPUs: stream syncs spin instead of yielding. With --merge-gu: localhost 0.38 -> 0.34 ms per request. |
| `P100E_FN_REXP_MIN_PAIRS=N` | `0` | Layer passes with fewer than N remote pairs skip node 1 and use the host path (changes output: the host path computes the 2-bit cold copies in adaptive mode). |

---

## 4. Exit codes

| code | meaning |
|---|---|
| `0` | Normal exit. For `--p100e-selftest`, all checks passed. |
| `1` | Bad command line, missing `-m`, or failed to start (model load error, HTTP error). |
| `3` | Fatal engine error (for example a GPU allreduce timeout). The server logs `FATAL engine error` and exits so a supervisor can restart it. |
| `2` | For `p100e-run` batch mode: not every run exited 0. |

---

## 5. Not confirmed

These items could not be fully confirmed from the source and may be wrong:

- Default values of the server's standard llama-server flags (`--host`, `--port`, `-c`, `-ub`) come from the llama.cpp argument parser, which is in a separate source tree. The values here are the usual llama-server ones.
- Defaults for `--cache-*`, `--ckpt-every` and `--min-split` in `p100e-run` are set inside the test code and are listed as "test defaults".
- `P100E_PF_SASS` default depends on whether Python was available at build time.
- `P100E_UNPACK`, `P100E_ATTN_PF`, `P100E_DEC2_*` and the `P100E_MV_*` geometry values are internal numeric codes. Their meanings are in the source comments only.
- The Flash-Next (`fn`) engine is for a different model family. Its variables are listed from the source but were not run.
- `--p100e-parse-test` requirements (template flags) are taken from the code path in `main`.
