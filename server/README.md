# p100e-server: llama-server-compatible HTTP server on the p100e engine

The server keeps llama-server's HTTP API, templates, tokenizer, tool-call and reasoning parsing and sampling, and runs
them on the p100e tensor-parallel engine with MTP, several chats at once, a prompt cache and image input.

## Build

`build.cmd` / `build.sh` build it with everything else (`p100e-server`, one file: static libs, no extra DLLs beyond the
CUDA runtime `p100e-run` already needs). `-DP100E_SERVER=OFF` skips it; `-DP100E_FORK_DIR=<path>` points at the fork
(default `third_party/llama.cpp-fork`, included in this repository). If the fork is not found the server is skipped
with a warning.

The server's "same behaviour as llama-server" only holds at the verified fork commit, which is the included copy. For
an outside fork checkout, use `-DP100E_FORK_CHECK=ON`: `cmake/check_fork.cmake` then fails the configure, and again
every build of the server (target `p100e_fork_check`), unless `git -C <fork> rev-parse HEAD` starts with
`P100E_FORK_COMMIT` and `git status` shows no change under the fork paths the build compiles (modified, staged or
untracked files under `CMakeLists.txt cmake ggml src include common vendor tools/server tools/mtmd licenses LICENSE`;
ignored build trees do not count).

`p100e-server-fake` (`-DP100E_SERVER_FAKE=OFF` skips it) is the same server on a deterministic CPU fake of the engine
(`tests/server/fake_engine.cpp`): slot loop, queue and MTP bookkeeping tests without a GPU. Never deploy it.

How the llama.cpp parts get in (`cmake/server.cmake`):

| piece | source | why |
|---|---|---|
| libllama (vocab-only load), libcommon, mtmd, cpp-httplib, nlohmann | the fork, `add_subdirectory(... EXCLUDE_FROM_ALL)`, static, **ggml CPU-only** | tokenizer, `common_chat_*` and the fork's Jinja engine, tool-call and reasoning parsers, sampler chain, reasoning budget, the llama-server argument parser. CPU-only ggml never touches the GPUs, so there is no second CUDA context next to the engine |
| `server-common / -chat / -task / -queue / -schema / -stream / -http.cpp` | the fork's `tools/server`, compiled unchanged | request parsing (`oaicompat_chat_params_parse`, Anthropic and Responses converters), task params, response formatting (OAI chat, Anthropic SSE, usage, timings), task queue, HTTP (cpp-httplib, threads, 503 while loading, disconnect detection) |
| `server/server_context.cpp` | replaces the fork's `server-context.cpp` | same `server_context` / `server_routes` declarations; the route handlers are the fork's code, the slot loop drives the engine |
| `server/srv_sampler.cpp` | port of `common/sampling.cpp` | `common_sampler` reads logits from a `llama_context`; the port takes a host logit row (same chain, prefilter, grammar, reasoning budget, sample-and-accept) |
| `server/fork/CMakeLists.txt` | wrapper | renames the fork's `ggml_abort` (`p100e_core` defines its own); no p100e object changes |

**Keep ggml-cuda out of this static build.** The vision code (`src/vision`) links the fork's mtmd/clip against a
shared build's import libraries and delay-loads the DLLs on the first image, so `p100e-server` stays a static,
CPU-only-ggml build and text-only runs never load ggml-cuda. Do not switch `GGML_CUDA=ON` in `server/fork`. With a
static ggml-cuda in the same exe, `p100e_core`'s `ggml_cuda_info`, `ggml_cuda_get_device`, `ggml_abort` (`src/types.cu`)
and the conversion functions (`third_party/ggml/convert.cu`) collide with ggml-cuda's, and with static libraries MSVC
may silently link one copy instead of failing.

**Image input.** `--mmproj` loads `src/vision` built as `p100e-vision.dll` (`cmake/vision.cmake`: linked against the
fork's import libraries, itself delay-loaded by the exe; it delay-loads mtmd, llama, ggml and ggml-cuda from
`P100E_VISION_DLLS`, the exe directory or the fork build). It is a DLL, not the static `p100e_vision`, because the
server links the fork's llama, mtmd and ggml statically and the import libraries export the same names, so they must
live in different modules. Put `p100e-vision.dll` next to `p100e-server.exe`; without `--mmproj` neither it nor the
fork DLLs are loaded. Linux links the fork's `.so` files directly (see the top README).

Flow: the HTTP thread runs the fork's `oaicompat_chat_params_parse` (data URIs, http and raw base64 become files, image
parts become the per-process media marker) and keeps `image_url.detail` per image. The task loop encodes when the task
gets a slot (serialized with the engine's GPU work, embedding cache first), sets the task's tokens to the row list
(image rows are `LLAMA_TOKEN_NULL`, so `n_tokens`, the context check and usage count image rows as llama-server does)
and prefills through `engine::prefill_mm` in groups of at most `n_ubatch` rows, images whole. Caps: `detail` "low" uses
`--image-min-tokens` (1024), anything else `--image-max-tokens` (default 2048). Count-tokens endpoints size images from
the decoded size or the cache without encoding. The encoder runs on the last engine GPU (`--p100e-vision-device N`;
CPU with `--no-mmproj-offload` or `--p100e-no-engine`).

## Engine interface (`server/engine.h`)

Plain C++, no CUDA, ggml or llama.cpp types. Per sequence: `seq_open / seq_keep(n) / seq_close`, `prefill(tokens)` (the
server feeds `n_ubatch` chunks so cancel and other tasks are seen between chunks), `decode(reqs)` for every generating
sequence at once (rows = last sampled token plus MTP drafts; per row the host logits, or only the GPU argmax for greedy
requests), `keep(n)` (the sampler accepted n rows, the rest is rolled back), `draft(next, k_max)` (MTP drafts for the
next step) and `prefill_mm(pieces)` (text and image embedding rows, interleaved M-RoPE; the default throws, so the fake
engine has no images). `engine_tp` (`server/engine_tp.cpp`) implements it on `tp_model` and `tp_mtp` through their
public API. `engine_fn` (`server/engine_fn.cpp`) does the same for the Flash-Next model.

## Sampling and MTP

- Greedy requests (temperature <= 0 and nothing in the chain that changes the maximum): GPU argmax, no logits copy.
- Everything else: host logits and the llama.cpp sampler chain (`srv_sampler`).
- MTP (default adaptive k <= 3, tau 0.5) under any sampler uses llama.cpp's sample-and-match rule
  (`common_sampler_sample_and_accept_n`): row i is sampled with the full chain and the step stops at the first sample
  that differs from draft i, so every emitted token is a sample of the target distribution. The accepted run also ends
  at the first end-of-generation token. If a greedy step meets a row the chain can no longer take from the argmax (the
  reasoning budget starts forcing), the step is cut there and the next one copies logits.
- `--p100e-refcheck` reruns every greedy task as plain greedy decoding (one forward per token, host argmax) and logs
  `REFCHECK IDENTICAL / MISMATCH`. **Debug only:** the replay runs synchronously inside `release()` on the server loop,
  so every other request and endpoint waits for it.

## Status endpoints

`/slots` and `/metrics` answer from a snapshot the task loop publishes after every loop iteration (one prompt chunk
and / or one decode step) and after every task it handles (`server/live_status.h`). The HTTP threads copy it under a
mutex and never wait for the loop, so they answer in milliseconds mid-prefill and mid-decode. `/props` and `/health`
never touch the loop. Shapes stay the fork's; additions:

- `/slots`: `n_prompt_tokens` = the whole prompt (text tokens and image rows, known at admission), `state` (idle /
  started / processing_prompt / generating), `p100e` (rows in the engine including cached, context used, unix-ms
  timestamps, prompt and generation ms and t/s, drafts). An idle slot keeps its last task's numbers.
- `/metrics`: `p100e:slot_busy|slot_prompt_tokens|slot_prompt_tokens_processed|slot_prompt_tokens_cached|
  slot_tokens_decoded{slot="i"}`, `p100e:slots_busy`, `p100e:prompt_tokens_processing`, `p100e:tokens_decoding`,
  `p100e:tokens_predicted_live_total`, `p100e:status_snapshot_age_seconds`, `p100e:prompt_cache_*`, `p100e:kv_pool_*`.
- `/props` `build_info`: `p100e-<commit>[-dirty]` plus the fork id (`cmake/build_stamp.cmake`).

## Failure handling

- f16 prefill wire: if a prefill clamps a row-parallel partial to +-65504 (`tp_model::last_prefill_saturated`),
  `engine_tp::prefill` resets and redoes the whole prompt fed so far on the f32 wire inside the same call (one stderr
  line, `engine_stats::n_prefill_redo`), so a prefill never leaves, and a prompt cache never stores, clamped state.
  `P100E_PF_TEST_SAT=1` exercises the path. Image prompts too: `prefill_mm` redoes every piece fed so far.
- P2P allreduce timeout (`tp_model::ar_failed()`: the instance is unusable): any engine call then throws
  `engine_fatal_error`; the server logs `FATAL engine error` and exits with code 3 so its supervisor restarts it.
  `P100E_AR_TEST_DROP=200` on GPUs and `P100E_FAKE_FATAL_AT=N` on the fake engine trigger it. Other CUDA errors fail
  only the request.

## Several chats at once and the prompt cache

`server/engine_tp.cpp` runs on `tp_model`'s sequences (a paged KV pool shared by all): `-np N` slots (default 3 when
`-np` is not given) prefill and decode concurrently. Every decode step is one `forward_batch_ids` over all generating
slots (at most 8 rows: the server trims the longest draft lists first), each row bitwise the row of that sequence
alone, logits copied only for sampling rows. `-c` is the total over the slots: the per-slot cap is `-c / -np`. The KV
pool itself takes the free VRAM after load (1024 MiB kept on the last GPU for the image encoder).

Prompt cache (`--p100e-cache 1`, default; per request `cache_prompt`): a closed slot's sequence stays as a cache entry
(resident, `--p100e-cache-resident 4` extra state slots; then offloaded to pinned host RAM,
`--p100e-host-budget-mb 20480`; dropped LRU last). A new request is matched by row keys (token ids; image rows by
image hash, so different images never match) against every idle entry. The reuse point is the entry's end (the same
conversation resent, decoded tokens included: the entry is taken over) or one of its checkpoints (delta-net state and
draft carry in pinned RAM, `--p100e-ckpt-budget-mb 8192`) taken at the prompt's last 4 message starts. A shared prefix
at a checkpoint is copied into a new sequence (`tp_model::seq_fork`), so another chat never takes over a conversation's
entry. Cached rows are reported as `cached_tokens`. A reused prefix was computed with other chunk boundaries than a
cold run of the same prompt, so greedy text may differ from the cold run within rounding. `P100E_FAKE_SEQS=3` runs the
fake engine with 3 sequences and the cache.

Latency switches, read at start:

| switch (default) | what it does |
|---|---|
| `P100E_LCP_CKPT=1024` | a checkpoint at the longest prefix the prompt shares with any cache entry, when that is at least N rows past the reuse point (0 = off) |
| `P100E_CKPT_EVERY=8192` | a checkpoint every N prefilled rows (0 = off) |
| `P100E_CKPT_ROOM=1` | a checkpoint that finds no room recycles the least recently used one of any entry, or drops the LRU offloaded entry, instead of being skipped |
| `P100E_TAKEOVER_TAIL=8192` | reuse inside an idle entry whose rows past the shared prefix are at most N: cut it in place instead of copying the prefix (0 = always copy) |
| `--p100e-decode-share 0.5` / `P100E_DECODE_SHARE` | while a slot prefills, generating slots get this share of the wall time (0 = one step per chunk) |
| `P100E_PROMPT_ORDER=srf` (`first` = old order) | the prompting slot with the fewest rows left runs the next chunk |

## Differences from llama-server

- `/v1/messages` honours `reasoning_budget_tokens` / `thinking_budget_tokens` at the top level of the body; llama.cpp's
  Anthropic converter drops them.
- Not supported (explicit errors): audio and video input, embeddings, rerank, LoRA, slot save / restore, `n_cmpl > 1`.
  `/props` reports `modalities.video false`.
- No sleeping (`sleep_idle_seconds` is forced off: the weights stay resident).

## Running

```
p100e-server -m model.gguf --host 127.0.0.1 --port 8080 -c 262144 -np 1 -ub 2048 --jinja ^
    [--log-file ... --log-timestamps --log-prefix]
```

llama-server flags are parsed by the fork's own parser (`-c` = total context over `-np` slots). Backend flags
(`--split-mode`, `-ngl`, `--flash-attn`, `--spec-*`, ...) are accepted and ignored. Every p100e flag is listed in
[docs/FLAGS.md](../docs/FLAGS.md). In production do not pass `--p100e-refcheck`, `--p100e-print-ids` (token ids of
every prompt in the log) or `--p100e-selftest`.

## Tests

- Slot loop on the CPU fake engine, no GPU: `python -I tests/server/fake_engine_test.py build/p100e-server-fake
  <model.gguf>` (the GGUF is only read for its tokenizer).
- Parser cases, no GPU: `p100e-server -m <gguf> --jinja --chat-template-file <template> --p100e-parse-test
  tests/server/parser_cases.json`.
- Built-in self-test on the GPUs: `p100e-server ... --p100e-selftest builtin`, with `--p100e-refcheck` to compare
  greedy tasks with plain decoding.
- Engine prompt cache and batched drafting: `p100e-engine-cache-test`, `p100e-engine-draft-test` on a tiny model from
  `tools/mk_tiny_qwen35.py`.

## Not done yet

- Video and audio input. The image encoder runs on the task loop (a 2048-token image blocks the loop for about 3 s).
- GPU top-K and penalties for sampled requests: today a sampled step copies the full logit rows to the host.
- The prefill pipeline has a floor of about 200 ms for short prompts (9 to about 100 tokens).
