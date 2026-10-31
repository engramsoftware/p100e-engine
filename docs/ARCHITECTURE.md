# Architecture overview

This is a short map of how p100e is built. For server internals, see [../server/README.md](../server/README.md).

## The big picture

```
 client (OpenAI / Anthropic API)
        |
   p100e-server  ----  text layer: HTTP, chat templates, tokenizer, tool-call and reasoning parsers, sampler
        |
   engine interface (server/engine.h)       plain C++, no CUDA types
        |
   tp_model + tp_mtp + scheduler            tensor-parallel model, MTP draft head, paged KV pool, prompt cache
        |
   fast kernels (src/fast)                  hand-written CUDA for sm_60
        |
   1, 2 or 4 GPUs (P2P over PCIe or NVLink)
```

The model is loaded straight from a GGUF file. Weights are decoded on the GPU and stay in VRAM. There is no ggml at run time.

## Kernels

All hot paths are custom CUDA kernels in `src/fast/`.

- **Decode (one token at a time).** The step is memory-bound: it reads every weight once. The matrix-vector kernels read the compressed weights directly (Q4, Q5, Q6, Q8, IQ-types and others) and use fp16 math in pairs (`HFMA2`) to match what the P100 can do. Related weights (for example gate and up) are joined into one launch.
- **Multi-column decode (2 to 8 rows).** Used to check MTP drafts. The weight decode is shared between the columns, so checking 3 tokens costs much less than 3 separate steps. Each column gives exactly the bits that a single-token step would give.
- **Prefill (reading a prompt).** Weights are unpacked to f16 on a side stream into a ring buffer, ahead of the matrix multiplies. A "fold" GEMM uses fp16 products and folds into fp32 regularly. Attention for prefill is two GEMMs per chunk of keys with a running softmax.
- **Decode attention.** One kernel layout that does not change with the number of rows or the depth, so a row in a batch gives the same bits as the same row alone. This is what makes MTP output identical to plain greedy output.
- **Delta net.** The hybrid model mixes full attention layers with linear-attention (delta-net) layers. p100e has fused kernels for these, including a batched version that writes rollback snapshots for rejected drafts.
- **CUDA graphs.** A whole decode step (all layers, the output head and the GPU-to-GPU exchange) is captured once per shape and replayed. This removes launch overhead.
- **KV cache.** Stored as `f16`, `q8_0` (default) or `q4_0`. Pages of 256 positions come from one pool per GPU, shared by all chats.

## Tensor parallel: 1 to 4 GPUs

`--p100e-devices` (server) and `--devices` (`p100e-run`) take 1 to 4 GPU indices. Each layer is split across the GPUs:

- Attention heads, KV heads, delta-net heads and the feed-forward rows are divided evenly when they divide by the GPU
  count (1, 2, 4 on the 27B). Otherwise (3 GPUs, or weights from `--p100e-split`) each GPU gets whole units: KV-head
  groups, delta-net head groups, 256-row FFN and vocab blocks. One range table (`src/tp_split.h`) holds every GPU's share.
- Row-parallel outputs are added together by an allreduce.
- During decode, the allreduce is a one-shot peer-to-peer exchange with a flag word. It runs inside the CUDA graph. A wait has a time limit; on timeout the process reports a fatal error and exits so a supervisor can restart it.
- During prefill, the partial results are exchanged between GPUs. By default they travel as f16 to save bandwidth. If a value would overflow f16, the engine redoes the prompt with f32 exchange, so no bad state is ever kept or cached.
- Across machines (`opts::net_nodes`, `src/net_link.h`): SPMD lockstep. Every node runs the same program,
  knows the global rank count and the split, and loads only its own ranks. The only cross-node traffic is data: per
  allreduce the local partials go D2H, one TCP message per node carries all of them, they land H2D in the event path's
  peer slots and the same add kernels sum all partials in global rank order (bit-identical to one process with
  `P100E_P2P_AR=0`). The prefill pipeline does the same per token chunk (f16 wire as in one process), and the logit
  slices or argmax pairs, the pool size, the f16-saturation count and the plan verdicts are gathered so every node
  takes the same path. Messages carry a sequence number and size, so a node that left the common control flow is
  caught. CUDA graphs, the P2P allreduce, MTP, sessions to host / disk and batched sequences are off in this mode.
  Speed-ups, all bit-exact: the receivers' sum is the left fold `((x + p0) + p1) + p2` in rank
  order and x is identical on every node, so node 0 (ranks 0..k-1, the fold's prefix) sends `s = (x + p0) + .. +
  p(k-1)` once and the receivers set `x = s` and add `-0.0f` (an exact identity) in those slots (`P100E_NET_PREADD`;
  `p0 + p1` alone would not be exact: float addition is not associative). T = 1 decode steps replay one CUDA graph
  per segment between exchanges (`P100E_NET_GRAPH`), and the socket loop busy-polls (`P100E_NET_SPIN`).

Three GPUs are refused: the model's 4 KV heads, 16 delta-net heads, feed-forward size and vocabulary do not divide by 3. The 4-GPU code path is tested with virtual ranks on one GPU, not on real four-card hardware.

## Server

`p100e-server` reuses the text layer of a llama.cpp fork (tokenizer, Jinja chat templates, tool-call and reasoning parsers, sampler chain, argument parser, HTTP routes). The slot loop is replaced so that it drives the p100e engine. That is why existing llama-server command lines and clients work.

- **Slots.** `-np N` chats run at once. Every decode step is one forward pass over all generating slots, up to 8 rows. Each row gives the same bits as that chat alone.
- **Prompt cache.** A finished chat stays as a cache entry on the GPU. After that it moves to pinned host RAM, and is dropped last-recently-used. A new request is matched by token ids (image rows by image hash) against idle entries. It reuses the entry's end, or a checkpoint (delta-net state plus draft state) taken at message starts. Shared prefixes are copied, never stolen from another chat.
- **Fairness.** While one slot reads a long prompt, generating slots get a share of the time (`--p100e-decode-share`).
- **Sampling.** Greedy requests use a GPU argmax and copy no logits. Other requests copy logits to the host and use the llama.cpp sampler chain. MTP works with any sampler (sample-and-match), so every emitted token is a sample of the real distribution.
- **Failure handling.** A fatal GPU error makes the server exit with code 3. A supervisor starts it again.
- **Live status.** `/slots` and `/metrics` answer from a snapshot the loop publishes after every step, so they respond in milliseconds even during a long prefill.

## Vision: native encoder (nvit)

Images are encoded by `src/nvit`, a CUDA implementation of the Qwen3-VL vision tower (27 blocks, 2D rotary positions, 2x2 merger). There is no ggml at run time for this part.

- **Two GPUs, sequence parallel.** Each GPU holds half of the image patches. A GPU swaps key and value tiles with its neighbour once per layer, hidden behind attention.
- **Low VRAM.** Weights stay in pinned host RAM and are streamed one layer ahead. Scratch memory is borrowed from the idle prefill ring. Permanent cost is about 28 MB per GPU, against about 850 MB for the llama.cpp encoder on one GPU.
- **Preprocessing** on the CPU follows the llama.cpp path bit for bit (resize, pad, normalise).
- **In the model.** Text and image pieces run as one row list. Image tokens use interleaved M-RoPE positions. A cache keyed by a hash of the file bytes avoids encoding the same image twice. An older path that uses the llama.cpp encoder (`P100E_NVIT=0`) is kept as a fallback.

## Drafting: MTP and n-gram

**MTP (multi-token prediction).** The model file carries an extra "nextn" block. p100e runs it as a draft head:

1. After each step, the draft head proposes up to `K` next tokens (default 3). It scores only the 32768 most frequent tokens, which is much cheaper than the full vocabulary and loses almost no accepted drafts.
2. The main model checks `[last token + drafts]` in one multi-column pass.
3. Accepted tokens are kept. Rejected rows are rolled back by one gather kernel that restores the delta-net state from a snapshot.
4. Adaptive length: a draft is checked only while the product of draft probabilities is at least `tau` (default 0.5).

Greedy output with MTP is bit-identical to plain greedy output. This is checked with `P100E_MTP_CHECK=1` and by the `IDENTICAL` lines of `p100e-run --mtp`. With sampling, the sample-and-match rule keeps the output a true sample.

**N-gram drafting.** An experimental host-side drafter proposes tokens by finding the longest earlier match of the recent text in the chat history. It is off by default (`P100E_NGRAM_DRAFT=0`). A telemetry-only "shadow" mode (`P100E_NGRAM_SHADOW`, on by default) measures what it would have proposed, without touching any output. This helped decide whether n-gram drafting is worth turning on.

## Other engines in the tree

`--p100e-engine fn` selects "Flash-Next", an experimental engine for a mixture-of-experts model family (hot experts resident on the GPU, cold experts streamed from host RAM, sparse attention, MTP, one slot). The main documented path is the `tp` engine described above.

## Testing approach

- Bit-exactness is a design rule: batched and single-row paths must give identical bits. Unit tests in `tests/` check kernels one by one, including on a small GTX 1660 (`sm_75`).
- `p100e-server --p100e-selftest` replays scripted requests. `--p100e-refcheck` re-runs greedy tasks as plain decoding and compares the ids.
- `p100e-run --kld-base` measures quality as KL divergence against a llama.cpp reference file.
