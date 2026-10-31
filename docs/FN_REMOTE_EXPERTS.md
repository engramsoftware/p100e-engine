# Flash-Next remote expert tier

## What it is
`p100e-fn-run` (node 0, for example 2 x P100) runs the whole Qwen3.8-Flash-Next model as before: every layer,
the embedding, the head, the MTP draft block, the KV cache, the PLE table, its resident and adaptive expert cache.
`p100e-fn-expert-server` (node 1, for example an RTX 3070 + GTX 1660) holds extra Q4 routed experts that
node 0 does not have resident, and computes them on request. It loads only those expert tensors from its own copy of
the same GGUF, so no dense weights, no KV cache and next to no host RAM (about 0.1 GB).

Per layer of a decode, MTP draft or MTP verify forward (4 rows or fewer), node 0 routes, sends one message (the FFN
input rows plus the expert jobs) over plain TCP, runs its own experts meanwhile, and adds node 1's results into the
same combine step. One round trip per layer is the floor: layer L+1's attention needs layer L's FFN output.

- Exact mode returns each (row, expert) output and gives the same bits as node 0's own grouped path on the same GPU
  type. Sum mode (`--rexp-sum`) returns one weighted sum per row: fewer bytes, different summation order.
- Node 1's starting set is the best-ranked experts (expert_split.rank_order) that node 0 does not hold. In adaptive
  mode, node 0's routing counts drive swaps on node 1's first GPU; the 1660 keeps a fixed set (it is on PCIe x4).
- With every switch off, outputs are bit-identical to the CPU expert path.

Protocol: `src/fn_rexp.h`. Node 0 side: `src/fn_model.cu` (`rexp_*`). Node 1: `tools/p100e_fn_expert_server.cu`.
Switch details: `docs/FLAGS.md`.

## Running it on two PCs
Start node 0 first; it waits up to 30 minutes for node 1. Port 29500 must be reachable from node 1.

Node 0 (Linux):
```
env P100E_FN_KV=q8 P100E_FN_PREFETCH_T=0 P100E_FN_PLE_ASYNC=1 P100E_REXP_SPIN_US=2000 \
    P100E_FN_ADAPT_MAX=2 P100E_FN_ADAPT_ASYNC=1 P100E_FN_ADAPT_CAP=16 \
  ./build/p100e-fn-run /models/model.gguf --devices 0,1 --slots 78,78 \
    --ctx 32768 --max-t 128 --chunk 128 --mtp 3 --gen 256 --tokens-file prompt.txt \
    --rexp-listen 0.0.0.0:29500 --rexp-sum
```
Node 1 (Windows, PowerShell):
```
$env:P100E_REXP_SPIN_US="2000"
& ".\build\p100e-fn-expert-server.exe" models\model.gguf `
    --connect <node-0-ip>:29500 --devices 0,1 --slots auto,auto --swap-gpus 0 --once
```
`--merge-gu --spin-sync` on node 1 are optional (no measurable change on the real link).

## Switches (all off unless noted)
Node 0, `p100e-fn-run`:

| Switch | Default | What it does |
|---|---|---|
| `--rexp-listen host:port` | off | Turns the tier on; node 0 listens there. |
| `--rexp-max-t N` | 4 | Forwards of at most N rows use node 1 (prompt chunks stay local). |
| `--rexp-sum` | off | Per-row weighted sums back (not bit-equal). |
| `--rexp-connect-s S` | 1800 | Wait for node 1 to connect and load. |
| `--plan "F0,F1:H"` | | Offline one-process byte plan (MiB free VRAM per GPU : MiB available RAM). |
| `P100E_FN_PLE_ASYNC=1` | off | Linux PLE rows read in parallel (madvise) + verify rows prefetched during drafting. |
| `P100E_REXP_SPIN_US=N` | 0 | Spin on socket reads before sleeping (set it on both nodes). |
| `P100E_FN_ADAPT_ASYNC=1` | off | Promotions copy and pack off the compute stream. |
| `P100E_FN_ADAPT_CAP=N` | 0 | At most N promotions per GPU per adapt step, biggest gains first. |
| `P100E_FN_ADAPT_MAX=N` | 8 | Existing: promotions per layer per adapt step. |
| `P100E_FN_REXP_ADAPT=0` | on in adaptive mode | Node 1's set does not follow node 0's routing. |
| `P100E_FN_REXP_ADAPT_MAX`, `_CAP` | 4, 96 | Node 1 swaps per layer / per adapt step. |
| `P100E_FN_REXP_EARLY_SHARED=0` | on with the tier | Shared expert after the reply instead of during the wait. |
| `P100E_FN_REXP_MIN_PAIRS=N` | 0 | Layers with fewer remote pairs use the host path (changes output). |
| `P100E_FN_REXP_HALF=1` | off | f16 sum replies (no gain measured). |
| `P100E_FN_HC_TILE=1` | off | Tiled hc down matvec (slower on the P100: 39.8 vs 36.8 us). |
| `P100E_FN_TIME=1` / `=2` | off | Phase timing with device syncs / with GPU events (no syncs). |
| `P100E_FN_PROF_HC=1` | off | Per-kernel hc_mix GPU time. |
| `P100E_FN_LOAD_DEBUG=2` | off | VRAM after every dense upload. |

Node 1, `p100e-fn-expert-server <gguf> --connect host:port`:

| Switch | Default | What it does |
|---|---|---|
| `--devices 0,1`, `--slots a,b\|auto` | 0, auto | GPUs and experts per layer per GPU. |
| `--swap-gpus 0\|0,1\|none` | 0 | GPUs whose experts follow node 0's routing; the others hold a fixed set. |
| `--margin-mb N` | 384 | VRAM left free per GPU. |
| `--no-keepalive` | keep-alive on | Without it a GeForce card idles at low clocks between requests. |
| `--no-threads` | threads on | One worker thread per extra GPU (both GPUs at once). |
| `--no-one-copy` | one copy on | Request inputs in one H2D copy per GPU instead of three. |
| `--merge-gu`, `--spin-sync` | off | Gate+up in one launch (same bits); spinning syncs. |
| `--once`, `--verbose`, `--connect-s S` | | Exit after one session; timing per GPU; connect window. |

Byte plan (always on): every cudaMalloc of 1 MiB or more takes whole 2 MiB pages; the plan counts
that, so `--slots auto` no longer OOMs at the prefetch arenas.

