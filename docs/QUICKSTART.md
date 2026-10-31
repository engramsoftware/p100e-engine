# Quickstart

Copy-paste command lines. First time here? Follow [GETTING_STARTED.md](GETTING_STARTED.md) instead. Replace the file names with your own. All examples are for Windows. Lines ending in `^` continue on the next line in `cmd`. In PowerShell, use a backtick instead, or put the command on one line.

See [FLAGS.md](FLAGS.md) for every option.

## 1. Build

Open a "x64 Native Tools Command Prompt for VS 2022" (or let `build.cmd` do it) and run:

```
build.cmd 60
```

`60` builds for the Tesla P100 only. To build for a P100 and a GTX 1660, run `build.cmd` with no argument.

To build without the server:

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=60 -DP100E_SERVER=OFF
cmake --build build -j 8
```

The server uses the llama.cpp fork code included in `third_party/llama.cpp-fork`. To use a different checkout, add `-DP100E_FORK_DIR=C:/path/to/llama.cpp-fork`.

## 2. Look at a model file

```
build\p100e-info.exe model.gguf
build\p100e-info.exe model.gguf --tensors
```

## 3. First run with `p100e-run`

Speed test on two GPUs: a 512-token prompt, then 64 decode steps.

```
build\p100e-run.exe model.gguf --devices 0,1 --bench 512 64
```

Same on one GPU (the model must fit):

```
build\p100e-run.exe model.gguf --devices 0 --bench 512 64
```

Same on four GPUs:

```
build\p100e-run.exe model.gguf --devices 0,1,2,3 --bench 512 64
```

Generate from token ids (greedy, 64 tokens). `p100e-run` takes token ids, not text. You can get ids from a running server with its `/tokenize` endpoint.

```
build\p100e-run.exe model.gguf --devices 0,1 --tokens 785,6722,315,9625,374 --gen 64
```

## 4. Speed with MTP drafting

Compare plain decoding with MTP at 1, 2 and 3 drafts. The tool prints speed, acceptance and `IDENTICAL` or `MISMATCH`.

```
build\p100e-run.exe model.gguf --devices 0,1 --tokens 785,6722,315,9625,374 --gen 128 --mtp 1,2,3
```

The server default (adaptive, up to 3 drafts, threshold 0.5):

```
build\p100e-run.exe model.gguf --devices 0,1 --tokens 785,6722,315,9625,374 --gen 128 --mtp 3 --mtp-adaptive 0.5
```

## 5. Long context speed

Prefill and decode speed at different context depths, with the smaller `q4_0` KV cache:

```
build\p100e-run.exe model.gguf --devices 0,1 --ctx 131072 --kv q4_0 --bench 2048 64 --bench-depth 0,8192,32768,65536 --load-mtp
```

## 6. Start the server

Local only, two GPUs, one chat at a time, MTP on:

```
build\p100e-server.exe -m model.gguf --host 127.0.0.1 --port 8080 -c 131072 -np 1 -ub 2048 --jinja --p100e-devices 0,1
```

Three chats at once (the default slot count), total context 262144 shared between them:

```
build\p100e-server.exe -m model.gguf --host 127.0.0.1 --port 8080 -c 262144 -np 3 -ub 2048 --jinja --p100e-devices 0,1
```

Offer it to other machines on your network:

```
build\p100e-server.exe -m model.gguf --host 0.0.0.0 --port 8080 -c 131072 -np 1 -ub 2048 --jinja --alias my-model --api-key-file keys.txt --p100e-no-remote-images --p100e-devices 0,1
```

Only do this on a network you trust. The key file holds one key per line. Without a key, anyone who can reach the port can use the server. See [GETTING_STARTED.md](GETTING_STARTED.md#9-share-it-on-your-network-safely).

With a custom chat template and a smaller KV cache:

```
build\p100e-server.exe -m model.gguf --jinja --chat-template-file template.jinja --p100e-kv q4_0 -c 200000 -np 1 --p100e-devices 0,1
```

With image input (the `p100e-vision.dll` file must sit next to the exe):

```
build\p100e-server.exe -m model.gguf --mmproj mmproj.gguf --image-min-tokens 1024 --image-max-tokens 2048 -c 131072 -np 1 --jinja --p100e-devices 0,1
```

Without MTP, and without the prompt cache:

```
build\p100e-server.exe -m model.gguf -c 65536 -np 1 --jinja --p100e-devices 0,1 --p100e-mtp 0 --p100e-cache 0
```

Wait for the line `model loaded` in the log. While the model loads, `/health` answers 503.

## 7. Talk to the server

Check it is ready:

```
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/props
```

Chat (OpenAI style):

```
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Explain what a GPU is in two sentences.\"}],\"max_tokens\":128}"
```

Chat with streaming:

```
curl -N http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Count to five.\"}]}"
```

Anthropic style:

```
curl http://127.0.0.1:8080/v1/messages -H "Content-Type: application/json" -d "{\"max_tokens\":128,\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}"
```

Turn tokens into ids (useful for `p100e-run`):

```
curl http://127.0.0.1:8080/tokenize -H "Content-Type: application/json" -d "{\"content\":\"The capital of France is\"}"
```

Live status (`/metrics` only works if the server was started with `--metrics`):

```
curl http://127.0.0.1:8080/slots
curl http://127.0.0.1:8080/metrics
```

An image request (OpenAI style, with a base64 data URL):

```
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What is in this picture?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,<BASE64>\",\"detail\":\"low\"}}]}],\"max_tokens\":128}"
```

## 8. Environment variables

Set a variable for one session, then start the program.

PowerShell:

```
$env:P100E_METRICS_LOG = "metrics.csv"
$env:P100E_DECODE_SHARE = "0.7"
build\p100e-server.exe -m model.gguf --jinja --p100e-devices 0,1
```

cmd:

```
set P100E_METRICS_LOG=metrics.csv
set P100E_DECODE_SHARE=0.7
build\p100e-server.exe -m model.gguf --jinja --p100e-devices 0,1
```

## 9. Checks and tests

Server self-test (starts, serves a scripted request set to itself, prints the results, exits):

```
build\p100e-server.exe -m model.gguf --jinja --p100e-devices 0,1 --p100e-selftest builtin
```

Check that MTP output equals plain greedy output bit for bit:

```
set P100E_MTP_CHECK=1
build\p100e-run.exe model.gguf --devices 0,1 --tokens 785,6722,315,9625,374 --gen 256 --mtp 3 --mtp-adaptive 0.5
```

Compare two settings in one process (here: f16 or f32 exchange between GPUs during prefill):

```
build\p100e-run.exe model.gguf --devices 0,1 --bench 2048 64 --ab "P100E_PF_WIRE16=0/P100E_PF_WIRE16=1" --ab-reps 3
```

Image through the command-line tool:

```
build\p100e-run.exe model.gguf --devices 0,1 --mmproj mmproj.gguf --image photo.png --chat "Describe this image." --gen 128
```

## 10. Keep it running

If the server finds a fatal GPU error it exits with code 3 on purpose. Run it under a supervisor that starts it again.

## Troubleshooting

- `error: unknown p100e flag ...`: the flag name is wrong, or your build is older than the flag. Check [FLAGS.md](FLAGS.md).
- `model has no MTP (nextn) block`: use `--p100e-mtp 0` or a model that has the block.
- Load fails with a message that names a part with no share: your `--p100e-split` weights leave a GPU with no KV head, head group or block. Use more even weights.
- Out of memory while loading: lower `-c`, use `--p100e-kv q4_0`, or lower `-ub`.
- Build fails on the fork check (only with `-DP100E_FORK_CHECK=ON`): the outside fork checkout is not at the expected commit. Use the right commit or turn the check off.
- CUDA 13 build errors about Pascal: use CUDA Toolkit 12.9.
