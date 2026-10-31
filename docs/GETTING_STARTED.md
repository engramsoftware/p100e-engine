# Getting started

This guide takes you from nothing to a running server you can chat with. Follow it in order. For more command lines see [QUICKSTART.md](QUICKSTART.md), and for every option see [FLAGS.md](FLAGS.md).

**In short:** build the project, get a model file, start `p100e-server`, then point any OpenAI-compatible or Anthropic-compatible app at it.

## The easy way

`p100e.py` does the setup for you. It only needs Python 3, and it works on Windows and Linux. On Linux, type `python3` instead of `python`.

| command | what it does |
|---|---|
| `python p100e.py check` | Lists what is installed, what is missing and how to fix it. |
| `python p100e.py setup` | Builds the server. Finds Visual Studio and CUDA for you. |
| `python p100e.py setup --vision` | Builds the server with image input. |
| `python p100e.py start model.gguf` | Finds your GPUs and starts the server. |

Options for `start`:

| option | use |
|---|---|
| `--gpus 0,1` | Choose the GPUs yourself. By default it uses your P100s, or every GPU if you have none. |
| `--context 65536` | Use a smaller total context if the model does not fit. |
| `--small-cache` | Use less GPU memory per token. |
| `--chats 3` | Run several chats at once. They share the context. |
| `--mmproj vision.gguf` | Turn on image input. |
| `--no-mtp` | For models without the MTP block. |
| `--share` | Let other machines connect. Makes a random key in `keys.txt` the first time. |
| `--keep-running` | Restart the server after a fatal GPU error. |
| `--dry-run` | Print the full command without running it. |

Anything after `--` goes straight to the server, for example `python p100e.py start model.gguf -- --metrics`.

### One GPU

One 16 GB P100 works, with three limits:

- **Use a small model file.** For Qwen3.8-27B, a file of about 13 GB such as `UD-IQ4_XS` fits. Larger files such as Q6_K (about 23 GB) do not.
- **There is no MTP drafting.** It needs 2 to 4 GPUs.
- **Context is smaller.** About 32K tokens fits next to a 13 GB model.

`python p100e.py start` sets all of this for you when it finds one GPU. By hand, the same thing is:

```
build\p100e-server.exe -m model.gguf --alias qwen -c 32768 -np 1 -ub 32 --jinja --p100e-devices 0 --p100e-mtp 0 --p100e-kv q4_0 --p100e-pool-tokens 32768
```

In testing, this gave about 25 tokens per second of output with about 350 MB of GPU memory left free. Prompts read more slowly with `-ub 32`, and long prompts were not measured.

The rest of this guide explains each step by hand, for when you want more control.

To run it in a cluster, see [KUBERNETES.md](KUBERNETES.md).

## 1. What you need

| item | details |
|---|---|
| GPUs | 1 to 4 NVIDIA GPUs. Made for the Tesla P100 16 GB. Also builds for the GTX 1660 and RTX 3070. |
| Operating system | Windows 10 or 11 (64-bit), or Linux (Ubuntu 24.04 is the tested build). |
| CUDA Toolkit | **12.9.** CUDA 13 dropped the P100, so do not use it. |
| Compiler | Windows: Visual Studio 2022 Build Tools with the C++ workload. Linux: GCC 11 to 14. |
| Build tools | CMake 3.24 or newer, Ninja, Python 3, and git. |
| Model | A Qwen 3.8 `qwen35` model in GGUF format, for example Qwen3.8-27B. See step 3. |
| Host RAM | The prompt cache uses up to 20 GB of pinned RAM by default. See step 7 to lower it. |

## 2. Get the code and build it

```
git clone https://github.com/engramsoftware/p100e-engine.git
cd p100e-engine
```

**Windows.** First open `build.cmd` and check that the Visual Studio and CUDA paths at the top match your PC. Then run:

```
build.cmd 60
```

**Linux:**

```
sudo apt install build-essential cmake ninja-build git python3
./build.sh 60
```

`60` means "build for the P100 only", which is the fastest choice. To also build for a GTX 1660 or RTX 3070, leave the number off.

The programs land in `build/`. Linux programs have no `.exe` at the end, so read `build\p100e-server.exe` as `build/p100e-server` in the rest of this guide.

If you run out of memory while building, use fewer parallel jobs by setting `JOBS=2` first.

### Optional: image input

To send pictures to the model, build the included llama.cpp code **once, before the main build**. The full command is in the README under [The llama.cpp fork](../README.md#the-llamacpp-fork). Skip this if you only want text.

On Windows, copy `build\p100e-vision.dll` next to `p100e-server.exe` if you move the program somewhere else.

## 3. Get a model

Models are not included in this repository because they are very large. You need a `qwen35` GGUF file. For image input you also need the matching `mmproj` GGUF.

Check that your file can be read:

```
build\p100e-info.exe model.gguf
```

For Qwen3.8-27B, a Q4-class file is the usual starting point on two 16 GB cards.

MTP drafting, which speeds up replies, needs a model that includes the MTP ("nextn") block. If yours does not have one, add `--p100e-mtp 0` when you start the server.

## 4. Start the server

For two GPUs on your own PC:

```
build\p100e-server.exe -m model.gguf --alias qwen --host 127.0.0.1 --port 8080 -c 131072 -np 1 -ub 2048 --jinja --p100e-devices 0,1
```

| option | what it does |
|---|---|
| `-m model.gguf` | The model file. |
| `--alias qwen` | The model name apps will see. Without it, replies show the model's full file path. |
| `--host 127.0.0.1` | Only this PC can connect. |
| `-c 131072` | Total context in tokens, shared by all chats. |
| `-np 1` | One chat at a time. |
| `-ub 2048` | The most prompt tokens handled in one pass. |
| `--jinja` | Uses the model's chat template. Needed for tools and reasoning. |
| `--p100e-devices 0,1` | Which GPUs to use. For one GPU, use `0`. |

Wait for `model loaded` in the log. Until then, `/health` answers with error 503.

## 5. Check that it works

Open a second terminal:

```
curl http://127.0.0.1:8080/health
```

You should see `{"status":"ok"}`. Now send a message. On Windows `cmd`:

```
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":64}"
```

On Linux or macOS:

```
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

To see how fast it runs, stop the server and try:

```
build\p100e-run.exe model.gguf --devices 0,1 --bench 512 64
```

## 6. Use it from your own apps

Most apps only need these settings:

| setting | value |
|---|---|
| Base URL (OpenAI style) | `http://127.0.0.1:8080/v1` |
| Base URL (Anthropic style) | `http://127.0.0.1:8080` |
| Model | The name you gave `--alias`, such as `qwen` |
| API key | Your key if you set one. Otherwise any text, if the app insists on having one. |

**Python, with the `openai` package:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
stream = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Explain what a GPU is in two sentences."}],
    max_tokens=128,
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**Python, with no extra packages:**

```python
import json
import urllib.request

body = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}
request = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request) as response:
    print(json.load(response)["choices"][0]["message"]["content"])
```

**Anthropic style:**

```
curl http://127.0.0.1:8080/v1/messages -H "Content-Type: application/json" -d "{\"max_tokens\":128,\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}"
```

The most useful endpoints:

| endpoint | use |
|---|---|
| `/v1/chat/completions` | OpenAI-style chat. |
| `/v1/responses` | OpenAI responses API. |
| `/v1/messages` | Anthropic-style chat. |
| `/v1/models` | Lists the model. |
| `/tokenize` | Turns text into token ids. |
| `/health` | Says if the server is ready. |
| `/props`, `/slots` | Server settings and live chat status. |
| `/metrics` | Prometheus numbers. Only works if the server was started with `--metrics`. |

## 7. Make it fit your GPUs and RAM

If the model does not load, or you want more context, change one thing at a time:

| goal | change |
|---|---|
| Fix "out of memory" at load | Lower `-c`, or add `--p100e-kv q4_0`, or lower `-ub`. |
| Fit more context | Add `--p100e-kv q4_0`. It is smaller than the default `q8_0`, which costs about 17 KiB per token per GPU. That is about 2.2 GB per GPU at `-c 131072`. |
| Use less host RAM | Add `--p100e-host-budget-mb 8192`. |
| Run several chats at once | Set `-np 3` and raise `-c`. Each chat gets `-c` divided by `-np`. |
| Use one GPU | See [One GPU](#one-gpu). The usual settings fail on one card. |
| Use GPUs of different sizes | Add `--p100e-split 2,1,1` to give each card a relative share. |

For example, three chats sharing a larger context:

```
build\p100e-server.exe -m model.gguf --alias qwen -c 262144 -np 3 -ub 2048 --jinja --p100e-devices 0,1
```

## 8. Send images

This needs the optional step in section 2 and an `mmproj` file:

```
build\p100e-server.exe -m model.gguf --alias qwen --mmproj mmproj.gguf -c 131072 -np 1 --jinja --p100e-devices 0,1
```

Send the picture as a base64 `data:` URL. There is an example in [QUICKSTART.md](QUICKSTART.md#7-talk-to-the-server).

## 9. Share it on your network safely

By default only your own PC can connect. To let other machines connect, listen on all addresses and **always require a key**:

```
build\p100e-server.exe -m model.gguf --alias qwen --host 0.0.0.0 --port 8080 -c 131072 -np 1 --jinja --api-key-file keys.txt --p100e-no-remote-images --p100e-devices 0,1
```

- Put one long random key per line in `keys.txt`, and keep that file out of git.
- `--api-key-file` keeps the key off the command line, where other programs could see it.
- `--p100e-no-remote-images` stops the server from downloading image URLs it is sent.
- The traffic is plain HTTP, so only use this on a network you trust. Put a TLS proxy in front of it for anything wider.

Clients then send the key:

```
curl http://192.0.2.10:8080/v1/chat/completions -H "Authorization: Bearer YOUR_KEY" -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":64}"
```

Anthropic-style clients can send `x-api-key: YOUR_KEY` instead.

## 10. Keep it running

If the server finds a fatal GPU error, it exits with code 3 on purpose. Run it under something that starts it again.

**Linux,** in a systemd service file. Change the paths and user first:

```ini
[Unit]
Description=p100e server
After=network.target

[Service]
ExecStart=/opt/p100e-engine/build/p100e-server -m /models/model.gguf --alias qwen -c 131072 -np 1 --jinja --p100e-devices 0,1
Restart=always
RestartSec=5
User=p100e
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
```

**Windows,** in a small `run-server.cmd` file:

```bat
:loop
build\p100e-server.exe -m model.gguf --alias qwen -c 131072 -np 1 --jinja --p100e-devices 0,1
timeout /t 5
goto loop
```

## Troubleshooting

| what you see | what to do |
|---|---|
| CUDA errors about Pascal at build time | Install CUDA Toolkit 12.9, not 13. |
| The compiler runs out of memory | Set `JOBS=2` and build again. |
| `error: unknown p100e flag ...` | Check the name in [FLAGS.md](FLAGS.md). Your build may be older than the flag. |
| `model has no MTP (nextn) block` | Add `--p100e-mtp 0`, or use a model that has the block. |
| `MTP needs 2 to 4 GPUs` or `no room for a single KV page` | You are on one GPU. See [One GPU](#one-gpu). |
| Out of memory at load | See section 7. |
| `/health` returns 503 | The model is still loading. Wait for `model loaded`. |
| `/metrics` returns 501 | Start the server with `--metrics`. |
| `Invalid API Key` (401) | Send `Authorization: Bearer YOUR_KEY` or `x-api-key: YOUR_KEY`. |
| `--mmproj` is refused | Do the image input step in section 2, then rebuild. |
| A reply shows your full model path | Add `--alias`. |

## What is not supported yet

- Audio or video input, embeddings, rerank, LoRA, slot save and restore, and `n_cmpl > 1`. The server returns a clear error for these.
- Models other than the `qwen35` family on the main engine.

The [README](../README.md#limits) lists what has and has not been tested on real hardware.
