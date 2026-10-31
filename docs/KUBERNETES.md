# Running on Kubernetes

This runs `p100e-server` as a single pod on the node that has the GPU. The files here are set up for **one 16 GB P100**.

**Do you need Kubernetes?** With one GPU it adds little, because there is nothing to spread the work across. If you do not already run a cluster, [GETTING_STARTED.md](GETTING_STARTED.md) is simpler.

## 1. Check the GPU node first

The P100 needs NVIDIA's **proprietary** driver from the **580 branch or older**. Open kernel modules do not support it, and the 590 branch drops it. This is the most likely thing to go wrong.

On the GPU node:

```
nvidia-smi
modinfo nvidia | grep license
```

`nvidia-smi` must list the P100, and the licence line must say `NVIDIA`. This project was run on driver 580.178.04.

The cluster also needs NVIDIA's device plugin, so that it can hand out `nvidia.com/gpu`. Check that the node offers a GPU:

```
kubectl describe node NODE_NAME | grep nvidia.com/gpu
```

If you use the NVIDIA GPU Operator with its own driver, set `driver.kernelModuleType=proprietary`, or set `driver.enabled=false` to keep the driver already on the node. Pick a driver version from the 580 branch.

## 2. Build the image

From the repository root:

```
docker build -t YOUR_REGISTRY/p100e-server:latest .
docker push YOUR_REGISTRY/p100e-server:latest
```

The build takes about 15 to 30 minutes. The image is for the P100 only, has no image input, and holds no model or keys.

Use a private registry. The image contains NVIDIA's CUDA software, which comes under NVIDIA's own licences, so check those before you publish the image anywhere public. Building it from this repository is covered by the same terms you accept when you pull NVIDIA's base images.

## 3. Put the model on the node

Copy the model into `/models` on the GPU node. For one P100, use a file of about 13 GB, such as `Qwen3.8-27B-UD-IQ4_XS.gguf`. The server runs as user 10001, so that user must be able to read the file:

```
sudo chmod 755 /models
sudo chmod 644 /models/*.gguf
```

## 4. Make an API key

This writes a random key to `keys.txt` and stores it in the cluster. Keep the file out of git.

```
python3 -c "import secrets; print(secrets.token_urlsafe(32))" > keys.txt
kubectl create secret generic p100e-keys --from-file=keys.txt
```

## 5. Start it

Open `deploy/kubernetes/p100e.yaml` and replace the three `CHANGE_ME` values: the node name, the image and the model file. Then run:

```
kubectl apply -f deploy/kubernetes/p100e.yaml
kubectl get pods -l app=p100e -w
```

Wait for `READY 1/1`. To watch the model load:

```
kubectl logs -f deployment/p100e
```

## 6. Try it

```
kubectl port-forward service/p100e 8080:8080
```

In a second terminal, with the key from `keys.txt`:

```
curl http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer YOUR_KEY" -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

Other pods can reach it at `http://p100e:8080/v1`. The service stays inside the cluster. If you add an ingress, put TLS in front of it, because the server speaks plain HTTP.

## Changing the settings

Everything is in the `args` list in `p100e.yaml`. See [FLAGS.md](FLAGS.md).

For two GPUs on one node:

- Set `nvidia.com/gpu: 2`.
- Set `--p100e-devices` to `0,1`.
- Remove `--p100e-mtp`, `--p100e-kv` and `--p100e-pool-tokens`.
- Set `-c` to `131072` and `-ub` to `2048`.
- Remove `--p100e-host-budget-mb` to use the default 20 GB, and raise the memory limit to suit.

## Limits

- **All GPUs must be on one node.** The server cannot split a model across nodes.
- **Keep `replicas: 1`.** More replicas would each need their own GPU and would keep separate prompt caches.
- **Image input is not in this image.** It needs the fork build described in the README.
- **The memory values are estimates.** If the pod is killed for running out of memory, raise `limits.memory`.

## Troubleshooting

| what you see | what to do |
|---|---|
| Pod stays `Pending` | The node offers no `nvidia.com/gpu`, or another pod holds it. See step 1. |
| `nvidia-smi` says no devices were found | The node is using open kernel modules. Install the proprietary driver. |
| Pod cannot find a GPU, but the node has one | Set `runtimeClassName: nvidia` in `p100e.yaml`. |
| `MTP needs 2 to 4 GPUs` | Keep `--p100e-mtp 0` on one GPU. |
| `no room for a single KV page` | Keep `--p100e-pool-tokens` on one GPU, or use a smaller model. |
| `CreateContainerConfigError` | The `p100e-keys` secret is missing. See step 4. |
| Pod restarts during load | The model is on a slow disk. Raise `failureThreshold` in `startupProbe`. |
| The log says `FATAL engine error` | The server exits on a fatal GPU error, and Kubernetes starts it again. |
