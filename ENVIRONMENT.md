# Test environment

## Cloud experiment

| Item | Recorded setup |
| --- | --- |
| Provider / region | RunPod Secure Cloud, EUR-IS-1 |
| Operating system | Ubuntu 24.04.3 LTS, container-based Pod |
| GPU | NVIDIA A100 SXM4, 80GB VRAM |
| System memory quota | 125GB decimal, about 116.4GiB |
| CPU | UI advertised 32 vCPU; cgroup quota was 27.2 CPU-equivalents |
| Storage | 250GB independent network volume; separate container disk |
| Python | 3.11.17, isolated Conda environment |
| Inference | vLLM 0.6.6.post1, PyTorch 2.5.1 with CUDA 12.4 |
| RAG framework | flashrag-dev 0.1.4.dev20250111 |
| Other key packages | transformers 4.48.0, datasets 3.2.0, faiss-cpu 1.8.0, numpy 1.26.4 |
| Generator | Official Llama-3.1-8B-Instruct BF16 weights |
| Retriever | intfloat/e5-base-v2, FAISS CPU FlatIP, top-3 per query |
| Generation | Input limit 4096, output limit 256 tokens, do_sample=false |
| Debate limits | Retrieval: 3 rounds; response: 3 rounds |

`runtime-pip-freeze.txt` and `runtime-conda-explicit.txt` preserve the installed
package lists. `environment-record.json` contains the original runtime check.
The Pod template initially included a different PyTorch version; the experiment
used its isolated environment, not the template's base Python.

## Observed resource use

Sampling recorded a process-tree RSS around 68GiB at peak and GPU allocation
around 67-68GiB. GPU allocation includes vLLM's reserved KV cache and is not
solely model weights. RSS sums can double-count shared pages; cgroup memory
includes file cache. These observations are not minimum hardware requirements.
No out-of-memory termination was recorded in the completed runs.

The model and index initialize once per worker process, usually taking around
4-5.5 minutes. Questions within that process reuse them. Starting a separate
process loads them again. The GPU is not needed to inspect the saved results.

## Earlier Mac trial

A MacBook with 32GB unified memory ran a basic RAG smoke test using Ollama's
quantized Llama 3.1 8B and the full corpus/index. One FAISS search took about
198 seconds. Total process wall time was about 306 seconds, including model and
index initialization. The answer to the trial question was correct.

This was a single-pass RAG test, not a timed full DRAG run. A 50-minute DRAG
estimate discussed during planning was an extrapolation, not a measured result.
The Mac and cloud trials differ in inference runtime and model precision, so
their times are not a controlled hardware comparison.

## Deployment notes

Use the mounted persistent volume for models, corpus, index, environment, and
outputs. A stopped Pod may not immediately regain GPU capacity. An independent
network volume persists separately and can still incur storage charges.
Stopping Python or disconnecting SSH does not stop Pod billing. Back up results
before terminating a Pod or deleting its storage.
