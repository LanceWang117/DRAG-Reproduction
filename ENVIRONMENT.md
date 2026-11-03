# Recorded test environment

| Item | Setup used |
| --- | --- |
| Cloud | RunPod, EUR-IS-1, Ubuntu 24.04.3 LTS |
| GPU | NVIDIA A100 SXM4, 80GB VRAM |
| RAM | 125GB decimal (about 116.4GiB) |
| CPU | UI: 32 vCPU; cgroup quota: 27.2 CPU-equivalents |
| Disk | 250GB network volume, separate container disk |
| Python | 3.11.17 in an isolated Conda environment |
| LLM | Llama-3.1-8B-Instruct, official BF16 weights, vLLM 0.6.6.post1 |
| Retrieval | E5-base-v2; full Wikipedia corpus; FAISS CPU FlatIP; top-3 per query |
| Generation | Input limit 4096, output limit 256 tokens, do_sample=false |
| Debate | Up to 3 retrieval rounds and 3 response rounds |

Runtime checks confirmed CUDA availability, PyTorch CUDA 12.4, successful
FlashRAG/DRAG imports, and a GPU matrix operation. These checks were recorded
during setup; they are not a check of the current Pod status.

Sampled process-tree RSS reached about 68GiB and GPU allocation about 67-68GiB.
GPU allocation includes vLLM cache reservation. RSS sums can count shared pages
more than once; cgroup memory includes file cache. These are observations, not
minimum requirements. Model/index initialization took roughly 4-5.5 minutes
per process; later questions reused them.

An earlier 32GB Mac trial used Ollama's quantized Llama 8B. One FAISS search
took about 198 seconds; one basic RAG question took about 306 seconds including
startup. This was not a timed full DRAG run or a controlled hardware comparison.

## Installed packages

Use `requirements.txt` for the cleaned installation list. The records below
are the exact Conda and pip listings saved from the experiment environment.
Conda lists its managed packages and builds; pip lists the Python packages.
Do not concatenate them into a requirements file.

<details>
<summary>Conda package/build record</summary>

```text
# This file may be used to create an environment using:
# $ conda create --name <env> --file <this file>
# platform: linux-64
# created-by: conda 26.7.2
@EXPLICIT
https://conda.anaconda.org/conda-forge/noarch/ca-certificates-2026.7.22-hbd8a1cb_0.conda
https://conda.anaconda.org/conda-forge/linux-64/libgomp-16.2.0-he0feb66_7.conda
https://conda.anaconda.org/conda-forge/linux-64/libzlib-1.3.2-h25fd6f3_3.conda
https://conda.anaconda.org/conda-forge/noarch/tzdata-2026c-h151e31d_0.conda
https://conda.anaconda.org/conda-forge/linux-64/_openmp_mutex-4.5-20_gnu.conda
https://conda.anaconda.org/conda-forge/linux-64/zstd-1.5.7-hb78ec9c_7.conda
https://conda.anaconda.org/conda-forge/linux-64/ld_impl_linux-64-2.46.1-default_hbd61a6d_103.conda
https://conda.anaconda.org/conda-forge/linux-64/libgcc-16.2.0-ha9f2e26_7.conda
https://conda.anaconda.org/conda-forge/linux-64/bzip2-1.0.8-hda65f42_10.conda
https://conda.anaconda.org/conda-forge/linux-64/libexpat-2.8.5-hd2095e1_0.conda
https://conda.anaconda.org/conda-forge/linux-64/libffi-3.7.0-h81df57d_1.conda
https://conda.anaconda.org/conda-forge/linux-64/liblzma-5.8.3-hb03c661_1.conda
https://conda.anaconda.org/conda-forge/linux-64/libnsl-2.0.1-hb9d3cd8_1.conda
https://conda.anaconda.org/conda-forge/linux-64/libpython-3.11.17-ha57cdbe_2_cpython.conda
https://conda.anaconda.org/conda-forge/linux-64/libsqlite-3.54.1-h0737f62_100.conda
https://conda.anaconda.org/conda-forge/linux-64/libuuid-2.42.4-hcfc3c73_0.conda
https://conda.anaconda.org/conda-forge/linux-64/libxcrypt-4.4.38-h280c20c_0.conda
https://conda.anaconda.org/conda-forge/linux-64/ncurses-6.6-hdb14827_1.conda
https://conda.anaconda.org/conda-forge/linux-64/openssl-3.6.5-h781a0a9_0.conda
https://conda.anaconda.org/conda-forge/linux-64/tk-8.6.13-noxft_h1df4ec4_4.conda
https://conda.anaconda.org/conda-forge/linux-64/readline-8.3-hd6e31c0_1.conda
https://conda.anaconda.org/conda-forge/linux-64/python-3.11.17-h5f976f7_2_cpython.conda
https://conda.anaconda.org/conda-forge/noarch/packaging-26.3-pyhc364b38_0.conda
https://conda.anaconda.org/conda-forge/noarch/setuptools-84.0.0-pyh332efcf_0.conda
https://conda.anaconda.org/conda-forge/noarch/wheel-0.48.0-pyhd8ed1ab_0.conda
https://conda.anaconda.org/conda-forge/noarch/pip-26.2.1-pyh8b19718_0.conda
```

</details>

<details>
<summary>Python package versions (pip freeze)</summary>

```text
accelerate==1.3.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.5
aiohttp-cors==0.8.1
aiosignal==1.4.0
airportsdata==20260905
altair==5.5.0
annotated-types==0.8.0
anyio==4.15.1
astor==0.8.1
attrs==26.1.0
autotiktokenizer==0.2.2
base58==2.1.1
blake3==1.0.11
blinker==1.9.0
blis==0.7.11
bm25s==0.2.0
cachetools==5.5.2
catalogue==2.0.10
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.2
chonkie==0.4.1
click==8.5.0
cloudpickle==3.1.2
colorful==0.5.8
compressed-tensors==0.8.1
confection==0.1.5
cryptography==50.0.2
cymem==2.0.13
datasets==3.2.0
defusedxml==0.7.1
depyf==0.18.0
dill==0.3.8
diskcache==5.6.3
distlib==0.4.4
distro==1.9.0
dominate==2.9.1
einops==0.8.2
faiss-cpu==1.8.0
fastapi==0.115.6
filelock==4.2.0
flashrag-dev==0.1.4.dev20250111
Flask==3.1.3
frozenlist==1.8.0
fschat==0.2.36
fsspec==2024.9.0
gguf==0.10.0
gitdb==4.0.12
GitPython==3.2.1
google-api-core==2.30.3
google-auth==2.62.0
googleapis-common-protos==1.75.0
grpcio==1.84.0
h11==0.16.0
HTML4Vision==0.4.3
httpcore==1.0.9
httptools==0.9.0
httpx==0.28.1
huggingface-hub==0.27.1
idna==3.20
importlib_metadata==9.0.1
iniconfig==2.3.1
interegular==0.3.3
itsdangerous==2.2.0
Jinja2==3.1.6
jiter==0.17.0
joblib==1.6.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
langcodes==3.5.1
lark==1.2.2
latex2mathml==3.81.1
Levenshtein==0.26.1
linkify-it-py==2.2.0
llvmlite==0.50.0
lm-format-enforcer==0.10.12
markdown-it-py==4.2.0
markdown2==2.5.5
MarkupSafe==3.0.4
mdit-py-plugins==0.6.1
mdurl==0.1.2
memray==1.20.0
mistral_common==1.5.1
mpmath==1.3.0
msgpack==1.2.3
msgspec==0.22.0
multidict==7.1.0
multiprocess==0.70.16
murmurhash==1.0.15
narwhals==2.27.1
nest-asyncio==1.6.0
networkx==3.6.1
nh3==0.3.7
nltk==3.10.3
numba==0.68.0
numpy==1.26.4
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-ml-py==13.615.71
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.4.127
openai==1.59.8
opencensus==0.11.4
opencensus-context==0.1.3
opencv-python-headless==4.11.0.86
orjson==3.13.0
outlines==0.1.11
outlines_core==0.1.26
packaging==24.2
pandas==2.2.3
partial-json-parser==0.2.1.1.post7
pathlib_abc==0.5.2
pathy==0.14.2
peft==0.14.0
pillow==10.4.0
platformdirs==4.13.3
pluggy==1.7.0
preshed==3.0.13
prometheus-fastapi-instrumentator==7.1.0
prometheus_client==0.26.0
prompt_toolkit==3.0.53
propcache==0.5.4
proto-plus==1.28.2
protobuf==5.29.3
psutil==6.1.1
py-cpuinfo==9.0.0
py-spy==0.4.2
pyarrow==19.0.0
pyasn1==0.6.4
pyasn1_modules==0.4.2
pybind11==3.1.0
pycountry==26.2.16
pycparser==3.11
pydantic==2.10.5
pydantic_core==2.27.2
pydeck==0.9.3
Pygments==2.21.0
PyStemmer==3.1.0
pytest==9.1.1
python-dateutil==2.9.0.post0
python-discovery==1.6.3
python-dotenv==1.2.4
python-Levenshtein==0.26.1
pytz==2026.5
PyYAML==6.0.2
pyzmq==27.2.0
rank-bm25==0.2.2
RapidFuzz==3.14.6
ray==2.40.0
referencing==0.37.0
regex==2026.9.29
requests==2.34.2
rich==13.9.4
rouge==1.0.1
rpds-py==2026.9.1
safetensors==0.5.2
scikit-learn==1.6.1
scipy==1.15.1
sentence-transformers==3.3.1
sentencepiece==0.2.0
shortuuid==1.0.13
six==1.17.0
smart-open==6.4.0
smmap==5.0.3
sniffio==1.3.1
spacy==3.6.1
spacy-legacy==3.0.12
spacy-loggers==1.0.5
srsly==2.5.4
starlette==0.41.3
streamlit==1.41.1
svgwrite==1.4.3
sympy==1.13.1
tenacity==9.2.1
textual==6.2.1
thinc==8.1.12
threadpoolctl==3.7.0
tiktoken==0.7.0
tokenizers==0.21.0
toml==0.10.2
torch==2.5.1
torchvision==0.20.1
tornado==6.5.10
tqdm==4.70.1
transformers==4.48.0
triton==3.1.0
typer==0.9.4
typing_extensions==4.16.0
tzdata==2026.5
urllib3==2.8.0
uvicorn==0.54.0
uvloop==0.23.0
virtualenv==21.14.5
vllm==0.6.6.post1
wasabi==1.1.3
watchdog==6.0.0
watchfiles==1.3.0
wavedrom==2.0.3.post3
wcwidth==0.9.2
websockets==17.2
Werkzeug==3.1.9
xformers==0.0.28.post3
xgrammar==0.1.10
xxhash==4.0.1
yarl==1.25.1
zipp==4.1.1
```

</details>
