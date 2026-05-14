# LazySlide-wrap

Docker/Apptainer image for whole slide image feature extraction using [LazySlide](https://github.com/rendeirolab/LazySlide).

## Quick start

### Docker

```bash
docker pull b3rse/lazyslide:latest

docker run --gpus all --rm \
    --shm-size 8g \
    --user $(id -u):$(id -g) \
    -v /path/to/hf/cache:/models/huggingface \
    -v /path/to/data:/workspace \
    b3rse/lazyslide:latest \
    python /opt/extract_features.py \
        -i /workspace/slide.ndpi \
        -o /workspace/embeddings.h5ad \
        -m gigapath --mpp 0.5 --tile-px 256 --amp
```

### Apptainer (HPC)

```bash
# Pull once
apptainer pull lazyslide_latest.sif docker://b3rse/lazyslide:latest

# Run
apptainer exec --nv \
    --bind /path/to/hf/cache:/models/huggingface \
    --bind /path/to/data:/workspace \
    lazyslide_latest.sif \
    python /opt/extract_features.py \
        -i /workspace/slide.ndpi \
        -o /workspace/embeddings.h5ad \
        -m gigapath --mpp 0.5 --tile-px 256 --amp
```

> Use `--gpus all` / `--nv` only on GPU nodes. Omit these flags on CPU nodes.

> **Singularity:** all `apptainer` commands work with `singularity` by swapping the command name — Apptainer is the renamed successor to Singularity.

> **Separate output directory:** input data and output embeddings can be mounted independently — add `-v /path/to/output:/output` and write to `-o /output/embeddings.h5ad`.

> **`--user` note:** when passing `--user $(id -u):$(id -g)` to Docker, ensure all mounted directories exist and are owned and writable by your user: `mkdir -p /path/to/output && sudo chown -R $(id -u):$(id -g) /path/to/hf/cache /path/to/data /path/to/output`.

### JupyterLab

**On the server** (use tmux to keep it running after disconnecting):
```bash
docker run --gpus all -it --rm \
    --shm-size 8g \
    --user $(id -u):$(id -g) \
    -e HOME=/workspace \
    -e HF_TOKEN=hf_xxxxxxxxxxxx \
    -p 8888:8888 \
    -v /path/to/hf/cache:/models/huggingface \
    -v /path/to/data:/workspace \
    b3rse/lazyslide:latest \
    jupyter lab --ip=0.0.0.0 --no-browser
```

**Apptainer** (use tmux to keep it running after disconnecting):
```bash
export HF_TOKEN=hf_xxxxxxxxxxxx

apptainer exec --nv \
    --bind /path/to/hf/cache:/models/huggingface \
    --bind /path/to/data:/workspace \
    lazyslide_latest.sif \
    jupyter lab --ip=0.0.0.0 --no-browser --port=8888
```

**On your laptop**, open a new terminal and create an SSH tunnel:
```bash
ssh -L 8888:localhost:8888 user@server
```

Then open the URL printed in the server output in your browser:
```
http://127.0.0.1:8888/lab?token=<token>
```

All code runs inside the container on the server. Files saved to `/workspace` persist on the host at `/path/to/data`.

To close the tunnel just close the terminal — JupyterLab keeps running on the server. Reconnect anytime with the same `ssh -L` command and token URL.

Multiple users can run simultaneously by mapping different host ports. Each user picks a unique port (e.g. 8888, 8889, 8890) on both the server `-p` flag and the SSH tunnel:
```bash
# server (each user uses a different host port)
docker run ... -p 8889:8888 ...
# laptop
ssh -L 8889:localhost:8889 user@server
# browser
http://127.0.0.1:8889/lab?token=<token>
```

### Interactive session

**Docker:**
```bash
docker run --gpus all -it --rm \
    --shm-size 8g \
    --user $(id -u):$(id -g) \
    -e HOME=/workspace \
    -v /path/to/hf/cache:/models/huggingface \
    -v /path/to/data:/workspace \
    b3rse/lazyslide:latest \
    ipython
```

**Apptainer:**
```bash
apptainer exec --nv \
    --bind /path/to/hf/cache:/models/huggingface \
    --bind /path/to/data:/workspace \
    lazyslide_latest.sif \
    ipython
```

```python
import lazyslide as zs
from wsidata import open_wsi

wsi = open_wsi("/workspace/slide.ndpi")
zs.pp.find_tissues(wsi)
zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)

# Limit tiles for quick testing
wsi.shapes["tiles"] = wsi.shapes["tiles"].iloc[:5]

zs.tl.feature_extraction(wsi, model="gigapath", amp=True)

adata = wsi.tables["gigapath_tiles"]
print(adata)
```

## Models

| Model | `--model` | HuggingFace | Access |
|---|---|---|---|
| UNI2 | `uni2` | [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) | Request required |
| GigaPath | `gigapath` | [prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) | Request required |
| Virchow2 | `virchow2` | [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2) | Request required |
| CONCH | `conch` | [MahmoodLab/conch](https://huggingface.co/MahmoodLab/conch) | Request required |
| PLIP | `plip` | [vinid/plip](https://huggingface.co/vinid/plip) | Open access |

Gated models require a HuggingFace token. Two ways to provide it:

- **Environment variable (recommended):** pass `-e HF_TOKEN=hf_xxxx` to `docker run`, or `export HF_TOKEN=hf_xxxx` before `apptainer exec`. This is the most reliable method as it works for all models including those loaded via `timm`.
- **Token file:** mount a directory containing a `token` file (created by `hf auth login`) to `/models/huggingface`. Pass `--token hf_xxxx` to the script as a fallback for models that read it directly from LazySlide.
- **Offline mode:** if weights are already cached, set `-e HF_HUB_OFFLINE=1` (Docker) or `export HF_HUB_OFFLINE=1` (Apptainer) to skip all network requests and use the local cache directly. Useful on compute nodes without internet access.

## Output formats

| Flag | Format | Notes |
|---|---|---|
| `-o out.h5ad` | AnnData | Embeddings + metadata + spatial coordinates |
| `-o out.pt` | PyTorch | For training pipelines |
| `-o out.npz` | NumPy | Maximally portable |

> **`.pt` note:** the file contains non-tensor objects (`obs`, `model`), so load with `torch.load("embeddings.pt", weights_only=False)`.
