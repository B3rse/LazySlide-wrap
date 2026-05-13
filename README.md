# LazySlide-wrap

Docker/Apptainer image for whole slide image feature extraction using [LazySlide](https://github.com/rendeirolab/LazySlide).

## Quick start

### Docker

```bash
docker pull b3rse/lazyslide:latest

docker run --gpus all --rm \
    --shm-size 8g \
    -v ~/.cache/huggingface:/models/huggingface \
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
    --bind ~/.cache/huggingface:/models/huggingface \
    --bind /path/to/data:/workspace \
    lazyslide_latest.sif \
    python /opt/extract_features.py \
        -i /workspace/slide.ndpi \
        -o /workspace/embeddings.h5ad \
        -m gigapath --mpp 0.5 --tile-px 256 --amp
```

> Use `--gpus all` / `--nv` only on GPU nodes. Omit it on CPU nodes — `cucim` falls back to CPU automatically.

## Models

| Model | `--model` | HuggingFace | Access |
|---|---|---|---|
| UNI2 | `uni2` | [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) | Request required |
| GigaPath | `gigapath` | [prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) | Request required |
| Virchow2 | `virchow2` | [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2) | Request required |
| CONCH | `conch` | [MahmoodLab/conch](https://huggingface.co/MahmoodLab/conch) | Request required |
| PLIP | `plip` | [vinid/plip](https://huggingface.co/vinid/plip) | Open access |

Gated models require a HuggingFace token passed via `--token` or stored at `~/.cache/huggingface/token` (set by `huggingface-cli login`).

## Output formats

| Flag | Format | Notes |
|---|---|---|
| `-o out.h5ad` | AnnData | Embeddings + metadata + spatial coordinates |
| `-o out.h5` | HDF5 | CLAM-compatible |
| `-o out.pt` | PyTorch | For training pipelines |
| `-o out.npz` | NumPy | Maximally portable |
