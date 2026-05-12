FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Runtime/cache configuration
# HF_HUB_OFFLINE=1 can optionally be set at runtime for no internet access / local-only models
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    XDG_CACHE_HOME=/models/.cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    JAVA_HOME=/usr/lib/jvm/default-java \
    PATH="/usr/lib/jvm/default-java/bin:$PATH"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    default-jdk \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libopenslide0 \
    libvips42 \
    build-essential \
    tini \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -f /usr/lib/python3*/EXTERNALLY-MANAGED \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN pip install --no-cache-dir --upgrade \
    pip setuptools wheel

# CUDA torch — must precede lazyslide to avoid pulling CPU variant
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Core lazyslide with all pip-available extras
RUN pip install --no-cache-dir \
    "lazyslide[all,models]"

# Segmentation / augmentation / generation ecosystem
RUN pip install --no-cache-dir \
    cellpose \
    segmentation-models-pytorch \
    albumentations \
    diffusers \
    accelerate \
    fairscale \
    sentencepiece \
    segment-anything \
    instanseg-torch

# Spatial analysis dependencies
RUN pip install --no-cache-dir \
    igraph \
    leidenalg \
    muon

# Gated pathology foundation models
# Weights still require HuggingFace approval/access
# --no-deps intentional: both repos conflict with current timm/transformers versions
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/mahmoodlab/CONCH.git
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/lilab-stanford/MUSK.git

# Visualization / notebook / IO extras
RUN pip install --no-cache-dir \
    scyjava \
    cucim \
    openslide-python \
    openslide-bin \
    spatialdata-plot \
    datashader \
    hf-xet \
    jupyterlab \
    ipywidgets \
    marsilea \
    mpl-fontkit \
    bokeh

# Create cache/work directories
RUN mkdir -p \
    /models/huggingface \
    /models/torch \
    /models/.cache \
    /workspace

COPY extract_features.py /opt/extract_features.py

WORKDIR /workspace
EXPOSE 8888
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
