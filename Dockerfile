FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

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
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    git \
    default-jdk \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libopenslide0 \
    libvips42 \
    build-essential \
    tini \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -f /usr/lib/python3*/EXTERNALLY-MANAGED \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN pip install --no-cache-dir --upgrade \
    pip setuptools wheel

# Pin transformers — TITAN remote code predates all_tied_weights_keys (added post 4.49)
RUN pip install --no-cache-dir "transformers==4.49.0"

# CUDA torch — must precede lazyslide to avoid pulling CPU variant
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Core lazyslide — pip (stable) or source (latest main)
# Build with: docker build --build-arg LAZYSLIDE_SOURCE=source .
ARG LAZYSLIDE_SOURCE=pip
RUN if [ "$LAZYSLIDE_SOURCE" = "source" ]; then \
        pip install --no-cache-dir \
        "lazyslide[all,models] @ git+https://github.com/rendeirolab/LazySlide.git"; \
    else \
        pip install --no-cache-dir \
        "lazyslide[all,models]"; \
    fi

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

# GigaPath slide encoder dependencies
# ninja speeds up CUDA extension compilation
# omegaconf, fvcore, iopath, einops required by prov-gigapath
RUN pip install --no-cache-dir \
    ninja \
    omegaconf \
    fvcore \
    iopath \
    einops

# GigaPath slide encoder (required for feature_aggregation with gigapath-slide-encoder)
RUN pip install --no-cache-dir \
    git+https://github.com/prov-gigapath/prov-gigapath

# FlashAttention — required by GigaPath slide encoder (LongNet attention)
# Pinned to 2.5.8 as specified in official prov-gigapath environment
# Force old C++ ABI to match PyTorch wheels from PyPI (compiled with cxx11abi=FALSE)
RUN FLASH_ATTENTION_FORCE_BUILD=TRUE \
    CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
    pip install --no-cache-dir "flash-attn==2.5.8" --no-build-isolation

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
