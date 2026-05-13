"""
extract_features.py

Run LazySlide feature extraction on a whole slide image and save embeddings.

Usage:
    python extract_features.py -i slide.ndpi -o embeddings.h5ad -m gigapath
    python extract_features.py -i slide.ndpi -o embeddings.npz -m uni2 --mpp 0.5 --tile-px 256
    python extract_features.py -i slide.ndpi -o embeddings.pt -m virchow2 --batch-size 64 --amp
    python extract_features.py -i slide.ndpi -o embeddings.h5ad --model-path /models/uni.pt --model-name uni
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import lazyslide as zs
from wsidata import open_wsi


## Output formats -------------------------------------------------------------------

SUPPORTED_FORMATS = {".h5ad", ".pt", ".npz"}


def save_embeddings(adata, output_path: Path, model: str, verbose: bool = False):
    """Save embeddings to file based on output extension."""
    ext = output_path.suffix.lower()

    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported output format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    X = adata.X

    if sp.issparse(X):
        X = X.toarray()
    elif not isinstance(X, np.ndarray):
        X = np.asarray(X)

    X = X.astype(np.float32, copy=False)

    if verbose:
        print(f"Saving embeddings ({adata.shape[0]} tiles, {adata.shape[1]} dims) to {output_path}")

    if ext == ".h5ad":
        # AnnData format — best for scverse ecosystem (scanpy, squidpy)
        # preserves embeddings + tile metadata + spatial coordinates
        adata.write_h5ad(output_path)

    elif ext == ".pt":
        # PyTorch tensor format — convenient for training pipelines
        import torch

        torch.save({
            "embeddings": torch.tensor(X),
            "obs": adata.obs.to_dict(orient="list"),
            "model": model,
        }, output_path)

    elif ext == ".npz":
        # Compressed numpy — maximally portable, no extra dependencies
        np.savez_compressed(
            output_path,
            embeddings=X,
            model=model,
        )

    if verbose:
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"Saved to {output_path} ({size_mb:.1f} MB)")


## Argument parsing -------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run LazySlide feature extraction on a WSI and save embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    io = parser.add_argument_group("I/O")
    io.add_argument(
        "-i", "--image",
        required=True,
        metavar="FILE",
        help="Path to input whole slide image (WSI) file",
    )
    io.add_argument(
        "-o", "--output",
        required=True,
        metavar="FILE",
        help=f"Path to output embeddings file. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
    )

    # Model
    model_group = parser.add_argument_group("Model")
    model_source = model_group.add_mutually_exclusive_group()
    model_source.add_argument(
        "-m", "--model",
        default=None,
        metavar="NAME",
        help="LazySlide model name. Run `python -c 'import lazyslide as zs; print(zs.models.list_models())'` for all options.",
    )
    model_source.add_argument(
        "--model-path",
        default=None,
        metavar="PATH",
        help="Path to a local model file. Use for air-gapped environments. Requires --model-name.",
    )
    model_group.add_argument(
        "--model-name",
        default=None,
        metavar="NAME",
        help="Model name used for the output table key. Required when using --model-path; optional override when using --model.",
    )
    model_group.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="Batch size for inference",
    )
    model_group.add_argument(
        "--num-workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of dataloader workers",
    )
    model_group.add_argument(
        "--device",
        default=None,
        metavar="STR",
        help="Device for inference (e.g. 'cuda', 'cpu'). Auto-detected if not set.",
    )
    model_group.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision (faster on GPU, reduces memory usage)",
    )
    model_group.add_argument(
        "--token",
        default=None,
        metavar="STR",
        help="HuggingFace token for gated models. Falls back to HF_HOME/token if not set.",
    )

    # Tiling
    tiling = parser.add_argument_group("Tiling")
    tiling.add_argument(
        "--tile-px",
        type=int,
        default=256,
        metavar="N",
        help="Tile size in pixels",
    )
    tiling.add_argument(
        "--mpp",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Target microns per pixel for tiling (0.5 = 20x, 0.25 = 40x)",
    )
    tiling.add_argument(
        "--slide-mpp",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Override the MPP value read from the slide header. Useful for slides with missing or incorrect metadata.",
    )
    tiling.add_argument(
        "--overlap",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Tile overlap. If in (0, 1) treated as ratio, if > 1 treated as pixels.",
    )
    tiling.add_argument(
        "--background-fraction",
        type=float,
        default=0.3,
        metavar="FLOAT",
        help="Maximum background fraction allowed per tile (0-1). Tiles above this are discarded.",
    )
    tiling.add_argument(
        "--include-edge",
        action="store_true",
        help="Include edge tiles (partial tiles at tissue boundary)",
    )

    # Misc
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        metavar="N",
        help="Limit processing to the first N tiles. For testing only.",
    )

    return parser.parse_args(argv)


## Main -------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)

    output_path = Path(args.output)

    # Validate output format early before doing any work
    if output_path.suffix.lower() not in SUPPORTED_FORMATS:
        print(
            f"ERROR: Unsupported output format '{output_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
            file=sys.stderr,
        )
        return 1

    if not args.model and not args.model_path:
        print("ERROR: Either --model or --model-path must be specified", file=sys.stderr)
        return 1

    if args.model_path and not args.model_name:
        print("ERROR: --model-name is required when using --model-path", file=sys.stderr)
        return 1

    model_name = args.model_name or args.model
    model_key = f"{model_name}_tiles"

    # Open slide
    if args.verbose:
        print(f"Opening {args.image}")
    wsi = open_wsi(args.image)
    if args.verbose:
        print(wsi)

    # Tissue detection
    if args.verbose:
        print("Finding tissues...")
    zs.pp.find_tissues(wsi)
    if args.verbose:
        n_tissues = len(wsi.shapes["tissues"])
        print(f"Found {n_tissues} tissue region(s)")

    # Tiling
    if args.verbose:
        print(f"Tiling at {args.mpp} MPP, {args.tile_px}px tiles...")
    zs.pp.tile_tissues(
        wsi,
        tile_px=args.tile_px,
        mpp=args.mpp,
        slide_mpp=args.slide_mpp,
        overlap=args.overlap,
        edge=args.include_edge,
        background_fraction=args.background_fraction,
    )

    if len(wsi.shapes["tiles"]) == 0:
        print("ERROR: No tiles generated. Adjust tiling parameters and try again.", file=sys.stderr)
        return 1

    if args.max_tiles is not None:
        wsi.shapes["tiles"] = wsi.shapes["tiles"].iloc[:args.max_tiles]

    if args.verbose:
        n_tiles = len(wsi.shapes["tiles"])
        print(f"Processing {n_tiles} tiles")

    # Feature extraction
    if args.verbose:
        print(f"Extracting features with model '{model_name}'...")
    t0 = time.time()
    zs.tl.feature_extraction(
        wsi,
        model=args.model if not args.model_path else None,
        model_path=args.model_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        amp=args.amp,
        token=args.token,
    )
    elapsed = time.time() - t0
    print(f"Extraction complete in {elapsed/60:.1f} min ({elapsed:.1f} sec)")

    if model_key not in wsi.tables:
        print(
            f"ERROR: Feature table '{model_key}' not found. "
            f"Available tables: {list(wsi.tables.keys())}",
            file=sys.stderr,
        )
        return 1

    adata = wsi.tables[model_key]
    if args.verbose:
        print(f"Embeddings shape: {adata.shape}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_embeddings(adata, output_path, model=model_name, verbose=args.verbose)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
