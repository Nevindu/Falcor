#!/usr/bin/env python3
"""
Stage 1 DirectL/lenticular display simulator.

This is a purely analytic display-side simulator:
    1. Compute the DirectL-style view-order matrix V[x, y, k].
    2. Either load an already-rendered encoded/interleaved image, or synthesize one.
    3. Approximate what a viewer in a selected viewing zone sees by selecting the
       subpixels assigned to that view.

It does NOT ray trace the lenticular optics. It is meant to debug/calibrate:
    - view count Nv
    - line count Lx
    - slant angle alpha
    - offset K_offset
    - reverse view order

Dependencies:
    pip install numpy pillow matplotlib

Example synthetic test:
    python directl_stage1_view_sim.py --width 480 --height 270 --views 16 \
        --line-count 5.333333 --tilt-deg -2.12 --selected-views 0,4,8,12

Example with your rendered encoded image:
    python directl_stage1_view_sim.py --input encoded.png --views 48 \
        --line-count 5.333333 --tilt-deg -2.12 --offset 0 \
        --selected-views 0,8,16,24,32,40
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib.pyplot as plt


# -----------------------------
# Core DirectL-style mapping
# -----------------------------

def compute_view_matrix(
    height: int,
    width: int,
    n_views: int,
    line_count: float,
    tilt_deg: float = 0.0,
    offset: float = 0.0,
    reverse: bool = False,
) -> np.ndarray:
    """
    Compute DirectL-style view-order matrix V with shape (H, W, 3).

    V[x, y, k] is the view index assigned to LCD subpixel:
        x = row
        y = pixel column
        k = RGB channel index, 0=R, 1=G, 2=B

    DirectL-style formula:
        d_offset = 3*y + 3*x*tan(alpha) + k - K_offset
        x_offset = d_offset mod Lx
        v = floor(Nv * x_offset / Lx)
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if n_views <= 0:
        raise ValueError("n_views must be positive")
    if line_count <= 0:
        raise ValueError("line_count must be positive")

    x = np.arange(height, dtype=np.float64)[:, None, None]   # row
    y = np.arange(width, dtype=np.float64)[None, :, None]    # column
    k = np.arange(3, dtype=np.float64)[None, None, :]        # RGB subpixel

    alpha = np.deg2rad(tilt_deg)
    d_offset = 3.0 * y + 3.0 * x * np.tan(alpha) + k - offset
    x_offset = np.mod(d_offset, line_count)

    V = np.floor(n_views * x_offset / line_count).astype(np.int32)
    V = np.clip(V, 0, n_views - 1)

    if reverse:
        V = n_views - 1 - V

    return V


# -----------------------------
# Image helpers
# -----------------------------

def to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def load_rgb_image(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def save_rgb(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(img)).save(path)


def save_gray(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


# -----------------------------
# Synthetic multi-view content
# -----------------------------

def make_synthetic_views(height: int, width: int, n_views: int) -> np.ndarray:
    """
    Create synthetic full-resolution views for debugging interleaving.

    Output shape: (N, H, W, 3).
    Each view contains a moving vertical bar and view-dependent color/phase.
    """
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        np.linspace(0.0, 1.0, height, dtype=np.float32),
    )

    views = np.zeros((n_views, height, width, 3), dtype=np.float32)

    for v in range(n_views):
        t = v / max(1, n_views - 1)

        # Background gradient with a view-dependent tint.
        views[v, :, :, 0] = 0.15 + 0.65 * xx
        views[v, :, :, 1] = 0.15 + 0.65 * (1.0 - xx) * (0.4 + 0.6 * t)
        views[v, :, :, 2] = 0.20 + 0.65 * yy * (1.0 - 0.5 * t)

        # Moving bright vertical bar = simple parallax cue.
        center = int(width * (0.20 + 0.60 * t))
        bar_w = max(2, width // 80)
        x0 = max(0, center - bar_w)
        x1 = min(width, center + bar_w + 1)
        views[v, :, x0:x1, :] = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # View ID encoded as small-ish color stripe blocks.
        stripe_y0 = int(height * 0.08)
        stripe_y1 = int(height * 0.18)
        views[v, stripe_y0:stripe_y1, :, :] *= 0.45
        block_w = max(1, width // max(8, n_views))
        bx0 = min(width - 1, v * block_w % width)
        bx1 = min(width, bx0 + block_w)
        views[v, stripe_y0:stripe_y1, bx0:bx1, :] = np.array([1.0, 0.8, 0.1], dtype=np.float32)

    return np.clip(views, 0.0, 1.0)


def interleave_views(views: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Interleave full-resolution view images into one encoded LCD image.

    views: shape (N, H, W, 3)
    V:     shape (H, W, 3)
    """
    n_views, height, width, channels = views.shape
    if channels != 3:
        raise ValueError("views must be RGB")
    if V.shape != (height, width, 3):
        raise ValueError(f"V shape {V.shape} does not match views {(height, width, 3)}")

    encoded = np.zeros((height, width, 3), dtype=np.float32)

    for k in range(3):
        for v in range(n_views):
            mask = (V[:, :, k] == v)
            encoded[:, :, k][mask] = views[v, :, :, k][mask]

    return encoded


# -----------------------------
# Viewer-zone extraction
# -----------------------------

def extract_sparse_eye_view(encoded: np.ndarray, V: np.ndarray, view_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select the subpixels visible from one viewing zone.

    Returns:
        sparse_rgb: encoded subpixels belonging to view_id, black elsewhere.
        mask_rgb:   1 where channel belongs to view_id, 0 elsewhere.
    """
    if encoded.shape != V.shape:
        raise ValueError(f"encoded image shape {encoded.shape} must equal V shape {V.shape}")

    mask = (V == view_id).astype(np.float32)
    sparse = encoded * mask
    return sparse, mask


def normalized_gaussian_reconstruct(sparse: np.ndarray, mask: np.ndarray, radius: float = 1.25) -> np.ndarray:
    """
    Cheap eye-view reconstruction from sparse subpixels.

    This is NOT physical optics. It just visualizes the approximate content carried
    by one view by blurring the sparse samples and normalizing by blurred coverage.
    """
    if radius <= 0:
        return sparse

    sparse_img = Image.fromarray(to_uint8(sparse))
    mask_img = Image.fromarray(to_uint8(mask))

    num = np.asarray(sparse_img.filter(ImageFilter.GaussianBlur(radius=radius))).astype(np.float32) / 255.0
    den = np.asarray(mask_img.filter(ImageFilter.GaussianBlur(radius=radius))).astype(np.float32) / 255.0

    recon = num / np.maximum(den, 1e-4)
    recon[den < 1e-4] = 0.0
    return np.clip(recon, 0.0, 1.0)


# -----------------------------
# Visualization outputs
# -----------------------------

def save_view_matrix_plot(V: np.ndarray, n_views: int, out_path: str | Path) -> None:
    """Save H x (3W) subpixel view-order matrix visualization."""
    M = V.reshape(V.shape[0], V.shape[1] * 3)

    plt.figure(figsize=(12, 5))
    plt.imshow(M, interpolation="nearest", aspect="auto", vmin=0, vmax=max(1, n_views - 1), cmap="turbo")
    plt.title("DirectL-style subpixel view-order matrix V[x, 3y+k]")
    plt.xlabel("physical LCD subpixel column")
    plt.ylabel("LCD row")
    plt.colorbar(label="view index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_single_view_mask_plot(V: np.ndarray, view_id: int, out_path: str | Path) -> None:
    M = V.reshape(V.shape[0], V.shape[1] * 3)
    mask = (M == view_id).astype(np.float32)

    plt.figure(figsize=(12, 5))
    plt.imshow(mask, interpolation="nearest", aspect="auto", cmap="gray")
    plt.title(f"Physical subpixel positions assigned to view {view_id}")
    plt.xlabel("physical LCD subpixel column")
    plt.ylabel("LCD row")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_contact_sheet(items: List[Tuple[str, Image.Image]], out_path: str | Path, thumb_w: int = 320) -> None:
    """Create a simple labeled contact sheet."""
    if not items:
        return

    font = ImageFont.load_default()
    thumbs = []
    label_h = 22
    pad = 12

    for label, img in items:
        img = img.convert("RGB")
        w, h = img.size
        scale = thumb_w / max(1, w)
        thumb_h = max(1, int(h * scale))
        img = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)

        tile = Image.new("RGB", (thumb_w, thumb_h + label_h), "white")
        tile.paste(img, (0, label_h))
        d = ImageDraw.Draw(tile)
        d.text((4, 4), label, fill="black", font=font)
        thumbs.append(tile)

    cols = min(3, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    tile_w = thumb_w
    tile_h = max(t.height for t in thumbs)

    sheet = Image.new("RGB", (cols * tile_w + (cols + 1) * pad, rows * tile_h + (rows + 1) * pad), "white")

    for i, tile in enumerate(thumbs):
        r = i // cols
        c = i % cols
        x = pad + c * (tile_w + pad)
        y = pad + r * (tile_h + pad)
        sheet.paste(tile, (x, y))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


# -----------------------------
# CLI
# -----------------------------

def parse_selected_views(s: str, n_views: int) -> List[int]:
    if not s:
        # Good default: a few spread-out views.
        vals = sorted(set([0, n_views // 4, n_views // 2, (3 * n_views) // 4, n_views - 1]))
    else:
        vals = [int(x.strip()) for x in s.split(",") if x.strip()]

    return [v for v in vals if 0 <= v < n_views]


def main() -> None:
    repo_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Stage 1 DirectL/lenticular view-order simulator")

    parser.add_argument("--input", type=str, default=None, help="Optional rendered encoded/interleaved RGB image")
    parser.add_argument("--outdir", type=str, default=str(repo_dir / "outputs" / "stage1_outputs"), help="Output directory")

    parser.add_argument("--width", type=int, default=480, help="Synthetic display width if --input is not given")
    parser.add_argument("--height", type=int, default=270, help="Synthetic display height if --input is not given")

    parser.add_argument("--views", type=int, default=16, help="Number of views Nv")
    parser.add_argument("--line-count", type=float, default=5.333333, help="Lenticular line count Lx in subpixel-width units")
    parser.add_argument("--tilt-deg", type=float, default=-2.12, help="Lenticular slant angle alpha in degrees")
    parser.add_argument("--offset", type=float, default=0.0, help="Offset K_offset in subpixel-width units")
    parser.add_argument("--reverse", action="store_true", help="Reverse view order")

    parser.add_argument("--selected-views", type=str, default="", help="Comma-separated view IDs to extract, e.g. 0,8,16")
    parser.add_argument("--blur-radius", type=float, default=1.25, help="Gaussian radius for approximate eye-view reconstruction")
    parser.add_argument("--save-all-masks", action="store_true", help="Save masks for all views, not only selected views")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.input is not None:
        encoded = load_rgb_image(args.input)
        height, width = encoded.shape[:2]
        print(f"Loaded encoded image: {args.input} ({width} x {height})")
    else:
        width = args.width
        height = args.height
        encoded = None
        print(f"No --input provided. Generating synthetic encoded image ({width} x {height}).")

    V = compute_view_matrix(
        height=height,
        width=width,
        n_views=args.views,
        line_count=args.line_count,
        tilt_deg=args.tilt_deg,
        offset=args.offset,
        reverse=args.reverse,
    )

    # If no encoded image was loaded, synthesize multi-view content and interleave it.
    if encoded is None:
        views = make_synthetic_views(height, width, args.views)
        encoded = interleave_views(views, V)
        save_rgb(outdir / "synthetic_encoded.png", encoded)

        # Save a few full view images for reference.
        selected = parse_selected_views(args.selected_views, args.views)
        for v in selected:
            save_rgb(outdir / f"source_view_{v:03d}.png", views[v])
    else:
        selected = parse_selected_views(args.selected_views, args.views)

    save_rgb(outdir / "encoded_input_or_synthetic.png", encoded)
    save_view_matrix_plot(V, args.views, outdir / "view_order_matrix.png")

    # Numeric diagnostics.
    M = V.reshape(height, width * 3)
    counts = np.bincount(M.reshape(-1), minlength=args.views)
    density = counts / float(height * width * 3)

    print("\nParameters")
    print(f"  Nv/views      = {args.views}")
    print(f"  Lx/line-count = {args.line_count}")
    print(f"  alpha/tilt    = {args.tilt_deg} deg")
    print(f"  K_offset      = {args.offset}")
    print(f"  reverse       = {args.reverse}")
    print("\nPer-view subpixel counts")
    for v in range(args.views):
        print(f"  view {v:3d}: {counts[v]:8d} subpixels, density={density[v]:.5f}")

    # Save selected view masks and approximate observed views.
    contact_items: List[Tuple[str, Image.Image]] = []
    contact_items.append(("encoded image", Image.fromarray(to_uint8(encoded))))

    # Add matrix preview via saved file.
    contact_items.append(("view-order matrix", Image.open(outdir / "view_order_matrix.png")))

    mask_views = list(range(args.views)) if args.save_all_masks else selected

    for v in mask_views:
        save_single_view_mask_plot(V, v, outdir / f"mask_view_{v:03d}.png")

    for v in selected:
        sparse, mask = extract_sparse_eye_view(encoded, V, v)
        recon = normalized_gaussian_reconstruct(sparse, mask, radius=args.blur_radius)

        save_rgb(outdir / f"eye_view_{v:03d}_sparse.png", sparse)
        save_rgb(outdir / f"eye_view_{v:03d}_recon.png", recon)
        save_rgb(outdir / f"eye_view_{v:03d}_mask_rgb.png", mask)

        contact_items.append((f"view {v} sparse", Image.fromarray(to_uint8(sparse))))
        contact_items.append((f"view {v} approx recon", Image.fromarray(to_uint8(recon))))

    make_contact_sheet(contact_items, outdir / "contact_sheet.png")

    print(f"\nSaved outputs to: {outdir.resolve()}")
    print("Key files:")
    print("  view_order_matrix.png")
    print("  encoded_input_or_synthetic.png")
    print("  eye_view_XXX_sparse.png")
    print("  eye_view_XXX_recon.png")
    print("  contact_sheet.png")


if __name__ == "__main__":
    main()
