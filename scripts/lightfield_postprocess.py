import argparse
import glob
import math
import os
import re
from pathlib import Path

from PIL import Image


VIEW_RE = re.compile(r"view_(\d+)", re.IGNORECASE)


def find_views(input_dir):
    paths = []
    for ext in ("png", "jpg", "jpeg", "bmp", "tif", "tiff"):
        paths.extend(glob.glob(os.path.join(input_dir, f"view_*.{ext}")))
        paths.extend(glob.glob(os.path.join(input_dir, f"view_*.*.{ext}")))

    indexed = []
    for path in paths:
        match = VIEW_RE.search(os.path.basename(path))
        if match:
            indexed.append((int(match.group(1)), path))

    # If a capture produced both RGB and RGBA variants for the same view, keep the newest file.
    by_index = {}
    for idx, path in indexed:
        if idx not in by_index or os.path.getmtime(path) > os.path.getmtime(by_index[idx]):
            by_index[idx] = path

    return [by_index[i] for i in sorted(by_index.keys())]


def load_views(input_dir, views_x=None, views_y=None):
    paths = find_views(input_dir)
    if not paths:
        raise RuntimeError(f"No view images found in {input_dir}")

    if views_x is None and views_y is None:
        views_x = len(paths)
        views_y = 1
    elif views_x is None:
        views_x = math.ceil(len(paths) / views_y)
    elif views_y is None:
        views_y = math.ceil(len(paths) / views_x)

    expected = views_x * views_y
    if len(paths) < expected:
        raise RuntimeError(f"Expected {expected} views for {views_x}x{views_y}, found {len(paths)}")

    images = [Image.open(path).convert("RGBA") for path in paths[:expected]]
    return images, paths[:expected], views_x, views_y


def make_mosaic(images, views_x, views_y):
    view_w, view_h = images[0].size
    mosaic = Image.new("RGBA", (views_x * view_w, views_y * view_h))

    for idx, image in enumerate(images):
        x = idx % views_x
        y = idx // views_x
        mosaic.paste(image, (x * view_w, y * view_h))

    return mosaic


def refocus(images, views_x, views_y, focus, crop=True):
    """Shift-and-add refocus.

    `focus` is in pixels per view-step. Positive values align farther views in
    the +x/+y direction relative to the center view; negative values flip that.
    """
    view_w, view_h = images[0].size
    acc = [0.0] * (view_w * view_h * 4)

    cx = (views_x - 1) * 0.5
    cy = (views_y - 1) * 0.5

    for idx, image in enumerate(images):
        vx = idx % views_x
        vy = idx // views_x
        dx = (vx - cx) * focus
        dy = (vy - cy) * focus
        shifted = image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1.0, 0.0, -dx, 0.0, 1.0, -dy),
            resample=Image.Resampling.BICUBIC,
            fillcolor=(0, 0, 0, 0),
        )

        for i, (r, g, b, a) in enumerate(shifted.getdata()):
            out = i * 4
            acc[out + 0] += r
            acc[out + 1] += g
            acc[out + 2] += b
            acc[out + 3] += a

    out_pixels = []
    for i in range(view_w * view_h):
        j = i * 4
        out_pixels.append(
            (
                int(round(acc[j + 0] / len(images))),
                int(round(acc[j + 1] / len(images))),
                int(round(acc[j + 2] / len(images))),
                int(round(acc[j + 3] / len(images))),
            )
        )

    result = Image.new("RGBA", (view_w, view_h))
    result.putdata(out_pixels)

    if crop:
        max_dx = int(math.ceil(abs(focus) * max(cx, 0.0)))
        max_dy = int(math.ceil(abs(focus) * max(cy, 0.0)))
        if max_dx * 2 < view_w and max_dy * 2 < view_h:
            result = result.crop((max_dx, max_dy, view_w - max_dx, view_h - max_dy))

    return result


def parse_focus_values(args):
    if args.focus_values:
        return [float(x.strip()) for x in args.focus_values.split(",") if x.strip()]

    if args.focus_count <= 1:
        return [args.focus_min]

    step = (args.focus_max - args.focus_min) / (args.focus_count - 1)
    return [args.focus_min + i * step for i in range(args.focus_count)]


def main():
    parser = argparse.ArgumentParser(description="Create a light-field mosaic and focal stack from rendered view images.")
    parser.add_argument("input_dir", help="Directory containing view_*.png images.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to input_dir/postprocess.")
    parser.add_argument("--views-x", type=int, default=None, help="Horizontal view count.")
    parser.add_argument("--views-y", type=int, default=None, help="Vertical view count.")
    parser.add_argument("--mosaic-name", default="mosaic.png", help="Mosaic output filename.")
    parser.add_argument("--focus-values", default=None, help="Comma-separated focus shifts, e.g. -12,-6,0,6,12.")
    parser.add_argument("--focus-min", type=float, default=-4.0, help="Minimum focus shift in pixels per view-step.")
    parser.add_argument("--focus-max", type=float, default=4.0, help="Maximum focus shift in pixels per view-step.")
    parser.add_argument("--focus-count", type=int, default=17, help="Number of focal-stack images.")
    parser.add_argument("--no-crop", action="store_true", help="Keep full frame with edge fill instead of cropping to common overlap.")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(input_dir, "postprocess"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    images, paths, views_x, views_y = load_views(input_dir, args.views_x, args.views_y)
    print(f"Loaded {len(images)} views as {views_x}x{views_y}")

    mosaic = make_mosaic(images, views_x, views_y)
    mosaic_path = os.path.join(output_dir, args.mosaic_name)
    mosaic.save(mosaic_path)
    print(f"Wrote mosaic: {mosaic_path}")

    stack_dir = os.path.join(output_dir, "focal_stack")
    Path(stack_dir).mkdir(parents=True, exist_ok=True)

    for i, focus in enumerate(parse_focus_values(args)):
        image = refocus(images, views_x, views_y, focus, crop=not args.no_crop)
        path = os.path.join(stack_dir, f"focus_{i:03d}_{focus:+.3f}.png")
        image.save(path)
        print(f"Wrote focal slice {i}: focus={focus:+.3f} -> {path}")


if __name__ == "__main__":
    main()
