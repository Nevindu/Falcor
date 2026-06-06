import argparse
import glob
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


VIEW_RE = re.compile(r"view_(\d+)", re.IGNORECASE)


def find_views(input_dir):
    paths = []
    for ext in ("png", "jpg", "jpeg", "bmp", "tif", "tiff"):
        paths.extend(glob.glob(os.path.join(input_dir, f"view_*.{ext}")))
        paths.extend(glob.glob(os.path.join(input_dir, f"view_*.*.{ext}")))

    by_index = {}
    for path in paths:
        match = VIEW_RE.search(os.path.basename(path))
        if not match:
            continue
        idx = int(match.group(1))
        if idx not in by_index or os.path.getmtime(path) > os.path.getmtime(by_index[idx]):
            by_index[idx] = path

    return by_index


def make_order(views_x, views_y, trajectory):
    if trajectory == "row-major":
        return [x + y * views_x for y in range(views_y) for x in range(views_x)]

    if trajectory == "serpentine":
        order = []
        for y in range(views_y):
            xs = range(views_x) if y % 2 == 0 else range(views_x - 1, -1, -1)
            order.extend(x + y * views_x for x in xs)
        return order

    if trajectory == "horizontal":
        y = views_y // 2
        return [x + y * views_x for x in range(views_x)]

    if trajectory == "vertical":
        x = views_x // 2
        return [x + y * views_x for y in range(views_y)]

    if trajectory == "perimeter":
        order = []
        for x in range(views_x):
            order.append(x)
        for y in range(1, views_y):
            order.append((views_x - 1) + y * views_x)
        if views_y > 1:
            for x in range(views_x - 2, -1, -1):
                order.append(x + (views_y - 1) * views_x)
        if views_x > 1:
            for y in range(views_y - 2, 0, -1):
                order.append(y * views_x)
        return order

    raise ValueError(f"Unknown trajectory '{trajectory}'")


def load_frame(path, size=None, label=None):
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    if label:
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 160, 34), fill=(0, 0, 0))
        draw.text((14, 14), label, fill=(255, 255, 255))
    return image


def write_video(frames, output_path, fps):
    ext = os.path.splitext(output_path)[1].lower()

    if ext == ".gif":
        duration_ms = int(round(1000.0 / fps))
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        return

    try:
        import imageio.v2 as imageio
        import numpy as np
    except ImportError as e:
        raise RuntimeError("MP4 output requires imageio and numpy. Use --output trajectory.gif as a fallback.") from e

    try:
        with imageio.get_writer(output_path, fps=fps, macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame))
        return
    except Exception:
        if shutil.which("ffmpeg") is None:
            raise

    with tempfile.TemporaryDirectory(prefix="lightfield_video_") as tmp_dir:
        for i, frame in enumerate(frames):
            frame.save(os.path.join(tmp_dir, f"frame_{i:05d}.png"))

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(tmp_dir, "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Create a video that moves through rendered light-field views.")
    parser.add_argument("input_dir", help="Directory containing view_*.png images.")
    parser.add_argument("--views-x", type=int, required=True, help="Horizontal view count.")
    parser.add_argument("--views-y", type=int, required=True, help="Vertical view count.")
    parser.add_argument("--output", default=None, help="Output .mp4 or .gif path. Defaults to input_dir/trajectory.mp4.")
    parser.add_argument("--trajectory", choices=["row-major", "serpentine", "horizontal", "vertical", "perimeter"], default="serpentine")
    parser.add_argument("--fps", type=float, default=12.0, help="Output frames per second.")
    parser.add_argument("--loops", type=int, default=1, help="Repeat the trajectory this many times.")
    parser.add_argument("--pingpong", action="store_true", help="Append the reverse trajectory for a smooth back-and-forth video.")
    parser.add_argument("--hold-ends", type=int, default=0, help="Duplicate first and last trajectory frame this many times.")
    parser.add_argument("--width", type=int, default=None, help="Optional output width.")
    parser.add_argument("--height", type=int, default=None, help="Optional output height.")
    parser.add_argument("--label", action="store_true", help="Draw view index labels on frames.")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output = os.path.abspath(args.output or os.path.join(input_dir, "trajectory.mp4"))
    Path(os.path.dirname(output)).mkdir(parents=True, exist_ok=True)

    views = find_views(input_dir)
    expected = args.views_x * args.views_y
    missing = [i for i in range(expected) if i not in views]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} views. First missing indices: {missing[:10]}")

    order = make_order(args.views_x, args.views_y, args.trajectory)
    if args.pingpong and len(order) > 1:
        order = order + order[-2:0:-1]
    order = order * max(1, args.loops)

    if args.hold_ends > 0 and order:
        order = [order[0]] * args.hold_ends + order + [order[-1]] * args.hold_ends

    size = None
    if args.width or args.height:
        first = Image.open(views[order[0]])
        w, h = first.size
        out_w = args.width or round(w * (args.height / h))
        out_h = args.height or round(h * (args.width / w))
        size = (out_w, out_h)

    frames = []
    for idx in order:
        x = idx % args.views_x
        y = idx // args.views_x
        label = f"view {idx:03d} ({x},{y})" if args.label else None
        frames.append(load_frame(views[idx], size=size, label=label))

    write_video(frames, output, args.fps)
    print(f"Wrote {len(frames)} frames to {output}")


if __name__ == "__main__":
    main()
