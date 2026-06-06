import argparse
import glob
import os
import subprocess
import sys
import tempfile


def _repo_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_mogwai(repo_dir, config):
    return os.path.join(repo_dir, "build", "windows-vs2022", "bin", config, "Mogwai.exe")


def _split_latest_quilt(output_dir, views_x, views_y):
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is not installed; saved the quilt atlas but skipped per-view splitting.")
        return

    pattern = os.path.join(output_dir, "classroom_reproj_*.ToneMapper.dst.*.png")
    captures = glob.glob(pattern)
    if not captures:
        print("No captured quilt found matching: {}".format(pattern))
        return

    quilt_path = max(captures, key=os.path.getmtime)
    image = Image.open(quilt_path)
    tile_w = image.width // views_x
    tile_h = image.height // views_y

    if tile_w * views_x != image.width or tile_h * views_y != image.height:
        print("Quilt size {}x{} is not divisible by {}x{}; skipped splitting.".format(image.width, image.height, views_x, views_y))
        return

    views_dir = os.path.join(output_dir, "views")
    os.makedirs(views_dir, exist_ok=True)

    for y in range(views_y):
        for x in range(views_x):
            view_index = x + y * views_x
            tile = image.crop((x * tile_w, y * tile_h, (x + 1) * tile_w, (y + 1) * tile_h))
            tile.save(os.path.join(views_dir, "view_{:03d}.png".format(view_index)))

    print("Split quilt into {} views in {}".format(views_x * views_y, views_dir))


def main():
    repo_dir = _repo_dir()

    parser = argparse.ArgumentParser(description="Render LightFieldReprojPathTracer in headless Mogwai and save the quilt/views.")
    parser.add_argument("--mogwai", default=None, help="Path to Mogwai.exe.")
    parser.add_argument("--config", default="Release", choices=["Debug", "Release"], help="VS2022 build config to use when --mogwai is omitted.")
    parser.add_argument("--scene", default=os.path.join(repo_dir, "media", "classroom", "scene-v4.pbrt"), help="Scene to load.")
    parser.add_argument("--width", type=int, default=1920, help="Quilt atlas width, or per-view width when --per-view-resolution is set.")
    parser.add_argument("--height", type=int, default=1080, help="Quilt atlas height, or per-view height when --per-view-resolution is set.")
    parser.add_argument("--frames", type=int, default=64, help="Accumulation frames before capture. Total rendered frames is frames + 1 capture frame.")
    parser.add_argument("--samples-per-pixel", type=int, default=1, help="LightFieldReprojPathTracer samplesPerPixel setting.")
    parser.add_argument("--output-dir", default=os.path.join(repo_dir, "outputs", "lightfield_classroom_reproj"), help="Directory for capture, log, temp script, summary JSON, and split views.")

    parser.add_argument("--views-x", type=int, default=6, help="Number of horizontal quilt views.")
    parser.add_argument("--views-y", type=int, default=6, help="Number of vertical quilt views.")
    parser.add_argument("--per-view-resolution", action="store_true", help="Treat --width/--height as each view's resolution and render a larger quilt atlas.")
    parser.add_argument("--source-view-x", type=int, default=2, help="Source view X index.")
    parser.add_argument("--source-view-y", type=int, default=2, help="Source view Y index.")
    parser.add_argument("--baseline-x", type=float, default=0.01, help="Horizontal light-field baseline.")
    parser.add_argument("--baseline-y", type=float, default=0.01, help="Vertical light-field baseline.")
    parser.add_argument("--position-threshold", type=float, default=0.01, help="World-position reprojection validity threshold.")
    parser.add_argument("--normal-dot-threshold", type=float, default=0.95, help="Normal-dot reprojection validity threshold.")
    parser.add_argument("--temporal-alpha", type=float, default=0.2, help="Temporal alpha parameter passed to the render pass.")
    parser.add_argument("--display-mode", type=int, default=0, choices=[0, 1, 2, 3, 4], help="0 final, 2 valid mask, 3 discard overlay, 4 view ID.")
    parser.add_argument("--no-denoiser", action="store_true", help="Render without NRD. Path trace the source tile, reproject, and fallback trace invalid pixels directly.")
    parser.add_argument("--no-split-views", action="store_true", help="Only save the quilt atlas; do not split it into view PNGs.")

    parser.add_argument("--device-type", default="d3d12", choices=["d3d12", "vulkan"], help="Mogwai graphics backend.")
    parser.add_argument("--gpu", type=int, default=None, help="Optional GPU index.")
    parser.add_argument("--precise", action="store_true", help="Pass --precise to Mogwai.")
    args = parser.parse_args()

    mogwai = args.mogwai or _default_mogwai(repo_dir, args.config)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    atlas_width = args.width * args.views_x if args.per_view_resolution else args.width
    atlas_height = args.height * args.views_y if args.per_view_resolution else args.height

    driver = os.path.join(repo_dir, "scripts", "RunLightFieldReprojPathTracer.py")
    script = f"""
LFR_SCENE = r"{args.scene}"
LFR_OUTPUT_DIR = r"{output_dir}"
LFR_WIDTH = {atlas_width}
LFR_HEIGHT = {atlas_height}
LFR_FRAMES = {args.frames}
LFR_SAMPLES_PER_PIXEL = {args.samples_per_pixel}
LFR_VIEW_GRID_X = {args.views_x}
LFR_VIEW_GRID_Y = {args.views_y}
LFR_SOURCE_VIEW_X = {args.source_view_x}
LFR_SOURCE_VIEW_Y = {args.source_view_y}
LFR_BASELINE_X = {args.baseline_x}
LFR_BASELINE_Y = {args.baseline_y}
LFR_POSITION_THRESHOLD = {args.position_threshold}
LFR_NORMAL_DOT_THRESHOLD = {args.normal_dot_threshold}
LFR_TEMPORAL_ALPHA = {args.temporal_alpha}
LFR_DISPLAY_MODE = {args.display_mode}
LFR_USE_DENOISER = {not args.no_denoiser}
m.script(r"{driver}")
"""

    fd, script_path = tempfile.mkstemp(prefix="lightfield_reproj_", suffix=".py", dir=output_dir, text=True)
    with os.fdopen(fd, "w") as f:
        f.write(script)

    logfile = os.path.join(output_dir, "log.txt")
    cmd = [
        mogwai,
        "--headless",
        "--device-type",
        args.device_type,
        "--script",
        script_path,
        "--logfile",
        logfile,
    ]
    if args.gpu is not None:
        cmd += ["--gpu", str(args.gpu)]
    if args.precise:
        cmd += ["--precise"]

    print("Running:", " ".join('"{}"'.format(x) if " " in x else x for x in cmd))
    if args.per_view_resolution:
        print("Per-view resolution: {}x{}; atlas: {}x{}".format(args.width, args.height, atlas_width, atlas_height))
    else:
        print("Atlas resolution: {}x{}".format(atlas_width, atlas_height))
    print("Output:", output_dir)
    result = subprocess.run(cmd, cwd=repo_dir)
    if result.returncode != 0:
        return result.returncode

    if not args.no_split_views:
        _split_latest_quilt(output_dir, args.views_x, args.views_y)

    return 0


if __name__ == "__main__":
    sys.exit(main())
