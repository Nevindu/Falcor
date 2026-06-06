import argparse
import os
import subprocess
import sys
import tempfile


def _repo_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_mogwai(repo_dir, config):
    return os.path.join(repo_dir, "build", "windows-vs2022", "bin", config, "Mogwai.exe")


def main():
    repo_dir = _repo_dir()

    parser = argparse.ArgumentParser(description="Open LightFieldReprojPathTracer interactively in Mogwai.")
    parser.add_argument("--mogwai", default=None, help="Path to Mogwai.exe.")
    parser.add_argument("--config", default="Release", choices=["Debug", "Release"], help="VS2022 build config to use when --mogwai is omitted.")
    parser.add_argument("--scene", default=os.path.join(repo_dir, "media", "classroom", "scene-v4.pbrt"), help="Scene to load.")
    parser.add_argument("--width", type=int, default=1920, help="Atlas width, or per-view width when --per-view-resolution is set.")
    parser.add_argument("--height", type=int, default=1080, help="Atlas height, or per-view height when --per-view-resolution is set.")
    parser.add_argument("--views-x", type=int, default=6, help="Number of horizontal quilt views.")
    parser.add_argument("--views-y", type=int, default=6, help="Number of vertical quilt views.")
    parser.add_argument("--per-view-resolution", action="store_true", help="Treat --width/--height as each view's resolution and open a larger quilt atlas.")
    parser.add_argument("--samples-per-pixel", type=int, default=1, help="LightFieldReprojPathTracer samplesPerPixel setting.")
    parser.add_argument("--source-view-x", type=int, default=2, help="Source view X index.")
    parser.add_argument("--source-view-y", type=int, default=2, help="Source view Y index.")
    parser.add_argument("--baseline-x", type=float, default=0.01, help="Horizontal light-field baseline.")
    parser.add_argument("--baseline-y", type=float, default=0.01, help="Vertical light-field baseline.")
    parser.add_argument("--position-threshold", type=float, default=0.01, help="World-position reprojection validity threshold.")
    parser.add_argument("--normal-dot-threshold", type=float, default=0.95, help="Normal-dot reprojection validity threshold.")
    parser.add_argument("--temporal-alpha", type=float, default=0.2, help="SVGF temporal alpha.")
    parser.add_argument("--display-mode", type=int, default=0, choices=[0, 1, 2, 3, 4], help="0 final, 2 valid mask, 3 discard overlay, 4 view ID.")
    parser.add_argument("--no-denoiser", action="store_true", help="Disable SVGF tile denoising.")
    parser.add_argument("--device-type", default="d3d12", choices=["d3d12", "vulkan"], help="Mogwai graphics backend.")
    parser.add_argument("--gpu", type=int, default=None, help="Optional GPU index.")
    parser.add_argument("--precise", action="store_true", help="Pass --precise to Mogwai.")
    args = parser.parse_args()

    mogwai = args.mogwai or _default_mogwai(repo_dir, args.config)
    driver = os.path.join(repo_dir, "scripts", "LightFieldReprojPathTracerInteractive.py")

    script = f"""
LFR_SCENE = r"{args.scene}"
LFR_WIDTH = {args.width}
LFR_HEIGHT = {args.height}
LFR_VIEW_GRID_X = {args.views_x}
LFR_VIEW_GRID_Y = {args.views_y}
LFR_PER_VIEW_RESOLUTION = {args.per_view_resolution}
LFR_SAMPLES_PER_PIXEL = {args.samples_per_pixel}
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

    fd, script_path = tempfile.mkstemp(prefix="lightfield_reproj_interactive_", suffix=".py", text=True)
    with os.fdopen(fd, "w") as f:
        f.write(script)

    cmd = [mogwai, "--device-type", args.device_type, "--script", script_path]
    if args.gpu is not None:
        cmd += ["--gpu", str(args.gpu)]
    if args.precise:
        cmd += ["--precise"]

    print("Running:", " ".join('"{}"'.format(x) if " " in x else x for x in cmd))
    return subprocess.run(cmd, cwd=repo_dir).returncode


if __name__ == "__main__":
    sys.exit(main())
