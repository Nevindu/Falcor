import argparse
import os
import subprocess
import sys
import tempfile


def _repo_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_mogwai(repo_dir, config):
    return os.path.join(repo_dir, "build", "windows-vs2022", "bin", config, "Mogwai.exe")


def _bool_literal(value):
    return "True" if value else "False"


def main():
    repo_dir = _repo_dir()

    parser = argparse.ArgumentParser(description="Render an LFSubPixelPathTracer interleaved image in headless Mogwai.")
    parser.add_argument("--mogwai", default=None, help="Path to Mogwai.exe.")
    parser.add_argument("--config", default="Release", choices=["Debug", "Release"], help="VS2022 build config to use when --mogwai is omitted.")
    parser.add_argument("--scene", default=os.path.join(repo_dir, "media", "classroom", "scene-v4.pbrt"), help="Scene to load.")
    parser.add_argument("--width", type=int, default=1280, help="Output width.")
    parser.add_argument("--height", type=int, default=720, help="Output height.")
    parser.add_argument("--frames", type=int, default=64, help="Accumulation frames before capture. Total rendered frames is frames + 1 capture frame.")
    parser.add_argument("--samples-per-pixel", type=int, default=1, help="LFSubPixelPathTracer samplesPerPixel setting.")
    parser.add_argument("--output-dir", default=os.path.join(repo_dir, "outputs", "lightfield_classroom"), help="Directory for capture, log, temp script, and summary JSON.")

    parser.add_argument("--view-count", type=int, default=48, help="Number of lenticular/light-field views.")
    parser.add_argument("--line-count", type=float, default=5.333333, help="DirectL lenticular pitch in subpixel-width units.")
    parser.add_argument("--tilt-degrees", type=float, default=0.0, help="Lenticular slant angle in degrees.")
    parser.add_argument("--offset", type=float, default=0.0, help="DirectL horizontal display/lens offset in subpixel-width units.")
    parser.add_argument("--baseline", type=float, default=1.0, help="Total virtual camera-array baseline in scene units.")
    parser.add_argument("--convergence-distance", type=float, default=4.0, help="Toe-in target distance for camera mode 1.")
    parser.add_argument("--camera-mode", type=int, default=0, choices=[0, 1], help="0 = parallel, 1 = toe-in/recenter.")
    parser.add_argument("--reverse-view-order", action=argparse.BooleanOptionalAction, default=False, help="Flip lenticular view index ordering.")

    parser.add_argument("--device-type", default="d3d12", choices=["d3d12", "vulkan"], help="Mogwai graphics backend.")
    parser.add_argument("--gpu", type=int, default=None, help="Optional GPU index.")
    parser.add_argument("--precise", action="store_true", help="Pass --precise to Mogwai.")
    args = parser.parse_args()

    mogwai = args.mogwai or _default_mogwai(repo_dir, args.config)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    driver = os.path.join(repo_dir, "scripts", "RunLFSubPixelPathTracer.py")
    script = f"""
LFSP_SCENE = r"{args.scene}"
LFSP_OUTPUT_DIR = r"{output_dir}"
LFSP_WIDTH = {args.width}
LFSP_HEIGHT = {args.height}
LFSP_FRAMES = {args.frames}
LFSP_SAMPLES_PER_PIXEL = {args.samples_per_pixel}
LFSP_VIEW_COUNT = {args.view_count}
LFSP_LINE_COUNT = {args.line_count}
LFSP_TILT_DEGREES = {args.tilt_degrees}
LFSP_OFFSET = {args.offset}
LFSP_BASELINE = {args.baseline}
LFSP_CONVERGENCE_DISTANCE = {args.convergence_distance}
LFSP_CAMERA_MODE = {args.camera_mode}
LFSP_REVERSE_VIEW_ORDER = {_bool_literal(args.reverse_view_order)}
m.script(r"{driver}")
"""

    fd, script_path = tempfile.mkstemp(prefix="lfsubpixel_", suffix=".py", dir=output_dir, text=True)
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
    print("Output:", output_dir)
    result = subprocess.run(cmd, cwd=repo_dir)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
