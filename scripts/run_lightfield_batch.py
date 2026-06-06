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

    parser = argparse.ArgumentParser(description="Run the LightFieldPathTracer batch renderer in headless Mogwai.")
    parser.add_argument("--mogwai", default=None, help="Path to Mogwai.exe.")
    parser.add_argument("--config", default="Debug", choices=["Debug", "Release"], help="VS2022 build config to use when --mogwai is omitted.")
    parser.add_argument("--scene", default="Arcade/Arcade.pyscene", help="Scene to load.")
    parser.add_argument("--views-x", type=int, default=8, help="Number of horizontal light-field views.")
    parser.add_argument("--views-y", type=int, default=1, help="Number of vertical light-field views.")
    parser.add_argument("--view-width", type=int, default=512, help="Per-view output width.")
    parser.add_argument("--view-height", type=int, default=512, help="Per-view output height.")
    parser.add_argument("--baseline-x", type=float, default=0.08, help="World-space horizontal camera-array width.")
    parser.add_argument("--baseline-y", type=float, default=0.0, help="World-space vertical camera-array height.")
    parser.add_argument("--frames-per-view", type=int, default=8, help="Accumulation frames to render before each capture.")
    parser.add_argument("--samples-per-pixel", type=int, default=1, help="LightFieldPathTracer samplesPerPixel setting.")
    parser.add_argument("--output-dir", default=os.path.join(repo_dir, "outputs", "lightfield"), help="Directory for images, log, temp script, and summary JSON.")
    parser.add_argument("--device-type", default="d3d12", choices=["d3d12", "vulkan"], help="Mogwai graphics backend.")
    parser.add_argument("--gpu", type=int, default=None, help="Optional GPU index.")
    parser.add_argument("--precise", action="store_true", help="Pass --precise to Mogwai.")
    args = parser.parse_args()

    mogwai = args.mogwai or _default_mogwai(repo_dir, args.config)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if args.samples_per_pixel > 16:
        print("Warning: LightFieldPathTracer clamps --samples-per-pixel to 16. Use --frames-per-view for more accumulated samples.")

    driver = os.path.join(repo_dir, "scripts", "RunLightFieldPathTracerArcade.py")
    script = f"""
LF_SCENE = r"{args.scene}"
LF_OUTPUT_DIR = r"{output_dir}"
LF_VIEW_COUNT_X = {args.views_x}
LF_VIEW_COUNT_Y = {args.views_y}
LF_VIEW_WIDTH = {args.view_width}
LF_VIEW_HEIGHT = {args.view_height}
LF_BASELINE_X = {args.baseline_x}
LF_BASELINE_Y = {args.baseline_y}
LF_FRAMES_PER_VIEW = {args.frames_per_view}
LF_SAMPLES_PER_PIXEL = {args.samples_per_pixel}
m.script(r"{driver}")
"""

    fd, script_path = tempfile.mkstemp(prefix="lightfield_batch_", suffix=".py", dir=output_dir, text=True)
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
