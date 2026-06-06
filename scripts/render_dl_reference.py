import argparse
import os
import subprocess
import sys


def _repo_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    repo_dir = _repo_dir()

    parser = argparse.ArgumentParser(description="Render a high-spp DirectLighting reference in headless Mogwai.")
    parser.add_argument("--scene", default=r"C:\workspace\Falcor\media\classroom\scene-v4.pbrt", help="Scene to load.")
    parser.add_argument("--width", type=int, default=1280, help="Output width.")
    parser.add_argument("--height", type=int, default=720, help="Output height.")
    parser.add_argument("--frames", type=int, default=512, help="Accumulation frames before capture.")
    parser.add_argument("--samples-per-pixel", type=int, default=16, help="DirectLighting samplesPerPixel per frame.")
    parser.add_argument("--vbuffer-samples", type=int, default=16, help="VBufferRT sampleCount setting.")
    parser.add_argument("--output-dir", default=os.path.join(repo_dir, "outputs", "lighting_validation", "directlighting_reference"), help="Output directory.")
    parser.add_argument("--config", default="Release", choices=["Debug", "Release"], help="Build config whose Mogwai.exe should be used.")
    parser.add_argument("--mogwai", default=None, help="Optional explicit Mogwai.exe path.")
    parser.add_argument("--device-type", default="d3d12", choices=["d3d12", "vulkan"], help="Mogwai graphics backend.")
    parser.add_argument("--gpu", type=int, default=None, help="Optional GPU index.")
    parser.add_argument("--precise", action="store_true", help="Pass --precise to Mogwai.")
    parser.add_argument("--use-brdf-sampling", action=argparse.BooleanOptionalAction, default=True, help="Toggle BSDF sampling in DirectLighting.")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        os.path.join(repo_dir, "scripts", "run_lighting_batch.py"),
        "--config",
        args.config,
        "--scene",
        args.scene,
        "--integrator",
        "directlighting",
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--frames",
        str(args.frames),
        "--samples-per-pixel",
        str(args.samples_per_pixel),
        "--vbuffer-samples",
        str(args.vbuffer_samples),
        "--output-dir",
        args.output_dir,
        "--device-type",
        args.device_type,
    ]
    cmd.append("--use-brdf-sampling" if args.use_brdf_sampling else "--no-use-brdf-sampling")
    if args.mogwai:
        cmd += ["--mogwai", args.mogwai]
    if args.gpu is not None:
        cmd += ["--gpu", str(args.gpu)]
    if args.precise:
        cmd.append("--precise")

    print("DirectLighting reference command:")
    print(" ".join('"{}"'.format(x) if " " in x else x for x in cmd))
    return subprocess.run(cmd, cwd=repo_dir).returncode


if __name__ == "__main__":
    sys.exit(main())
