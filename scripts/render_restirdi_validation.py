import argparse
import json
import os
import subprocess
import sys
import time


def _repo_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _flag(name, enabled):
    return name if enabled else "--no-" + name[2:]


def _variant_commands(args, repo_dir, output_root):
    base = [
        sys.executable,
        os.path.join(repo_dir, "scripts", "run_lighting_batch.py"),
        "--config",
        args.config,
        "--scene",
        args.scene,
        "--integrator",
        "restirdi",
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--samples-per-pixel",
        "2",
        "--restir-light-candidates",
        "4",
        "--restir-temporal-m-cap",
        "1",
        "--restir-spatial-radius",
        str(args.spatial_radius),
        "--restir-spatial-m-cap",
        str(args.spatial_m_cap),
        "--restir-spatial-mis-strategy",
        "pairwise",
        "--restir-visibility-in-target",
        "--restir-visibility-in-temporal-mis",
        "--vbuffer-samples",
        str(args.vbuffer_samples),
        "--device-type",
        args.device_type,
        "--no-accumulate",
    ]
    if args.mogwai:
        base += ["--mogwai", args.mogwai]
    if args.gpu is not None:
        base += ["--gpu", str(args.gpu)]
    if args.precise:
        base.append("--precise")

    variants = [
        {
            "name": "01_initial_ris_light_only",
            "description": "Initial RIS only: M_ris=4, no BRDF, no temporal, no spatial.",
            "brdf": False,
            "brdf_candidates": 0,
            "temporal": False,
            "spatial_neighbors": 0,
            "warmup_frames": 0,
            "capture_label": "ReSTIRDI_initial_light_only_2spp_1frame",
        },
        {
            "name": "02_initial_ris_plus_brdf",
            "description": "Initial RIS with BRDF candidates: M_ris=4, M_brdf=1, no temporal, no spatial.",
            "brdf": True,
            "brdf_candidates": 1,
            "temporal": False,
            "spatial_neighbors": 0,
            "warmup_frames": 0,
            "capture_label": "ReSTIRDI_initial_brdf_2spp_1frame",
        },
        {
            "name": "03_temporal_only",
            "description": "Temporal reuse only: M_ris=4, M_brdf=1, temporal M cap=1, no spatial.",
            "brdf": True,
            "brdf_candidates": 1,
            "temporal": True,
            "spatial_neighbors": 0,
            "warmup_frames": 1,
            "capture_label": "ReSTIRDI_temporal_only_2spp_2frames",
        },
        {
            "name": "04_spatial_pairwise_only",
            "description": "Spatial reuse only: M_ris=4, M_brdf=1, pairwise MIS, 2 neighbors, no temporal.",
            "brdf": True,
            "brdf_candidates": 1,
            "temporal": False,
            "spatial_neighbors": 2,
            "warmup_frames": 0,
            "capture_label": "ReSTIRDI_spatial_pairwise_2spp_1frame",
        },
        {
            "name": "05_full_temporal_spatial_pairwise",
            "description": "Full requested setup: M_ris=4, M_brdf=1, temporal M cap=1, pairwise spatial reuse with 2 neighbors.",
            "brdf": True,
            "brdf_candidates": 1,
            "temporal": True,
            "spatial_neighbors": 2,
            "warmup_frames": 1,
            "capture_label": "ReSTIRDI_full_temporal_spatial_2spp_2frames",
        },
    ]

    commands = []
    for variant in variants:
        out_dir = os.path.join(output_root, variant["name"])
        cmd = list(base)
        cmd += [
            "--frames",
            str(variant["warmup_frames"]),
            "--restir-brdf-candidates",
            str(variant["brdf_candidates"]),
            "--restir-spatial-neighbors",
            str(variant["spatial_neighbors"]),
            "--output-dir",
            out_dir,
            "--capture-label",
            variant["capture_label"],
        ]
        cmd.append(_flag("--use-brdf-sampling", variant["brdf"]))
        cmd.append(_flag("--restir-temporal-reuse", variant["temporal"]))
        commands.append((variant, cmd))
    return commands


def main():
    repo_dir = _repo_dir()

    parser = argparse.ArgumentParser(description="Render incremental ReSTIR DI validation variants in headless Mogwai.")
    parser.add_argument("--scene", default=r"C:\workspace\Falcor\media\classroom\scene-v4.pbrt", help="Scene to load.")
    parser.add_argument("--width", type=int, default=1280, help="Output width.")
    parser.add_argument("--height", type=int, default=720, help="Output height.")
    parser.add_argument("--frames", type=int, default=None, help="Deprecated/ignored. ReSTIR validation captures one frame, or two frames for temporal variants.")
    parser.add_argument("--vbuffer-samples", type=int, default=16, help="VBufferRT sampleCount setting.")
    parser.add_argument("--spatial-radius", type=int, default=16, help="Spatial reuse radius for spatial variants.")
    parser.add_argument("--spatial-m-cap", type=int, default=20, help="Spatial M cap for spatial variants.")
    parser.add_argument("--output-dir", default=os.path.join(repo_dir, "outputs", "lighting_validation", "restirdi_variants"), help="Output root directory.")
    parser.add_argument("--config", default="Release", choices=["Debug", "Release"], help="Build config whose Mogwai.exe should be used.")
    parser.add_argument("--mogwai", default=None, help="Optional explicit Mogwai.exe path.")
    parser.add_argument("--device-type", default="d3d12", choices=["d3d12", "vulkan"], help="Mogwai graphics backend.")
    parser.add_argument("--gpu", type=int, default=None, help="Optional GPU index.")
    parser.add_argument("--precise", action="store_true", help="Pass --precise to Mogwai.")
    args = parser.parse_args()

    output_root = os.path.abspath(args.output_dir)
    os.makedirs(output_root, exist_ok=True)

    results = []
    start_all = time.perf_counter()
    for variant, cmd in _variant_commands(args, repo_dir, output_root):
        start = time.perf_counter()
        print("Rendering {}...".format(variant["name"]))
        print(" ".join('"{}"'.format(x) if " " in x else x for x in cmd))
        result = subprocess.run(cmd, cwd=repo_dir)
        elapsed = time.perf_counter() - start
        results.append(
            {
                "name": variant["name"],
                "description": variant["description"],
                "returnCode": result.returncode,
                "seconds": elapsed,
                "outputDir": os.path.join(output_root, variant["name"]),
            }
        )
        if result.returncode != 0:
            break

    summary = {
        "scene": args.scene,
        "width": args.width,
        "height": args.height,
        "frames": "non-temporal variants capture frame 1; temporal variants capture frame 2 after one warmup frame",
        "config": args.config,
        "totalSeconds": time.perf_counter() - start_all,
        "variants": results,
    }
    summary_path = os.path.join(output_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote summary to: {}".format(summary_path))

    if results and results[-1]["returnCode"] != 0:
        return results[-1]["returnCode"]
    return 0


if __name__ == "__main__":
    sys.exit(main())
