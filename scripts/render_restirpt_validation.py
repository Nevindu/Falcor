import argparse
import json
import os
import subprocess
import sys
import tempfile
import time


def _repo_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_mogwai(repo_dir, config):
    return os.path.join(repo_dir, "build", "windows-vs2022", "bin", config, "Mogwai.exe")


def _bool_literal(value):
    return "True" if value else "False"


def _run_variant(args, repo_dir, output_root, variant):
    mogwai = args.mogwai or _default_mogwai(repo_dir, args.config)
    out_dir = os.path.join(output_root, variant["name"])
    os.makedirs(out_dir, exist_ok=True)

    driver = os.path.join(repo_dir, "scripts", "RunReSTIRPTBatch.py")
    script = f"""
RESTIRPT_SCENE = r"{args.scene}"
RESTIRPT_OUTPUT_DIR = r"{out_dir}"
RESTIRPT_INTEGRATOR = "{variant["integrator"]}"
RESTIRPT_WIDTH = {args.width}
RESTIRPT_HEIGHT = {args.height}
RESTIRPT_FRAMES = {variant["frames"]}
RESTIRPT_SAMPLES_PER_PIXEL = {variant["samples_per_pixel"]}
RESTIRPT_INITIAL_CANDIDATE_COUNT = {variant["initial_candidate_count"]}
RESTIRPT_ACCUMULATE = {_bool_literal(variant["accumulate"])}
RESTIRPT_CAPTURE_LABEL = {repr(variant["capture_label"])}
RESTIRPT_VBUFFER_SAMPLES = {args.vbuffer_samples}
RESTIRPT_PATH_TRACER_USE_NRD_DEMODULATION = {_bool_literal(variant.get("path_tracer_use_nrd_demodulation", False))}
RESTIRPT_USE_SPATIAL_REUSE = {_bool_literal(variant.get("use_spatial_reuse", False))}
RESTIRPT_SPATIAL_NEIGHBOR_COUNT = {variant.get("spatial_neighbor_count", args.spatial_neighbor_count)}
RESTIRPT_SPATIAL_RADIUS = {variant.get("spatial_radius", args.spatial_radius)}
RESTIRPT_SPATIAL_ITERATIONS = {variant.get("spatial_iterations", args.spatial_iterations)}
m.script(r"{driver}")
"""

    fd, script_path = tempfile.mkstemp(prefix="restirpt_validation_", suffix=".py", dir=out_dir, text=True)
    with os.fdopen(fd, "w") as f:
        f.write(script)

    logfile = os.path.join(out_dir, "log.txt")
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
        cmd.append("--precise")

    print("Rendering {}...".format(variant["name"]))
    print(" ".join('"{}"'.format(x) if " " in x else x for x in cmd))
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=repo_dir)
    elapsed = time.perf_counter() - start

    return {
        "name": variant["name"],
        "description": variant["description"],
        "returnCode": result.returncode,
        "seconds": elapsed,
        "outputDir": out_dir,
        "integrator": variant["integrator"],
        "samplesPerPixel": variant["samples_per_pixel"],
        "initialCandidateCount": variant["initial_candidate_count"],
        "useSpatialReuse": variant.get("use_spatial_reuse", False),
        "spatialNeighborCount": variant.get("spatial_neighbor_count", args.spatial_neighbor_count),
        "spatialRadius": variant.get("spatial_radius", args.spatial_radius),
        "spatialIterations": variant.get("spatial_iterations", args.spatial_iterations),
        "nrdDenoising": variant.get("nrd_denoising", False),
        "pathTracerUseNRDDemodulation": variant.get("path_tracer_use_nrd_demodulation", False),
        "frames": variant["frames"],
        "framesRendered": variant["frames"] + 1,
        "accumulate": variant["accumulate"],
    }


def _variants(args):
    return [
        {
            "name": "00_pathtracern_reference",
            "role": "reference",
            "description": "High-spp PathTracerN reference.",
            "integrator": "pathtracern",
            "samples_per_pixel": args.reference_spp,
            "initial_candidate_count": args.reference_spp,
            "nrd_denoising": False,
            "path_tracer_use_nrd_demodulation": False,
            "use_spatial_reuse": False,
            "frames": args.reference_frames,
            "accumulate": True,
            "capture_label": "PathTracerN_reference_{}spp_{}frames".format(args.reference_spp, args.reference_frames + 1),
        },
        {
            "name": "01_pathtracer_1spp_baseline_no_nrd",
            "role": "baseline",
            "description": "Falcor PathTracer 1spp baseline with no NRD pass wired into the graph.",
            "integrator": "pathtracer",
            "samples_per_pixel": 1,
            "initial_candidate_count": 1,
            "nrd_denoising": False,
            "path_tracer_use_nrd_demodulation": False,
            "use_spatial_reuse": False,
            "frames": args.baseline_frames,
            "accumulate": args.baseline_accumulate,
            "capture_label": "PathTracer_baseline_1spp_no_nrd_{}frames".format(args.baseline_frames + 1),
        },
        {
            "name": "02_restirpt_initial_ris",
            "role": "restirpt",
            "description": "ReSTIRPT initial path-tree RIS only: no temporal reuse and no spatial reuse.",
            "integrator": "restirpt",
            "samples_per_pixel": 1,
            "initial_candidate_count": args.initial_candidate_count,
            "nrd_denoising": False,
            "path_tracer_use_nrd_demodulation": False,
            "use_spatial_reuse": False,
            "frames": args.restirpt_frames,
            "accumulate": args.restirpt_accumulate,
            "capture_label": "ReSTIRPT_initial_ris_M{}_{}frames".format(args.initial_candidate_count, args.restirpt_frames + 1),
        },
        {
            "name": "03_restirpt_spatial_identity_reuse",
            "role": "restirpt",
            "description": "ReSTIRPT spatial reuse skeleton using identity-style reservoir evaluation.",
            "integrator": "restirpt",
            "samples_per_pixel": 1,
            "initial_candidate_count": args.initial_candidate_count,
            "nrd_denoising": False,
            "path_tracer_use_nrd_demodulation": False,
            "use_spatial_reuse": True,
            "spatial_neighbor_count": args.spatial_neighbor_count,
            "spatial_radius": args.spatial_radius,
            "spatial_iterations": args.spatial_iterations,
            "frames": args.restirpt_frames,
            "accumulate": args.restirpt_accumulate,
            "capture_label": "ReSTIRPT_spatial_identity_M{}_N{}_R{}_{}frames".format(args.initial_candidate_count, args.spatial_neighbor_count, args.spatial_radius, args.restirpt_frames + 1),
        },
    ]


def main():
    repo_dir = _repo_dir()

    parser = argparse.ArgumentParser(description="Render incremental ReSTIR PT validation variants in headless Mogwai.")
    parser.add_argument("--scene", default=r"C:\workspace\Falcor\media\classroom\scene-v4.pbrt", help="Scene to load.")
    parser.add_argument("--width", type=int, default=1280, help="Output width.")
    parser.add_argument("--height", type=int, default=720, help="Output height.")
    parser.add_argument("--vbuffer-samples", type=int, default=16, help="VBufferRT sampleCount setting.")
    parser.add_argument("--reference-spp", type=int, default=16, help="PathTracerN samplesPerPixel for the reference. Must be <= PathTracerN max spp.")
    parser.add_argument("--reference-frames", type=int, default=63, help="Pre-capture accumulation frames for the PathTracerN reference.")
    parser.add_argument("--baseline-frames", type=int, default=0, help="Pre-capture frames for the 1spp Falcor PathTracer baseline.")
    parser.add_argument("--baseline-accumulate", action=argparse.BooleanOptionalAction, default=False, help="Enable AccumulatePass for the Falcor PathTracer baseline.")
    parser.add_argument("--initial-candidate-count", type=int, default=1, help="ReSTIRPT initial candidate path trees for current RIS-only variant.")
    parser.add_argument("--restirpt-frames", type=int, default=0, help="Pre-capture frames for ReSTIRPT.")
    parser.add_argument("--restirpt-accumulate", action=argparse.BooleanOptionalAction, default=False, help="Enable AccumulatePass for ReSTIRPT.")
    parser.add_argument("--spatial-neighbor-count", type=int, default=3, help="Spatial reuse neighbor count for the identity-reuse skeleton variant.")
    parser.add_argument("--spatial-radius", type=int, default=20, help="Spatial reuse search radius in pixels.")
    parser.add_argument("--spatial-iterations", type=int, default=1, help="Spatial reuse iterations. Currently only one is implemented.")
    parser.add_argument("--only", default="all", choices=["all", "reference", "baseline", "restirpt"], help="Render all variants or a single validation role.")
    parser.add_argument("--output-dir", default=os.path.join(repo_dir, "outputs", "lighting_validation", "restirpt_variants"), help="Output root directory.")
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
    variants = _variants(args)
    if args.only != "all":
        variants = [v for v in variants if v["role"] == args.only]

    for variant in variants:
        result = _run_variant(args, repo_dir, output_root, variant)
        results.append(result)
        if result["returnCode"] != 0:
            break

    summary = {
        "scene": args.scene,
        "width": args.width,
        "height": args.height,
        "config": args.config,
        "reference": "PathTracerN high-spp accumulated reference",
        "baseline": "Falcor PathTracer 1spp baseline with NRD denoising disabled",
        "currentReSTIRPTVariant": "Initial path-tree RIS only",
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
