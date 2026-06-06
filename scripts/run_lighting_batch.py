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


def _spatial_mis_strategy_value(value):
    values = {
        "none": 0,
        "cheap": 0,
        "contribution": 1,
        "contribution-mis": 1,
        "gbh": 2,
        "generalized-balance": 2,
        "pairwise": 3,
        "pairwise-mis": 3,
    }
    return values[value]


def main():
    repo_dir = _repo_dir()

    parser = argparse.ArgumentParser(description="Run DirectLighting and/or ReSTIRDI in headless Mogwai.")
    parser.add_argument("--mogwai", default=None, help="Path to Mogwai.exe.")
    parser.add_argument("--config", default="Debug", choices=["Debug", "Release"], help="VS2022 build config to use when --mogwai is omitted.")
    parser.add_argument("--scene", default=r"C:\workspace\Falcor\media\classroom\scene-v4.pbrt", help="Scene to load.")
    parser.add_argument("--integrator", default="both", choices=["directlighting", "direct", "dl", "restirdi", "restir", "both"], help="Which graph to render.")
    parser.add_argument("--width", type=int, default=1280, help="Output width.")
    parser.add_argument("--height", type=int, default=720, help="Output height.")
    parser.add_argument("--frames", type=int, default=1, help="Accumulation frames before capture. Total rendered frames is frames + 1 capture frame.")
    parser.add_argument("--samples-per-pixel", type=int, default=2, help="DirectLighting samplesPerPixel and legacy ReSTIRDI light candidate setting.")
    parser.add_argument("--use-brdf-sampling", action=argparse.BooleanOptionalAction, default=True, help="Toggle BRDF/BSDF sampling where supported.")
    parser.add_argument("--accumulate", action=argparse.BooleanOptionalAction, default=True, help="Enable AccumulatePass. Disable for single-frame ReSTIR validation captures.")
    parser.add_argument("--capture-label", default=None, help="Optional base filename label for the captured image.")
    parser.add_argument("--restir-light-candidates", type=int, default=None, help="ReSTIRDI lightCandidateCount setting. Defaults to --samples-per-pixel.")
    parser.add_argument("--restir-brdf-candidates", type=int, default=1, help="ReSTIRDI brdfCandidateCount setting.")
    parser.add_argument("--vbuffer-samples", type=int, default=16, help="VBufferRT sampleCount setting.")
    parser.add_argument("--output-dir", default=os.path.join(repo_dir, "outputs", "lighting_batch"), help="Directory for captures, log, temp script, and summary JSON.")
    parser.add_argument("--restir-spatial-neighbors", type=int, default=2, help="ReSTIRDI spatialNeighborCount setting.")
    parser.add_argument("--restir-spatial-radius", type=int, default=16, help="ReSTIRDI spatialRadius setting.")
    parser.add_argument("--restir-spatial-m-cap", type=int, default=20, help="ReSTIRDI spatialMCap setting.")
    parser.add_argument("--restir-temporal-m-cap", type=int, default=8, help="ReSTIRDI temporalMCap setting.")
    parser.add_argument(
        "--restir-spatial-mis-strategy",
        default="none",
        choices=["none", "cheap", "contribution", "contribution-mis", "gbh", "generalized-balance", "pairwise", "pairwise-mis"],
        help="ReSTIRDI spatialMISStrategy setting.",
    )
    parser.add_argument("--restir-visibility-in-target", action=argparse.BooleanOptionalAction, default=False, help="Toggle ReSTIRDI useVisibilityInTarget.")
    parser.add_argument("--restir-visibility-in-spatial-gbh", action=argparse.BooleanOptionalAction, default=False, help="Toggle ReSTIRDI useVisibilityInSpatialGBH.")
    parser.add_argument("--restir-temporal-reuse", action=argparse.BooleanOptionalAction, default=True, help="Toggle ReSTIRDI useTemporalReuse.")
    parser.add_argument("--restir-visibility-in-temporal-mis", action=argparse.BooleanOptionalAction, default=False, help="Toggle ReSTIRDI useVisibilityInTemporalMIS.")
    parser.add_argument("--device-type", default="d3d12", choices=["d3d12", "vulkan"], help="Mogwai graphics backend.")
    parser.add_argument("--gpu", type=int, default=None, help="Optional GPU index.")
    parser.add_argument("--precise", action="store_true", help="Pass --precise to Mogwai.")
    args = parser.parse_args()
    restir_light_candidates = args.restir_light_candidates if args.restir_light_candidates is not None else args.samples_per_pixel

    mogwai = args.mogwai or _default_mogwai(repo_dir, args.config)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    driver = os.path.join(repo_dir, "scripts", "RunLightingBatch.py")
    script = f"""
LIGHTING_SCENE = r"{args.scene}"
LIGHTING_OUTPUT_DIR = r"{output_dir}"
LIGHTING_INTEGRATOR = "{args.integrator}"
LIGHTING_WIDTH = {args.width}
LIGHTING_HEIGHT = {args.height}
LIGHTING_FRAMES = {args.frames}
LIGHTING_SAMPLES_PER_PIXEL = {args.samples_per_pixel}
LIGHTING_USE_BRDF_SAMPLING = {_bool_literal(args.use_brdf_sampling)}
LIGHTING_ACCUMULATE = {_bool_literal(args.accumulate)}
LIGHTING_CAPTURE_LABEL = {repr(args.capture_label)}
LIGHTING_VBUFFER_SAMPLES = {args.vbuffer_samples}
RESTIR_LIGHT_CANDIDATES = {restir_light_candidates}
RESTIR_BRDF_CANDIDATES = {args.restir_brdf_candidates}
RESTIR_SPATIAL_NEIGHBORS = {args.restir_spatial_neighbors}
RESTIR_SPATIAL_RADIUS = {args.restir_spatial_radius}
RESTIR_SPATIAL_M_CAP = {args.restir_spatial_m_cap}
RESTIR_TEMPORAL_M_CAP = {args.restir_temporal_m_cap}
RESTIR_SPATIAL_MIS_STRATEGY = {_spatial_mis_strategy_value(args.restir_spatial_mis_strategy)}
RESTIR_VISIBILITY_IN_TARGET = {_bool_literal(args.restir_visibility_in_target)}
RESTIR_VISIBILITY_IN_SPATIAL_GBH = {_bool_literal(args.restir_visibility_in_spatial_gbh)}
RESTIR_TEMPORAL_REUSE = {_bool_literal(args.restir_temporal_reuse)}
RESTIR_VISIBILITY_IN_TEMPORAL_MIS = {_bool_literal(args.restir_visibility_in_temporal_mis)}
m.script(r"{driver}")
"""

    fd, script_path = tempfile.mkstemp(prefix="lighting_batch_", suffix=".py", dir=output_dir, text=True)
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
