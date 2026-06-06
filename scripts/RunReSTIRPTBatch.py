from falcor import *
import json
import os
import time


script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)

scene_path = globals().get("RESTIRPT_SCENE", r"C:\workspace\Falcor\media\classroom\scene-v4.pbrt")
output_dir = os.path.abspath(globals().get("RESTIRPT_OUTPUT_DIR", os.path.join(repo_dir, "outputs", "restirpt_batch")))
integrator = globals().get("RESTIRPT_INTEGRATOR", "restirpt").lower()
width = globals().get("RESTIRPT_WIDTH", 1280)
height = globals().get("RESTIRPT_HEIGHT", 720)
frames = globals().get("RESTIRPT_FRAMES", 0)
samples_per_pixel = globals().get("RESTIRPT_SAMPLES_PER_PIXEL", 1)
initial_candidate_count = globals().get("RESTIRPT_INITIAL_CANDIDATE_COUNT", samples_per_pixel)
accumulate_enabled = globals().get("RESTIRPT_ACCUMULATE", False)
capture_label = globals().get("RESTIRPT_CAPTURE_LABEL", None)
vbuffer_samples = globals().get("RESTIRPT_VBUFFER_SAMPLES", 16)
path_tracer_use_nrd_demodulation = globals().get("RESTIRPT_PATH_TRACER_USE_NRD_DEMODULATION", False)
use_spatial_reuse = globals().get("RESTIRPT_USE_SPATIAL_REUSE", False)
spatial_neighbor_count = globals().get("RESTIRPT_SPATIAL_NEIGHBOR_COUNT", 3)
spatial_radius = globals().get("RESTIRPT_SPATIAL_RADIUS", 20)
spatial_iterations = globals().get("RESTIRPT_SPATIAL_ITERATIONS", 1)


def _make_graph(name, pass_name, pass_props):
    g = RenderGraph(name)

    vbuffer = createPass(
        "VBufferRT",
        {
            "samplePattern": "Stratified",
            "sampleCount": vbuffer_samples,
            "useAlphaTest": True,
        },
    )
    g.addPass(vbuffer, "VBufferRT")

    lighting = createPass(pass_name, pass_props)
    g.addPass(lighting, pass_name)

    accumulate = createPass("AccumulatePass", {"enabled": accumulate_enabled, "precisionMode": "Single"})
    g.addPass(accumulate, "AccumulatePass")

    tonemapper = createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0})
    g.addPass(tonemapper, "ToneMapper")

    g.addEdge("VBufferRT.vbuffer", "{}.vbuffer".format(pass_name))
    g.addEdge("VBufferRT.viewW", "{}.viewW".format(pass_name))
    g.addEdge("VBufferRT.mvec", "{}.mvec".format(pass_name))
    g.addEdge("{}.color".format(pass_name), "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g


def _graph_spec():
    if integrator in ("pathtracern", "ptn", "reference"):
        return (
            "PathTracerN",
            _make_graph("PathTracerN", "PathTracerN", {"samplesPerPixel": samples_per_pixel}),
        )

    if integrator in ("pathtracer", "pt", "baseline"):
        # This graph intentionally wires only PathTracer.color into accumulation/tone mapping.
        # No NRD pass consumes PathTracer's NRD outputs, so NRD denoising is disabled.
        return (
            "PathTracer",
            _make_graph(
                "PathTracer",
                "PathTracer",
                {
                    "samplesPerPixel": samples_per_pixel,
                    "useNRDDemodulation": path_tracer_use_nrd_demodulation,
                },
            ),
        )

    if integrator in ("restirpt", "restir", "ris"):
        props = {
            "initialCandidateCount": initial_candidate_count,
            "useSpatialReuse": use_spatial_reuse,
            "spatialNeighborCount": spatial_neighbor_count,
            "spatialRadius": spatial_radius,
            "spatialIterations": spatial_iterations,
        }
        return (
            "ReSTIRPT",
            _make_graph("ReSTIRPT", "ReSTIRPT", props),
        )

    raise RuntimeError("Unknown RESTIRPT_INTEGRATOR '{}'.".format(integrator))


os.makedirs(output_dir, exist_ok=True)
m.loadScene(scene_path)
m.resizeFrameBuffer(width, height)

old_capture_dir = m.frameCapture.outputDir
old_base_filename = m.frameCapture.baseFilename
total_start = time.perf_counter()
graph_name, graph = _graph_spec()

print(
    "ReSTIRPT batch: scene='{}', integrator='{}', {}x{}, frames={}, spp={}, initialCandidateCount={}, accumulate={}".format(
        scene_path, integrator, width, height, frames, samples_per_pixel, initial_candidate_count, accumulate_enabled
    )
)
print("Saving capture to: {}".format(output_dir))

try:
    m.frameCapture.outputDir = output_dir
    m.addGraph(graph)
    m.setActiveGraph(graph)

    start = time.perf_counter()
    for _ in range(frames):
        m.renderFrame()

    if capture_label:
        m.frameCapture.baseFilename = capture_label
    else:
        m.frameCapture.baseFilename = "{}_{}spp_{}frames".format(graph_name, samples_per_pixel, frames)
    m.renderFrame()
    m.frameCapture.capture()
    elapsed = time.perf_counter() - start
finally:
    m.frameCapture.outputDir = old_capture_dir
    m.frameCapture.baseFilename = old_base_filename

summary = {
    "scene": scene_path,
    "integrator": integrator,
    "graph": graph_name,
    "width": width,
    "height": height,
    "frames": frames,
    "framesRendered": frames + 1,
    "samplesPerPixel": samples_per_pixel,
    "initialCandidateCount": initial_candidate_count,
    "useSpatialReuse": use_spatial_reuse,
    "spatialNeighborCount": spatial_neighbor_count,
    "spatialRadius": spatial_radius,
    "spatialIterations": spatial_iterations,
    "nrdDenoising": False,
    "pathTracerUseNRDDemodulation": path_tracer_use_nrd_demodulation,
    "effectivePathTracerSamples": samples_per_pixel * (frames + 1),
    "accumulate": accumulate_enabled,
    "captureLabel": capture_label,
    "outputDir": output_dir,
    "seconds": elapsed,
    "totalSeconds": time.perf_counter() - total_start,
}

summary_path = os.path.join(output_dir, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print("Wrote summary to: {}".format(summary_path))
exit()
