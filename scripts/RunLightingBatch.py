from falcor import *
import json
import os
import time


script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)

scene_path = globals().get("LIGHTING_SCENE", r"C:\workspace\Falcor\media\classroom\scene-v4.pbrt")
output_dir = os.path.abspath(globals().get("LIGHTING_OUTPUT_DIR", os.path.join(repo_dir, "outputs", "lighting_batch")))
integrator = globals().get("LIGHTING_INTEGRATOR", "both").lower()
width = globals().get("LIGHTING_WIDTH", 1280)
height = globals().get("LIGHTING_HEIGHT", 720)
frames = globals().get("LIGHTING_FRAMES", 1)
samples_per_pixel = globals().get("LIGHTING_SAMPLES_PER_PIXEL", 2)
use_brdf_sampling = globals().get("LIGHTING_USE_BRDF_SAMPLING", True)
accumulate_enabled = globals().get("LIGHTING_ACCUMULATE", True)
capture_label = globals().get("LIGHTING_CAPTURE_LABEL", None)
vbuffer_samples = globals().get("LIGHTING_VBUFFER_SAMPLES", 16)
restir_light_candidates = globals().get("RESTIR_LIGHT_CANDIDATES", samples_per_pixel)
restir_brdf_candidates = globals().get("RESTIR_BRDF_CANDIDATES", 1)
restir_spatial_neighbors = globals().get("RESTIR_SPATIAL_NEIGHBORS", 2)
restir_spatial_radius = globals().get("RESTIR_SPATIAL_RADIUS", 16)
restir_spatial_m_cap = globals().get("RESTIR_SPATIAL_M_CAP", 20)
restir_temporal_m_cap = globals().get("RESTIR_TEMPORAL_M_CAP", 8)
restir_spatial_mis_strategy = globals().get("RESTIR_SPATIAL_MIS_STRATEGY", 0)
restir_visibility_in_target = globals().get("RESTIR_VISIBILITY_IN_TARGET", False)
restir_visibility_in_spatial_gbh = globals().get("RESTIR_VISIBILITY_IN_SPATIAL_GBH", False)
restir_temporal_reuse = globals().get("RESTIR_TEMPORAL_REUSE", True)
restir_visibility_in_temporal_mis = globals().get("RESTIR_VISIBILITY_IN_TEMPORAL_MIS", False)


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
    if pass_name == "ReSTIRDI":
        g.addEdge("VBufferRT.mvec", "{}.mvec".format(pass_name))
    g.addEdge("{}.color".format(pass_name), "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g


def _graph_specs():
    specs = []

    if integrator in ("directlighting", "direct", "dl", "both"):
        specs.append(
            (
                "DirectLighting",
                _make_graph(
                    "DirectLighting",
                    "DirectLighting",
                    {
                        "samplesPerPixel": samples_per_pixel,
                        "useAnalyticLights": True,
                        "useEmissiveLights": True,
                        "useEnvLight": True,
                        "useEmissiveMaterials": True,
                        "useEnvBackground": True,
                        "useMIS": True,
                        "useBRDFSampling": use_brdf_sampling,
                    },
                ),
            )
        )

    if integrator in ("restirdi", "restir", "both"):
        specs.append(
            (
                "ReSTIRDI",
                _make_graph(
                    "ReSTIRDI",
                    "ReSTIRDI",
                    {
                        "samplesPerPixel": samples_per_pixel,
                        "lightCandidateCount": restir_light_candidates,
                        "brdfCandidateCount": restir_brdf_candidates,
                        "spatialNeighborCount": restir_spatial_neighbors,
                        "spatialRadius": restir_spatial_radius,
                        "spatialMCap": restir_spatial_m_cap,
                        "temporalMCap": restir_temporal_m_cap,
                        "useTemporalReuse": restir_temporal_reuse,
                        "useBRDFSampling": use_brdf_sampling,
                        "spatialMISStrategy": restir_spatial_mis_strategy,
                        "useVisibilityInTarget": restir_visibility_in_target,
                        "useVisibilityInSpatialGBH": restir_visibility_in_spatial_gbh,
                        "useVisibilityInTemporalMIS": restir_visibility_in_temporal_mis,
                        "useAnalyticLights": True,
                        "useEmissiveLights": True,
                        "useEnvLight": True,
                        "useEmissiveMaterials": True,
                        "useEnvBackground": True,
                    },
                ),
            )
        )

    if not specs:
        raise RuntimeError("Unknown LIGHTING_INTEGRATOR '{}'.".format(integrator))

    return specs


os.makedirs(output_dir, exist_ok=True)
m.loadScene(scene_path)
m.resizeFrameBuffer(width, height)

old_capture_dir = m.frameCapture.outputDir
old_base_filename = m.frameCapture.baseFilename
results = []
total_start = time.perf_counter()

print(
    "Lighting batch: scene='{}', integrator='{}', {}x{}, frames={}, samplesPerPixel={}, restirLightCandidates={}, restirBRDFCandidates={}".format(
        scene_path, integrator, width, height, frames, samples_per_pixel, restir_light_candidates, restir_brdf_candidates
    )
)
print("Saving captures to: {}".format(output_dir))

try:
    m.frameCapture.outputDir = output_dir

    for graph_name, graph in _graph_specs():
        start = time.perf_counter()
        print("Rendering {}...".format(graph_name))
        m.addGraph(graph)
        m.setActiveGraph(graph)

        for _ in range(frames):
            m.renderFrame()

        if capture_label:
            m.frameCapture.baseFilename = capture_label
        else:
            m.frameCapture.baseFilename = "{}_{}spp_{}frames".format(graph_name, samples_per_pixel, frames)
        m.renderFrame()
        m.frameCapture.capture()

        elapsed = time.perf_counter() - start
        results.append({"graph": graph_name, "seconds": elapsed, "framesRendered": frames + 1})
        print("Captured {} in {:.3f}s".format(graph_name, elapsed))
finally:
    m.frameCapture.outputDir = old_capture_dir
    m.frameCapture.baseFilename = old_base_filename

summary = {
    "scene": scene_path,
    "integrator": integrator,
    "width": width,
    "height": height,
    "frames": frames,
    "samplesPerPixel": samples_per_pixel,
    "useBRDFSampling": use_brdf_sampling,
    "accumulate": accumulate_enabled,
    "captureLabel": capture_label,
    "restirLightCandidates": restir_light_candidates,
    "restirBRDFCandidates": restir_brdf_candidates,
    "outputDir": output_dir,
    "totalSeconds": time.perf_counter() - total_start,
    "results": results,
}

summary_path = os.path.join(output_dir, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print("Wrote summary to: {}".format(summary_path))
exit()
