from falcor import *
import json
import os
import time


script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)

scene_path = globals().get("LFSP_SCENE", os.path.join(repo_dir, "media", "classroom", "scene-v4.pbrt"))
output_dir = os.path.abspath(globals().get("LFSP_OUTPUT_DIR", os.path.join(repo_dir, "outputs", "lightfield_classroom")))
width = globals().get("LFSP_WIDTH", 1280)
height = globals().get("LFSP_HEIGHT", 720)
frames = globals().get("LFSP_FRAMES", 64)
samples_per_pixel = globals().get("LFSP_SAMPLES_PER_PIXEL", 1)

lf_view_count = globals().get("LFSP_VIEW_COUNT", 48)
lf_line_count = globals().get("LFSP_LINE_COUNT", 5.333333)
lf_tilt_degrees = globals().get("LFSP_TILT_DEGREES", 0.0)
lf_offset = globals().get("LFSP_OFFSET", 0.0)
lf_baseline = globals().get("LFSP_BASELINE", 1.0)
lf_convergence_distance = globals().get("LFSP_CONVERGENCE_DISTANCE", 4.0)
lf_camera_mode = globals().get("LFSP_CAMERA_MODE", 0)
lf_reverse_view_order = globals().get("LFSP_REVERSE_VIEW_ORDER", False)


def _make_graph():
    g = RenderGraph("LFSubPixelPathTracer")

    tracer = createPass(
        "LFSubPixelPathTracer",
        {
            "samplesPerPixel": samples_per_pixel,
            "lfViewCount": lf_view_count,
            "lfLineCount": lf_line_count,
            "lfTiltDegrees": lf_tilt_degrees,
            "lfOffset": lf_offset,
            "lfBaseline": lf_baseline,
            "lfConvergenceDistance": lf_convergence_distance,
            "lfCameraMode": lf_camera_mode,
            "lfReverseViewOrder": lf_reverse_view_order,
        },
    )
    g.addPass(tracer, "LFSubPixelPathTracer")

    accumulate = createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single"})
    g.addPass(accumulate, "AccumulatePass")

    tonemapper = createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0})
    g.addPass(tonemapper, "ToneMapper")

    g.addEdge("LFSubPixelPathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g


os.makedirs(output_dir, exist_ok=True)

print("LFSubPixelPathTracer batch")
print("  scene: {}".format(scene_path))
print("  output: {}".format(output_dir))
print("  resolution: {}x{}".format(width, height))
print("  frames: {}, samplesPerPixel: {}".format(frames, samples_per_pixel))

m.loadScene(scene_path)
m.resizeFrameBuffer(width, height)

graph = _make_graph()
m.addGraph(graph)
m.setActiveGraph(graph)

old_capture_dir = m.frameCapture.outputDir
old_base_filename = m.frameCapture.baseFilename
start = time.perf_counter()

try:
    m.frameCapture.outputDir = output_dir

    for _ in range(frames):
        m.renderFrame()

    base = "classroom_lfsubpixel_{}x{}_{}spp_{}frames".format(width, height, samples_per_pixel, frames)
    m.frameCapture.baseFilename = base
    m.renderFrame()
    m.frameCapture.capture()
finally:
    m.frameCapture.outputDir = old_capture_dir
    m.frameCapture.baseFilename = old_base_filename

summary = {
    "scene": scene_path,
    "outputDir": output_dir,
    "width": width,
    "height": height,
    "frames": frames,
    "samplesPerPixel": samples_per_pixel,
    "lfViewCount": lf_view_count,
    "lfLineCount": lf_line_count,
    "lfTiltDegrees": lf_tilt_degrees,
    "lfOffset": lf_offset,
    "lfBaseline": lf_baseline,
    "lfConvergenceDistance": lf_convergence_distance,
    "lfCameraMode": lf_camera_mode,
    "lfReverseViewOrder": lf_reverse_view_order,
    "seconds": time.perf_counter() - start,
}

summary_path = os.path.join(output_dir, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print("Wrote summary to: {}".format(summary_path))
exit()
