from falcor import *
import json
import math
import os
import time


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_SCRIPT_DIR)


def render_graph_LightFieldPathTracer(samples_per_pixel=1, view_count_x=8, view_count_y=1, view_width=512, view_height=512, baseline_x=0.08, baseline_y=0.0):
    g = RenderGraph("LightFieldPathTracer")

    VBufferRT = createPass("VBufferRT", {"samplePattern": "Stratified", "sampleCount": 16, "useAlphaTest": True})
    g.addPass(VBufferRT, "VBufferRT")

    LightFieldPathTracer = createPass(
        "LightFieldPathTracer",
        {
            "samplesPerPixel": samples_per_pixel,
            "enableLightField": True,
            "viewCountX": view_count_x,
            "viewCountY": view_count_y,
            "viewWidth": view_width,
            "viewHeight": view_height,
            "baselineX": baseline_x,
            "baselineY": baseline_y,
            "focusDistance": 2.0,
            "quiltCols": view_count_x,
            "quiltRows": view_count_y,
            "saveViews": True,
        },
    )
    g.addPass(LightFieldPathTracer, "LightFieldPathTracer")

    AccumulatePass = createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single"})
    g.addPass(AccumulatePass, "AccumulatePass")

    ToneMapper = createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("VBufferRT.vbuffer", "LightFieldPathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "LightFieldPathTracer.viewW")
    g.addEdge("VBufferRT.mvec", "LightFieldPathTracer.mvec")
    g.addEdge("LightFieldPathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")

    g.markOutput("ToneMapper.dst")
    return g


def _vec3(v):
    try:
        return [float(v.x), float(v.y), float(v.z)]
    except AttributeError:
        return [float(v[0]), float(v[1]), float(v[2])]


def _add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _mul(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _normalize(v):
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length <= 0.0:
        return [0.0, 0.0, 0.0]
    return [v[0] / length, v[1] / length, v[2] / length]


def render_light_field_views(
    m,
    view_count_x=8,
    view_count_y=1,
    baseline_x=0.08,
    baseline_y=0.0,
    frames_per_view=8,
    output_dir=None,
    base_seed=1,
    summary_name="summary.json",
):
    """Render a sequential parallel-camera light field and capture one image per view."""
    if output_dir is None:
        output_dir = os.path.join(_REPO_DIR, "outputs", "lightfield")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print("Saving light-field views to: {}".format(output_dir))

    camera = m.scene.camera
    center_pos = _vec3(camera.position)
    center_target = _vec3(camera.target)
    center_up = _normalize(_vec3(camera.up))
    forward = _normalize(_sub(center_target, center_pos))
    right = _normalize(_cross(forward, center_up))

    try:
        path_tracer = LightFieldPathTracer.getPass("LightFieldPathTracer")
    except Exception:
        path_tracer = None

    old_capture_dir = m.frameCapture.outputDir
    old_base_filename = m.frameCapture.baseFilename

    timings = []
    total_start = time.perf_counter()

    try:
        m.frameCapture.outputDir = output_dir

        for view_y in range(view_count_y):
            for view_x in range(view_count_x):
                view_start = time.perf_counter()
                view_idx = view_x + view_y * view_count_x
                print("Rendering light-field view {} / {}".format(view_idx + 1, view_count_x * view_count_y))
                u = ((view_x + 0.5) / view_count_x - 0.5) * baseline_x
                v = ((view_y + 0.5) / view_count_y - 0.5) * baseline_y
                offset = _add(_mul(right, u), _mul(center_up, v))

                camera.position = _add(center_pos, offset)
                camera.target = _add(center_target, offset)
                camera.up = center_up

                if path_tracer is not None:
                    path_tracer.useFixedSeed = False
                    path_tracer.reset()

                for _ in range(frames_per_view):
                    m.renderFrame()

                m.frameCapture.baseFilename = "view_{:03d}".format(view_idx)
                m.renderFrame()
                m.frameCapture.capture()
                elapsed = time.perf_counter() - view_start
                timings.append(
                    {
                        "viewIndex": view_idx,
                        "viewX": view_x,
                        "viewY": view_y,
                        "seconds": elapsed,
                        "framesRendered": frames_per_view + 1,
                    }
                )
                print("Captured view_{:03d} in {:.3f}s".format(view_idx, elapsed))
    finally:
        camera.position = center_pos
        camera.target = center_target
        camera.up = center_up
        m.frameCapture.outputDir = old_capture_dir
        m.frameCapture.baseFilename = old_base_filename

    total_seconds = time.perf_counter() - total_start
    summary = {
        "viewCountX": view_count_x,
        "viewCountY": view_count_y,
        "viewCount": view_count_x * view_count_y,
        "framesPerView": frames_per_view,
        "baselineX": baseline_x,
        "baselineY": baseline_y,
        "outputDir": output_dir,
        "totalSeconds": total_seconds,
        "averageSecondsPerView": total_seconds / max(1, view_count_x * view_count_y),
        "views": timings,
    }
    with open(os.path.join(output_dir, summary_name), "w") as f:
        json.dump(summary, f, indent=2)
    print("Light-field render finished in {:.3f}s ({:.3f}s/view).".format(summary["totalSeconds"], summary["averageSecondsPerView"]))
    print("Wrote summary to: {}".format(os.path.join(output_dir, summary_name)))
    return summary


LightFieldPathTracer = render_graph_LightFieldPathTracer(
    samples_per_pixel=globals().get("LF_SAMPLES_PER_PIXEL", 1),
    view_count_x=globals().get("LF_VIEW_COUNT_X", 8),
    view_count_y=globals().get("LF_VIEW_COUNT_Y", 1),
    view_width=globals().get("LF_VIEW_WIDTH", 512),
    view_height=globals().get("LF_VIEW_HEIGHT", 512),
    baseline_x=globals().get("LF_BASELINE_X", 0.08),
    baseline_y=globals().get("LF_BASELINE_Y", 0.0),
)
try:
    m.addGraph(LightFieldPathTracer)
except NameError:
    None
