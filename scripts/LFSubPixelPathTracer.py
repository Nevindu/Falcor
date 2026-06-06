from falcor import *


def render_graph_LFSubPixelPathTracer():
    g = RenderGraph("LFSubPixelPathTracer")

    # DirectL-style/offline lenticular encoded path tracer.
    # The pass traces primary rays directly, so no VBufferRT/GBuffer pass is needed.
    LFSubPixelPathTracer = createPass("LFSubPixelPathTracer", {
        "samplesPerPixel": 1,

        # Lenticular / DirectL parameters.
        # Tune these to your display calibration.
        "lfViewCount": 48,
        "lfLineCount": 5.333333,
        "lfTiltDegrees": 0.0,
        "lfOffset": 0.0,
        "lfBaseline": 1.0,
        "lfConvergenceDistance": 4.0,
        "lfCameraMode": 0,          # 0 = parallel, 1 = toe-in/recenter
        "lfReverseViewOrder": False,
    })
    g.addPass(LFSubPixelPathTracer, "LFSubPixelPathTracer")

    # Accumulate multiple frames for offline convergence.
    AccumulatePass = createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single"})
    g.addPass(AccumulatePass, "AccumulatePass")

    ToneMapper = createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("LFSubPixelPathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")

    g.markOutput("ToneMapper.dst")
    return g


LFSubPixelPathTracer = render_graph_LFSubPixelPathTracer()
try:
    m.addGraph(LFSubPixelPathTracer)
except NameError:
    None
