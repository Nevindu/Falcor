from falcor import *


def render_graph_PathTracerN():
    g = RenderGraph("PathTracerN")

    PathTracerN = createPass("PathTracerN", {"samplesPerPixel": 1})
    g.addPass(PathTracerN, "PathTracerN")

    VBufferRT = createPass("VBufferRT", {"samplePattern": "Stratified", "sampleCount": 16, "useAlphaTest": True})
    g.addPass(VBufferRT, "VBufferRT")

    AccumulatePass = createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single"})
    g.addPass(AccumulatePass, "AccumulatePass")

    ToneMapper = createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("VBufferRT.vbuffer", "PathTracerN.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracerN.viewW")
    g.addEdge("VBufferRT.mvec", "PathTracerN.mvec")
    g.addEdge("PathTracerN.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")

    g.markOutput("ToneMapper.dst")
    return g


PathTracerN = render_graph_PathTracerN()
try:
    m.addGraph(PathTracerN)
except NameError:
    None
