from falcor import *


def render_graph_DirectLighting():
    g = RenderGraph("DirectLighting")

    VBufferRT = createPass("VBufferRT", {"samplePattern": "Stratified", "sampleCount": 16, "useAlphaTest": True})
    g.addPass(VBufferRT, "VBufferRT")

    DirectLighting = createPass("DirectLighting", {"samplesPerPixel": 4})
    g.addPass(DirectLighting, "DirectLighting")

    AccumulatePass = createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single"})
    g.addPass(AccumulatePass, "AccumulatePass")

    ToneMapper = createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("VBufferRT.vbuffer", "DirectLighting.vbuffer")
    g.addEdge("VBufferRT.viewW", "DirectLighting.viewW")
    g.addEdge("DirectLighting.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")

    g.markOutput("ToneMapper.dst")
    return g


DirectLighting = render_graph_DirectLighting()
try:
    m.addGraph(DirectLighting)
except NameError:
    None
