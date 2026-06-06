from falcor import *


def render_graph_ReSTIRPT():
    g = RenderGraph("ReSTIRPT")

    ReSTIRPT = createPass("ReSTIRPT", {"initialCandidateCount": 1})
    g.addPass(ReSTIRPT, "ReSTIRPT")

    VBufferRT = createPass("VBufferRT", {"samplePattern": "Stratified", "sampleCount": 16, "useAlphaTest": True})
    g.addPass(VBufferRT, "VBufferRT")

    AccumulatePass = createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single"})
    g.addPass(AccumulatePass, "AccumulatePass")

    ToneMapper = createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("VBufferRT.vbuffer", "ReSTIRPT.vbuffer")
    g.addEdge("VBufferRT.viewW", "ReSTIRPT.viewW")
    g.addEdge("VBufferRT.mvec", "ReSTIRPT.mvec")
    g.addEdge("ReSTIRPT.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")

    g.markOutput("ToneMapper.dst")
    return g


ReSTIRPT = render_graph_ReSTIRPT()
try:
    m.addGraph(ReSTIRPT)
except NameError:
    None
