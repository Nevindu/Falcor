from falcor import *


def render_graph_ReSTIRDI():
    g = RenderGraph("ReSTIRDI")

    VBufferRT = createPass("VBufferRT", {"samplePattern": "Stratified", "sampleCount": 16, "useAlphaTest": True})
    g.addPass(VBufferRT, "VBufferRT")

    ReSTIRDI = createPass("ReSTIRDI", {"lightCandidateCount": 4, "brdfCandidateCount": 1, "useTemporalReuse": True, "spatialNeighborCount": 2, "spatialMISStrategy": 0, "useVisibilityInTarget": False})
    g.addPass(ReSTIRDI, "ReSTIRDI")

    AccumulatePass = createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single"})
    g.addPass(AccumulatePass, "AccumulatePass")

    ToneMapper = createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("VBufferRT.vbuffer", "ReSTIRDI.vbuffer")
    g.addEdge("VBufferRT.viewW", "ReSTIRDI.viewW")
    g.addEdge("VBufferRT.mvec", "ReSTIRDI.mvec")
    g.addEdge("ReSTIRDI.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")

    g.markOutput("ToneMapper.dst")
    return g


ReSTIRDI = render_graph_ReSTIRDI()
try:
    m.addGraph(ReSTIRDI)
except NameError:
    None
