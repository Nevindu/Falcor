from falcor import *

# Experimental light-field reprojection path tracing graph.
# Display in Mogwai: ToneMapper.dst. It is a quilt/atlas: the 6x6 tiles are the LF views.
# For debug, change lfDisplayMode on LightFieldReprojPathTracer:
#   0 = final/reprojected quilt
#   1 = source-only quilt
#   2 = valid mask
#   3 = discard overlay (invalid reprojection in red, like paper Fig. 8)
#   4 = view ID visualization

def render_graph_LightFieldReprojPathTracer():
    g = RenderGraph("LightFieldReprojPathTracer")

    # Paper defaults: 6x6 views, 1 spp, 8 max bounces, NEE on, temporal/reprojection alpha 0.2.
    # Important: the output texture is a quilt atlas. The per-view tile resolution is
    # output_resolution / (6,6). To exactly reproduce the paper's 6x6 of 1280x720,
    # you would need a 7680x4320 atlas, which exceeds this PathTracer's default 4096
    # per-axis shader packing limit. For Mogwai/debugging, use the window-sized quilt.
    LFVBufferRT = createPass("LightFieldVBufferRT", {
        "samplePattern": "Stratified",
        "sampleCount": 16,
        "useAlphaTest": True,
        "useDOF": False,
        "lfViewGridX": 6,
        "lfViewGridY": 6,
        "lfSourceViewX": 2,
        "lfSourceViewY": 2,
        "lfBaselineX": 0.01,
        "lfBaselineY": 0.01,
    })
    g.addPass(LFVBufferRT, "LFVBufferRT")

    SourceTracer = createPass("LightFieldReprojPathTracer", {
        "samplesPerPixel": 1,
        "maxSurfaceBounces": 8,
        "maxDiffuseBounces": 8,
        "maxSpecularBounces": 8,
        "maxTransmissionBounces": 8,
        "useNEE": True,
        "useMIS": True,
        "useRTXDI": False,
        "lfViewGridX": 6,
        "lfViewGridY": 6,
        "lfSourceViewX": 2,
        "lfSourceViewY": 2,
        "lfBaselineX": 0.01,
        "lfBaselineY": 0.01,
        "lfSourceOnly": 1,
        "lfTemporalAlpha": 0.2,
        "lfPositionThreshold": 0.01,
        "lfNormalDotThreshold": 0.95,
        "lfDisplayMode": 1,
    })
    g.addPass(SourceTracer, "SourceTracer")

    Reprojector = createPass("LightFieldReprojPathTracer", {
        "samplesPerPixel": 1,
        "maxSurfaceBounces": 8,
        "maxDiffuseBounces": 8,
        "maxSpecularBounces": 8,
        "maxTransmissionBounces": 8,
        "useNEE": True,
        "useMIS": True,
        "useRTXDI": False,
        "lfViewGridX": 6,
        "lfViewGridY": 6,
        "lfSourceViewX": 2,
        "lfSourceViewY": 2,
        "lfBaselineX": 0.01,
        "lfBaselineY": 0.01,
        "lfSourceOnly": 1,
        "lfTemporalAlpha": 0.2,
        "lfPositionThreshold": 0.01,
        "lfNormalDotThreshold": 0.95,
        "lfDisplayMode": 0,
    })
    g.addPass(Reprojector, "LightFieldReprojPathTracer")

    Denoiser = createPass("NRD", {
        "maxIntensity": 250.0,
        "worldSpaceMotion": False,
        "enableReprojectionTestSkippingWithoutMotion": True,
        "spatialVarianceEstimationHistoryThreshold": 1,
    })
    g.addPass(Denoiser, "Denoiser")

    ModulateIllumination = createPass("ModulateIllumination", {"useResidualRadiance": False})
    g.addPass(ModulateIllumination, "ModulateIllumination")

    AccumulatePass = createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single"})
    g.addPass(AccumulatePass, "AccumulatePass")

    ToneMapper = createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("LFVBufferRT.vbuffer", "SourceTracer.vbuffer")
    g.addEdge("LFVBufferRT.viewW", "SourceTracer.viewW")
    g.addEdge("LFVBufferRT.mvec", "SourceTracer.mvec")
    g.markOutput("SourceTracer.color")

    # Denoise only the path-traced source-view atlas, then reproject that denoised source.
    g.addEdge("SourceTracer.nrdDiffuseRadianceHitDist", "Denoiser.diffuseRadianceHitDist")
    g.addEdge("SourceTracer.nrdSpecularRadianceHitDist", "Denoiser.specularRadianceHitDist")
    g.addEdge("LFVBufferRT.linearZ", "Denoiser.viewZ")
    g.addEdge("LFVBufferRT.normWRoughnessMaterialID", "Denoiser.normWRoughnessMaterialID")
    g.addEdge("LFVBufferRT.mvec", "Denoiser.mvec")
    g.addEdge("SourceTracer.nrdEmission", "ModulateIllumination.emission")
    g.addEdge("SourceTracer.nrdDiffuseReflectance", "ModulateIllumination.diffuseReflectance")
    g.addEdge("Denoiser.filteredDiffuseRadianceHitDist", "ModulateIllumination.diffuseRadiance")
    g.addEdge("SourceTracer.nrdSpecularReflectance", "ModulateIllumination.specularReflectance")
    g.addEdge("Denoiser.filteredSpecularRadianceHitDist", "ModulateIllumination.specularRadiance")

    g.addEdge("LFVBufferRT.vbuffer", "LightFieldReprojPathTracer.vbuffer")
    g.addEdge("LFVBufferRT.viewW", "LightFieldReprojPathTracer.viewW")
    g.addEdge("LFVBufferRT.mvec", "LightFieldReprojPathTracer.mvec")
    g.addEdge("ModulateIllumination.output", "LightFieldReprojPathTracer.sourceColor")
    g.addEdge("LightFieldReprojPathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")

    return g

LightFieldReprojPathTracer = render_graph_LightFieldReprojPathTracer()
try:
    m.addGraph(LightFieldReprojPathTracer)
except NameError:
    None
