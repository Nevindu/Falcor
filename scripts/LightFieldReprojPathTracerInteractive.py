from falcor import *
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)

scene_path = globals().get("LFR_SCENE", os.path.join(repo_dir, "media", "classroom", "scene-v4.pbrt"))
view_grid_x = globals().get("LFR_VIEW_GRID_X", 6)
view_grid_y = globals().get("LFR_VIEW_GRID_Y", 6)
source_view_x = globals().get("LFR_SOURCE_VIEW_X", 2)
source_view_y = globals().get("LFR_SOURCE_VIEW_Y", 2)
per_view_resolution = globals().get("LFR_PER_VIEW_RESOLUTION", False)
width = globals().get("LFR_WIDTH", 1280 if per_view_resolution else 1920)
height = globals().get("LFR_HEIGHT", 720 if per_view_resolution else 1080)
atlas_width = width * view_grid_x if per_view_resolution else width
atlas_height = height * view_grid_y if per_view_resolution else height
view_width = max(1, atlas_width // max(1, view_grid_x))
view_height = max(1, atlas_height // max(1, view_grid_y))

samples_per_pixel = globals().get("LFR_SAMPLES_PER_PIXEL", 1)
baseline_x = globals().get("LFR_BASELINE_X", 0.01)
baseline_y = globals().get("LFR_BASELINE_Y", 0.01)
position_threshold = globals().get("LFR_POSITION_THRESHOLD", 0.01)
normal_dot_threshold = globals().get("LFR_NORMAL_DOT_THRESHOLD", 0.95)
temporal_alpha = globals().get("LFR_TEMPORAL_ALPHA", 0.2)
display_mode = globals().get("LFR_DISPLAY_MODE", 0)
use_denoiser = globals().get("LFR_USE_DENOISER", True)
load_scene = globals().get("LFR_LOAD_SCENE", True)


def render_graph_LightFieldReprojPathTracer():
    g = RenderGraph("LightFieldReprojPathTracerInteractive")

    vbuffer = createPass(
        "LightFieldVBufferRT",
        {
            "samplePattern": "Stratified",
            "sampleCount": 16,
            "useAlphaTest": True,
            "useDOF": False,
            "lfViewGridX": view_grid_x,
            "lfViewGridY": view_grid_y,
            "lfSourceViewX": source_view_x,
            "lfSourceViewY": source_view_y,
            "lfBaselineX": baseline_x,
            "lfBaselineY": baseline_y,
        },
    )
    g.addPass(vbuffer, "LFVBufferRT")

    source_display_mode = 1 if use_denoiser else display_mode
    source_tracer = createPass(
        "LightFieldReprojPathTracer",
        {
            "samplesPerPixel": samples_per_pixel,
            "maxSurfaceBounces": 8,
            "maxDiffuseBounces": 8,
            "maxSpecularBounces": 8,
            "maxTransmissionBounces": 8,
            "useNEE": True,
            "useMIS": True,
            "useRTXDI": False,
            "lfViewGridX": view_grid_x,
            "lfViewGridY": view_grid_y,
            "lfSourceViewX": source_view_x,
            "lfSourceViewY": source_view_y,
            "lfBaselineX": baseline_x,
            "lfBaselineY": baseline_y,
            "lfSourceOnly": 1,
            "lfTemporalAlpha": temporal_alpha,
            "lfPositionThreshold": position_threshold,
            "lfNormalDotThreshold": normal_dot_threshold,
            "lfDisplayMode": source_display_mode,
        },
    )
    g.addPass(source_tracer, "SourceTracer" if use_denoiser else "LightFieldReprojPathTracer")

    if use_denoiser:
        reprojector = createPass(
            "LightFieldReprojPathTracer",
            {
                "samplesPerPixel": samples_per_pixel,
                "maxSurfaceBounces": 8,
                "maxDiffuseBounces": 8,
                "maxSpecularBounces": 8,
                "maxTransmissionBounces": 8,
                "useNEE": True,
                "useMIS": True,
                "useRTXDI": False,
                "lfViewGridX": view_grid_x,
                "lfViewGridY": view_grid_y,
                "lfSourceViewX": source_view_x,
                "lfSourceViewY": source_view_y,
                "lfBaselineX": baseline_x,
                "lfBaselineY": baseline_y,
                "lfSourceOnly": 1,
                "lfTemporalAlpha": temporal_alpha,
                "lfPositionThreshold": position_threshold,
                "lfNormalDotThreshold": normal_dot_threshold,
                "lfDisplayMode": display_mode,
            },
        )
        g.addPass(reprojector, "LightFieldReprojPathTracer")

        source_tile_extract = createPass(
            "LightFieldTilePass",
            {
                "mode": "Extract",
                "lfViewGridX": view_grid_x,
                "lfViewGridY": view_grid_y,
                "lfSourceViewX": source_view_x,
                "lfSourceViewY": source_view_y,
                "fixedOutputSize": uint2(view_width, view_height),
            },
        )
        g.addPass(source_tile_extract, "SourceTileExtract")

        denoiser = createPass(
            "SVGFPass",
            {
                "Enabled": True,
                "Iterations": 4,
                "FeedbackTap": 1,
                "VarianceEpsilon": 9.999999747378752e-05,
                "PhiColor": 10.0,
                "PhiNormal": 128.0,
                "Alpha": temporal_alpha,
                "MomentsAlpha": 0.2,
                "outputSize": "Fixed",
                "fixedOutputSize": uint2(view_width, view_height),
            },
        )
        g.addPass(denoiser, "Denoiser")

        source_tile_insert = createPass(
            "LightFieldTilePass",
            {
                "mode": "Insert",
                "lfViewGridX": view_grid_x,
                "lfViewGridY": view_grid_y,
                "lfSourceViewX": source_view_x,
                "lfSourceViewY": source_view_y,
            },
        )
        g.addPass(source_tile_insert, "SourceTileInsert")

    accumulate = createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single"})
    g.addPass(accumulate, "AccumulatePass")

    tonemapper = createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0})
    g.addPass(tonemapper, "ToneMapper")

    if use_denoiser:
        g.addEdge("LFVBufferRT.vbuffer", "SourceTracer.vbuffer")
        g.addEdge("LFVBufferRT.viewW", "SourceTracer.viewW")
        g.addEdge("LFVBufferRT.mvec", "SourceTracer.mvec")
        g.addEdge("SourceTracer.color", "SourceTileExtract.srcColor")
        g.addEdge("SourceTracer.albedo", "SourceTileExtract.srcAlbedo")
        g.addEdge("SourceTracer.nrdEmission", "SourceTileExtract.srcEmission")
        g.addEdge("LFVBufferRT.posW", "SourceTileExtract.srcPosW")
        g.addEdge("LFVBufferRT.guideNormalW", "SourceTileExtract.srcGuideNormalW")
        g.addEdge("LFVBufferRT.pnFwidth", "SourceTileExtract.srcPNFwidth")
        g.addEdge("LFVBufferRT.linearZ", "SourceTileExtract.srcLinearZ")
        g.addEdge("LFVBufferRT.mvec", "SourceTileExtract.srcMvec")

        g.addEdge("SourceTileExtract.color", "Denoiser.Color")
        g.addEdge("SourceTileExtract.albedo", "Denoiser.Albedo")
        g.addEdge("SourceTileExtract.emission", "Denoiser.Emission")
        g.addEdge("SourceTileExtract.posW", "Denoiser.WorldPosition")
        g.addEdge("SourceTileExtract.guideNormalW", "Denoiser.WorldNormal")
        g.addEdge("SourceTileExtract.pnFwidth", "Denoiser.PositionNormalFwidth")
        g.addEdge("SourceTileExtract.linearZ", "Denoiser.LinearZ")
        g.addEdge("SourceTileExtract.mvec", "Denoiser.MotionVec")
        g.addEdge("Denoiser.Filtered image", "SourceTileInsert.srcColor")

        g.addEdge("LFVBufferRT.vbuffer", "LightFieldReprojPathTracer.vbuffer")
        g.addEdge("LFVBufferRT.viewW", "LightFieldReprojPathTracer.viewW")
        g.addEdge("LFVBufferRT.mvec", "LightFieldReprojPathTracer.mvec")
        g.addEdge("SourceTileInsert.dstColor", "LightFieldReprojPathTracer.sourceColor")
    else:
        g.addEdge("LFVBufferRT.vbuffer", "LightFieldReprojPathTracer.vbuffer")
        g.addEdge("LFVBufferRT.viewW", "LightFieldReprojPathTracer.viewW")
        g.addEdge("LFVBufferRT.mvec", "LightFieldReprojPathTracer.mvec")

    g.addEdge("LightFieldReprojPathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g


if load_scene:
    m.loadScene(scene_path)
m.resizeFrameBuffer(atlas_width, atlas_height)

graph = render_graph_LightFieldReprojPathTracer()
m.addGraph(graph)
m.setActiveGraph(graph)

print("Interactive LightFieldReprojPathTracer")
print("  scene: {}".format(scene_path))
print("  atlas resolution: {}x{}".format(atlas_width, atlas_height))
print("  per-view resolution: {}x{}".format(view_width, view_height))
print("  view grid: {}x{}".format(view_grid_x, view_grid_y))
print("  denoiser: {}".format("enabled" if use_denoiser else "disabled"))
