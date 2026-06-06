add_plugin(LFSubPixelPathTracer)

target_sources(LFSubPixelPathTracer PRIVATE
    ColorType.slang
    GeneratePaths.cs.slang
    GuideData.slang
    LoadShadingData.slang
    Params.slang
    PathState.slang
    LFSubPixelPathTracer.cpp
    LFSubPixelPathTracer.h
    LFSubPixelPathTracer.slang
    ReflectTypes.cs.slang
    ResolvePass.cs.slang
    StaticParams.slang
    TracePass.rt.slang
)

target_copy_shaders(LFSubPixelPathTracer RenderPasses/LFSubPixelPathTracer)

target_source_group(LFSubPixelPathTracer "RenderPasses")
