add_plugin(LightFieldPathTracer)

target_sources(LightFieldPathTracer PRIVATE
    ColorType.slang
    GeneratePaths.cs.slang
    GuideData.slang
    LoadShadingData.slang
    Params.slang
    PathState.slang
    LightFieldPathTracer.cpp
    LightFieldPathTracer.h
    LightFieldPathTracer.slang
    ReflectTypes.cs.slang
    ResolvePass.cs.slang
    StaticParams.slang
    TracePass.rt.slang
)

target_copy_shaders(LightFieldPathTracer RenderPasses/LightFieldPathTracer)

target_source_group(LightFieldPathTracer "RenderPasses")
