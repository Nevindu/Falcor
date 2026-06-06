add_plugin(PathTracerN)

target_sources(PathTracerN PRIVATE
    ColorType.slang
    GeneratePaths.cs.slang
    GuideData.slang
    LoadShadingData.slang
    Params.slang
    PathState.slang
    PathTracerN.cpp
    PathTracerN.h
    PathTracerN.slang
    ReflectTypes.cs.slang
    ResolvePass.cs.slang
    StaticParams.slang
    TracePass.rt.slang
)

target_copy_shaders(PathTracerN RenderPasses/PathTracerN)

target_source_group(PathTracerN "RenderPasses")
