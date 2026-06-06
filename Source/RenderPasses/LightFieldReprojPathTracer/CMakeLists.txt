add_plugin(LightFieldReprojPathTracer)

target_sources(LightFieldReprojPathTracer PRIVATE
    ColorType.slang
    GeneratePaths.cs.slang
    GuideData.slang
    LoadShadingData.slang
    NRDHelpers.slang
    Params.slang
    PathState.slang
    LightFieldReprojPathTracer.slang
    LightFieldReprojPathTracer.cpp
    LightFieldReprojPathTracer.h
    LightFieldReprojPathTracerNRD.slang
    LightFieldSpatialReproject.cs.slang
    ReflectTypes.cs.slang
    ResolvePass.cs.slang
    StaticParams.slang
    TracePass.rt.slang
)

target_copy_shaders(LightFieldReprojPathTracer RenderPasses/LightFieldReprojPathTracer)

target_source_group(LightFieldReprojPathTracer "RenderPasses")
