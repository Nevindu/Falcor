add_plugin(ReSTIRPT)

target_sources(ReSTIRPT PRIVATE
    ColorType.slang
    GeneratePaths.cs.slang
    GuideData.slang
    LoadShadingData.slang
    Params.slang
    PathState.slang
    ReSTIRPT.cpp
    ReSTIRPT.h
    ReSTIRPT.slang
    ReflectTypes.cs.slang
    ResolvePass.cs.slang
    ShiftMapping_Hybrid.slang
    ShiftMapping_RandomReplay.slang
    ShiftMapping_Reconnection.slang
    ShiftMappings.slang
    SpatialReuse.cs.slang
    StaticParams.slang
    TracePass.rt.slang
)

target_copy_shaders(ReSTIRPT RenderPasses/ReSTIRPT)

target_source_group(ReSTIRPT "RenderPasses")
