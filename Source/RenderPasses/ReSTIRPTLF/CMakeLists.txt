add_plugin(ReSTIRPTLF)

target_sources(ReSTIRPTLF PRIVATE
    ColorType.slang
    GenerateLightFieldVBuffer.cs.slang
    GeneratePaths.cs.slang
    GuideData.slang
    LightField.slang
    LoadShadingData.slang
    Params.slang
    PathState.slang
    ReSTIRPTLF.cpp
    ReSTIRPTLF.h
    ReSTIRPTLF.slang
    ReflectTypes.cs.slang
    ResolvePass.cs.slang
    ShiftMapping_RandomReplay.slang
    ShiftMapping_Reconnection.slang
    ShiftMappings.slang
    SpatialReuse.cs.slang
    StaticParams.slang
    TemporalReuse.cs.slang
    TracePass.rt.slang
)

target_copy_shaders(ReSTIRPTLF RenderPasses/ReSTIRPTLF)

target_source_group(ReSTIRPTLF "RenderPasses")
