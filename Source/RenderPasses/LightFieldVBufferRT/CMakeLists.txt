add_plugin(LightFieldVBufferRT)

target_sources(LightFieldVBufferRT PRIVATE
    LightFieldVBufferRT.cpp
    LightFieldVBufferRT.h
    LightFieldVBufferRT.slang
    LightFieldVBufferRT.rt.slang
    LightFieldVBufferRT.cs.slang
)

target_copy_shaders(LightFieldVBufferRT RenderPasses/LightFieldVBufferRT)

target_source_group(LightFieldVBufferRT "RenderPasses")

target_sources(LightFieldVBufferRT PRIVATE
    ../GBuffer/GBufferBase.cpp
    ../GBuffer/GBufferBase.h
)

target_compile_definitions(LightFieldVBufferRT PRIVATE FALCOR_GBUFFERBASE_NO_REGISTER_PLUGIN)
