add_plugin(DirectLighting)

target_sources(DirectLighting PRIVATE
    DirectLighting.cpp
    DirectLighting.h
    DirectLighting.rt.slang
)

target_copy_shaders(DirectLighting RenderPasses/DirectLighting)

target_source_group(DirectLighting "RenderPasses")
