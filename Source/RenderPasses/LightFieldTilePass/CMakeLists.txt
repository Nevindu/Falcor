add_plugin(LightFieldTilePass)

target_sources(LightFieldTilePass PRIVATE
    LightFieldTilePass.cpp
    LightFieldTilePass.h
    LightFieldTilePass.cs.slang
)

target_copy_shaders(LightFieldTilePass RenderPasses/LightFieldTilePass)

target_source_group(LightFieldTilePass "RenderPasses")
