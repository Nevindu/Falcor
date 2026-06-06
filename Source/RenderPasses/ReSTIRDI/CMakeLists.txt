add_plugin(ReSTIRDI)

target_sources(ReSTIRDI PRIVATE
    ReSTIRDI.cpp
    ReSTIRDI.h
    ReSTIRDI.rt.slang
)

target_copy_shaders(ReSTIRDI RenderPasses/ReSTIRDI)

target_source_group(ReSTIRDI "RenderPasses")
