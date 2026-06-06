/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Rendering/Lights/LightBVHSampler.h"
#include "Rendering/Lights/EmissivePowerSampler.h"
#include "Rendering/Lights/EnvMapSampler.h"

using namespace Falcor;

class ReSTIRDI : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ReSTIRDI, "ReSTIRDI", "Direct lighting integrator.");

    enum class SpatialMISStrategy : uint32_t
    {
        None = 0,
        Contribution = 1,
        GBH = 2,
        Pairwise = 3,
    };

    static ref<ReSTIRDI> create(ref<Device> pDevice, const Properties& props) { return make_ref<ReSTIRDI>(pDevice, props); }

    ReSTIRDI(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    void parseProperties(const Properties& props);
    bool prepareLighting(RenderContext* pRenderContext);
    void prepareVars();

    ref<Scene> mpScene;
    ref<SampleGenerator> mpSampleGenerator;
    std::unique_ptr<EnvMapSampler> mpEnvMapSampler;
    std::unique_ptr<EmissiveLightSampler> mpEmissiveSampler;
    ref<ParameterBlock> mpReSTIRDIBlock;
    ref<Buffer> mpInitialReservoirs;
    ref<Buffer> mpTemporalReservoirs;
    ref<Buffer> mpSpatialReservoirs;
    ref<Buffer> mpPrevReservoirs;
    ref<Buffer> mpCurrentSurfaces;
    ref<Buffer> mpPrevSurfaces;

    uint32_t mLightCandidateCount = 1;
    uint32_t mBRDFCandidateCount = 1;
    uint32_t mSpatialNeighborCount = 2;
    uint32_t mSpatialRadius = 16;
    uint32_t mSpatialMCap = 20;
    uint32_t mTemporalMCap = 8;
    uint2 mReservoirDim = {};
    bool mUseAnalyticLights = true;
    bool mUseEmissiveLights = true;
    bool mUseEnvLight = true;
    bool mUseEmissiveMaterials = true;
    bool mUseEnvBackground = true;
    bool mUseTemporalReuse = true;
    bool mUseBRDFSampling = true;
    SpatialMISStrategy mSpatialMISStrategy = SpatialMISStrategy::None;
    bool mUseVisibilityInTarget = false;
    bool mUseVisibilityInSpatialGBH = false;
    bool mUseVisibilityInTemporalMIS = false;
    bool mOptionsChanged = false;
    uint32_t mFrameCount = 0;

    struct TracePass
    {
        ref<Program> pProgram;
        ref<RtBindingTable> pBindingTable;
        ref<RtProgramVars> pVars;
    };

    TracePass mInitialPass;
    TracePass mTemporalPass;
    TracePass mSpatialPass;
    TracePass mShadePass;
};
