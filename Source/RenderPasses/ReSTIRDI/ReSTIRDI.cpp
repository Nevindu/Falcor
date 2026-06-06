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
#include "ReSTIRDI.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReSTIRDI>();
}

namespace
{
const char kShaderFile[] = "RenderPasses/ReSTIRDI/ReSTIRDI.rt.slang";

const uint32_t kMaxPayloadSizeBytes = 64u;
const uint32_t kMaxRecursionDepth = 2u;

const char kInputVBuffer[] = "vbuffer";
const char kInputViewDir[] = "viewW";
const char kInputMotionVectors[] = "mvec";
const char kOutputColor[] = "color";

const ChannelList kInputChannels = {
    { kInputVBuffer, "gVBuffer", "Visibility buffer in packed format" },
    { kInputViewDir, "gViewW", "World-space view direction (xyz float format)", true },
    { kInputMotionVectors, "gMotionVectors", "Motion vector buffer", true },
};

const ChannelList kOutputChannels = {
    { kOutputColor, "gOutputColor", "Output color (linear)", false, ResourceFormat::RGBA32Float },
};

const char kUseAnalyticLights[] = "useAnalyticLights";
const char kUseEmissiveLights[] = "useEmissiveLights";
const char kUseEnvLight[] = "useEnvLight";
const char kUseEmissiveMaterials[] = "useEmissiveMaterials";
const char kUseEnvBackground[] = "useEnvBackground";
const char kUseTemporalReuse[] = "useTemporalReuse";
const char kUseBRDFSampling[] = "useBRDFSampling";
const char kSpatialMISStrategy[] = "spatialMISStrategy";
const char kUseVisibilityInTarget[] = "useVisibilityInTarget";
const char kUseVisibilityInSpatialGBH[] = "useVisibilityInSpatialGBH";
const char kUseVisibilityInTemporalMIS[] = "useVisibilityInTemporalMIS";
const char kSamplesPerPixel[] = "samplesPerPixel";
const char kLightCandidateCount[] = "lightCandidateCount";
const char kBRDFCandidateCount[] = "brdfCandidateCount";
const char kSpatialNeighborCount[] = "spatialNeighborCount";
const char kSpatialRadius[] = "spatialRadius";
const char kSpatialMCap[] = "spatialMCap";
const char kTemporalMCap[] = "temporalMCap";
const uint32_t kReservoirStructSize = 112u;
const uint32_t kTemporalSurfaceStructSize = 32u;

const Gui::DropdownList kSpatialMISStrategyList = {
    { (uint32_t)ReSTIRDI::SpatialMISStrategy::None, "None (cheap)" },
    { (uint32_t)ReSTIRDI::SpatialMISStrategy::Contribution, "Contribution MIS" },
    { (uint32_t)ReSTIRDI::SpatialMISStrategy::GBH, "Generalized balance" },
    { (uint32_t)ReSTIRDI::SpatialMISStrategy::Pairwise, "Pairwise MIS" },
};
} // namespace

/// Create the render pass, validate required raytracing support, and apply graph properties.
ReSTIRDI::ReSTIRDI(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_5))
        FALCOR_THROW("ReSTIRDI requires Shader Model 6.5 support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
        FALCOR_THROW("ReSTIRDI requires Raytracing Tier 1.1 support.");

    parseProperties(props);

    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_TINY_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);
}

/// Read serialized graph properties and clamp them to the ranges supported by the shader.
void ReSTIRDI::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kSamplesPerPixel)
            mLightCandidateCount = value;
        else if (key == kLightCandidateCount)
            mLightCandidateCount = value;
        else if (key == kBRDFCandidateCount)
            mBRDFCandidateCount = value;
        else if (key == kUseAnalyticLights)
            mUseAnalyticLights = value;
        else if (key == kUseEmissiveLights)
            mUseEmissiveLights = value;
        else if (key == kUseEnvLight)
            mUseEnvLight = value;
        else if (key == kUseEmissiveMaterials)
            mUseEmissiveMaterials = value;
        else if (key == kUseEnvBackground)
            mUseEnvBackground = value;
        else if (key == kUseTemporalReuse)
            mUseTemporalReuse = value;
        else if (key == kUseBRDFSampling)
            mUseBRDFSampling = value;
        else if (key == kSpatialMISStrategy)
            mSpatialMISStrategy = static_cast<SpatialMISStrategy>((uint32_t)value);
        else if (key == kUseVisibilityInTarget)
            mUseVisibilityInTarget = value;
        else if (key == kUseVisibilityInSpatialGBH)
            mUseVisibilityInSpatialGBH = value;
        else if (key == kUseVisibilityInTemporalMIS)
            mUseVisibilityInTemporalMIS = value;
        else if (key == kSpatialNeighborCount)
            mSpatialNeighborCount = value;
        else if (key == kSpatialRadius)
            mSpatialRadius = value;
        else if (key == kSpatialMCap)
            mSpatialMCap = value;
        else if (key == kTemporalMCap)
            mTemporalMCap = value;
        else
            logWarning("Unknown property '{}' in ReSTIRDI properties.", key);
    }

    mLightCandidateCount = std::min(64u, std::max(1u, mLightCandidateCount));
    mBRDFCandidateCount = std::min(64u, mBRDFCandidateCount);
    mSpatialNeighborCount = std::min(32u, mSpatialNeighborCount);
    mSpatialRadius = std::max(1u, mSpatialRadius);
    mSpatialMCap = std::max(1u, mSpatialMCap);
    mTemporalMCap = std::max(1u, mTemporalMCap);
    if ((uint32_t)mSpatialMISStrategy > (uint32_t)SpatialMISStrategy::Pairwise)
        mSpatialMISStrategy = SpatialMISStrategy::Contribution;
}

/// Serialize current ReSTIRDI settings so scripts and render graphs can reproduce the pass state.
Properties ReSTIRDI::getProperties() const
{
    Properties props;
    props[kSamplesPerPixel] = mLightCandidateCount;
    props[kLightCandidateCount] = mLightCandidateCount;
    props[kBRDFCandidateCount] = mBRDFCandidateCount;
    props[kUseAnalyticLights] = mUseAnalyticLights;
    props[kUseEmissiveLights] = mUseEmissiveLights;
    props[kUseEnvLight] = mUseEnvLight;
    props[kUseEmissiveMaterials] = mUseEmissiveMaterials;
    props[kUseEnvBackground] = mUseEnvBackground;
    props[kUseTemporalReuse] = mUseTemporalReuse;
    props[kUseBRDFSampling] = mUseBRDFSampling;
    props[kSpatialMISStrategy] = (uint32_t)mSpatialMISStrategy;
    props[kUseVisibilityInTarget] = mUseVisibilityInTarget;
    props[kUseVisibilityInSpatialGBH] = mUseVisibilityInSpatialGBH;
    props[kUseVisibilityInTemporalMIS] = mUseVisibilityInTemporalMIS;
    props[kSpatialNeighborCount] = mSpatialNeighborCount;
    props[kSpatialRadius] = mSpatialRadius;
    props[kSpatialMCap] = mSpatialMCap;
    props[kTemporalMCap] = mTemporalMCap;
    return props;
}

/// Declare the VBuffer, optional motion/vector inputs, and linear color output used by the pass.
RenderPassReflection ReSTIRDI::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);
    return reflector;
}

/// Run the ReSTIRDI frame pipeline: initial RIS, optional temporal reuse, optional spatial reuse, and shading.
void ReSTIRDI::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[kRenderPassRefreshFlags] = flags | RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    const auto pOutput = renderData.getTexture(kOutputColor);
    FALCOR_ASSERT(pOutput);

    if (!mpScene)
    {
        pRenderContext->clearUAV(pOutput->getUAV().get(), float4(0.f));
        return;
    }

    if (is_set(mpScene->getUpdates(), IScene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), IScene::UpdateFlags::GeometryChanged))
    {
        mInitialPass.pProgram = nullptr;
        mTemporalPass.pProgram = nullptr;
        mSpatialPass.pProgram = nullptr;
        mShadePass.pProgram = nullptr;
        mInitialPass.pVars = nullptr;
        mTemporalPass.pVars = nullptr;
        mSpatialPass.pVars = nullptr;
        mShadePass.pVars = nullptr;
        setScene(pRenderContext, mpScene);
    }

    const bool useDOF = mpScene->getCamera()->getApertureRadius() > 0.f;
    if (useDOF && renderData[kInputViewDir] == nullptr)
        logWarning("Depth-of-field requires the '{}' input. Expect incorrect shading.", kInputViewDir);

    bool lightingChanged = prepareLighting(pRenderContext);
    if (lightingChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[kRenderPassRefreshFlags] = flags | RenderPassRefreshFlags::LightingChanged;
    }

    // Push UI/script settings into shader specialization constants.
    auto addDefines = [&](const ref<Program>& pProgram)
    {
        bool changed = false;
        changed |= pProgram->addDefine("LIGHT_CANDIDATE_COUNT", std::to_string(mLightCandidateCount));
        changed |= pProgram->addDefine("BRDF_CANDIDATE_COUNT", std::to_string(mBRDFCandidateCount));
        changed |= pProgram->addDefine("SPATIAL_NEIGHBOR_COUNT", std::to_string(mSpatialNeighborCount));
        changed |= pProgram->addDefine("SPATIAL_RADIUS", std::to_string(mSpatialRadius));
        changed |= pProgram->addDefine("SPATIAL_M_CAP", std::to_string(mSpatialMCap));
        changed |= pProgram->addDefine("TEMPORAL_M_CAP", std::to_string(mTemporalMCap));
        changed |= pProgram->addDefine("USE_ANALYTIC_LIGHTS", (mUseAnalyticLights && mpScene->useAnalyticLights()) ? "1" : "0");
        changed |= pProgram->addDefine("USE_EMISSIVE_LIGHTS", (mUseEmissiveLights && mpScene->useEmissiveLights()) ? "1" : "0");
        changed |= pProgram->addDefine("USE_ENV_LIGHT", (mUseEnvLight && mpScene->useEnvLight()) ? "1" : "0");
        changed |= pProgram->addDefine("USE_EMISSIVE_MATERIALS", mUseEmissiveMaterials ? "1" : "0");
        changed |= pProgram->addDefine("USE_ENV_BACKGROUND", (mUseEnvBackground && mpScene->useEnvBackground()) ? "1" : "0");
        changed |= pProgram->addDefine("USE_TEMPORAL_REUSE", mUseTemporalReuse ? "1" : "0");
        changed |= pProgram->addDefine("USE_BRDF_SAMPLING", mUseBRDFSampling ? "1" : "0");
        changed |= pProgram->addDefine("SPATIAL_MIS_STRATEGY", std::to_string((uint32_t)mSpatialMISStrategy));
        changed |= pProgram->addDefine("USE_VISIBILITY_IN_TARGET", mUseVisibilityInTarget ? "1" : "0");
        changed |= pProgram->addDefine("USE_VISIBILITY_IN_SPATIAL_GBH", mUseVisibilityInSpatialGBH ? "1" : "0");
        changed |= pProgram->addDefine("USE_VISIBILITY_IN_TEMPORAL_MIS", mUseVisibilityInTemporalMIS ? "1" : "0");
        changed |= pProgram->addDefines(getValidResourceDefines(kInputChannels, renderData));
        changed |= pProgram->addDefines(getValidResourceDefines(kOutputChannels, renderData));
        return changed;
    };

    bool definesChanged = false;
    definesChanged |= addDefines(mInitialPass.pProgram);
    definesChanged |= addDefines(mTemporalPass.pProgram);
    definesChanged |= addDefines(mSpatialPass.pProgram);
    definesChanged |= addDefines(mShadePass.pProgram);

    if (definesChanged)
    {
        mInitialPass.pVars = nullptr;
        mTemporalPass.pVars = nullptr;
        mSpatialPass.pVars = nullptr;
        mShadePass.pVars = nullptr;
        mpReSTIRDIBlock = nullptr;
        mFrameCount = 0;
    }

    if (!mInitialPass.pVars || !mTemporalPass.pVars || !mSpatialPass.pVars || !mShadePass.pVars)
        prepareVars();

    auto blockVar = mpReSTIRDIBlock->getRootVar();
    if (mpEnvMapSampler && mUseEnvLight && mpScene->useEnvLight())
        mpEnvMapSampler->bindShaderData(blockVar["envMapSampler"]);
    if (mpEmissiveSampler && mUseEmissiveLights && mpScene->useEmissiveLights())
        mpEmissiveSampler->bindShaderData(blockVar["emissiveSampler"]);

    // Bind shared textures, frame constants, and the ReSTIRDI parameter block for each raytracing pass.
    auto bindCommon = [&](const ref<RtProgramVars>& pVars)
    {
        auto var = pVars->getRootVar();
        var["gReSTIRDI"] = mpReSTIRDIBlock;

        var["CB"]["gFrameCount"] = mFrameCount;
        var["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;

        for (const auto& channel : kInputChannels)
        {
            if (!channel.texname.empty())
                var[channel.texname] = renderData.getTexture(channel.name);
        }
        for (const auto& channel : kOutputChannels)
        {
            if (!channel.texname.empty())
                var[channel.texname] = renderData.getTexture(channel.name);
        }
    };

    bindCommon(mInitialPass.pVars);
    bindCommon(mTemporalPass.pVars);
    bindCommon(mSpatialPass.pVars);
    bindCommon(mShadePass.pVars);

    const uint3 dispatchDims(pOutput->getWidth(), pOutput->getHeight(), 1);
    if (dispatchDims.x == 0 || dispatchDims.y == 0)
        FALCOR_THROW("ReSTIRDI: Invalid ray dispatch dimensions {}x{}.", dispatchDims.x, dispatchDims.y);

    if (!mpInitialReservoirs || !mpTemporalReservoirs || !mpSpatialReservoirs || !mpPrevReservoirs || !mpCurrentSurfaces || !mpPrevSurfaces ||
        mReservoirDim.x != dispatchDims.x || mReservoirDim.y != dispatchDims.y)
    {
        const uint32_t pixelCount = dispatchDims.x * dispatchDims.y;
        mpInitialReservoirs = mpDevice->createStructuredBuffer(
            kReservoirStructSize,
            pixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        mpTemporalReservoirs = mpDevice->createStructuredBuffer(
            kReservoirStructSize,
            pixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        mpSpatialReservoirs = mpDevice->createStructuredBuffer(
            kReservoirStructSize,
            pixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        mpPrevReservoirs = mpDevice->createStructuredBuffer(
            kReservoirStructSize,
            pixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        mpCurrentSurfaces = mpDevice->createStructuredBuffer(
            kTemporalSurfaceStructSize,
            pixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        mpPrevSurfaces = mpDevice->createStructuredBuffer(
            kTemporalSurfaceStructSize,
            pixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        mReservoirDim = uint2(dispatchDims.x, dispatchDims.y);
        mFrameCount = 0;
    }

    if (mFrameCount == 0)
    {
        logInfo(
            "ReSTIRDI: Dispatching rays {}x{}x{}, scene geometries {}, SBT geometries {}, ray types {}, miss records {}, hit records {}.",
            dispatchDims.x,
            dispatchDims.y,
            dispatchDims.z,
            mpScene->getGeometryCount(),
            mInitialPass.pVars->getGeometryCount(),
            mInitialPass.pVars->getRayTypeCount(),
            mInitialPass.pVars->getMissVarsCount(),
            mInitialPass.pVars->getTotalHitVarsCount()
        );
    }

    // Wire the reservoir buffers according to the stage that is about to run.
    auto bindReservoirs = [&](const ref<RtProgramVars>& pVars, const ref<Buffer>& pInput, const ref<Buffer>& pOutputReservoirs, const ref<Buffer>& pFinal)
    {
        auto var = pVars->getRootVar();
        var["gInputReservoirs"] = pInput;
        var["gOutputReservoirs"] = pOutputReservoirs;
        var["gFinalReservoirs"] = pFinal;
        var["gPrevReservoirs"] = mpPrevReservoirs;
        var["gCurrentSurfaces"] = mpCurrentSurfaces;
        var["gPrevSurfaces"] = mpPrevSurfaces;
    };

    const bool logPassBoundaries = mFrameCount < 2;

    if (logPassBoundaries) logInfo("ReSTIRDI: Initial reservoir pass begin.");
    bindReservoirs(mInitialPass.pVars, mpSpatialReservoirs, mpInitialReservoirs, mpSpatialReservoirs);
    mpScene->raytrace(pRenderContext, mInitialPass.pProgram.get(), mInitialPass.pVars, dispatchDims);
    pRenderContext->uavBarrier(mpInitialReservoirs.get());
    pRenderContext->uavBarrier(mpCurrentSurfaces.get());
    if (logPassBoundaries) logInfo("ReSTIRDI: Initial reservoir pass end.");

    ref<Buffer> pFinalReservoirs = mpInitialReservoirs;
    const bool useTemporalReuse = mUseTemporalReuse && mFrameCount > 0 && renderData[kInputMotionVectors] != nullptr;
    if (useTemporalReuse)
    {
        if (logPassBoundaries) logInfo("ReSTIRDI: Temporal reuse pass begin.");
        bindReservoirs(mTemporalPass.pVars, mpInitialReservoirs, mpTemporalReservoirs, mpTemporalReservoirs);
        mpScene->raytrace(pRenderContext, mTemporalPass.pProgram.get(), mTemporalPass.pVars, dispatchDims);
        pRenderContext->uavBarrier(mpTemporalReservoirs.get());
        pFinalReservoirs = mpTemporalReservoirs;
        if (logPassBoundaries) logInfo("ReSTIRDI: Temporal reuse pass end.");
    }

    if (mSpatialNeighborCount > 0)
    {
        if (logPassBoundaries) logInfo("ReSTIRDI: Spatial reuse pass begin.");
        bindReservoirs(mSpatialPass.pVars, pFinalReservoirs, mpSpatialReservoirs, mpSpatialReservoirs);
        mpScene->raytrace(pRenderContext, mSpatialPass.pProgram.get(), mSpatialPass.pVars, dispatchDims);
        pRenderContext->uavBarrier(mpSpatialReservoirs.get());
        pFinalReservoirs = mpSpatialReservoirs;
        if (logPassBoundaries) logInfo("ReSTIRDI: Spatial reuse pass end.");
    }

    if (logPassBoundaries) logInfo("ReSTIRDI: Shading pass begin.");
    bindReservoirs(mShadePass.pVars, mpInitialReservoirs, mpInitialReservoirs, pFinalReservoirs);
    mpScene->raytrace(pRenderContext, mShadePass.pProgram.get(), mShadePass.pVars, dispatchDims);
    if (logPassBoundaries) logInfo("ReSTIRDI: Shading pass end.");

    pRenderContext->copyResource(mpPrevReservoirs.get(), mpInitialReservoirs.get());
    pRenderContext->copyResource(mpPrevSurfaces.get(), mpCurrentSurfaces.get());
    mFrameCount++;
}

/// Draw ReSTIRDI controls and invalidate shader vars when a setting changes.
void ReSTIRDI::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;
    dirty |= widget.var("Light candidates/pixel", mLightCandidateCount, 1u, 64u);
    widget.tooltip("Number of light-sampled initial candidates considered by reservoir importance sampling.");
    dirty |= widget.var("BRDF candidates/pixel", mBRDFCandidateCount, 0u, 64u);
    widget.tooltip("Number of BRDF-sampled initial candidates. Set to 1 for a cheap realtime specular/glossy helper ray.");
    dirty |= widget.checkbox("Use temporal reuse", mUseTemporalReuse);
    widget.tooltip("Reuse the previous frame reservoir via motion-vector reprojection.");
    dirty |= widget.checkbox("Use BRDF sampling", mUseBRDFSampling);
    widget.tooltip("Add BRDF-sampled initial candidates and balance them against light-sampled candidates.");
    dirty |= widget.var("Temporal M cap", mTemporalMCap, 1u, 128u);
    widget.tooltip("Maximum confidence value after temporal reuse.");
    dirty |= widget.var("Spatial neighbors", mSpatialNeighborCount, 0u, 32u);
    widget.tooltip("Number of neighboring reservoirs considered during spatial reuse.");
    dirty |= widget.var("Spatial radius", mSpatialRadius, 1u, 128u);
    widget.tooltip("Neighbor search radius in pixels.");
    dirty |= widget.var("Spatial M cap", mSpatialMCap, 1u, 128u);
    widget.tooltip("Maximum confidence value after spatial reuse.");
    dirty |= widget.checkbox("Use analytic lights", mUseAnalyticLights);
    dirty |= widget.checkbox("Use emissive lights", mUseEmissiveLights);
    dirty |= widget.checkbox("Use environment light", mUseEnvLight);
    dirty |= widget.checkbox("Use emissive materials", mUseEmissiveMaterials);
    dirty |= widget.checkbox("Use environment background", mUseEnvBackground);
    uint32_t spatialMISStrategy = (uint32_t)mSpatialMISStrategy;
    if (widget.dropdown("Spatial MIS strategy", kSpatialMISStrategyList, spatialMISStrategy))
    {
        mSpatialMISStrategy = static_cast<SpatialMISStrategy>(spatialMISStrategy);
        dirty = true;
    }
    widget.tooltip("MIS strategy used when combining the current pixel reservoir with spatial neighbor reservoirs.");
    dirty |= widget.checkbox("Use visibility in RIS target", mUseVisibilityInTarget);
    widget.tooltip("If enabled, each candidate traces visibility before reservoir weighting. If disabled, only the selected reservoir sample is shadow-tested.");
    dirty |= widget.checkbox("Use visibility in spatial GBH", mUseVisibilityInSpatialGBH);
    widget.tooltip("If enabled, spatial balance weights trace visibility for every candidate/source pair. This is expensive and intended for validation.");
    dirty |= widget.checkbox("Use visibility in temporal MIS", mUseVisibilityInTemporalMIS);
    widget.tooltip("If enabled, temporal MIS target estimates trace visibility at the reprojected pixel.");

    if (dirty)
    {
        mOptionsChanged = true;
        mInitialPass.pVars = nullptr;
        mTemporalPass.pVars = nullptr;
        mSpatialPass.pVars = nullptr;
        mShadePass.pVars = nullptr;
        mpReSTIRDIBlock = nullptr;
        mFrameCount = 0;
    }
}

/// Rebuild raytracing programs and samplers when a new scene is attached.
void ReSTIRDI::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mInitialPass = {};
    mTemporalPass = {};
    mSpatialPass = {};
    mShadePass = {};
    mpReSTIRDIBlock = nullptr;
    mFrameCount = 0;
    mpEnvMapSampler = nullptr;
    mpEmissiveSampler = nullptr;

    mpScene = pScene;

    if (!mpScene)
        return;

    if (mpScene->hasGeometryType(Scene::GeometryType::Custom))
        logWarning("ReSTIRDI: This render pass does not support custom primitives.");

    TypeConformanceList globalTypeConformances;
    mpScene->getTypeConformances(globalTypeConformances);

    // All four stages share the same hit/miss setup and differ only by ray-generation entry point.
    auto createTracePass = [&](const char* rayGenName)
    {
        TracePass pass;

        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        pass.pBindingTable = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
        pass.pBindingTable->setRayGen(desc.addRayGen(rayGenName, globalTypeConformances));
        pass.pBindingTable->setMiss(0, desc.addMiss("shadowMiss"));
        pass.pBindingTable->setMiss(1, desc.addMiss("brdfMiss"));

        auto materialTypes = mpScene->getMaterialSystem().getMaterialTypes();
        for (const auto materialType : materialTypes)
        {
            auto typeConformances = mpScene->getMaterialSystem().getTypeConformances(materialType);

            if (auto geometryIDs = mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh, materialType); !geometryIDs.empty())
            {
                auto shaderID =
                    desc.addHitGroup("shadowTriangleMeshClosestHit", "shadowTriangleMeshAnyHit", "", typeConformances, to_string(materialType));
                pass.pBindingTable->setHitGroup(0, geometryIDs, shaderID);
                auto brdfShaderID =
                    desc.addHitGroup("brdfTriangleMeshClosestHit", "brdfTriangleMeshAnyHit", "", typeConformances, to_string(materialType));
                pass.pBindingTable->setHitGroup(1, geometryIDs, brdfShaderID);
            }

            if (auto geometryIDs = mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh, materialType); !geometryIDs.empty())
            {
                auto shaderID =
                    desc.addHitGroup("shadowDisplacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection", typeConformances, to_string(materialType));
                pass.pBindingTable->setHitGroup(0, geometryIDs, shaderID);
                auto brdfShaderID =
                    desc.addHitGroup("brdfDisplacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection", typeConformances, to_string(materialType));
                pass.pBindingTable->setHitGroup(1, geometryIDs, brdfShaderID);
            }

            if (auto geometryIDs = mpScene->getGeometryIDs(Scene::GeometryType::Curve, materialType); !geometryIDs.empty())
            {
                auto shaderID = desc.addHitGroup("shadowCurveClosestHit", "", "curveIntersection", typeConformances, to_string(materialType));
                pass.pBindingTable->setHitGroup(0, geometryIDs, shaderID);
                auto brdfShaderID = desc.addHitGroup("brdfCurveClosestHit", "", "curveIntersection", typeConformances, to_string(materialType));
                pass.pBindingTable->setHitGroup(1, geometryIDs, brdfShaderID);
            }

            if (auto geometryIDs = mpScene->getGeometryIDs(Scene::GeometryType::SDFGrid, materialType); !geometryIDs.empty())
            {
                auto shaderID = desc.addHitGroup("shadowSdfGridClosestHit", "", "sdfGridIntersection", typeConformances, to_string(materialType));
                pass.pBindingTable->setHitGroup(0, geometryIDs, shaderID);
                auto brdfShaderID = desc.addHitGroup("brdfSdfGridClosestHit", "", "sdfGridIntersection", typeConformances, to_string(materialType));
                pass.pBindingTable->setHitGroup(1, geometryIDs, brdfShaderID);
            }
        }

        pass.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
        return pass;
    };

    mInitialPass = createTracePass("initialRayGen");
    mTemporalPass = createTracePass("temporalRayGen");
    mSpatialPass = createTracePass("spatialRayGen");
    mShadePass = createTracePass("shadeRayGen");
}

/// Keep environment and emissive-light samplers synchronized with scene lighting changes.
bool ReSTIRDI::prepareLighting(RenderContext* pRenderContext)
{
    bool lightingChanged = false;
    auto invalidateVars = [&]()
    {
        mInitialPass.pVars = nullptr;
        mTemporalPass.pVars = nullptr;
        mSpatialPass.pVars = nullptr;
        mShadePass.pVars = nullptr;
        mpReSTIRDIBlock = nullptr;
    };

    if (mUseEnvLight && mpScene->useEnvLight())
    {
        if (!mpEnvMapSampler)
        {
            mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, mpScene->getEnvMap());
            lightingChanged = true;
            invalidateVars();
        }
    }
    else if (mpEnvMapSampler)
    {
        mpEnvMapSampler = nullptr;
        lightingChanged = true;
        invalidateVars();
    }

    if (mUseEmissiveLights && mpScene->getRenderSettings().useEmissiveLights)
        mpScene->getILightCollection(pRenderContext);

    if (mUseEmissiveLights && mpScene->useEmissiveLights())
    {
        if (!mpEmissiveSampler)
        {
            const auto& pLights = mpScene->getILightCollection(pRenderContext);
            FALCOR_ASSERT(pLights && pLights->getActiveLightCount(pRenderContext) > 0);
            mpEmissiveSampler = std::make_unique<LightBVHSampler>(pRenderContext, pLights);
            lightingChanged = true;
            invalidateVars();
        }
    }
    else if (mpEmissiveSampler)
    {
        mpEmissiveSampler = nullptr;
        lightingChanged = true;
        invalidateVars();
    }

    if (mpEmissiveSampler)
    {
        lightingChanged |= mpEmissiveSampler->update(pRenderContext, mpScene->getILightCollection(pRenderContext));
        if (lightingChanged)
            invalidateVars();
    }

    return lightingChanged;
}

/// Create raytracing variables and bind samplers after shader defines or scene state changes.
void ReSTIRDI::prepareVars()
{
    FALCOR_ASSERT(mpScene);
    FALCOR_ASSERT(mInitialPass.pProgram);
    FALCOR_ASSERT(mTemporalPass.pProgram);
    FALCOR_ASSERT(mSpatialPass.pProgram);
    FALCOR_ASSERT(mShadePass.pProgram);

    mInitialPass.pProgram->addDefines(mpSampleGenerator->getDefines());
    mTemporalPass.pProgram->addDefines(mpSampleGenerator->getDefines());
    mSpatialPass.pProgram->addDefines(mpSampleGenerator->getDefines());
    mShadePass.pProgram->addDefines(mpSampleGenerator->getDefines());
    if (mpEmissiveSampler)
    {
        mInitialPass.pProgram->addDefines(mpEmissiveSampler->getDefines());
        mTemporalPass.pProgram->addDefines(mpEmissiveSampler->getDefines());
        mSpatialPass.pProgram->addDefines(mpEmissiveSampler->getDefines());
        mShadePass.pProgram->addDefines(mpEmissiveSampler->getDefines());
    }

    mInitialPass.pVars = RtProgramVars::create(mpDevice, mInitialPass.pProgram, mInitialPass.pBindingTable);
    mTemporalPass.pVars = RtProgramVars::create(mpDevice, mTemporalPass.pProgram, mTemporalPass.pBindingTable);
    mSpatialPass.pVars = RtProgramVars::create(mpDevice, mSpatialPass.pProgram, mSpatialPass.pBindingTable);
    mShadePass.pVars = RtProgramVars::create(mpDevice, mShadePass.pProgram, mShadePass.pBindingTable);
    mpReSTIRDIBlock = ParameterBlock::create(mpDevice, mInitialPass.pProgram->getReflector()->getParameterBlock("gReSTIRDI"));
    FALCOR_ASSERT(mpReSTIRDIBlock);

    mpSampleGenerator->bindShaderData(mInitialPass.pVars->getRootVar());
    mpSampleGenerator->bindShaderData(mTemporalPass.pVars->getRootVar());
    mpSampleGenerator->bindShaderData(mSpatialPass.pVars->getRootVar());
    mpSampleGenerator->bindShaderData(mShadePass.pVars->getRootVar());
}
