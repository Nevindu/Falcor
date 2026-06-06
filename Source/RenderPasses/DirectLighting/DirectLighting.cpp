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
#include "DirectLighting.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, DirectLighting>();
}

namespace
{
const char kShaderFile[] = "RenderPasses/DirectLighting/DirectLighting.rt.slang";

const uint32_t kMaxPayloadSizeBytes = 64u;
const uint32_t kMaxRecursionDepth = 2u;

const char kInputVBuffer[] = "vbuffer";
const char kInputViewDir[] = "viewW";
const char kOutputColor[] = "color";

const ChannelList kInputChannels = {
    { kInputVBuffer, "gVBuffer", "Visibility buffer in packed format" },
    { kInputViewDir, "gViewW", "World-space view direction (xyz float format)", true },
};

const ChannelList kOutputChannels = {
    { kOutputColor, "gOutputColor", "Output color (linear)", false, ResourceFormat::RGBA32Float },
};

const char kUseAnalyticLights[] = "useAnalyticLights";
const char kUseEmissiveLights[] = "useEmissiveLights";
const char kUseEnvLight[] = "useEnvLight";
const char kUseEmissiveMaterials[] = "useEmissiveMaterials";
const char kUseEnvBackground[] = "useEnvBackground";
const char kUseMIS[] = "useMIS";
const char kUseBRDFSampling[] = "useBRDFSampling";
const char kSamplesPerPixel[] = "samplesPerPixel";
} // namespace

DirectLighting::DirectLighting(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_5))
        FALCOR_THROW("DirectLighting requires Shader Model 6.5 support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
        FALCOR_THROW("DirectLighting requires Raytracing Tier 1.1 support.");

    parseProperties(props);

    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_TINY_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);
}

void DirectLighting::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kSamplesPerPixel)
            mSamplesPerPixel = value;
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
        else if (key == kUseMIS)
            mUseMIS = value;
        else if (key == kUseBRDFSampling)
            mUseBRDFSampling = value;
        else
            logWarning("Unknown property '{}' in DirectLighting properties.", key);
    }

    mSamplesPerPixel = std::max(1u, mSamplesPerPixel);
}

Properties DirectLighting::getProperties() const
{
    Properties props;
    props[kSamplesPerPixel] = mSamplesPerPixel;
    props[kUseAnalyticLights] = mUseAnalyticLights;
    props[kUseEmissiveLights] = mUseEmissiveLights;
    props[kUseEnvLight] = mUseEnvLight;
    props[kUseEmissiveMaterials] = mUseEmissiveMaterials;
    props[kUseEnvBackground] = mUseEnvBackground;
    props[kUseMIS] = mUseMIS;
    props[kUseBRDFSampling] = mUseBRDFSampling;
    return props;
}

RenderPassReflection DirectLighting::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);
    return reflector;
}

void DirectLighting::execute(RenderContext* pRenderContext, const RenderData& renderData)
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
        mTracer.pProgram = nullptr;
        mTracer.pVars = nullptr;
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

    bool definesChanged = false;
    definesChanged |= mTracer.pProgram->addDefine("SAMPLES_PER_PIXEL", std::to_string(mSamplesPerPixel));
    definesChanged |= mTracer.pProgram->addDefine("USE_ANALYTIC_LIGHTS", (mUseAnalyticLights && mpScene->useAnalyticLights()) ? "1" : "0");
    definesChanged |= mTracer.pProgram->addDefine("USE_EMISSIVE_LIGHTS", (mUseEmissiveLights && mpScene->useEmissiveLights()) ? "1" : "0");
    definesChanged |= mTracer.pProgram->addDefine("USE_ENV_LIGHT", (mUseEnvLight && mpScene->useEnvLight()) ? "1" : "0");
    definesChanged |= mTracer.pProgram->addDefine("USE_EMISSIVE_MATERIALS", mUseEmissiveMaterials ? "1" : "0");
    definesChanged |= mTracer.pProgram->addDefine("USE_ENV_BACKGROUND", (mUseEnvBackground && mpScene->useEnvBackground()) ? "1" : "0");
    definesChanged |= mTracer.pProgram->addDefine("USE_MIS", mUseMIS ? "1" : "0");
    definesChanged |= mTracer.pProgram->addDefine("USE_BRDF_SAMPLING", mUseBRDFSampling ? "1" : "0");
    definesChanged |= mTracer.pProgram->addDefines(getValidResourceDefines(kInputChannels, renderData));
    definesChanged |= mTracer.pProgram->addDefines(getValidResourceDefines(kOutputChannels, renderData));

    if (definesChanged)
    {
        mTracer.pVars = nullptr;
        mpDirectLightingBlock = nullptr;
    }

    if (!mTracer.pVars)
        prepareVars();

    auto var = mTracer.pVars->getRootVar();
    auto blockVar = mpDirectLightingBlock->getRootVar();
    if (mpEnvMapSampler && mUseEnvLight && mpScene->useEnvLight())
        mpEnvMapSampler->bindShaderData(blockVar["envMapSampler"]);
    if (mpEmissiveSampler && mUseEmissiveLights && mpScene->useEmissiveLights())
        mpEmissiveSampler->bindShaderData(blockVar["emissiveSampler"]);
    var["gDirectLighting"] = mpDirectLightingBlock;

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

    const uint3 dispatchDims(pOutput->getWidth(), pOutput->getHeight(), 1);
    if (dispatchDims.x == 0 || dispatchDims.y == 0)
        FALCOR_THROW("DirectLighting: Invalid ray dispatch dimensions {}x{}.", dispatchDims.x, dispatchDims.y);

    if (mFrameCount == 0)
    {
        logInfo(
            "DirectLighting: Dispatching rays {}x{}x{}, scene geometries {}, SBT geometries {}, ray types {}, miss records {}, hit records {}.",
            dispatchDims.x,
            dispatchDims.y,
            dispatchDims.z,
            mpScene->getGeometryCount(),
            mTracer.pVars->getGeometryCount(),
            mTracer.pVars->getRayTypeCount(),
            mTracer.pVars->getMissVarsCount(),
            mTracer.pVars->getTotalHitVarsCount()
        );
    }

    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, dispatchDims);

    mFrameCount++;
}

void DirectLighting::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;
    dirty |= widget.var("Samples/pixel", mSamplesPerPixel, 1u, 64u);
    dirty |= widget.checkbox("Use analytic lights", mUseAnalyticLights);
    dirty |= widget.checkbox("Use emissive lights", mUseEmissiveLights);
    dirty |= widget.checkbox("Use environment light", mUseEnvLight);
    dirty |= widget.checkbox("Use emissive materials", mUseEmissiveMaterials);
    dirty |= widget.checkbox("Use environment background", mUseEnvBackground);
    dirty |= widget.checkbox("Use MIS", mUseMIS);
    dirty |= widget.checkbox("Use BRDF sampling", mUseBRDFSampling);

    if (dirty)
    {
        mOptionsChanged = true;
        mTracer.pVars = nullptr;
        mpDirectLightingBlock = nullptr;
    }
}

void DirectLighting::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mTracer.pProgram = nullptr;
    mTracer.pBindingTable = nullptr;
    mTracer.pVars = nullptr;
    mpDirectLightingBlock = nullptr;
    mFrameCount = 0;
    mpEnvMapSampler = nullptr;
    mpEmissiveSampler = nullptr;

    mpScene = pScene;

    if (!mpScene)
        return;

    if (mpScene->hasGeometryType(Scene::GeometryType::Custom))
        logWarning("DirectLighting: This render pass does not support custom primitives.");

    ProgramDesc desc;
    desc.addShaderModules(mpScene->getShaderModules());
    desc.addShaderLibrary(kShaderFile);
    desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
    desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
    desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

    TypeConformanceList globalTypeConformances;
    mpScene->getTypeConformances(globalTypeConformances);

    mTracer.pBindingTable = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
    auto& sbt = mTracer.pBindingTable;
    sbt->setRayGen(desc.addRayGen("rayGen", globalTypeConformances));
    sbt->setMiss(0, desc.addMiss("shadowMiss"));
    sbt->setMiss(1, desc.addMiss("brdfMiss"));

    auto materialTypes = mpScene->getMaterialSystem().getMaterialTypes();
    for (const auto materialType : materialTypes)
    {
        auto typeConformances = mpScene->getMaterialSystem().getTypeConformances(materialType);

        if (auto geometryIDs = mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh, materialType); !geometryIDs.empty())
        {
            auto shaderID =
                desc.addHitGroup("shadowTriangleMeshClosestHit", "shadowTriangleMeshAnyHit", "", typeConformances, to_string(materialType));
            sbt->setHitGroup(0, geometryIDs, shaderID);
            auto brdfShaderID =
                desc.addHitGroup("brdfTriangleMeshClosestHit", "brdfTriangleMeshAnyHit", "", typeConformances, to_string(materialType));
            sbt->setHitGroup(1, geometryIDs, brdfShaderID);
        }

        if (auto geometryIDs = mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh, materialType); !geometryIDs.empty())
        {
            auto shaderID =
                desc.addHitGroup("shadowDisplacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection", typeConformances, to_string(materialType));
            sbt->setHitGroup(0, geometryIDs, shaderID);
            auto brdfShaderID =
                desc.addHitGroup("brdfDisplacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection", typeConformances, to_string(materialType));
            sbt->setHitGroup(1, geometryIDs, brdfShaderID);
        }

        if (auto geometryIDs = mpScene->getGeometryIDs(Scene::GeometryType::Curve, materialType); !geometryIDs.empty())
        {
            auto shaderID = desc.addHitGroup("shadowCurveClosestHit", "", "curveIntersection", typeConformances, to_string(materialType));
            sbt->setHitGroup(0, geometryIDs, shaderID);
            auto brdfShaderID = desc.addHitGroup("brdfCurveClosestHit", "", "curveIntersection", typeConformances, to_string(materialType));
            sbt->setHitGroup(1, geometryIDs, brdfShaderID);
        }

        if (auto geometryIDs = mpScene->getGeometryIDs(Scene::GeometryType::SDFGrid, materialType); !geometryIDs.empty())
        {
            auto shaderID = desc.addHitGroup("shadowSdfGridClosestHit", "", "sdfGridIntersection", typeConformances, to_string(materialType));
            sbt->setHitGroup(0, geometryIDs, shaderID);
            auto brdfShaderID = desc.addHitGroup("brdfSdfGridClosestHit", "", "sdfGridIntersection", typeConformances, to_string(materialType));
            sbt->setHitGroup(1, geometryIDs, brdfShaderID);
        }
    }

    mTracer.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
}

bool DirectLighting::prepareLighting(RenderContext* pRenderContext)
{
    bool lightingChanged = false;

    if (mUseEnvLight && mpScene->useEnvLight())
    {
        if (!mpEnvMapSampler)
        {
            mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, mpScene->getEnvMap());
            lightingChanged = true;
            mTracer.pVars = nullptr;
            mpDirectLightingBlock = nullptr;
        }
    }
    else if (mpEnvMapSampler)
    {
        mpEnvMapSampler = nullptr;
        lightingChanged = true;
        mTracer.pVars = nullptr;
        mpDirectLightingBlock = nullptr;
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
            mTracer.pVars = nullptr;
            mpDirectLightingBlock = nullptr;
        }
    }
    else if (mpEmissiveSampler)
    {
        mpEmissiveSampler = nullptr;
        lightingChanged = true;
        mTracer.pVars = nullptr;
        mpDirectLightingBlock = nullptr;
    }

    if (mpEmissiveSampler)
    {
        lightingChanged |= mpEmissiveSampler->update(pRenderContext, mpScene->getILightCollection(pRenderContext));
        if (lightingChanged)
        {
            mTracer.pVars = nullptr;
            mpDirectLightingBlock = nullptr;
        }
    }

    return lightingChanged;
}

void DirectLighting::prepareVars()
{
    FALCOR_ASSERT(mpScene);
    FALCOR_ASSERT(mTracer.pProgram);

    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());
    if (mpEmissiveSampler)
        mTracer.pProgram->addDefines(mpEmissiveSampler->getDefines());
    mTracer.pProgram->setTypeConformances(mpScene->getTypeConformances());

    mTracer.pVars = RtProgramVars::create(mpDevice, mTracer.pProgram, mTracer.pBindingTable);
    mpDirectLightingBlock = ParameterBlock::create(mpDevice, mTracer.pProgram->getReflector()->getParameterBlock("gDirectLighting"));
    FALCOR_ASSERT(mpDirectLightingBlock);

    auto var = mTracer.pVars->getRootVar();
    mpSampleGenerator->bindShaderData(var);
}
