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
#include "ReSTIRPTLF.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Rendering/Lights/EmissiveUniformSampler.h"
#include "Scene/HitInfo.h"


namespace
{
    const std::string kGeneratePathsFilename = "RenderPasses/ReSTIRPTLF/GeneratePaths.cs.slang";
    const std::string kTracePassFilename = "RenderPasses/ReSTIRPTLF/TracePass.rt.slang";
    const std::string kResolvePassFilename = "RenderPasses/ReSTIRPTLF/ResolvePass.cs.slang";
    const std::string kTemporalReuseFilename = "RenderPasses/ReSTIRPTLF/TemporalReuse.cs.slang";
    const std::string kSpatialReuseFilename = "RenderPasses/ReSTIRPTLF/SpatialReuse.cs.slang";
    const std::string kGenerateLightFieldVBufferFilename = "RenderPasses/ReSTIRPTLF/GenerateLightFieldVBuffer.cs.slang";
    const std::string kReflectTypesFile = "RenderPasses/ReSTIRPTLF/ReflectTypes.cs.slang";

    // Render pass inputs and outputs.
    const std::string kInputVBuffer = "vbuffer";
    const std::string kInputMotionVectors = "mvec";
    const std::string kInputViewDir = "viewW";

    const Falcor::ChannelList kInputChannels =
    {
        { kInputVBuffer,        "gVBuffer",         "Visibility buffer in packed format", true /* optional */ },
        { kInputMotionVectors,  "gMotionVectors",   "Motion vector buffer (float format)", true /* optional */ },
        { kInputViewDir,        "gViewW",           "World-space view direction (xyz float format)", true /* optional */ },
    };

    const std::string kOutputColor = "color";
    const std::string kOutputAlbedo = "albedo";
    const std::string kOutputSpecularAlbedo = "specularAlbedo";
    const std::string kOutputIndirectAlbedo = "indirectAlbedo";
    const std::string kOutputGuideNormal = "guideNormal";
    const std::string kOutputReflectionPosW = "reflectionPosW";
    const std::string kOutputRayCount = "rayCount";
    const std::string kOutputPathLength = "pathLength";

    const Falcor::ChannelList kOutputChannels =
    {
        { kOutputColor,                                     "",     "Output color (linear)", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputAlbedo,                                    "",     "Output albedo (linear)", true /* optional */, ResourceFormat::RGBA8Unorm },
        { kOutputSpecularAlbedo,                            "",     "Output specular albedo (linear)", true /* optional */, ResourceFormat::RGBA8Unorm },
        { kOutputIndirectAlbedo,                            "",     "Output indirect albedo (linear)", true /* optional */, ResourceFormat::RGBA8Unorm },
        { kOutputGuideNormal,                               "",     "Output guide normal (linear)", true /* optional */, ResourceFormat::RGBA16Float },
        { kOutputReflectionPosW,                            "",     "Output reflection pos (world space)", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputRayCount,                                  "",     "Per-pixel ray count", true /* optional */, ResourceFormat::R32Uint },
        { kOutputPathLength,                                "",     "Per-pixel path length", true /* optional */, ResourceFormat::R32Uint },
    };

    // Scripting options.
    const std::string kSamplesPerPixel = "samplesPerPixel";
    const std::string kInitialCandidateCount = "initialCandidateCount";
    const std::string kMaxSurfaceBounces = "maxSurfaceBounces";
    const std::string kMaxDiffuseBounces = "maxDiffuseBounces";
    const std::string kMaxSpecularBounces = "maxSpecularBounces";
    const std::string kMaxTransmissionBounces = "maxTransmissionBounces";

    const std::string kSampleGenerator = "sampleGenerator";
    const std::string kFixedSeed = "fixedSeed";
    const std::string kUseBSDFSampling = "useBSDFSampling";
    const std::string kUseRussianRoulette = "useRussianRoulette";
    const std::string kUseNEE = "useNEE";
    const std::string kUseMIS = "useMIS";
    const std::string kMISHeuristic = "misHeuristic";
    const std::string kMISPowerExponent = "misPowerExponent";
    const std::string kEmissiveSampler = "emissiveSampler";
    const std::string kLightBVHOptions = "lightBVHOptions";

    const std::string kUseAlphaTest = "useAlphaTest";
    const std::string kAdjustShadingNormals = "adjustShadingNormals";
    const std::string kMaxNestedMaterials = "maxNestedMaterials";
    const std::string kUseLightsInDielectricVolumes = "useLightsInDielectricVolumes";
    const std::string kDisableCaustics = "disableCaustics";
    const std::string kSpecularRoughnessThreshold = "specularRoughnessThreshold";
    const std::string kMinReconnectDistance = "minReconnectDistance";
    const std::string kMaxReconnectJacobian = "maxReconnectJacobian";
    const std::string kPrimaryLodMode = "primaryLodMode";
    const std::string kLODBias = "lodBias";

    const std::string kOutputSize = "outputSize";
    const std::string kFixedOutputSize = "fixedOutputSize";
    const std::string kColorFormat = "colorFormat";
    const std::string kDebugView = "debugView";
    const std::string kUseTemporalReuse = "useTemporalReuse";
    const std::string kTemporalHistoryLength = "temporalHistoryLength";
    const std::string kTemporalReuseForceSamePixel = "temporalReuseForceSamePixel";
    const std::string kUseSpatialReuse = "useSpatialReuse";
    const std::string kSpatialNeighborCount = "spatialNeighborCount";
    const std::string kSpatialRadius = "spatialRadius";
    const std::string kSpatialIterations = "spatialIterations";
    const std::string kSpatialMISStrategy = "spatialMISStrategy";
    const std::string kFeatureBasedRejection = "featureBasedRejection";
    const std::string kShiftMapping = "shiftMapping";
    const std::string kEnableAngularReuse = "enableAngularReuse";
    const std::string kLightFieldEnabled = "lightFieldEnabled";
    const std::string kLightFieldViewCount = "lightFieldViewCount";
    const std::string kLightFieldViewsPerRow = "lightFieldViewsPerRow";
    const std::string kLightFieldViewDim = "lightFieldViewDim";
    const std::string kLightFieldViewWidth = "lightFieldViewWidth";
    const std::string kLightFieldViewHeight = "lightFieldViewHeight";
    const std::string kLightFieldViewSpacing = "lightFieldViewSpacing";
    const std::string kLightFieldFocalDistance = "lightFieldFocalDistance";
    const std::string kLightFieldAngularNeighborRadius = "lightFieldAngularNeighborRadius";
    const std::string kLightFieldMaxReprojectionErrorPx = "lightFieldMaxReprojectionErrorPx";
    const std::string kLightFieldNormalThreshold = "lightFieldNormalThreshold";
    const std::string kLightFieldDepthRelThreshold = "lightFieldDepthRelThreshold";

    const Gui::DropdownList kDebugViewList =
    {
        { 0u, "Beauty" },
        { 1u, "Selected target" },
        { 2u, "Reservoir weight" },
        { 3u, "Path length" },
        { 4u, "Valid mask" },
        { 5u, "Selected candidate" },
        { 6u, "Shifted target" },
        { 7u, "Target ratio" },
        { 8u, "Shift valid mask" },
        { 9u, "Shift rejection reason" },
        { 10u, "Reuse accepted count" },
        { 11u, "Reuse rejected count" },
        { 12u, "Reuse combined M" },
        { 13u, "Reuse source distance" },
        { 14u, "Reuse rejection reason" },
        { 15u, "Reuse shift mask" },
        { 16u, "Replay mismatch mask" },
        { 17u, "Reuse MIS weight sum" },
        { 18u, "Reuse accepted fraction" },
    };

    const Gui::DropdownList kShiftMappingList =
    {
        { uint32_t(ShiftMapping::Reconnection), "Reconnection" },
        { uint32_t(ShiftMapping::RandomReplay), "Random replay" },
    };
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReSTIRPTLF>();
    ScriptBindings::registerBinding(ReSTIRPTLF::registerBindings);
}

void ReSTIRPTLF::registerBindings(pybind11::module& m)
{
    pybind11::class_<ReSTIRPTLF, RenderPass, ref<ReSTIRPTLF>> pass(m, "ReSTIRPTLF");
    pass.def("reset", &ReSTIRPTLF::reset);
    pass.def_property_readonly("pixelStats", &ReSTIRPTLF::getPixelStats);

    pass.def_property("useFixedSeed",
        [](const ReSTIRPTLF* pt) { return pt->mParams.useFixedSeed ? true : false; },
        [](ReSTIRPTLF* pt, bool value) { pt->mParams.useFixedSeed = value ? 1 : 0; }
    );
    pass.def_property("fixedSeed",
        [](const ReSTIRPTLF* pt) { return pt->mParams.fixedSeed; },
        [](ReSTIRPTLF* pt, uint32_t value) { pt->mParams.fixedSeed = value; }
    );
}

ReSTIRPTLF::ReSTIRPTLF(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_5))
        FALCOR_THROW("ReSTIRPTLF requires Shader Model 6.5 support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
        FALCOR_THROW("ReSTIRPTLF requires Raytracing Tier 1.1 support.");

    parseProperties(props);
    validateOptions();

    // Create sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, mStaticParams.sampleGenerator);

    // Create resolve pass. This doesn't depend on the scene so can be created here.
    auto defines = mStaticParams.getDefines(*this);
    mpResolvePass = ComputePass::create(mpDevice, ProgramDesc().addShaderLibrary(kResolvePassFilename).csEntry("main"), defines, false);

    // Note: The other programs are lazily created in updatePrograms() because a scene needs to be present when creating them.

    mpPixelStats = std::make_unique<PixelStats>(mpDevice);
    mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);
}

void ReSTIRPTLF::setProperties(const Properties& props)
{
    parseProperties(props);
    validateOptions();
    if (auto lightBVHSampler = dynamic_cast<LightBVHSampler*>(mpEmissiveSampler.get()))
        lightBVHSampler->setOptions(mLightBVHOptions);
    mRecompile = true;
    mOptionsChanged = true;
}

void ReSTIRPTLF::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        // Rendering parameters
        if (key == kSamplesPerPixel || key == kInitialCandidateCount) mStaticParams.samplesPerPixel = value;
        else if (key == kMaxSurfaceBounces) mStaticParams.maxSurfaceBounces = value;
        else if (key == kMaxDiffuseBounces) mStaticParams.maxDiffuseBounces = value;
        else if (key == kMaxSpecularBounces) mStaticParams.maxSpecularBounces = value;
        else if (key == kMaxTransmissionBounces) mStaticParams.maxTransmissionBounces = value;

        // Sampling parameters
        else if (key == kSampleGenerator) mStaticParams.sampleGenerator = value;
        else if (key == kFixedSeed) { mParams.fixedSeed = value; mParams.useFixedSeed = true; }
        else if (key == kUseBSDFSampling) mStaticParams.useBSDFSampling = value;
        else if (key == kUseRussianRoulette) mStaticParams.useRussianRoulette = value;
        else if (key == kUseNEE) mStaticParams.useNEE = value;
        else if (key == kUseMIS) mStaticParams.useMIS = value;
        else if (key == kMISHeuristic) mStaticParams.misHeuristic = value;
        else if (key == kMISPowerExponent) mStaticParams.misPowerExponent = value;
        else if (key == kEmissiveSampler) mStaticParams.emissiveSampler = value;
        else if (key == kLightBVHOptions) mLightBVHOptions = value;

        // Material parameters
        else if (key == kUseAlphaTest) mStaticParams.useAlphaTest = value;
        else if (key == kAdjustShadingNormals) mStaticParams.adjustShadingNormals = value;
        else if (key == kMaxNestedMaterials) mStaticParams.maxNestedMaterials = value;
        else if (key == kUseLightsInDielectricVolumes) mStaticParams.useLightsInDielectricVolumes = value;
        else if (key == kDisableCaustics) mStaticParams.disableCaustics = value;
        else if (key == kSpecularRoughnessThreshold) mParams.specularRoughnessThreshold = value;
        else if (key == kMinReconnectDistance) mParams.minReconnectDistance = value;
        else if (key == kMaxReconnectJacobian) mParams.maxReconnectJacobian = value;
        else if (key == kPrimaryLodMode) mStaticParams.primaryLodMode = value;
        else if (key == kLODBias) mParams.lodBias = value;

        // Output parameters
        else if (key == kOutputSize) mOutputSizeSelection = value;
        else if (key == kFixedOutputSize) mFixedOutputSize = value;
        else if (key == kColorFormat) mStaticParams.colorFormat = value;
        else if (key == kDebugView) mDebugView = value;
        else if (key == kUseTemporalReuse) mUseTemporalReuse = value;
        else if (key == kTemporalHistoryLength) mTemporalHistoryLength = value;
        else if (key == kTemporalReuseForceSamePixel)
        {
            const bool forceSamePixel = value;
            if (mTemporalReuseForceSamePixel != forceSamePixel)
            {
                mTemporalReuseForceSamePixel = forceSamePixel;
                reset();
            }
        }
        else if (key == kUseSpatialReuse) mUseSpatialReuse = value;
        else if (key == kSpatialNeighborCount) mSpatialNeighborCount = value;
        else if (key == kSpatialRadius) mSpatialRadius = value;
        else if (key == kSpatialIterations) mSpatialIterations = value;
        else if (key == kSpatialMISStrategy) mSpatialMISStrategy = value;
        else if (key == kFeatureBasedRejection) mFeatureBasedRejection = value;
        else if (key == kShiftMapping) mParams.shiftMapping = value;
        else if (key == kEnableAngularReuse) mEnableAngularReuse = value;
        else if (key == kLightFieldEnabled)
        {
            const bool enabled = value;
            mParams.lfEnabled = enabled ? 1u : 0u;
        }
        else if (key == kLightFieldViewCount) mParams.lfViewCount = value;
        else if (key == kLightFieldViewsPerRow) mParams.lfViewsPerRow = value;
        else if (key == kLightFieldViewDim) mParams.lfViewDim = value;
        else if (key == kLightFieldViewWidth) mParams.lfViewDim.x = value;
        else if (key == kLightFieldViewHeight) mParams.lfViewDim.y = value;
        else if (key == kLightFieldViewSpacing) mParams.lfViewSpacing = value;
        else if (key == kLightFieldFocalDistance) mParams.lfFocalDistance = value;
        else if (key == kLightFieldAngularNeighborRadius) mParams.lfAngularNeighborRadius = value;
        else if (key == kLightFieldMaxReprojectionErrorPx) mParams.lfMaxReprojectionErrorPx = value;
        else if (key == kLightFieldNormalThreshold) mParams.lfNormalThreshold = value;
        else if (key == kLightFieldDepthRelThreshold) mParams.lfDepthRelThreshold = value;

        else logWarning("Unknown property '{}' in ReSTIRPTLF properties.", key);
    }

    if (props.has(kMaxSurfaceBounces))
    {
        // Initialize bounce counts to 'maxSurfaceBounces' if they weren't explicitly set.
        if (!props.has(kMaxDiffuseBounces)) mStaticParams.maxDiffuseBounces = mStaticParams.maxSurfaceBounces;
        if (!props.has(kMaxSpecularBounces)) mStaticParams.maxSpecularBounces = mStaticParams.maxSurfaceBounces;
        if (!props.has(kMaxTransmissionBounces)) mStaticParams.maxTransmissionBounces = mStaticParams.maxSurfaceBounces;
    }
    else
    {
        // Initialize surface bounces.
        mStaticParams.maxSurfaceBounces = std::max(mStaticParams.maxDiffuseBounces, std::max(mStaticParams.maxSpecularBounces, mStaticParams.maxTransmissionBounces));
    }

    bool maxSurfaceBouncesNeedsAdjustment =
        mStaticParams.maxSurfaceBounces < mStaticParams.maxDiffuseBounces ||
        mStaticParams.maxSurfaceBounces < mStaticParams.maxSpecularBounces ||
        mStaticParams.maxSurfaceBounces < mStaticParams.maxTransmissionBounces;

    // Show a warning if maxSurfaceBounces will be adjusted in validateOptions().
    if (props.has(kMaxSurfaceBounces) && maxSurfaceBouncesNeedsAdjustment)
    {
        logWarning("'{}' is set lower than '{}', '{}' or '{}' and will be increased.", kMaxSurfaceBounces, kMaxDiffuseBounces, kMaxSpecularBounces, kMaxTransmissionBounces);
    }
}

void ReSTIRPTLF::validateOptions()
{
    if (mParams.specularRoughnessThreshold < 0.f || mParams.specularRoughnessThreshold > 1.f)
    {
        logWarning("'specularRoughnessThreshold' has invalid value. Clamping to range [0,1].");
        mParams.specularRoughnessThreshold = std::clamp(mParams.specularRoughnessThreshold, 0.f, 1.f);
    }
    if (mParams.minReconnectDistance < 0.f)
    {
        logWarning("'minReconnectDistance' has invalid value. Clamping to 0.");
        mParams.minReconnectDistance = 0.f;
    }
    if (mParams.maxReconnectJacobian <= 0.f)
    {
        logWarning("'maxReconnectJacobian' has invalid value. Resetting to 1e4.");
        mParams.maxReconnectJacobian = 1e4f;
    }

    // Static parameters.
    if (mStaticParams.samplesPerPixel < 1 || mStaticParams.samplesPerPixel > kMaxSamplesPerPixel)
    {
        logWarning("'samplesPerPixel' must be in the range [1, {}]. Clamping to this range.", kMaxSamplesPerPixel);
        mStaticParams.samplesPerPixel = std::clamp(mStaticParams.samplesPerPixel, 1u, kMaxSamplesPerPixel);
    }

    auto clampBounces = [] (uint32_t& bounces, const std::string& name)
    {
        if (bounces > kMaxBounces)
        {
            logWarning("'{}' exceeds the maximum supported bounces. Clamping to {}.", name, kMaxBounces);
            bounces = kMaxBounces;
        }
    };

    clampBounces(mStaticParams.maxSurfaceBounces, kMaxSurfaceBounces);
    clampBounces(mStaticParams.maxDiffuseBounces, kMaxDiffuseBounces);
    clampBounces(mStaticParams.maxSpecularBounces, kMaxSpecularBounces);
    clampBounces(mStaticParams.maxTransmissionBounces, kMaxTransmissionBounces);

    // Make sure maxSurfaceBounces is at least as many as any of diffuse, specular or transmission.
    uint32_t minSurfaceBounces = std::max(mStaticParams.maxDiffuseBounces, std::max(mStaticParams.maxSpecularBounces, mStaticParams.maxTransmissionBounces));
    mStaticParams.maxSurfaceBounces = std::max(mStaticParams.maxSurfaceBounces, minSurfaceBounces);

    if (mStaticParams.primaryLodMode == TexLODMode::RayCones)
    {
        logWarning("Unsupported tex lod mode. Defaulting to Mip0.");
        mStaticParams.primaryLodMode = TexLODMode::Mip0;
    }

    mSpatialNeighborCount = std::clamp(mSpatialNeighborCount, 0u, 32u);
    mSpatialRadius = std::clamp(mSpatialRadius, 1u, 128u);
    mSpatialIterations = std::clamp(mSpatialIterations, 1u, 1u);
    mTemporalHistoryLength = std::clamp(mTemporalHistoryLength, 0u, 1024u);
    mParams.shiftMapping = std::min<uint32_t>(mParams.shiftMapping, uint32_t(ShiftMapping::RandomReplay));
    mParams.lfEnabled = mParams.lfEnabled != 0u ? 1u : 0u;
    mParams.lfViewCount = std::clamp(mParams.lfViewCount, 1u, 64u);
    mParams.lfViewsPerRow = std::clamp(mParams.lfViewsPerRow, 1u, mParams.lfViewCount);
    mParams.lfAngularNeighborRadius = std::clamp(mParams.lfAngularNeighborRadius, 0u, 16u);
    mParams.lfViewSpacing = std::max(0.f, mParams.lfViewSpacing);
    mParams.lfFocalDistance = std::max(1e-4f, mParams.lfFocalDistance);
    mParams.lfMaxReprojectionErrorPx = std::max(0.f, mParams.lfMaxReprojectionErrorPx);
    mParams.lfNormalThreshold = std::clamp(mParams.lfNormalThreshold, 0.f, 1.f);
    mParams.lfDepthRelThreshold = std::max(0.f, mParams.lfDepthRelThreshold);
}

Properties ReSTIRPTLF::getProperties() const
{
    if (auto lightBVHSampler = dynamic_cast<LightBVHSampler*>(mpEmissiveSampler.get()))
    {
        mLightBVHOptions = lightBVHSampler->getOptions();
    }

    Properties props;

    // Rendering parameters
    props[kInitialCandidateCount] = mStaticParams.samplesPerPixel;
    props[kMaxSurfaceBounces] = mStaticParams.maxSurfaceBounces;
    props[kMaxDiffuseBounces] = mStaticParams.maxDiffuseBounces;
    props[kMaxSpecularBounces] = mStaticParams.maxSpecularBounces;
    props[kMaxTransmissionBounces] = mStaticParams.maxTransmissionBounces;

    // Sampling parameters
    props[kSampleGenerator] = mStaticParams.sampleGenerator;
    if (mParams.useFixedSeed) props[kFixedSeed] = mParams.fixedSeed;
    props[kUseBSDFSampling] = mStaticParams.useBSDFSampling;
    props[kUseRussianRoulette] = mStaticParams.useRussianRoulette;
    props[kUseNEE] = mStaticParams.useNEE;
    props[kUseMIS] = mStaticParams.useMIS;
    props[kMISHeuristic] = mStaticParams.misHeuristic;
    props[kMISPowerExponent] = mStaticParams.misPowerExponent;
    props[kEmissiveSampler] = mStaticParams.emissiveSampler;
    if (mStaticParams.emissiveSampler == EmissiveLightSamplerType::LightBVH) props[kLightBVHOptions] = mLightBVHOptions;

    // Material parameters
    props[kUseAlphaTest] = mStaticParams.useAlphaTest;
    props[kAdjustShadingNormals] = mStaticParams.adjustShadingNormals;
    props[kMaxNestedMaterials] = mStaticParams.maxNestedMaterials;
    props[kUseLightsInDielectricVolumes] = mStaticParams.useLightsInDielectricVolumes;
    props[kDisableCaustics] = mStaticParams.disableCaustics;
    props[kSpecularRoughnessThreshold] = mParams.specularRoughnessThreshold;
    props[kMinReconnectDistance] = mParams.minReconnectDistance;
    props[kMaxReconnectJacobian] = mParams.maxReconnectJacobian;
    props[kPrimaryLodMode] = mStaticParams.primaryLodMode;
    props[kLODBias] = mParams.lodBias;

    // Output parameters
    props[kOutputSize] = mOutputSizeSelection;
    if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed) props[kFixedOutputSize] = mFixedOutputSize;
    props[kColorFormat] = mStaticParams.colorFormat;
    props[kDebugView] = mDebugView;
    props[kUseTemporalReuse] = mUseTemporalReuse;
    props[kTemporalHistoryLength] = mTemporalHistoryLength;
    props[kTemporalReuseForceSamePixel] = mTemporalReuseForceSamePixel;
    props[kUseSpatialReuse] = mUseSpatialReuse;
    props[kSpatialNeighborCount] = mSpatialNeighborCount;
    props[kSpatialRadius] = mSpatialRadius;
    props[kSpatialIterations] = mSpatialIterations;
    props[kSpatialMISStrategy] = mSpatialMISStrategy;
    props[kFeatureBasedRejection] = mFeatureBasedRejection;
    props[kShiftMapping] = mParams.shiftMapping;
    props[kEnableAngularReuse] = mEnableAngularReuse;
    props[kLightFieldEnabled] = mParams.lfEnabled != 0u;
    props[kLightFieldViewCount] = mParams.lfViewCount;
    props[kLightFieldViewsPerRow] = mParams.lfViewsPerRow;
    props[kLightFieldViewDim] = mParams.lfViewDim;
    props[kLightFieldViewWidth] = mParams.lfViewDim.x;
    props[kLightFieldViewHeight] = mParams.lfViewDim.y;
    props[kLightFieldViewSpacing] = mParams.lfViewSpacing;
    props[kLightFieldFocalDistance] = mParams.lfFocalDistance;
    props[kLightFieldAngularNeighborRadius] = mParams.lfAngularNeighborRadius;
    props[kLightFieldMaxReprojectionErrorPx] = mParams.lfMaxReprojectionErrorPx;
    props[kLightFieldNormalThreshold] = mParams.lfNormalThreshold;
    props[kLightFieldDepthRelThreshold] = mParams.lfDepthRelThreshold;

    return props;
}

RenderPassReflection ReSTIRPTLF::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);

    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels, ResourceBindFlags::UnorderedAccess, sz);
    return reflector;
}

void ReSTIRPTLF::setFrameDim(const uint2 frameDim)
{
    auto prevFrameDim = mParams.frameDim;
    auto prevScreenTiles = mParams.screenTiles;

    mParams.frameDim = frameDim;
    if (mParams.frameDim.x > kMaxFrameDimension || mParams.frameDim.y > kMaxFrameDimension)
    {
        FALCOR_THROW("Frame dimensions up to {} pixels width/height are supported.", kMaxFrameDimension);
    }

    if (mParams.lfEnabled != 0u && (mParams.lfViewDim.x == 0u || mParams.lfViewDim.y == 0u))
    {
        const uint32_t viewsPerRow = std::max(1u, mParams.lfViewsPerRow);
        const uint32_t rows = (std::max(1u, mParams.lfViewCount) + viewsPerRow - 1u) / viewsPerRow;
        mParams.lfViewDim = uint2(
            std::max(1u, mParams.frameDim.x / viewsPerRow),
            std::max(1u, mParams.frameDim.y / std::max(1u, rows))
        );
    }

    // Tile dimensions have to be powers-of-two.
    FALCOR_ASSERT(isPowerOf2(kScreenTileDim.x) && isPowerOf2(kScreenTileDim.y));
    FALCOR_ASSERT(kScreenTileDim.x == (1 << kScreenTileBits.x) && kScreenTileDim.y == (1 << kScreenTileBits.y));
    mParams.screenTiles = div_round_up(mParams.frameDim, kScreenTileDim);

    if (any(mParams.frameDim != prevFrameDim) || any(mParams.screenTiles != prevScreenTiles))
    {
        mVarsChanged = true;
    }
}

void ReSTIRPTLF::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mUpdateFlagsConnection = {};
    mUpdateFlags = IScene::UpdateFlags::None;

    mpScene = pScene;
    mParams.frameCount = 0;
    mParams.frameDim = {};
    mParams.screenTiles = {};

    resetPrograms();
    resetLighting();

    if (mpScene)
    {
        mUpdateFlagsConnection = mpScene->getUpdateFlagsSignal().connect([&](IScene::UpdateFlags flags) { mUpdateFlags |= flags; });

        if (pScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("ReSTIRPTLF: This render pass does not support custom primitives.");
        }

        validateOptions();
    }
}


void ReSTIRPTLF::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!beginFrame(pRenderContext, renderData)) return;

    // Update shader program specialization.
    updatePrograms();

    // Prepare resources.
    prepareResources(pRenderContext, renderData);

    if (mParams.lfEnabled != 0u)
    {
        generateLightFieldVBuffer(pRenderContext);
    }

    // Prepare the path tracer parameter block.
    // This should be called after all resources have been created.
    prepareReSTIRPTLF(renderData);

    // Generate paths at primary hits.
    generatePaths(pRenderContext, renderData);

    // Trace pass.
    FALCOR_ASSERT(mpTracePass);
    tracePass(pRenderContext, renderData, *mpTracePass);

    const bool canRunTemporalReuse =
        mUseTemporalReuse &&
        mTemporalHistoryLength > 0 &&
        mParams.frameCount > 0 &&
        (mTemporalReuseForceSamePixel || renderData.getTexture(kInputMotionVectors) != nullptr);
    mTemporalReuseActive = canRunTemporalReuse;
    if (mTemporalReuseActive)
    {
        temporalReusePass(pRenderContext, renderData);
    }

    ref<Buffer> pReuseInputReservoirs = mTemporalReuseActive ? mpTemporalReservoirs : mpCurrentReservoirs;

    // The spatial skeleton writes the final color directly from the combined
    // reservoir. When disabled, keep the original PathTracerN-style resolve path.
    if (mUseSpatialReuse)
    {
        spatialReusePass(pRenderContext, renderData, pReuseInputReservoirs);
        pReuseInputReservoirs = mpSpatialReservoirs;
    }

    mAngularReuseActive =
        mParams.lfEnabled != 0u &&
        mEnableAngularReuse &&
        mParams.lfAngularNeighborRadius > 0u;
    if (mAngularReuseActive)
    {
        angularReusePass(pRenderContext, renderData, pReuseInputReservoirs);
    }
    else if (!mTemporalReuseActive && !mUseSpatialReuse)
    {
        // Resolve pass.
        resolvePass(pRenderContext, renderData);
    }

    endFrame(pRenderContext, renderData);
}

void ReSTIRPTLF::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    // Rendering options.
    dirty |= renderRenderingUI(widget);

    // Stats and debug options.
    renderStatsUI(widget);
    dirty |= renderDebugUI(widget);

    if (dirty)
    {
        validateOptions();
        mOptionsChanged = true;
    }
}

bool ReSTIRPTLF::renderRenderingUI(Gui::Widgets& widget)
{
    bool dirty = false;
    bool runtimeDirty = false;

    dirty |= widget.var("Initial candidate path trees", mStaticParams.samplesPerPixel, 1u, kMaxSamplesPerPixel);
    widget.tooltip("Number of independent path trees traced per pixel for initial ReSTIR PT candidate generation. Each path tree is reduced to one RIS-selected contribution.");

    if (widget.var("Max surface bounces", mStaticParams.maxSurfaceBounces, 0u, kMaxBounces))
    {
        // Allow users to change the max surface bounce parameter in the UI to clamp all other surface bounce parameters.
        mStaticParams.maxDiffuseBounces = std::min(mStaticParams.maxDiffuseBounces, mStaticParams.maxSurfaceBounces);
        mStaticParams.maxSpecularBounces = std::min(mStaticParams.maxSpecularBounces, mStaticParams.maxSurfaceBounces);
        mStaticParams.maxTransmissionBounces = std::min(mStaticParams.maxTransmissionBounces, mStaticParams.maxSurfaceBounces);
        dirty = true;
    }
    widget.tooltip("Maximum number of surface bounces (diffuse + specular + transmission).\n"
        "Note that specular reflection events from a material with a roughness greater than specularRoughnessThreshold are also classified as diffuse events.");

    dirty |= widget.var("Max diffuse bounces", mStaticParams.maxDiffuseBounces, 0u, kMaxBounces);
    widget.tooltip("Maximum number of diffuse bounces.\n0 = direct only\n1 = one indirect bounce etc.");

    dirty |= widget.var("Max specular bounces", mStaticParams.maxSpecularBounces, 0u, kMaxBounces);
    widget.tooltip("Maximum number of specular bounces.\n0 = direct only\n1 = one indirect bounce etc.");

    dirty |= widget.var("Max transmission bounces", mStaticParams.maxTransmissionBounces, 0u, kMaxBounces);
    widget.tooltip("Maximum number of transmission bounces.\n0 = no transmission\n1 = one transmission bounce etc.");

    // Sampling options.

    if (widget.dropdown("Sample generator", SampleGenerator::getGuiDropdownList(), mStaticParams.sampleGenerator))
    {
        mpSampleGenerator = SampleGenerator::create(mpDevice, mStaticParams.sampleGenerator);
        dirty = true;
    }

    dirty |= widget.checkbox("BSDF importance sampling", mStaticParams.useBSDFSampling);
    widget.tooltip("BSDF importance sampling should normally be enabled.\n\n"
        "If disabled, cosine-weighted hemisphere sampling is used for debugging purposes");

    dirty |= widget.checkbox("Russian roulette", mStaticParams.useRussianRoulette);
    widget.tooltip("Use russian roulette to terminate low throughput paths.");

    dirty |= widget.checkbox("Next-event estimation (NEE)", mStaticParams.useNEE);
    widget.tooltip("Use next-event estimation.\nThis option enables direct illumination sampling at each path vertex.");

    if (mStaticParams.useNEE)
    {
        dirty |= widget.checkbox("Multiple importance sampling (MIS)", mStaticParams.useMIS);
        widget.tooltip("When enabled, BSDF sampling is combined with light sampling for the environment map and emissive lights.\n"
            "Note that MIS has currently no effect on analytic lights.");

        if (mStaticParams.useMIS)
        {
            dirty |= widget.dropdown("MIS heuristic", mStaticParams.misHeuristic);

            if (mStaticParams.misHeuristic == MISHeuristic::PowerExp)
            {
                dirty |= widget.var("MIS power exponent", mStaticParams.misPowerExponent, 0.01f, 10.f);
            }
        }

        if (mpScene && mpScene->useEmissiveLights())
        {
            if (auto group = widget.group("Emissive sampler"))
            {
                if (widget.dropdown("Emissive sampler", mStaticParams.emissiveSampler))
                {
                    resetLighting();
                    dirty = true;
                }
                widget.tooltip("Selects which light sampler to use for importance sampling of emissive geometry.", true);

                if (mpEmissiveSampler)
                {
                    if (mpEmissiveSampler->renderUI(group)) mOptionsChanged = true;
                }
            }
        }
    }

    if (auto group = widget.group("Light field"))
    {
        bool lfEnabled = mParams.lfEnabled != 0u;
        if (group.checkbox("Enable light field", lfEnabled))
        {
            mParams.lfEnabled = lfEnabled ? 1u : 0u;
            runtimeDirty = true;
            mVarsChanged = true;
        }

        auto updateFixedLightFieldSize = [&]()
        {
            if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed && mParams.lfViewDim.x > 0u && mParams.lfViewDim.y > 0u)
            {
                const uint32_t rows = (std::max(1u, mParams.lfViewCount) + std::max(1u, mParams.lfViewsPerRow) - 1u) / std::max(1u, mParams.lfViewsPerRow);
                mFixedOutputSize = uint2(mParams.lfViewDim.x * std::max(1u, mParams.lfViewsPerRow), mParams.lfViewDim.y * rows);
                requestRecompile();
            }
        };

        if (mParams.lfEnabled != 0u)
        {
            if (group.var("View count", mParams.lfViewCount, 1u, 64u))
            {
                runtimeDirty = true;
                updateFixedLightFieldSize();
            }
            if (group.var("Views per row", mParams.lfViewsPerRow, 1u, 64u))
            {
                runtimeDirty = true;
                updateFixedLightFieldSize();
            }
            if (group.var("View size", mParams.lfViewDim, 1u, 4096u))
            {
                runtimeDirty = true;
                updateFixedLightFieldSize();
            }
            runtimeDirty |= group.var("View spacing", mParams.lfViewSpacing, 0.f, 1.f, 0.001f);
            runtimeDirty |= group.var("Focal distance", mParams.lfFocalDistance, 0.001f, 1000.f, 0.01f);
            runtimeDirty |= group.checkbox("Enable angular reuse", mEnableAngularReuse);
            if (mEnableAngularReuse)
            {
                runtimeDirty |= group.var("Angular radius", mParams.lfAngularNeighborRadius, 0u, 16u);
                runtimeDirty |= group.var("Max reprojection error", mParams.lfMaxReprojectionErrorPx, 0.f, 16.f, 0.1f);
                runtimeDirty |= group.var("Normal threshold", mParams.lfNormalThreshold, 0.f, 1.f, 0.01f);
                runtimeDirty |= group.var("Depth threshold", mParams.lfDepthRelThreshold, 0.f, 1.f, 0.001f);
            }
        }
    }

    if (auto group = widget.group("Spatial reuse"))
    {
        runtimeDirty |= group.checkbox("Enable spatial reuse", mUseSpatialReuse);
        group.tooltip("Combines nearby ReSTIR PT reservoirs using a canonical self sample, direct terminal reconnection, and conservative source-tail replay for deeper selected NEE/terminal events. Primary NEE neighbor import is disabled.");

        if (mUseSpatialReuse)
        {
            // Runtime-only knobs for the spatial skeleton. They do not change shader
            // declarations, so they are bound through the spatial compute pass constants.
            runtimeDirty |= group.var("Neighbor count", mSpatialNeighborCount, 0u, 32u);
            runtimeDirty |= group.var("Radius", mSpatialRadius, 1u, 128u);
            runtimeDirty |= group.var("Iterations", mSpatialIterations, 1u, 1u);
            runtimeDirty |= group.dropdown("Spatial MIS", mSpatialMISStrategy);
            runtimeDirty |= group.checkbox("Feature-based rejection", mFeatureBasedRejection);
            group.tooltip("Reject spatial neighbors whose primary surface differs too much in normal or camera distance. Reduces fireflies/variance from low-support shifts; the estimator stays unbiased.");
            runtimeDirty |= group.dropdown("Shift mapping", kShiftMappingList, mParams.shiftMapping);
        }
    }

    if (auto group = widget.group("Temporal reuse"))
    {
        runtimeDirty |= group.checkbox("Enable temporal reuse", mUseTemporalReuse);
        group.tooltip("Reprojects the previous frame reservoir with motion vectors and shifts it into the current pixel.");

        if (mUseTemporalReuse)
        {
            runtimeDirty |= group.var("History length", mTemporalHistoryLength, 0u, 1024u);
            group.tooltip("Caps previous-frame reservoir confidence to this multiple of the current reservoir confidence. 0 disables previous-frame import.");
            if (group.checkbox("Force same temporal pixel", mTemporalReuseForceSamePixel))
            {
                runtimeDirty = true;
                reset();
            }
            group.tooltip("Debug mode: reuse the previous-frame reservoir from the same pixel and treat it as an identity-domain sample. This ignores motion vectors and bypasses the temporal shift map.");
            if (!mUseSpatialReuse && !mTemporalReuseForceSamePixel)
            {
                runtimeDirty |= group.dropdown("Shift mapping", kShiftMappingList, mParams.shiftMapping);
            }
        }
    }

    if (auto group = widget.group("Material controls"))
    {
        dirty |= widget.checkbox("Alpha test", mStaticParams.useAlphaTest);
        widget.tooltip("Use alpha testing on non-opaque triangles.");

        dirty |= widget.checkbox("Adjust shading normals on secondary hits", mStaticParams.adjustShadingNormals);
        widget.tooltip("Enables adjustment of the shading normals to reduce the risk of black pixels due to back-facing vectors.\nDoes not apply to primary hits which is configured in GBuffer.", true);

        dirty |= widget.var("Max nested materials", mStaticParams.maxNestedMaterials, 2u, 4u);
        widget.tooltip("Maximum supported number of nested materials.");

        dirty |= widget.checkbox("Use lights in dielectric volumes", mStaticParams.useLightsInDielectricVolumes);
        widget.tooltip("Use lights inside of volumes (transmissive materials). We typically don't want this because lights are occluded by the interface.");

        dirty |= widget.checkbox("Disable caustics", mStaticParams.disableCaustics);
        widget.tooltip("Disable sampling of caustic light paths (i.e. specular events after diffuse events).");

        runtimeDirty |= widget.var("Specular roughness threshold", mParams.specularRoughnessThreshold, 0.f, 1.f);
        widget.tooltip("Specular reflection events are only classified as specular if the material's roughness value is equal or smaller than this threshold. Otherwise they are classified diffuse.");

        runtimeDirty |= widget.var("Min reconnect distance", mParams.minReconnectDistance, 0.f, 1.f, 0.001f);
        widget.tooltip("Reject path-space reconnections shorter than this scene-space distance.");

        runtimeDirty |= widget.var("Max reconnect Jacobian", mParams.maxReconnectJacobian, 1.f, 1e6f);
        widget.tooltip("Reject path-space reconnections with larger Jacobians instead of clamping them.");

        dirty |= widget.dropdown("Primary LOD Mode", mStaticParams.primaryLodMode);
        widget.tooltip("Texture LOD mode at primary hit");

        runtimeDirty |= widget.var("TexLOD bias", mParams.lodBias, -16.f, 16.f, 0.01f);
    }

    if (auto group = widget.group("Output options"))
    {
        // Switch to enable/disable path tracer output.
        dirty |= widget.checkbox("Enable output", mEnabled);

        // Controls for output size.
        // When output size requirements change, we'll trigger a graph recompile to update the render pass I/O sizes.
        if (widget.dropdown("Output size", mOutputSizeSelection)) requestRecompile();
        if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed)
        {
            if (widget.var("Size in pixels", mFixedOutputSize, 32u, 16384u)) requestRecompile();
        }

        dirty |= widget.dropdown("Color format", mStaticParams.colorFormat);
        widget.tooltip("Selects the color format used for internal per-sample color buffers");
    }

    if (dirty) mRecompile = true;
    return dirty || runtimeDirty;
}

bool ReSTIRPTLF::renderDebugUI(Gui::Widgets& widget)
{
    bool dirty = false;

    if (auto group = widget.group("Debugging"))
    {
        dirty |= group.checkbox("Use fixed seed", mParams.useFixedSeed);
        group.tooltip("Forces a fixed random seed for each frame.\n\n"
            "This should produce exactly the same image each frame, which can be useful for debugging.");
        if (mParams.useFixedSeed)
        {
            dirty |= group.var("Seed", mParams.fixedSeed);
        }

        if (group.dropdown("Reservoir view", kDebugViewList, mDebugView))
        {
            requestRecompile();
            mRecompile = true;
            dirty = true;
        }

        mpPixelDebug->renderUI(group);
    }

    return dirty;
}

void ReSTIRPTLF::renderStatsUI(Gui::Widgets& widget)
{
    if (auto g = widget.group("Statistics"))
    {
        // Show ray stats
        mpPixelStats->renderUI(g);
    }
}

bool ReSTIRPTLF::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpPixelDebug->onMouseEvent(mouseEvent);
}

void ReSTIRPTLF::reset()
{
    mParams.frameCount = 0;
    mTemporalReuseActive = false;
    mAngularReuseActive = false;
}

ReSTIRPTLF::TracePass::TracePass(ref<Device> pDevice, const std::string& name, const std::string& passDefine, const ref<Scene>& pScene, const DefineList& defines, const TypeConformanceList& globalTypeConformances)
    : name(name)
    , passDefine(passDefine)
{
    const uint32_t kRayTypeScatter = 0;
    const uint32_t kMissScatter = 0;

    ProgramDesc desc;
    desc.addShaderModules(pScene->getShaderModules());
    desc.addShaderLibrary(kTracePassFilename);
    // The ReSTIR PT payload extends PathTracerN with reusable NEE light data for
    // spatial reconnection. Keep this comfortably above the reflected payload size
    // so DispatchRays does not fail validation when the shader table is launched.
    desc.setMaxPayloadSize(320);
    desc.setMaxAttributeSize(pScene->getRaytracingMaxAttributeSize());
    desc.setMaxTraceRecursionDepth(1);
    if (!pScene->hasProceduralGeometry()) desc.setRtPipelineFlags(RtPipelineFlags::SkipProceduralPrimitives);

    // Create ray tracing binding table.
    pBindingTable = RtBindingTable::create(1, 1, pScene->getGeometryCount());

    // Specify entry point for raygen and miss shaders.
    // The raygen shader needs type conformances for *all* materials in the scene.
    // The miss shader doesn't need need any type conformances because it does not use materials.
    pBindingTable->setRayGen(desc.addRayGen("rayGen", globalTypeConformances));
    pBindingTable->setMiss(kMissScatter, desc.addMiss("scatterMiss"));

    // Specify hit group entry points for every combination of geometry and material type.
    // The code for each hit group gets specialized for the actual types it's operating on.
    // First query which material types the scene has.
    auto materialTypes = pScene->getMaterialSystem().getMaterialTypes();

    for (const auto materialType : materialTypes)
    {
        auto typeConformances = pScene->getMaterialSystem().getTypeConformances(materialType);

        // Add hit groups for triangles.
        if (auto geometryIDs = pScene->getGeometryIDs(Scene::GeometryType::TriangleMesh, materialType); !geometryIDs.empty())
        {
            auto shaderID = desc.addHitGroup("scatterTriangleClosestHit", "scatterTriangleAnyHit", "", typeConformances, to_string(materialType));
            pBindingTable->setHitGroup(kRayTypeScatter, geometryIDs, shaderID);
        }

        // Add hit groups for displaced triangle meshes.
        if (auto geometryIDs = pScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh, materialType); !geometryIDs.empty())
        {
            auto shaderID = desc.addHitGroup("scatterDisplacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection", typeConformances, to_string(materialType));
            pBindingTable->setHitGroup(kRayTypeScatter, geometryIDs, shaderID);
        }

        // Add hit groups for curves.
        if (auto geometryIDs = pScene->getGeometryIDs(Scene::GeometryType::Curve, materialType); !geometryIDs.empty())
        {
            auto shaderID = desc.addHitGroup("scatterCurveClosestHit", "", "curveIntersection", typeConformances, to_string(materialType));
            pBindingTable->setHitGroup(kRayTypeScatter, geometryIDs, shaderID);
        }

        // Add hit groups for SDF grids.
        if (auto geometryIDs = pScene->getGeometryIDs(Scene::GeometryType::SDFGrid, materialType); !geometryIDs.empty())
        {
            auto shaderID = desc.addHitGroup("scatterSdfGridClosestHit", "", "sdfGridIntersection", typeConformances, to_string(materialType));
            pBindingTable->setHitGroup(kRayTypeScatter, geometryIDs, shaderID);
        }
    }

    pProgram = Program::create(pDevice, desc, defines);
}


void ReSTIRPTLF::TracePass::prepareProgram(ref<Device> pDevice, const DefineList& defines)
{
    FALCOR_ASSERT(pProgram != nullptr && pBindingTable != nullptr);
    pProgram->setDefines(defines);
    if (!passDefine.empty()) pProgram->addDefine(passDefine);
    pVars = RtProgramVars::create(pDevice, pProgram, pBindingTable);
}

void ReSTIRPTLF::resetPrograms()
{
    mpTracePass = nullptr;
    mpGenerateLightFieldVBuffer = nullptr;
    mpGeneratePaths = nullptr;
    mpTemporalReusePass = nullptr;
    mpSpatialReusePass = nullptr;
    mpReflectTypes = nullptr;

    mRecompile = true;
}

void ReSTIRPTLF::updatePrograms()
{
    FALCOR_ASSERT(mpScene);

    if (mRecompile == false) return;

    // If we get here, a change that require recompilation of shader programs has occurred.
    // This may be due to change of scene defines, type conformances, shader modules, or other changes that require recompilation.
    // When type conformances and/or shader modules change, the programs need to be recreated. We assume programs have been reset upon such changes.
    // When only defines have changed, it is sufficient to update the existing programs and recreate the program vars.

    auto defines = mStaticParams.getDefines(*this);
    TypeConformanceList globalTypeConformances;
    mpScene->getTypeConformances(globalTypeConformances);

    // Create trace pass.
    if (!mpTracePass)
        mpTracePass = TracePass::create(mpDevice, "tracePass", "", mpScene, defines, globalTypeConformances);

    mpTracePass->prepareProgram(mpDevice, defines);

    // Create compute passes.
    ProgramDesc baseDesc;
    mpScene->getShaderModules(baseDesc.shaderModules);
    baseDesc.addTypeConformances(globalTypeConformances);

    if (!mpGeneratePaths)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kGeneratePathsFilename).csEntry("main");
        mpGeneratePaths = ComputePass::create(mpDevice, desc, defines, false);
    }
    if (!mpGenerateLightFieldVBuffer)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kGenerateLightFieldVBufferFilename).csEntry("main");
        mpGenerateLightFieldVBuffer = ComputePass::create(mpDevice, desc, defines, false);
    }
    if (!mpSpatialReusePass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kSpatialReuseFilename).csEntry("main");
        mpSpatialReusePass = ComputePass::create(mpDevice, desc, defines, false);
    }
    if (!mpTemporalReusePass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kTemporalReuseFilename).csEntry("main");
        mpTemporalReusePass = ComputePass::create(mpDevice, desc, defines, false);
    }
    if (!mpReflectTypes)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kReflectTypesFile).csEntry("main");
        mpReflectTypes = ComputePass::create(mpDevice, desc, defines, false);
    }

    auto preparePass = [&](ref<ComputePass> pass)
    {
        // Note that we must use set instead of add defines to replace any stale state.
        pass->getProgram()->setDefines(defines);

        // Recreate program vars. This may trigger recompilation if needed.
        // Note that program versions are cached, so switching to a previously used specialization is faster.
        pass->setVars(nullptr);
    };
    preparePass(mpGenerateLightFieldVBuffer);
    preparePass(mpGeneratePaths);
    preparePass(mpTemporalReusePass);
    preparePass(mpSpatialReusePass);
    preparePass(mpResolvePass);
    preparePass(mpReflectTypes);

    mVarsChanged = true;
    mRecompile = false;
}

void ReSTIRPTLF::prepareResources(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Compute allocation requirements for paths and output samples.
    // Note that the sample buffers are padded to whole tiles, while the max path count depends on actual frame dimension.
    uint32_t spp = mStaticParams.samplesPerPixel;
    uint32_t tileCount = mParams.screenTiles.x * mParams.screenTiles.y;
    const uint32_t sampleBufferCount = tileCount * kScreenTileDim.x * kScreenTileDim.y * spp;

    auto var = mpReflectTypes->getRootVar();

    // Allocate per-sample buffers.
    // For the special case of fixed 1 spp, the output is written out directly and this buffer is not needed.
    if (mStaticParams.samplesPerPixel > 1)
    {
        if (!mpSampleColor || mpSampleColor->getElementCount() < sampleBufferCount || mVarsChanged)
        {
            mpSampleColor = mpDevice->createStructuredBuffer(var["sampleColor"], sampleBufferCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            mVarsChanged = true;
        }
    }

    if (mOutputGuideData && (!mpSampleGuideData || mpSampleGuideData->getElementCount() < sampleBufferCount || mVarsChanged))
    {
        mpSampleGuideData = mpDevice->createStructuredBuffer(var["sampleGuideData"], sampleBufferCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
        mVarsChanged = true;
    }

    const uint32_t reservoirCount = mParams.frameDim.x * mParams.frameDim.y;
    ref<Texture> pInputVBuffer = renderData.getTexture(kInputVBuffer);

    if (mParams.lfEnabled != 0u)
    {
        ResourceFormat vbufferFormat = pInputVBuffer ? pInputVBuffer->getFormat() : ResourceFormat::Unknown;
        if (vbufferFormat == ResourceFormat::Unknown)
        {
            if (auto pScene = dynamic_ref_cast<Scene>(mpScene))
            {
                vbufferFormat = pScene->getHitInfo().getFormat();
            }
        }

        FALCOR_ASSERT(vbufferFormat != ResourceFormat::Unknown);
        if (!mpLightFieldVBuffer ||
            mpLightFieldVBuffer->getWidth() != mParams.frameDim.x ||
            mpLightFieldVBuffer->getHeight() != mParams.frameDim.y ||
            mpLightFieldVBuffer->getFormat() != vbufferFormat ||
            mVarsChanged)
        {
            mpLightFieldVBuffer = mpDevice->createTexture2D(
                mParams.frameDim.x,
                mParams.frameDim.y,
                vbufferFormat,
                1,
                1,
                nullptr,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
            );
            mVarsChanged = true;
        }
    }
    else
    {
        mpLightFieldVBuffer = nullptr;
    }

    ref<Texture> pVBuffer = getActiveVBuffer(renderData);

    // Reservoir buffers are per-pixel, not per-candidate. The trace pass writes
    // the initial RIS reservoir, the optional spatial pass writes a combined
    // reservoir, and previousReservoirs preserves the final reservoir for future
    // temporal reuse work.
    if (!mpCurrentReservoirs || mpCurrentReservoirs->getElementCount() < reservoirCount || mVarsChanged)
    {
        mpCurrentReservoirs = mpDevice->createStructuredBuffer(var["currentReservoirs"], reservoirCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
        pRenderContext->clearUAV(mpCurrentReservoirs->getUAV().get(), uint4(0));
        mVarsChanged = true;
    }

    if (!mpTemporalReservoirs || mpTemporalReservoirs->getElementCount() < reservoirCount || mVarsChanged)
    {
        mpTemporalReservoirs = mpDevice->createStructuredBuffer(var["temporalReservoirs"], reservoirCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
        pRenderContext->clearUAV(mpTemporalReservoirs->getUAV().get(), uint4(0));
        mVarsChanged = true;
    }

    if (!mpSpatialReservoirs || mpSpatialReservoirs->getElementCount() < reservoirCount || mVarsChanged)
    {
        mpSpatialReservoirs = mpDevice->createStructuredBuffer(var["spatialReservoirs"], reservoirCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
        pRenderContext->clearUAV(mpSpatialReservoirs->getUAV().get(), uint4(0));
        mVarsChanged = true;
    }

    if (!mpAngularReservoirs || mpAngularReservoirs->getElementCount() < reservoirCount || mVarsChanged)
    {
        mpAngularReservoirs = mpDevice->createStructuredBuffer(var["spatialReservoirs"], reservoirCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
        pRenderContext->clearUAV(mpAngularReservoirs->getUAV().get(), uint4(0));
        mVarsChanged = true;
    }

    if (!mpPreviousReservoirs || mpPreviousReservoirs->getElementCount() < reservoirCount || mVarsChanged)
    {
        mpPreviousReservoirs = mpDevice->createStructuredBuffer(var["previousReservoirs"], reservoirCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
        pRenderContext->clearUAV(mpPreviousReservoirs->getUAV().get(), uint4(0));
        mVarsChanged = true;
    }

    if (pVBuffer && (!mpPreviousVBuffer ||
        mpPreviousVBuffer->getWidth() != pVBuffer->getWidth() ||
        mpPreviousVBuffer->getHeight() != pVBuffer->getHeight() ||
        mpPreviousVBuffer->getFormat() != pVBuffer->getFormat() ||
        mVarsChanged))
    {
        mpPreviousVBuffer = mpDevice->createTexture2D(
            pVBuffer->getWidth(),
            pVBuffer->getHeight(),
            pVBuffer->getFormat(),
            1,
            1,
            nullptr,
            ResourceBindFlags::ShaderResource
        );
        mVarsChanged = true;
    }

}

void ReSTIRPTLF::prepareReSTIRPTLF(const RenderData& renderData)
{
    // Create path tracer parameter block if needed.
    if (!mpReSTIRPTLFBlock || mVarsChanged)
    {
        auto reflector = mpReflectTypes->getProgram()->getReflector()->getParameterBlock("ReSTIRPTLF");
        mpReSTIRPTLFBlock = ParameterBlock::create(mpDevice, reflector);
        FALCOR_ASSERT(mpReSTIRPTLFBlock);
        mVarsChanged = true;
    }

    // Bind resources.
    auto var = mpReSTIRPTLFBlock->getRootVar();
    bindShaderData(var, renderData);
}

void ReSTIRPTLF::resetLighting()
{
    // Retain the options for the emissive sampler.
    if (auto lightBVHSampler = dynamic_cast<LightBVHSampler*>(mpEmissiveSampler.get()))
    {
        mLightBVHOptions = lightBVHSampler->getOptions();
    }

    mpEmissiveSampler = nullptr;
    mpEnvMapSampler = nullptr;
    mRecompile = true;
}

void ReSTIRPTLF::prepareMaterials(RenderContext* pRenderContext)
{
    // This functions checks for scene changes that require shader recompilation.
    // Whenever materials or geometry is added/removed to the scene, we reset the shader programs to trigger
    // recompilation with the correct defines, type conformances, shader modules, and binding table.

    if (is_set(mUpdateFlags, IScene::UpdateFlags::RecompileNeeded) ||
        is_set(mUpdateFlags, IScene::UpdateFlags::GeometryChanged))
    {
        resetPrograms();
    }
}

bool ReSTIRPTLF::prepareLighting(RenderContext* pRenderContext)
{
    bool lightingChanged = false;

    if (is_set(mUpdateFlags, IScene::UpdateFlags::RenderSettingsChanged))
    {
        lightingChanged = true;
        mRecompile = true;
    }

    if (is_set(mUpdateFlags, IScene::UpdateFlags::SDFGridConfigChanged))
    {
        mRecompile = true;
    }

    if (is_set(mUpdateFlags, IScene::UpdateFlags::EnvMapChanged))
    {
        mpEnvMapSampler = nullptr;
        lightingChanged = true;
        mRecompile = true;
    }

    if (mpScene->useEnvLight())
    {
        if (!mpEnvMapSampler)
        {
            mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, mpScene->getEnvMap());
            lightingChanged = true;
            mRecompile = true;
        }
    }
    else
    {
        if (mpEnvMapSampler)
        {
            mpEnvMapSampler = nullptr;
            lightingChanged = true;
            mRecompile = true;
        }
    }

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getILightCollection(pRenderContext);
    }

    if (mpScene->useEmissiveLights())
    {
        if (!mpEmissiveSampler)
        {
            const auto& pLights = mpScene->getILightCollection(pRenderContext);
            FALCOR_ASSERT(pLights && pLights->getActiveLightCount(pRenderContext) > 0);
            FALCOR_ASSERT(!mpEmissiveSampler);

            switch (mStaticParams.emissiveSampler)
            {
            case EmissiveLightSamplerType::Uniform:
                mpEmissiveSampler = std::make_unique<EmissiveUniformSampler>(pRenderContext, mpScene->getILightCollection(pRenderContext));
                break;
            case EmissiveLightSamplerType::LightBVH:
                mpEmissiveSampler = std::make_unique<LightBVHSampler>(pRenderContext, mpScene->getILightCollection(pRenderContext), mLightBVHOptions);
                break;
            case EmissiveLightSamplerType::Power:
                mpEmissiveSampler = std::make_unique<EmissivePowerSampler>(pRenderContext, mpScene->getILightCollection(pRenderContext));
                break;
            default:
                FALCOR_THROW("Unknown emissive light sampler type");
            }
            lightingChanged = true;
            mRecompile = true;
        }
    }
    else
    {
        if (mpEmissiveSampler)
        {
            // Retain the options for the emissive sampler.
            if (auto lightBVHSampler = dynamic_cast<LightBVHSampler*>(mpEmissiveSampler.get()))
            {
                mLightBVHOptions = lightBVHSampler->getOptions();
            }

            mpEmissiveSampler = nullptr;
            lightingChanged = true;
            mRecompile = true;
        }
    }

    if (mpEmissiveSampler)
    {
        lightingChanged |= mpEmissiveSampler->update(pRenderContext, mpScene->getILightCollection(pRenderContext));
        auto defines = mpEmissiveSampler->getDefines();
        if (mpTracePass && mpTracePass->pProgram->addDefines(defines)) mRecompile = true;
    }

    return lightingChanged;
}

void ReSTIRPTLF::bindShaderData(const ShaderVar& var, const RenderData& renderData, bool useLightSampling) const
{
    // Bind static resources that don't change per frame.
    if (mVarsChanged)
    {
        if (useLightSampling && mpEnvMapSampler) mpEnvMapSampler->bindShaderData(var["envMapSampler"]);

        var["sampleColor"] = mpSampleColor;
        var["sampleGuideData"] = mpSampleGuideData;
        var["currentReservoirs"] = mpCurrentReservoirs;
        var["previousReservoirs"] = mpPreviousReservoirs;
    }

    ref<Texture> pViewDir;
    if (mParams.lfEnabled == 0u && mpScene && mpScene->getCamera()->getApertureRadius() > 0.f)
    {
        pViewDir = renderData.getTexture(kInputViewDir);
        if (!pViewDir) logWarning("Depth-of-field requires the '{}' input. Expect incorrect rendering.", kInputViewDir);
    }

    var["params"].setBlob(mParams);
    var["vbuffer"] = getActiveVBuffer(renderData);
    var["viewDir"] = pViewDir; // Can be nullptr
    var["outputColor"] = renderData.getTexture(kOutputColor);

    if (useLightSampling && mpEmissiveSampler)
    {
        // TODO: Do we have to bind this every frame?
        mpEmissiveSampler->bindShaderData(var["emissiveSampler"]);
    }
}

ref<Texture> ReSTIRPTLF::getActiveVBuffer(const RenderData& renderData) const
{
    return (mParams.lfEnabled != 0u && mpLightFieldVBuffer) ? mpLightFieldVBuffer : renderData.getTexture(kInputVBuffer);
}

bool ReSTIRPTLF::beginFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pOutputColor = renderData.getTexture(kOutputColor);
    FALCOR_ASSERT(pOutputColor);

    // Set output frame dimension.
    setFrameDim(uint2(pOutputColor->getWidth(), pOutputColor->getHeight()));

    // Validate all I/O sizes match the expected size.
    // If not, we'll disable the path tracer to give the user a chance to fix the configuration before re-enabling it.
    bool resolutionMismatch = false;
    auto validateChannels = [&](const auto& channels) {
        for (const auto& channel : channels)
        {
            auto pTexture = renderData.getTexture(channel.name);
            if (pTexture && (pTexture->getWidth() != mParams.frameDim.x || pTexture->getHeight() != mParams.frameDim.y)) resolutionMismatch = true;
        }
    };
    validateChannels(kInputChannels);
    validateChannels(kOutputChannels);

    if (mEnabled && resolutionMismatch)
    {
        logError("ReSTIRPTLF I/O sizes don't match. The pass will be disabled.");
        mEnabled = false;
    }

    if (mEnabled && mParams.lfEnabled == 0u && !renderData.getTexture(kInputVBuffer))
    {
        logError("ReSTIRPTLF requires '{}' when light-field mode is disabled. The pass will be disabled.", kInputVBuffer);
        mEnabled = false;
    }

    if (mpScene == nullptr || !mEnabled)
    {
        pRenderContext->clearUAV(pOutputColor->getUAV().get(), float4(0.f));

        // Set refresh flag if changes that affect the output have occured.
        // This is needed to ensure other passes get notified when the path tracer is enabled/disabled.
        if (mOptionsChanged)
        {
            auto& dict = renderData.getDictionary();
            auto flags = dict.getValue(kRenderPassRefreshFlags, Falcor::RenderPassRefreshFlags::None);
            if (mOptionsChanged) flags |= Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
            dict[Falcor::kRenderPassRefreshFlags] = flags;
        }

        return false;
    }

    // Update materials.
    prepareMaterials(pRenderContext);

    // Update the env map and emissive sampler to the current frame.
    bool lightingChanged = prepareLighting(pRenderContext);

    // Update refresh flag if changes that affect the output have occured.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged || lightingChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, Falcor::RenderPassRefreshFlags::None);
        if (mOptionsChanged) flags |= Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        if (lightingChanged) flags |= Falcor::RenderPassRefreshFlags::LightingChanged;
        dict[Falcor::kRenderPassRefreshFlags] = flags;
        mOptionsChanged = false;
    }

    // Check if GBuffer has adjusted shading normals enabled.
    bool gbufferAdjustShadingNormals = dict.getValue(Falcor::kRenderPassGBufferAdjustShadingNormals, false);
    if (gbufferAdjustShadingNormals != mGBufferAdjustShadingNormals)
    {
        mGBufferAdjustShadingNormals = gbufferAdjustShadingNormals;
        mRecompile = true;
    }

    // Check if guide data should be generated.
    mOutputGuideData = renderData[kOutputAlbedo] != nullptr || renderData[kOutputSpecularAlbedo] != nullptr
        || renderData[kOutputIndirectAlbedo] != nullptr || renderData[kOutputGuideNormal] != nullptr
        || renderData[kOutputReflectionPosW] != nullptr;

    // Enable pixel stats if rayCount or pathLength outputs are connected.
    if (renderData[kOutputRayCount] != nullptr || renderData[kOutputPathLength] != nullptr)
    {
        mpPixelStats->setEnabled(true);
    }

    mpPixelStats->beginFrame(pRenderContext, mParams.frameDim);
    mpPixelDebug->beginFrame(pRenderContext, mParams.frameDim);

    // Update the random seed.
    mParams.seed = mParams.useFixedSeed ? mParams.fixedSeed : mParams.frameCount;

    mUpdateFlags = IScene::UpdateFlags::None;

    return true;
}

void ReSTIRPTLF::endFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    mpPixelStats->endFrame(pRenderContext);
    mpPixelDebug->endFrame(pRenderContext);

    auto copyTexture = [pRenderContext](Texture* pDst, const Texture* pSrc)
    {
        if (pDst && pSrc)
        {
            FALCOR_ASSERT(pDst && pSrc);
            FALCOR_ASSERT(pDst->getFormat() == pSrc->getFormat());
            FALCOR_ASSERT(pDst->getWidth() == pSrc->getWidth() && pDst->getHeight() == pSrc->getHeight());
            pRenderContext->copyResource(pDst, pSrc);
        }
        else if (pDst)
        {
            pRenderContext->clearUAV(pDst->getUAV().get(), uint4(0, 0, 0, 0));
        }
    };

    // Copy pixel stats to outputs if available.
    copyTexture(renderData.getTexture(kOutputRayCount).get(), mpPixelStats->getRayCountTexture(pRenderContext).get());
    copyTexture(renderData.getTexture(kOutputPathLength).get(), mpPixelStats->getPathLengthTexture().get());

    if (mpCurrentReservoirs && mpPreviousReservoirs)
    {
        // Store the reservoir that actually produced this frame. Spatial reuse
        // becomes final when enabled; otherwise temporal reuse is final if it ran.
        Buffer* pFinalReservoirs = mpCurrentReservoirs.get();
        if (mAngularReuseActive && mpAngularReservoirs)
        {
            pFinalReservoirs = mpAngularReservoirs.get();
        }
        else if (mUseSpatialReuse && mpSpatialReservoirs)
        {
            pFinalReservoirs = mpSpatialReservoirs.get();
        }
        else if (mTemporalReuseActive && mpTemporalReservoirs)
        {
            pFinalReservoirs = mpTemporalReservoirs.get();
        }
        pRenderContext->copyResource(mpPreviousReservoirs.get(), pFinalReservoirs);
    }
    ref<Texture> pVBuffer = getActiveVBuffer(renderData);
    if (mpPreviousVBuffer && pVBuffer)
    {
        pRenderContext->copyResource(mpPreviousVBuffer.get(), pVBuffer.get());
    }

    mVarsChanged = false;
    mTemporalReuseActive = false;
    mAngularReuseActive = false;
    mParams.frameCount++;
}

void ReSTIRPTLF::generateLightFieldVBuffer(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "generateLightFieldVBuffer");

    FALCOR_ASSERT(mpGenerateLightFieldVBuffer);
    FALCOR_ASSERT(mpLightFieldVBuffer);

    auto var = mpGenerateLightFieldVBuffer->getRootVar();
    var["CB"]["params"].setBlob(mParams);
    var["outputVBuffer"] = mpLightFieldVBuffer;

    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);
    mpGenerateLightFieldVBuffer->execute(pRenderContext, { mParams.frameDim, 1u });
}

void ReSTIRPTLF::generatePaths(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "generatePaths");

    // Check shader assumptions.
    // We launch one thread group per screen tile, with threads linearly indexed.
    const uint32_t tileSize = kScreenTileDim.x * kScreenTileDim.y;
    FALCOR_ASSERT(kScreenTileDim.x == 16 && kScreenTileDim.y == 16); // TODO: Remove this temporary limitation when Slang bug has been fixed, see comments in shader.
    FALCOR_ASSERT(kScreenTileBits.x <= 4 && kScreenTileBits.y <= 4); // Since we use 8-bit deinterleave.
    FALCOR_ASSERT(mpGeneratePaths->getThreadGroupSize().x == tileSize);
    FALCOR_ASSERT(mpGeneratePaths->getThreadGroupSize().y == 1 && mpGeneratePaths->getThreadGroupSize().z == 1);

    // Additional specialization. This shouldn't change resource declarations.
    mpGeneratePaths->addDefine("USE_VIEW_DIR", (mParams.lfEnabled == 0u && mpScene->getCamera()->getApertureRadius() > 0 && renderData[kInputViewDir] != nullptr) ? "1" : "0");
    mpGeneratePaths->addDefine("OUTPUT_GUIDE_DATA", mOutputGuideData ? "1" : "0");

    // Bind resources.
    auto var = mpGeneratePaths->getRootVar()["CB"]["gPathGenerator"];
    bindShaderData(var, renderData, false);

    mpScene->bindShaderData(mpGeneratePaths->getRootVar()["gScene"]);

    // Launch one thread per pixel.
    // The dimensions are padded to whole tiles to allow re-indexing the threads in the shader.
    mpGeneratePaths->execute(pRenderContext, { mParams.screenTiles.x * tileSize, mParams.screenTiles.y, 1u });
}

void ReSTIRPTLF::tracePass(RenderContext* pRenderContext, const RenderData& renderData, TracePass& tracePass)
{
    FALCOR_PROFILE(pRenderContext, tracePass.name);

    FALCOR_ASSERT(tracePass.pProgram != nullptr && tracePass.pBindingTable != nullptr && tracePass.pVars != nullptr);

    // Additional specialization. This shouldn't change resource declarations.
    tracePass.pProgram->addDefine("USE_VIEW_DIR", (mParams.lfEnabled == 0u && mpScene->getCamera()->getApertureRadius() > 0 && renderData[kInputViewDir] != nullptr) ? "1" : "0");
    tracePass.pProgram->addDefine("OUTPUT_GUIDE_DATA", mOutputGuideData ? "1" : "0");

    // Bind global resources.
    auto var = tracePass.pVars->getRootVar();

    if (mVarsChanged) mpSampleGenerator->bindShaderData(var);
    mpPixelStats->prepareProgram(tracePass.pProgram, var);
    mpPixelDebug->prepareProgram(tracePass.pProgram, var);

    // Bind the path tracer.
    var["gReSTIRPTLF"] = mpReSTIRPTLFBlock;

    // Full screen dispatch.
    mpScene->raytrace(pRenderContext, tracePass.pProgram.get(), tracePass.pVars, uint3(mParams.frameDim, 1));
}

void ReSTIRPTLF::temporalReusePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "temporalReusePass");

    FALCOR_ASSERT(mpTemporalReusePass);

    mpTemporalReusePass->addDefine("USE_VIEW_DIR", (mParams.lfEnabled == 0u && mpScene->getCamera()->getApertureRadius() > 0 && renderData[kInputViewDir] != nullptr) ? "1" : "0");

    // Temporal reuse shifts the previous frame's final reservoir into the
    // current pixel domain and writes a combined reservoir/color. The same-pixel
    // debug path bypasses the temporal shift map entirely.
    auto var = mpTemporalReusePass->getRootVar()["CB"]["gTemporalReusePass"];
    var["params"].setBlob(mParams);
    var["currentReservoirs"] = mpCurrentReservoirs;
    var["previousReservoirs"] = mpPreviousReservoirs;
    var["outputReservoirs"] = mpTemporalReservoirs;
    var["outputColor"] = renderData.getTexture(kOutputColor);
    ref<Texture> pViewDir;
    if (mParams.lfEnabled == 0u) pViewDir = renderData.getTexture(kInputViewDir);
    var["vbuffer"] = getActiveVBuffer(renderData);
    var["previousVBuffer"] = mpPreviousVBuffer;
    var["viewDir"] = pViewDir;
    var["motionVectors"] = renderData.getTexture(kInputMotionVectors);
    var["historyLength"] = mTemporalHistoryLength;
    var["debugView"] = mDebugView;
    var["featureBasedRejection"] = mFeatureBasedRejection ? 1u : 0u;
    var["forceSamePixel"] = mTemporalReuseForceSamePixel ? 1u : 0u;

    mpTemporalReusePass->getRootVar()["gReSTIRPTLF"] = mpReSTIRPTLFBlock;
    mpScene->bindShaderData(mpTemporalReusePass->getRootVar()["gScene"]);
    mpTemporalReusePass->execute(pRenderContext, { mParams.frameDim, 1u });
}

void ReSTIRPTLF::spatialReusePass(RenderContext* pRenderContext, const RenderData& renderData, const ref<Buffer>& pInputReservoirs)
{
    FALCOR_PROFILE(pRenderContext, "spatialReusePass");

    FALCOR_ASSERT(mpSpatialReusePass);

    mpSpatialReusePass->addDefine("USE_VIEW_DIR", (mParams.lfEnabled == 0u && mpScene->getCamera()->getApertureRadius() > 0 && renderData[kInputViewDir] != nullptr) ? "1" : "0");

    // This pass consumes the initial reservoirs written by ray tracing and
    // produces a second reservoir set. It binds both the scene and ReSTIRPTLF
    // parameter block because light reconnection evaluates destination-side
    // env/emissive PDFs using the same samplers as the initial NEE pass.
    auto var = mpSpatialReusePass->getRootVar()["CB"]["gSpatialReusePass"];
    var["params"].setBlob(mParams);
    var["inputReservoirs"] = pInputReservoirs ? pInputReservoirs : mpCurrentReservoirs;
    var["outputReservoirs"] = mpSpatialReservoirs;
    var["outputColor"] = renderData.getTexture(kOutputColor);
    ref<Texture> pViewDir;
    if (mParams.lfEnabled == 0u) pViewDir = renderData.getTexture(kInputViewDir);
    var["vbuffer"] = getActiveVBuffer(renderData);
    var["viewDir"] = pViewDir;
    var["neighborCount"] = mSpatialNeighborCount;
    var["radius"] = mSpatialRadius;
    var["debugView"] = mDebugView;
    var["misStrategy"] = static_cast<uint32_t>(mSpatialMISStrategy);
    var["featureBasedRejection"] = mFeatureBasedRejection ? 1u : 0u;
    var["angularReuse"] = 0u;

    mpSpatialReusePass->getRootVar()["gReSTIRPTLF"] = mpReSTIRPTLFBlock;
    mpScene->bindShaderData(mpSpatialReusePass->getRootVar()["gScene"]);
    mpSpatialReusePass->execute(pRenderContext, { mParams.frameDim, 1u });
}

void ReSTIRPTLF::angularReusePass(RenderContext* pRenderContext, const RenderData& renderData, const ref<Buffer>& pInputReservoirs)
{
    FALCOR_PROFILE(pRenderContext, "angularReusePass");

    FALCOR_ASSERT(mpSpatialReusePass);
    FALCOR_ASSERT(mpAngularReservoirs);

    mpSpatialReusePass->addDefine("USE_VIEW_DIR", "0");

    auto var = mpSpatialReusePass->getRootVar()["CB"]["gSpatialReusePass"];
    var["params"].setBlob(mParams);
    var["inputReservoirs"] = pInputReservoirs ? pInputReservoirs : mpCurrentReservoirs;
    var["outputReservoirs"] = mpAngularReservoirs;
    var["outputColor"] = renderData.getTexture(kOutputColor);
    var["vbuffer"] = getActiveVBuffer(renderData);
    ref<Texture> pViewDir;
    var["viewDir"] = pViewDir;
    var["neighborCount"] = 0u;
    var["radius"] = mSpatialRadius;
    var["debugView"] = mDebugView;
    var["misStrategy"] = static_cast<uint32_t>(mSpatialMISStrategy);
    var["featureBasedRejection"] = mFeatureBasedRejection ? 1u : 0u;
    var["angularReuse"] = 1u;

    mpSpatialReusePass->getRootVar()["gReSTIRPTLF"] = mpReSTIRPTLFBlock;
    mpScene->bindShaderData(mpSpatialReusePass->getRootVar()["gScene"]);
    mpSpatialReusePass->execute(pRenderContext, { mParams.frameDim, 1u });
}

void ReSTIRPTLF::resolvePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mOutputGuideData && mStaticParams.samplesPerPixel == 1) return;

    FALCOR_PROFILE(pRenderContext, "resolvePass");

    // This pass is executed when multiple samples per pixel are used.
    // We launch one thread per pixel that computes the resolved color by iterating over the samples.
    // The samples are arranged in tiles with pixels in Morton order, with samples stored consecutively for each pixel.
    // Additional specialization. This shouldn't change resource declarations.
    mpResolvePass->addDefine("OUTPUT_GUIDE_DATA", mOutputGuideData ? "1" : "0");

    // Bind resources.
    auto var = mpResolvePass->getRootVar()["CB"]["gResolvePass"];
    var["params"].setBlob(mParams);
    var["outputColor"] = renderData.getTexture(kOutputColor);
    var["outputAlbedo"] = renderData.getTexture(kOutputAlbedo);
    var["outputSpecularAlbedo"] = renderData.getTexture(kOutputSpecularAlbedo);
    var["outputIndirectAlbedo"] = renderData.getTexture(kOutputIndirectAlbedo);
    var["outputGuideNormal"] = renderData.getTexture(kOutputGuideNormal);
    var["outputReflectionPosW"] = renderData.getTexture(kOutputReflectionPosW);

    if (mVarsChanged)
    {
        var["sampleColor"] = mpSampleColor;
        var["sampleGuideData"] = mpSampleGuideData;
    }

    // Launch one thread per pixel.
    mpResolvePass->execute(pRenderContext, { mParams.frameDim, 1u });
}

DefineList ReSTIRPTLF::StaticParams::getDefines(const ReSTIRPTLF& owner) const
{
    DefineList defines;

    // Path tracer configuration.
    defines.add("SAMPLES_PER_PIXEL", std::to_string(samplesPerPixel));
    defines.add("MAX_SURFACE_BOUNCES", std::to_string(maxSurfaceBounces));
    defines.add("MAX_DIFFUSE_BOUNCES", std::to_string(maxDiffuseBounces));
    defines.add("MAX_SPECULAR_BOUNCES", std::to_string(maxSpecularBounces));
    defines.add("MAX_TRANSMISSON_BOUNCES", std::to_string(maxTransmissionBounces));
    defines.add("ADJUST_SHADING_NORMALS", adjustShadingNormals ? "1" : "0");
    defines.add("USE_BSDF_SAMPLING", useBSDFSampling ? "1" : "0");
    defines.add("USE_NEE", useNEE ? "1" : "0");
    defines.add("USE_MIS", useMIS ? "1" : "0");
    defines.add("USE_RUSSIAN_ROULETTE", useRussianRoulette ? "1" : "0");
    defines.add("USE_ALPHA_TEST", useAlphaTest ? "1" : "0");
    defines.add("USE_LIGHTS_IN_DIELECTRIC_VOLUMES", useLightsInDielectricVolumes ? "1" : "0");
    defines.add("DISABLE_CAUSTICS", disableCaustics ? "1" : "0");
    defines.add("PRIMARY_LOD_MODE", std::to_string((uint32_t)primaryLodMode));
    defines.add("COLOR_FORMAT", std::to_string((uint32_t)colorFormat));
    defines.add("MIS_HEURISTIC", std::to_string((uint32_t)misHeuristic));
    defines.add("MIS_POWER_EXPONENT", std::to_string(misPowerExponent));
    defines.add("RESTIRPTLF_DEBUG_VIEW", std::to_string(owner.mDebugView));

    // Sampling utilities configuration.
    FALCOR_ASSERT(owner.mpSampleGenerator);
    defines.add(owner.mpSampleGenerator->getDefines());

    if (owner.mpEmissiveSampler) defines.add(owner.mpEmissiveSampler->getDefines());
    defines.add("INTERIOR_LIST_SLOT_COUNT", std::to_string(maxNestedMaterials));

    defines.add("GBUFFER_ADJUST_SHADING_NORMALS", owner.mGBufferAdjustShadingNormals ? "1" : "0");

    // Scene-specific configuration.
    // Set defaults
    defines.add("USE_ENV_LIGHT", "0");
    defines.add("USE_ANALYTIC_LIGHTS", "0");
    defines.add("USE_EMISSIVE_LIGHTS", "0");
    defines.add("USE_CURVES", "0");
    defines.add("USE_SDF_GRIDS", "0");
    defines.add("USE_HAIR_MATERIAL", "0");

    if (auto scene = dynamic_ref_cast<Scene>(owner.mpScene))
    {
        defines.add(scene->getSceneDefines());
        defines.add("USE_ENV_LIGHT", scene->useEnvLight() ? "1" : "0");
        defines.add("USE_ANALYTIC_LIGHTS", scene->useAnalyticLights() ? "1" : "0");
        defines.add("USE_EMISSIVE_LIGHTS", scene->useEmissiveLights() ? "1" : "0");
        defines.add("USE_CURVES", (scene->hasGeometryType(Scene::GeometryType::Curve)) ? "1" : "0");
        defines.add("USE_SDF_GRIDS", scene->hasGeometryType(Scene::GeometryType::SDFGrid) ? "1" : "0");
        defines.add("USE_HAIR_MATERIAL", scene->getMaterialCountByType(MaterialType::Hair) > 0u ? "1" : "0");
    }

    // Set default (off) values for additional features.
    defines.add("USE_VIEW_DIR", "0");
    defines.add("OUTPUT_GUIDE_DATA", "0");

    return defines;
}
