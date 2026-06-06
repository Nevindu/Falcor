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
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Rendering.Materials.IsotropicGGX;
import Rendering.Materials.InteriorListHelpers;
import Rendering.Volumes.HomogeneousVolumeSampler;
import Rendering.Utils.PixelStats;
import Scene.RaytracingInline;
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Debug.PixelDebug;
import Utils.Sampling.TinyUniformSampleGenerator;
import LoadShadingData;
import ColorType;
__exported import PathState;
__exported import Params;

/** Interface for querying visibility in the scene.
    This is used in `handleHit`.
*/
interface IVisibilityQuery
{
    /** Trace a visibility ray against the scene.
        \param[in] ray Ray.
        \return Returns true if the ray endpoints are mutually visible (i.e. the ray does NOT intersect the scene).
    */
    [mutating] bool traceVisibilityRay(const Ray ray);
};

/** Path tracer.

    This implements the high-level path tracing logic and is shared among
    different scheduling implementations.
*/
struct ReSTIRPT
{
    ReSTIRPTParams params;                        ///< Runtime parameters.

    // Samplers
    EnvMapSampler envMapSampler;                    ///< Environment map sampler. Only valid when kUseEnvLight == true.
    EmissiveLightSampler emissiveSampler;           ///< Emissive light sampler. Only valid when kUseEmissiveLights == true.

    // Inputs
    Texture2D<PackedHitInfo> vbuffer;               ///< Fullscreen V-buffer for the primary hits.
    Texture2D<float3> viewDir;                      ///< Optional view direction. Only valid when kUseViewDir == true.

    // Outputs
    RWStructuredBuffer<ColorType> sampleColor;      ///< Output per-sample color if kSamplesPerPixel != 1.
    RWStructuredBuffer<GuideData> sampleGuideData;  ///< Output per-sample guide data.

    RWTexture2D<float4> outputColor;                ///< Output color buffer if kSamplesPerPixel == 1.
    RWStructuredBuffer<PathReservoir> currentReservoirs; ///< Current-frame reservoirs, one per pixel.
    StructuredBuffer<PathReservoir> previousReservoirs;  ///< Previous-frame reservoirs for future temporal reuse.

    /*******************************************************************
                                Static members
    *******************************************************************/

    // Render settings that depend on the scene.
    // TODO: Move into scene defines.
    static const bool kUseEnvLight = USE_ENV_LIGHT;
    static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
    static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
    static const bool kUseCurves = USE_CURVES;
    static const bool kUseHairMaterial = USE_HAIR_MATERIAL;
    static const bool kUseAlphaTest = USE_ALPHA_TEST;

    // Additional specialization.
    static const bool kOutputGuideData = OUTPUT_GUIDE_DATA;
    static const uint kDebugView = RESTIRPT_DEBUG_VIEW;

    /** Types of samplable lights.
    */
    enum class SampledLightType
    {
        EnvMap,
        Emissive,
        Analytic
    };

    static const uint kReusableLightFlagUseMIS = 1u << 0;
    static const uint kReusableLightFlagSampleUpper = 1u << 1;
    static const uint kReusableLightFlagSampleLower = 1u << 2;

    /** Describes a light sample.
    */
    struct LightSample
    {
        float3  Li;         ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
        float3  Le;         ///< Light radiance before solid-angle PDF division, used for shifted reconnection.
        float   pdf;        ///< Pdf with respect to solid angle at the shading point.
        float3  origin;     ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
        float   distance;   ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
        float3  dir;        ///< Ray direction for visibility evaluation (normalized).
        float3  normal;     ///< Light normal for finite emissive samples.
        float   scatterPdf; ///< BSDF PDF for the sampled light direction, used by MIS.
        uint    lightType;  ///< Sampled light class this sample comes from (SampledLightType casted to uint).
        uint    lightIndex; ///< Analytic light index when lightType is Analytic.
        uint    triangleIndex; ///< Emissive triangle index when lightType is Emissive.
        uint    flags;      ///< Reusable-light metadata flags.

        Ray getVisibilityRay() { return Ray(origin, dir, 0.f, distance); }
    };

    /** Describes a path vertex.
    */
    struct PathVertex
    {
        uint index;         ///< Vertex index (0 = camera, 1 = primary hit, 2 = secondary hit, etc.).
        float3 pos;         ///< Vertex position.
        float3 faceNormal;  ///< Geometry normal at the vertex (zero if not on a surface).
        bool frontFacing;   ///< True if path vertex is on the front-facing side (if on a surface).

        /** Initializes a path vertex.
            \param[in] index Vertex index.
            \param[in] pos Vertex position.
            \param[in] faceNormal Geometry normal.
            \param[in] frontFacing Front-facing flag.
        */
        __init(uint index, float3 pos, float3 faceNormal = float3(0.f), bool frontFacing = true)
        {
            this.index = index;
            this.pos = pos;
            this.faceNormal = faceNormal;
            this.frontFacing = frontFacing;
        }

        /** Get position with offset applied in direction of the geometry normal to avoid self-intersection
            for visibility rays.
            \param[in] rayDir Direction of the visibility ray (does not need to be normalized).
            \return Returns the offseted position.
        */
        float3 getRayOrigin(float3 rayDir)
        {
            return computeRayOrigin(pos, dot(faceNormal, rayDir) >= 0 ? faceNormal : -faceNormal);
        }

        /** Returns the oriented face normal.
            \return Face normal flipped to the same side as the view vector.
        */
        float3 getOrientedFaceNormal()
        {
            return frontFacing ? faceNormal : -faceNormal;
        }
    };

    /** Set guiding data when background is hit.
    */
    static void setBackgroundGuideData(inout GuideData guideData, const float3 dir, const float3 Le)
    {
        if (kOutputGuideData)
        {
            guideData.setGuideNormal(-dir);
            // Compress dynamic range similar to UE4.
            const float3 compressedColor = pow(Le / (Le + 1.f), 0.454545f);
            guideData.setAlbedo(compressedColor);
            guideData.setSpecularAlbedo(float3(0.f));
            guideData.setIndirectAlbedo(float3(0.f));
            guideData.setReflectionPos(float3(0.f));
        }
    }

    /** Set guiding data when primary surface is hit.
    */
    static void setPrimarySurfaceGuideData(inout GuideData guideData, const ShadingData sd, const BSDFProperties bsdfProperties)
    {
        if (kOutputGuideData)
        {
            guideData.setGuideNormal(bsdfProperties.guideNormal);
            // Use sum of reflection/transmission albedo so the guide data covers both lobes.
            guideData.setAlbedo(bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.diffuseTransmissionAlbedo);
            guideData.setReflectionPos(sd.posW);

            const float NdotV = saturate(dot(bsdfProperties.guideNormal, sd.V));
            const float ggxAlpha = bsdfProperties.roughness * bsdfProperties.roughness;

            // We use an approximate shading model with a single specular lobe.
            // TODO: Generalize this to arbitrary materials.
            if (sd.mtl.getMaterialType() == MaterialType::Hair)
            {
                // Use a simple heuristic for hair material.
                const float3 preintegratedSpecular = approxSpecularIntegralGGX(float3(0.04f), ggxAlpha, NdotV);
                guideData.setSpecularAlbedo(preintegratedSpecular);
            }
            else
            {
                // Use pre-integrated FG term for specular
                const float3 preintegratedSpecular = approxSpecularIntegralGGX(bsdfProperties.specularReflectance, ggxAlpha, NdotV);
                guideData.setSpecularAlbedo(preintegratedSpecular);
            }
        }
    }

    /** Set guiding data when indirect surface is hit.
    */
    static void setIndirectSurfaceGuideData(inout GuideData guideData, const ShadingData sd, const BSDFProperties bsdfProperties)
    {
        if (any(bsdfProperties.emission > 0.f))
        {
            float3 indirectBaseColor = pow(bsdfProperties.emission / (bsdfProperties.emission + 1.0f), 0.454545f);
            guideData.setIndirectAlbedo(indirectBaseColor);
        }
        else
        {
            // Use sum of reflection/transmission albedo so the guide data covers both lobes.
            guideData.setIndirectAlbedo(bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.diffuseTransmissionAlbedo);
        }
        guideData.setReflectionPos(sd.posW);
    }

    /*******************************************************************
                              Member functions
    *******************************************************************/

    /** Check if the path has finished all surface bounces and needs to be terminated.
        Note: This is expected to be called after generateScatterRay(), which increments the bounce counters.
        \param[in] path Path state.
        \return Returns true if path has processed all bounces.
    */
    bool hasFinishedSurfaceBounces(const PathState path)
    {
        const uint diffuseBounces = path.getBounces(BounceType::Diffuse);
        const uint specularBounces = path.getBounces(BounceType::Specular);
        const uint transmissionBounces = path.getBounces(BounceType::Transmission);
        const uint surfaceBounces = diffuseBounces + specularBounces + transmissionBounces;
        return
            (surfaceBounces > kMaxSurfaceBounces) ||
            (diffuseBounces > kMaxDiffuseBounces) ||
            (specularBounces > kMaxSpecularBounces) ||
            (transmissionBounces > kMaxTransmissionBounces);
    }

    /** Compute the total length of a terminated path.
        \param[in] path Path state.
        \return Returns the total number of bounces a path took.
    */
    uint getTerminatedPathLength(const PathState path)
    {
        // Account for the fact that we may have counted one bounce too many (scatter ray at the last path vertex).
        uint diffuseBounces = min(kMaxDiffuseBounces, path.getBounces(BounceType::Diffuse));
        uint specularBounces = min(kMaxSpecularBounces, path.getBounces(BounceType::Specular));
        uint transmissionBounces = min(kMaxTransmissionBounces, path.getBounces(BounceType::Transmission));
        uint surfaceBounces = min(kMaxSurfaceBounces, diffuseBounces + specularBounces + transmissionBounces);
        return surfaceBounces;
    }

    /** Generate the path state for a primary hit in screen space.
        This is treated equivalent to subsequent path vertices to reduce code divergence.
        \param[in] pathID Path ID which encodes pixel and sample index.
        \param[out] path Path state for the primary hit.
    */
    void generatePath(const uint pathID, const uint pathSeed, out PathState path)
    {
        path = {};
        path.setActive();
        path.id = pathID;
        path.pathSeed = pathSeed;
        path.thp = float3(1.f);
        path.reconnectVertex = kInvalidReconnectVertex;

        const uint2 pixel = path.getPixel();

        // Create primary ray.
        Ray cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        if (kUseViewDir) cameraRay.dir = -viewDir[pixel];
        path.origin = cameraRay.origin;
        path.dir = cameraRay.dir;

        // Create sample generator.
        path.sg = SampleGenerator(pixel, path.pathSeed);
        path.risState = hashPathState(path.pathSeed ^ 0xa511e9b3u);
        path.selectedKind = kPathSampleKindInvalid;

        // Load the primary hit info from the V-buffer.
        const HitInfo hit = unpackHitInfo(vbuffer[pixel]);

        // If invalid, the path is still active and treated as a miss.
        if (hit.isValid())
        {
            path.setHit(hit);
            path.setVertexIndex(1);
        }
    }

    void generatePath(const uint pathID, out PathState path)
    {
        generatePath(pathID, makePathSeed(pathID, params.seed), path);
    }

    /** Set up path for logging and debugging.
        \param[in] path Path state.
    */
    void setupPathLogging(const PathState path)
    {
        printSetPixel(path.getPixel());
        logSetPixel(path.getPixel());
    }

    /** Compute a set of coherence hints for a given path state.
        These hints predict the code path taken in the next call to handleHit() with the given path state.
        They are used by reordering schedulers to extract coherence.
        \param[in] path Path state.
        \param[in] isDeltaSpecular Hit material is pure delta specular.
        \param[in] isEmissive Hit material is emissive.
        \return Returns a bit mask of coherence hints. MSB is highest priority hint. Currently 4 bits are used.
    */
    uint getCoherenceHints(const PathState path, const bool isDeltaSpecular, const bool isEmissive)
    {
        bool terminatedByRussianRoulette = {}; ///< handleHit() will be terminated early due to russian roulette.
        bool terminated = {};                  ///< handleHit() will be terminated early.
        bool samplesLight = {};                ///< handleHit() will sample a light.
        bool computesEmissive = {};            ///< handleHit() will compute emission.

        // Determines if handleHit() will return early due to russian roulette.
        if (kUseRussianRoulette)
        {
            // This is under the assumption that the first random number drawn from the sample generator is used for russian roulette.
            // Keep in sync with handleHit().
            PathState tmpPath = path;
            terminatedByRussianRoulette = terminatePathByRussianRoulette(tmpPath, sampleNext1D(tmpPath.sg));
        }

        // Determines if handleHit() will return early.
        terminated = hasFinishedSurfaceBounces(path);
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && isDeltaSpecular)
            terminated = true;

        // Determines if handleHit() will sample a light.
        samplesLight = true;
        if (isDeltaSpecular)
            samplesLight = false;
        if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume())
            samplesLight = false;

        // Determines if handleHit() will compute emission.
        computesEmissive = isEmissive && path.isLightSamplable();

        // MSB = highest priority hint.
        return
            select(computesEmissive, 0x1, 0x0) |
            select(samplesLight, 0x2, 0x0) |
            select(terminated, 0x4, 0x0) |
            select(terminatedByRussianRoulette, 0x8, 0x0);
    }

    /** Update the path throughouput.
        \param[in,out] path Path state.
        \param[in] weight Vertex throughput.
    */
    void updatePathThroughput(inout PathState path, const float3 weight)
    {
        path.thp *= weight;
    }

    /** True for BSDF lobes compatible with changing the incident reconnection edge.
    */
    bool isNonDeltaReflectionLobe(uint lobeType)
    {
        return (lobeType & (uint)LobeType::NonDeltaReflection) != 0u && (lobeType & (uint)LobeType::Transmission) == 0u;
    }

    /** Check whether a material's active lobes contain the stored replay lobe.
    */
    bool canRealizeReplayLobe(uint availableLobes, uint lobeType)
    {
        return lobeType != 0u && (availableLobes & lobeType) == lobeType;
    }

    /** Fetch a sampled lobe tag from the live path state for replay metadata.
    */
    uint getPathReplayLobeTag(const PathState path, uint vertex)
    {
        return getReplayLobeTag(path.scatterLobeTags0, path.scatterLobeTags1, vertex);
    }

    /** Fetch a sampled lobe tag from a compact reusable path sample.
    */
    uint getSampleReplayLobeTag(const PathSample sample, uint vertex)
    {
        return getReplayLobeTag(sample.scatterLobeTags0, sample.scatterLobeTags1, vertex);
    }

    /** Store the sampled scatter lobe at the current vertex for later replay.
    */
    void recordReplayScatterLobe(inout PathState path, uint lobeType)
    {
        setReplayLobeTag(path.scatterLobeTags0, path.scatterLobeTags1, path.getVertexIndex(), lobeType);
    }

    /** Conservative support test for the reused x2 vertex.

        Fixed-x2 pure reconnection changes the incoming direction at x2, so this
        rejects surfaces/volumes/lobes where that mapped direction has no support.
    */
    bool isFixedX2ReconnectibleSurface(
        const PathState path,
        const BSDFProperties bsdfProperties,
        const uint lobeTypes,
        const bool isTriangleHit,
        const bool isCurveHit)
    {
        if (path.getVertexIndex() != 2u) return false;
        if (!isTriangleHit || isCurveHit) return false;
        if (!path.interiorList.isEmpty() || path.isInsideDielectricVolume()) return false;
        if (float(path.sceneLength) < params.minReconnectDistance) return false;
        if (bsdfProperties.roughness <= params.specularRoughnessThreshold) return false;
        return (lobeTypes & (uint)LobeType::NonDeltaReflection) != 0u;
    }

    /** Mark the path reusable only when the current vertex is the supported x2.
    */
    void updateFixedX2ReconnectVertex(
        inout PathState path,
        const BSDFProperties bsdfProperties,
        const uint lobeTypes,
        const bool isTriangleHit,
        const bool isCurveHit)
    {
        if (path.reconnectVertex != kInvalidReconnectVertex) return;
        if (path.getVertexIndex() != 2u) return;
        if (isFixedX2ReconnectibleSurface(path, bsdfProperties, lobeTypes, isTriangleHit, isCurveHit))
            path.reconnectVertex = 2u;
    }

    void addPathContributionCandidate(
        inout PathState path,
        const float3 radiance,
        const uint sampleKind,
        const bool reusableLightValid,
        const uint reusableLightType,
        const uint reusableLightIndex,
        const uint reusableLightTriangleIndex,
        const float3 reusableLightPosW,
        const float3 reusableLightNormalW,
        const float3 reusableLightLe,
        const float3 reusableLightDir,
        const float reusableLightDistance,
        const float reusableLightPdf,
        const float reusableScatterPdf,
        const uint reusableLightFlags)
    {
        const float3 contribution = path.thp * radiance;
        const float target = max(0.f, luminance(contribution));
        if (target <= 0.f) return;

        const uint candidateIndex = path.risCandidateCount;
        path.risCandidateCount++;

        const float newWeightSum = path.risWeightSum + target;
        if (sampleNextRIS(path.risState) * newWeightSum < target)
        {
            path.L = contribution;
            path.risTarget = target;
            path.selectedCandidate = candidateIndex;
            path.selectedKind = sampleKind;
            path.selectedVertex = path.getVertexIndex();
            path.reusableLightValid = reusableLightValid ? 1u : 0u;
            path.reusableLightType = reusableLightType;
            path.reusableLightIndex = reusableLightIndex;
            path.reusableLightTriangleIndex = reusableLightTriangleIndex;
            path.reusableLightPosW = reusableLightPosW;
            path.reusableLightNormalW = reusableLightNormalW;
            path.reusableLightLi = reusableLightLe;
            path.reusableLightDir = reusableLightDir;
            path.reusableLightDistance = reusableLightDistance;
            path.reusableLightPdf = reusableLightPdf;
            path.reusableScatterPdf = reusableScatterPdf;
            path.reusableLightFlags = reusableLightFlags;
        }
        path.risWeightSum = newWeightSum;
    }

    /** Add a generic path-tree candidate to the initial RIS reservoir.
        \param[in,out] path Path state.
        \param[in] radiance Vertex radiance.
    */
    void addToPathContribution(inout PathState path, const float3 radiance)
    {
        addPathContributionCandidate(path, radiance, kPathSampleKindHitLight, false, 0u, 0u, 0u, float3(0.f), float3(0.f), float3(0.f), float3(0.f), 0.f, 0.f, 0.f, 0u);
    }

    /** Add a BSDF-sampled emissive hit candidate with enough endpoint data for spatial shifts.
    */
    void addHitLightPathContribution(
        inout PathState path,
        const float3 radiance,
        const float3 emission,
        const ShadingData sd,
        const uint triangleIndex,
        const float lightPdf,
        const float scatterPdf,
        const uint flags)
    {
        const bool reusable = path.getVertexIndex() > 1u && triangleIndex != 0xffffffffu && scatterPdf > 0.f;
        addPathContributionCandidate(
            path,
            radiance,
            kPathSampleKindHitLight,
            reusable,
            uint(SampledLightType::Emissive),
            0u,
            triangleIndex,
            sd.posW,
            sd.getOrientedFaceNormal(),
            emission,
            path.dir,
            distance(sd.posW, path.origin),
            lightPdf,
            scatterPdf,
            flags);
    }

    /** Add an environment miss candidate to the initial RIS reservoir.
        \param[in,out] path Path state.
        \param[in] radiance Environment radiance.
    */
    void addEnvPathContribution(inout PathState path, const float3 radiance, const float3 Le, const float lightPdf, const float scatterPdf, const uint flags)
    {
        const bool reusable = path.getVertexIndex() > 1u && scatterPdf > 0.f;
        addPathContributionCandidate(
            path,
            radiance,
            kPathSampleKindEnv,
            reusable,
            uint(SampledLightType::EnvMap),
            0u,
            0u,
            float3(0.f),
            -path.dir,
            Le,
            path.dir,
            kRayTMax,
            lightPdf,
            scatterPdf,
            flags);
    }

    /** Add an NEE direct-light candidate that may be spatially reconnected later.

        All NEE light samples carry enough geometry for the spatial pass to
        reconnect the same light sample at a neighboring destination surface.
    */
    void addNEEPathContribution(inout PathState path, const float3 radiance, const LightSample ls)
    {
        const bool finite = ls.distance < kRayTMax;
        const float3 lightPosW = finite ? ls.origin + ls.dir * ls.distance : float3(0.f);
        addPathContributionCandidate(path, radiance, kPathSampleKindNEE, true, ls.lightType, ls.lightIndex, ls.triangleIndex, lightPosW, ls.normal, ls.Le, ls.dir, ls.distance, ls.pdf, ls.scatterPdf, ls.flags);
    }

    /** Load compact primary-surface data for the pixel that generated a sample.

        Spatial reuse keeps this primary data for diagnostics and future temporal
        shifts. Current spatial correctness is decided by the shift map rather than
        by a material/normal/depth heuristic filter.
    */
    bool loadPrimarySurfaceData(const uint2 pixel, out float3 posW, out float3 normalW, out float depth, out uint materialID)
    {
        posW = float3(0.f);
        normalW = float3(0.f, 1.f, 0.f);
        depth = 0.f;
        materialID = 0u;

        const HitInfo hit = unpackHitInfo(vbuffer[pixel]);
        if (!hit.isValid()) return false;

        Ray cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        if (kUseViewDir) cameraRay.dir = -viewDir[pixel];

        ShadingData sd = loadShadingData(hit, cameraRay.origin, cameraRay.dir);
        uint loadedMaterialID = 0u;
        loadVertexData(hit, cameraRay.origin, cameraRay.dir, loadedMaterialID);

        posW = sd.posW;
        normalW = normalize(sd.faceN);
        depth = distance(cameraRay.origin, sd.posW);
        materialID = loadedMaterialID;
        return true;
    }

    /** Extract the selected RIS candidate from live path state into a reusable sample.

        The live PathState contains transient traversal state and cannot be carried
        across pixels or frames directly. PathSample is the compact record that later
        shift mappings will attempt to re-evaluate in another pixel's path space.
    */
    PathSample makePathSample(const PathState path)
    {
        PathSample sample = {};
        sample.pathID = path.id;
        sample.pathSeed = path.pathSeed;
        sample.selectedCandidate = path.selectedCandidate;
        sample.risCandidateCount = path.risCandidateCount;
        sample.selectedKind = path.selectedKind;
        sample.selectedVertex = path.selectedVertex;
        sample.reconnectVertex = path.reconnectVertex;
        sample.scatterLobeTags0 = path.scatterLobeTags0;
        sample.scatterLobeTags1 = path.scatterLobeTags1;
        sample.pathLength = getTerminatedPathLength(path);
        sample.flags = path.risTarget > 0.f ? kPathSampleFlagValid : 0u;
        const uint primaryScatterLobe = getPathReplayLobeTag(path, 1u);
        const bool primaryScatterReconnectible = path.isDiffusePrimaryHit() && isNonDeltaReflectionLobe(primaryScatterLobe);
        if (path.reusableLightValid != 0u && path.selectedVertex == 2u && primaryScatterReconnectible && (path.selectedKind == kPathSampleKindHitLight || path.selectedKind == kPathSampleKindEnv))
        {
            sample.flags |= kPathSampleFlagCanReconnectAtX2;
        }
        const bool hasSupportedTailReconnect = path.reconnectVertex == 2u && primaryScatterReconnectible;
        if (hasSupportedTailReconnect &&
            ((path.selectedKind == kPathSampleKindNEE && path.selectedVertex > 1u) ||
             ((path.selectedKind == kPathSampleKindHitLight || path.selectedKind == kPathSampleKindEnv) && path.selectedVertex > 2u)))
        {
            sample.flags |= kPathSampleFlagNeedsReplay;
        }
        sample.rngState = path.pathSeed;
        sample.target = path.risTarget;
        sample.contribution = path.L;
        sample.reusableLightValid = path.reusableLightValid;
        sample.reusableLightType = path.reusableLightType;
        sample.reusableLightIndex = path.reusableLightIndex;
        sample.reusableLightTriangleIndex = path.reusableLightTriangleIndex;
        sample.reusableLightPosW = path.reusableLightPosW;
        sample.reusableLightNormalW = path.reusableLightNormalW;
        sample.reusableLightLi = path.reusableLightLi;
        sample.reusableLightDir = path.reusableLightDir;
        sample.reusableLightDistance = path.reusableLightDistance;
        sample.reusableLightPdf = path.reusableLightPdf;
        sample.reusableScatterPdf = path.reusableScatterPdf;
        sample.reusableLightFlags = path.reusableLightFlags;

        float3 primaryPosW;
        float3 primaryNormalW;
        float primaryDepth;
        uint primaryMaterialID;
        if (loadPrimarySurfaceData(path.getPixel(), primaryPosW, primaryNormalW, primaryDepth, primaryMaterialID))
        {
            sample.primaryPosW = primaryPosW;
            sample.primaryNormalW = primaryNormalW;
            sample.primaryDepth = primaryDepth;
            sample.primaryMaterialID = primaryMaterialID;
        }
        else
        {
            // Directly visible environment samples have no primary surface, but
            // remain valid for the canonical identity domain. They simply cannot
            // be imported into surface pixels by the x2 reconnection shift.
            if (sample.selectedKind == kPathSampleKindEnv && sample.selectedVertex <= 1u)
            {
                sample.primaryPosW = float3(0.f);
                sample.primaryNormalW = float3(0.f, 1.f, 0.f);
                sample.primaryDepth = 0.f;
                sample.primaryMaterialID = 0xffffffffu;
            }
            else
            {
                sample.flags = 0u;
            }
        }
        return sample;
    }

    /** Build the initial per-pixel reservoir after all path-tree candidates have been traced.

        At this stage the reservoir represents only initial path-tree RIS. Spatial and
        temporal reuse operate on this explicit reservoir instead of on PathState.
    */
    PathReservoir makeInitialReservoir(const PathState path)
    {
        PathReservoir reservoir = {};
        reservoir.sample = makePathSample(path);
        reservoir.wSum = path.risWeightSum;
        reservoir.target = path.risTarget;
        reservoir.M = path.risTarget > 0.f ? 1u : 0u;
        reservoir.W = path.risTarget > 0.f ? path.risWeightSum / path.risTarget : 0.f;
        reservoir.shiftedTarget = path.risTarget;
        reservoir.targetRatio = path.risTarget > 0.f ? 1.f : 0.f;
        reservoir.jacobian = 1.f;
        reservoir.rejectionReason = path.risTarget > 0.f ? 0u : 1u;
        return reservoir;
    }

    bool isReservoirValid(const PathReservoir reservoir)
    {
        return (reservoir.sample.flags & kPathSampleFlagValid) != 0u && reservoir.target > 0.f && reservoir.sample.target > 0.f && reservoir.wSum > 0.f;
    }

    float getInitialReservoirCandidateWeight(const PathReservoir reservoir)
    {
        return isReservoirValid(reservoir) ? reservoir.wSum : 0.f;
    }

    PathReservoir finalizeInitialReservoir(PathReservoir reservoir, const float weightSum, const uint candidateCount)
    {
        if (!isReservoirValid(reservoir) || weightSum <= 0.f || candidateCount == 0u)
        {
            PathReservoir empty = {};
            empty.rejectionReason = 1u;
            return empty;
        }

        reservoir.wSum = weightSum;
        reservoir.M = candidateCount;
        reservoir.shiftedTarget = reservoir.target;
        reservoir.targetRatio = 1.f;
        reservoir.jacobian = 1.f;
        reservoir.W = reservoir.target > 0.f ? weightSum / (reservoir.target * float(candidateCount)) : 0.f;
        reservoir.rejectionReason = 0u;
        return reservoir;
    }

    /** Evaluate a selected sample under the identity shift.

        This is the first shift-mapping validation path: the sample remains in the
        same pixel/path space, so the shifted target should equal the stored target
        and the Jacobian is one. Real spatial/temporal shift mappings should keep
        this interface and replace the body with replay/reconnection logic.
    */
    ShiftMappingResult evaluateIdentityShift(const PathReservoir reservoir, const uint2 dstPixel)
    {
        ShiftMappingResult result = {};
        result.originalTarget = reservoir.target;
        result.jacobian = 1.f;

        if ((reservoir.sample.flags & kPathSampleFlagValid) == 0u)
        {
            result.rejectionReason = 1u;
            return result;
        }

        if (reservoir.target <= 0.f || reservoir.sample.target <= 0.f)
        {
            result.rejectionReason = 2u;
            return result;
        }

        // Identity re-evaluation: same pixel, same selected sample, unit Jacobian.
        // Future shift mappings will replace these assignments with replayed or
        // reconnected path evaluation in the destination pixel's path space.
        result.valid = true;
        result.shiftedTarget = reservoir.sample.target;
        result.targetRatio = result.shiftedTarget / reservoir.target;
        result.shiftedContribution = reservoir.sample.contribution;
        result.rejectionReason = 0u;
        return result;
    }

    /** Rebuild a reservoir from the result of identity-shift evaluation.

        Keeping this step explicit lets us validate the same accept/reject/weight
        path that future spatial and temporal shifts will use.
    */
    PathReservoir applyIdentityShift(const PathReservoir reservoir, const uint2 pixel)
    {
        ShiftMappingResult shift = evaluateIdentityShift(reservoir, pixel);

        PathReservoir shifted = reservoir;
        shifted.shiftedTarget = shift.shiftedTarget;
        shifted.targetRatio = shift.targetRatio;
        shifted.jacobian = shift.jacobian;
        shifted.rejectionReason = shift.rejectionReason;

        if (!shift.valid)
        {
            shifted.sample.flags = 0u;
            shifted.target = 0.f;
            shifted.wSum = 0.f;
            shifted.M = 0u;
            shifted.W = 0.f;
            return shifted;
        }

        shifted.sample.contribution = shift.shiftedContribution;
        shifted.sample.target = shift.shiftedTarget;
        shifted.target = shift.shiftedTarget;
        shifted.W = shifted.target > 0.f && shifted.M > 0u ? shifted.wSum * shift.jacobian / (shifted.target * float(shifted.M)) : 0.f;
        return shifted;
    }

    /** Convert a reservoir back to a color contribution for beauty output.

        W is the reservoir contribution weight. Spatial resampling may still use
        W * M as the source proposal compensation, but final shading evaluates the
        selected shifted contribution once with W.
    */
    float3 evaluateReservoirContribution(const PathReservoir reservoir)
    {
        return ((reservoir.sample.flags & kPathSampleFlagValid) != 0u && reservoir.target > 0.f) ? reservoir.sample.contribution * reservoir.W : float3(0.f);
    }

    /** Debug/beauty visualization for the selected reservoir.

        Values intentionally stay simple and untonemapped-looking; these views are
        for checking reservoir validity, weights, and shift diagnostics during
        incremental implementation.
    */
    float3 getDebugReservoirColor(const PathReservoir reservoir)
    {
        switch (kDebugView)
        {
        case 1: return reservoir.target.xxx * 0.02f;
        case 2: return reservoir.W.xxx;
        case 3: return float(reservoir.sample.pathLength).xxx / 10.f;
        case 4: return (reservoir.sample.flags & kPathSampleFlagValid) != 0u ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);
        case 5: return float(reservoir.sample.selectedCandidate).xxx / max(1.f, float(max(1u, reservoir.sample.risCandidateCount) - 1u));
        case 6: return reservoir.shiftedTarget.xxx * 0.02f;
        case 7: return reservoir.targetRatio.xxx;
        case 8: return reservoir.rejectionReason == 0u ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);
        case 9: return float(reservoir.rejectionReason).xxx / 8.f;
        default: return evaluateReservoirContribution(reservoir);
        }
    }

    /** Generates a new scatter ray using BSDF importance sampling.
        \param[in] sd Shading data.
        \param[in] mi Material instance at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    bool generateScatterRay(const ShadingData sd, const IMaterialInstance mi, inout PathState path)
    {
        BSDFSample result;
        bool valid = mi.sample(sd, path.sg, result, kUseBSDFSampling);
        if (valid) valid = generateScatterRay(result, sd, mi, path);
        return valid;
    }

    /** Generates a new scatter ray given a valid BSDF sample.
        \param[in] bs BSDF sample (assumed to be valid).
        \param[in] sd Shading data.
        \param[in] mi Material instance at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    bool generateScatterRay(const BSDFSample bs, const ShadingData sd, const IMaterialInstance mi, inout PathState path)
    {
        const bool isTriangleHit = path.hit.getType() == HitType::Triangle;
        const bool isCurveHit = kUseCurves && (path.hit.getType() == HitType::Curve);
        const bool isHairMaterial = kUseHairMaterial && (sd.mtl.getMaterialType() == MaterialType::Hair);
        // TODO: Decouple geometry from the material.
        const bool isCurvePolyTubeHit = isTriangleHit && isHairMaterial;

        path.dir = bs.wo;
        recordReplayScatterLobe(path, bs.lobeType);
        updatePathThroughput(path, bs.weight);
        path.pdf = bs.pdf;

        path.clearEventFlags();

        // Handle reflection events.
        if (bs.isLobe(LobeType::Reflection))
        {
            // We classify specular events as diffuse if the roughness is above some threshold.
            float roughness = mi.getProperties(sd).roughness;
            bool isDiffuse = bs.isLobe(LobeType::DiffuseReflection) || roughness > params.specularRoughnessThreshold;

            if (isDiffuse)
            {
                path.incrementBounces(BounceType::Diffuse);
                if (path.getVertexIndex() == 1u) path.setDiffusePrimaryHit();
            }
            else
            {
                path.incrementBounces(BounceType::Specular);
                path.setSpecular();
                if (path.getVertexIndex() == 1u) path.setSpecularPrimaryHit();
            }
        }

        // Handle delta events.
        if (bs.isLobe(LobeType::Delta))
        {
            path.setDelta();
            if (path.getVertexIndex() == 1u && bs.isLobe(LobeType::Reflection)) path.setDeltaReflectionPrimaryHit();
        }

        // Handle transmission events.
        if (bs.isLobe(LobeType::Transmission))
        {
            path.incrementBounces(BounceType::Transmission);
            path.setTransmission();

            if (isCurveHit)
            {
                // No need to offset the origin in this case.
            }
            else if (isCurvePolyTubeHit)
            {
                // For curves tessellated into poly-tubes, we make sure that the origin is on the same side as a scatter ray direction
                // so there is no self-intersection.
                path.origin = sd.posW - sd.frame.N * sd.curveRadius * 2.1f;
            }
            else
            {
                // Compute ray origin for next ray segment.
                path.origin = sd.computeRayOrigin(false);

                // Update interior list and inside volume flag.
                if (!sd.mtl.isThinSurface())
                {
                    uint nestedPriority = sd.mtl.getNestedPriority();
                    path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                    path.setInsideDielectricVolume(!path.interiorList.isEmpty());
                }
            }
        }

        // Save the normal used for NEE. This is needed for MIS.
        path.normal = sd.getOrientedFaceNormal();

        // Mark the path as valid only if it has a non-zero throughput.
        bool valid = any(path.thp > 0.f);

        return valid;
    }

    /** Evaluates the currently configured heuristic for multiple importance sampling (MIS).
        \param[in] n0 Number of samples taken from the first sampling strategy.
        \param[in] p0 Pdf for the first sampling strategy.
        \param[in] n1 Number of samples taken from the second sampling strategy.
        \param[in] p1 Pdf for the second sampling strategy.
        \return Weight for the contribution from the first strategy (p0).
    */
    float evalMIS(float n0, float p0, float n1, float p1)
    {
        switch (MISHeuristic(kMISHeuristic))
        {
        case MISHeuristic::Balance:
        {
            // Balance heuristic
            float q0 = n0 * p0;
            float q1 = n1 * p1;
            return q0 / (q0 + q1);
        }
        case MISHeuristic::PowerTwo:
        {
            // Power two heuristic
            float q0 = (n0 * p0) * (n0 * p0);
            float q1 = (n1 * p1) * (n1 * p1);
            return q0 / (q0 + q1);
        }
        case MISHeuristic::PowerExp:
        {
            // Power exp heuristic
            float q0 = pow(n0 * p0, kMISPowerExponent);
            float q1 = pow(n1 * p1, kMISPowerExponent);
            return q0 / (q0 + q1);
        }
        default:
            return 0.f;
        }
    }

    /** Generates a light sample on the environment map.
        \param[in] vertex Path vertex.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEnvMapSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.

        if (!kUseEnvLight) return false;

        // Sample environment map.
        EnvMapSample lightSample;
        if (!envMapSampler.sample(sampleNext2D(sg), lightSample)) return false;

        // Setup returned sample.
        ls.Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
        ls.Le = lightSample.Le;
        ls.pdf = lightSample.pdf;
        ls.origin = vertex.getRayOrigin(lightSample.dir);
        ls.distance = kRayTMax;
        ls.dir = lightSample.dir;
        ls.normal = -lightSample.dir;

        return any(ls.Li > 0.f);
    }

    /** Generates a light sample on the emissive geometry.
        \param[in] vertex Path vertex.
        \param[in] upperHemisphere True if only upper hemisphere should be considered.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEmissiveSample(const PathVertex vertex, const bool upperHemisphere, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.
        if (!kUseEmissiveLights) return false;

        TriangleLightSample tls;
        if (!emissiveSampler.sampleLight(vertex.pos, vertex.getOrientedFaceNormal(), upperHemisphere, sg, tls)) return false;

        // Setup returned sample.
        ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
        ls.Le = tls.Le;
        ls.pdf = tls.pdf;
        // Offset shading and light position to avoid self-intersection.
        float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
        ls.origin = vertex.getRayOrigin(lightPos - vertex.pos);
        float3 toLight = lightPos - ls.origin;
        ls.distance = length(toLight);
        ls.dir = normalize(toLight);
        ls.normal = tls.normalW;
        ls.triangleIndex = tls.triangleIndex;

        return any(ls.Li > 0.f);
    }

    /** Generates a light sample on the analytic lights.
        \param[in] vertex Path vertex.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateAnalyticLightSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.

        uint lightCount = gScene.getLightCount();
        if (!kUseAnalyticLights || lightCount == 0) return false;

        // Sample analytic light source selected uniformly from the light list.
        // TODO: Sample based on estimated contributions as pdf.
        uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);
        const LightData light = gScene.getLight(lightIndex);

        // Sample local light source.
        AnalyticLightSample lightSample;
        if (!sampleLight(vertex.pos, light, sg, lightSample)) return false;

        // Setup returned sample.
        ls.pdf = lightSample.pdf / lightCount;
        ls.Li = lightSample.Li * lightCount;
        ls.Le = light.intensity;
        // Offset shading position to avoid self-intersection.
        ls.origin = vertex.getRayOrigin(lightSample.dir);
        // Analytic lights do not currently have a geometric representation in the scene.
        // Do not worry about adjusting the ray length to avoid self-intersections at the light.
        ls.distance = lightSample.distance;
        ls.dir = lightSample.dir;
        ls.normal = lightSample.normalW;
        ls.lightIndex = lightIndex;
        if (light.type == LightType::Point || light.type == LightType::Rect || light.type == LightType::Disc || light.type == LightType::Sphere)
        {
            ls.origin = vertex.getRayOrigin(lightSample.posW - vertex.pos);
        }

        return any(ls.Li > 0.f);
    }

    /** Return the probabilities for selecting different light types.
        \param[out] p Probabilities.
    */
    void getLightTypeSelectionProbabilities(out float p[3])
    {
        // Set relative probabilities of the different sampling techniques.
        // TODO: These should use estimated irradiance from each light type. Using equal probabilities for now.
        p[0] = kUseEnvLight ? 1.f : 0.f;
        p[1] = kUseEmissiveLights ? 1.f : 0.f;
        p[2] = kUseAnalyticLights ? 1.f : 0.f;

        // Normalize probabilities. Early out if zero.
        float sum = p[0] + p[1] + p[2];
        if (sum == 0.f) return;

        float invSum = 1.f / sum;
        p[0] *= invSum;
        p[1] *= invSum;
        p[2] *= invSum;
    }

    float getEnvMapSelectionProbability()   { float p[3]; getLightTypeSelectionProbabilities(p); return p[0]; }
    float getEmissiveSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[1]; }
    float getAnalyicSelectionProbability()  { float p[3]; getLightTypeSelectionProbabilities(p); return p[2]; }

    /** Select a light type for sampling.
        \param[out] lightType Selected light type.
        \param[out] pdf Probability for selected type.
        \param[in,out] sg Sample generator.
        \return Return true if selection is valid.
    */
    bool selectLightType(out uint lightType, out float pdf, inout SampleGenerator sg)
    {
        float p[3];
        getLightTypeSelectionProbabilities(p);

        float u = sampleNext1D(sg);

        [unroll]
        for (lightType = 0; lightType < 3; ++lightType)
        {
            if (u < p[lightType])
            {
                pdf = p[lightType];
                return true;
            }
            u -= p[lightType];
        }

        lightType = {};
        pdf = {};

        return false;
    }

    /** Samples a light source in the scene.
        This function first stochastically selects a type of light source to sample,
        and then calls that the sampling function for the chosen light type.
        \param[in] vertex Path vertex.
        \param[in] sampleUpperHemisphere True if the upper hemisphere should be sampled.
        \param[in] sampleLowerHemisphere True if the lower hemisphere should be sampled.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateLightSample(const PathVertex vertex, const bool sampleUpperHemisphere, const bool sampleLowerHemisphere, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {};

        uint lightType;
        float selectionPdf;
        if (!selectLightType(lightType, selectionPdf, sg)) return false;

        bool valid = false;
        if (kUseEnvLight && lightType == (uint)SampledLightType::EnvMap) valid = generateEnvMapSample(vertex, sg, ls);
        if (kUseEmissiveLights && lightType == (uint)SampledLightType::Emissive)
        {
            // Emissive light samplers have an option to exclusively sample the upper hemisphere.
            bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;
            valid = generateEmissiveSample(vertex, upperHemisphere, sg, ls);
        }
        if (kUseAnalyticLights && lightType == (uint)SampledLightType::Analytic)
        {
            valid = generateAnalyticLightSample(vertex, sg, ls);
        }
        if (!valid) return false;

        // Reject samples in non-requested hemispheres.
        float NdotL = dot(vertex.getOrientedFaceNormal(), ls.dir);
        if ((!sampleUpperHemisphere && NdotL >= -kMinCosTheta) || (!sampleLowerHemisphere && NdotL <= kMinCosTheta))
            return false;

        // Account for light type selection.
        ls.lightType = lightType;
        ls.pdf *= selectionPdf;
        ls.Li /= selectionPdf;
        ls.flags = select(sampleUpperHemisphere, kReusableLightFlagSampleUpper, 0u) |
                   select(sampleLowerHemisphere, kReusableLightFlagSampleLower, 0u);

        return true;
    }

    /** Trace the next replay ray and advance the replay vertex index.

        Replay needs to reproduce the source suffix exactly; this helper mirrors
        normal path traversal while keeping the compact replay loop readable.
    */
    bool traceReplayNextHit(inout PathState path, out float hitT)
    {
        path.incrementVertexIndex();
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        const HitInfo hit = sceneRayQuery.traceRay(path.getScatterRay(), hitT);
        path.clearHit();
        if (!hit.isValid()) return false;

        path.setHit(hit);
        path.sceneLength += float16_t(hitT);
        return true;
    }

    /** Per-channel ratio that safely zeros channels with invalid source values.
    */
    float3 safeColorRatio(float3 numerator, float3 denominator)
    {
        const float eps = 1e-6f;
        return float3(
            denominator.x > eps ? numerator.x / denominator.x : 0.f,
            denominator.y > eps ? numerator.y / denominator.y : 0.f,
            denominator.z > eps ? numerator.z / denominator.z : 0.f);
    }

    /** Area-measure Jacobian for the fixed-x2 pure reconnection edge.
    */
    float computeVertexReconnectJacobian(float3 srcPrimaryPosW, float3 dstPrimaryPosW, float3 reconnectPosW, float3 reconnectNormalW)
    {
        // Fixed-x2 pure reconnection changes only the edge into the reused x2
        // vertex. The finite-vertex area-measure Jacobian is
        // (cos_new / |y1-x2|^2) / (cos_old / |x1-x2|^2).
        const float3 oldEdge = reconnectPosW - srcPrimaryPosW;
        const float3 newEdge = reconnectPosW - dstPrimaryPosW;
        const float oldDistSqr = dot(oldEdge, oldEdge);
        const float newDistSqr = dot(newEdge, newEdge);
        const float minDistSqr = params.minReconnectDistance * params.minReconnectDistance;
        if (oldDistSqr <= minDistSqr || newDistSqr <= minDistSqr) return 0.f;

        const float3 n = normalize(reconnectNormalW);
        const float oldCos = abs(dot(n, -oldEdge * rsqrt(oldDistSqr)));
        const float newCos = abs(dot(n, -newEdge * rsqrt(newDistSqr)));
        if (oldCos <= kMinCosTheta || newCos <= kMinCosTheta) return 0.f;

        const float jacobian = (newCos / oldCos) * (oldDistSqr / newDistSqr);
        return !isnan(jacobian) && !isinf(jacobian) && jacobian > 0.f && jacobian <= params.maxReconnectJacobian ? jacobian : 0.f;
    }

    /** Re-evaluate the stored NEE light sample at an arbitrary replay vertex.

        Used both for the source replay vertex and the shifted destination
        version of that vertex, preserving the original light-type PDF convention.
    */
    bool evaluateStoredLightGeometryAtVertex(PathSample sample, ShadingData sd, out float3 origin, out float3 dir, out float distance, out float3 lightFactor, out float lightPdf, out uint rejectionReason)
    {
        origin = sd.computeRayOrigin();
        dir = float3(0.f, 1.f, 0.f);
        distance = 0.f;
        lightFactor = float3(0.f);
        lightPdf = 0.f;
        rejectionReason = 0u;

        if (sample.reusableLightValid == 0u)
        {
            rejectionReason = 8u;
            return false;
        }

        if (sample.reusableLightType == uint(SampledLightType::EnvMap))
        {
#if USE_ENV_LIGHT
            const float selectionPdf = getEnvMapSelectionProbability();
            if (selectionPdf <= 0.f)
            {
                rejectionReason = 8u;
                return false;
            }

            dir = normalize(sample.reusableLightDir);
            lightPdf = selectionPdf * envMapSampler.evalPdf(dir);
            if (lightPdf <= 0.f)
            {
                rejectionReason = 9u;
                return false;
            }

            origin = sd.computeRayOrigin();
            distance = kRayTMax;
            lightFactor = sample.reusableLightLi / lightPdf;
            return any(lightFactor > 0.f);
#else
            rejectionReason = 8u;
            return false;
#endif
        }

        if (sample.reusableLightType == uint(SampledLightType::Emissive))
        {
#if USE_EMISSIVE_LIGHTS
            const float selectionPdf = getEmissiveSelectionProbability();
            if (selectionPdf <= 0.f)
            {
                rejectionReason = 8u;
                return false;
            }

            TriangleLightHit hit;
            hit.triangleIndex = sample.reusableLightTriangleIndex;
            hit.posW = sample.reusableLightPosW;
            hit.normalW = sample.reusableLightNormalW;

            const bool sampleUpperHemisphere = (sample.reusableLightFlags & kReusableLightFlagSampleUpper) != 0u;
            const bool sampleLowerHemisphere = (sample.reusableLightFlags & kReusableLightFlagSampleLower) != 0u;
            const bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;
            lightPdf = selectionPdf * emissiveSampler.evalPdf(sd.posW, sd.getOrientedFaceNormal(), upperHemisphere, hit);
            if (lightPdf <= 0.f)
            {
                rejectionReason = 9u;
                return false;
            }

            const float3 toLightPoint = sample.reusableLightPosW - sd.posW;
            const float distSqr = dot(toLightPoint, toLightPoint);
            if (distSqr <= 1e-8f || dot(sample.reusableLightNormalW, -normalize(toLightPoint)) <= 0.f)
            {
                rejectionReason = 9u;
                return false;
            }

            distance = sqrt(distSqr);
            dir = toLightPoint / distance;
            origin = sd.computeRayOrigin();
            lightFactor = sample.reusableLightLi / lightPdf;
            return any(lightFactor > 0.f);
#else
            rejectionReason = 8u;
            return false;
#endif
        }

        if (sample.reusableLightType == uint(SampledLightType::Analytic))
        {
#if USE_ANALYTIC_LIGHTS
            const float selectionPdf = getAnalyicSelectionProbability();
            const uint lightCount = gScene.getLightCount();
            if (selectionPdf <= 0.f || sample.reusableLightIndex >= lightCount || lightCount == 0u)
            {
                rejectionReason = 8u;
                return false;
            }

            const LightData light = gScene.getLight(sample.reusableLightIndex);
            lightPdf = selectionPdf / float(lightCount);
            if (lightPdf <= 0.f)
            {
                rejectionReason = 9u;
                return false;
            }

            if (light.type == LightType::Directional || light.type == LightType::Distant)
            {
                dir = normalize(sample.reusableLightDir);
                origin = sd.computeRayOrigin();
                distance = kRayTMax;
                lightFactor = light.intensity / lightPdf;
                return any(lightFactor > 0.f);
            }

            const float3 toLightPoint = sample.reusableLightPosW - sd.posW;
            const float distSqr = max(dot(toLightPoint, toLightPoint), kMinLightDistSqr);
            distance = sqrt(distSqr);
            dir = toLightPoint / distance;
            origin = sd.computeRayOrigin();

            if (light.type == LightType::Point)
            {
                const float cosTheta = -dot(dir, light.dirW);
                float falloff = 1.f;
                if (cosTheta < light.cosOpeningAngle) falloff = 0.f;
                else if (light.penumbraAngle > 0.f)
                {
                    const float deltaAngle = light.openingAngle - acos(cosTheta);
                    falloff = smoothstep(0.f, light.penumbraAngle, deltaAngle);
                }
                lightFactor = light.intensity * falloff / distSqr / lightPdf;
                return any(lightFactor > 0.f);
            }

            if (light.type == LightType::Rect || light.type == LightType::Sphere || light.type == LightType::Disc)
            {
                const float cosLight = dot(sample.reusableLightNormalW, -dir);
                if (cosLight <= 0.f)
                {
                    rejectionReason = 9u;
                    return false;
                }
                lightFactor = light.intensity * (light.surfaceArea * cosLight / distSqr) / lightPdf;
                return any(lightFactor > 0.f);
            }
#endif
        }

        rejectionReason = 8u;
        return false;
    }

    /** Evaluate the selected stored NEE contribution at a replayed surface.

        The returned value is the local light-sampling contribution before the
        path-prefix ratios are applied by evaluateReplayReconnection().
    */
    bool evaluateStoredNEEAtVertex(PathSample sample, ShadingData sd, HitInfo hit, uint2 evalPixel, out float3 contribution, out uint rejectionReason)
    {
        contribution = float3(0.f);
        rejectionReason = 0u;

        float3 origin;
        float3 dir;
        float distance;
        float3 lightFactor;
        float lightPdf;
        if (!evaluateStoredLightGeometryAtVertex(sample, sd, origin, dir, distance, lightFactor, lightPdf, rejectionReason))
            return false;

        let lod = ExplicitLodTextureSampler(0.f);
        let hints = getMaterialInstanceHints(hit, false);
        const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(evalPixel, params.frameCount);

        if ((sample.reusableLightFlags & kReusableLightFlagUseMIS) != 0u)
        {
            const float scatterPdf = mi.evalPdf(sd, dir, kUseBSDFSampling);
            lightFactor *= evalMIS(1.f, lightPdf, 1.f, scatterPdf);
        }

        contribution = mi.eval(sd, dir, sg) * lightFactor;
        if (!any(contribution > 0.f))
        {
            rejectionReason = 10u;
            return false;
        }
        return true;
    }

    /** Consume NEE-side random choices during replay without adding candidates.

        This keeps the replay RNG stream aligned with the original trace, even
        though visibility and RIS selection are not recomputed while replaying.
    */
    bool consumeReplayNEE(ShadingData sd, const IMaterialInstance mi, uint lobeTypes, bool isCurveHit, bool isCurvePolyTubeHit, inout PathState path)
    {
        const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0;
        const bool applyNEE = kUseNEE && hasNonDeltaLobes;

        path.setLightSampled(false, false);
        if (!applyNEE) return true;

        LightSample ls = {};
        PathVertex vertex = PathVertex(path.getVertexIndex(), sd.posW, sd.faceN, sd.frontFacing);

        const bool canNeeSampleSphere = isCurveHit || isCurvePolyTubeHit;
        bool sampleUpperHemisphere = canNeeSampleSphere || ((lobeTypes & (uint)LobeType::NonDeltaReflection) != 0);
        bool sampleLowerHemisphere = canNeeSampleSphere || ((lobeTypes & (uint)LobeType::NonDeltaTransmission) != 0);
        if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume()) sampleUpperHemisphere = false;

        bool validSample = generateLightSample(vertex, sampleUpperHemisphere, sampleLowerHemisphere, path.sg, ls);
        path.setLightSampled(sampleUpperHemisphere, sampleLowerHemisphere);
        if (!validSample) return true;

        if (kUseMIS && ls.lightType != (uint)SampledLightType::Analytic)
        {
            float scatterPdf = mi.evalPdf(sd, ls.dir, kUseBSDFSampling);
            ls.Li *= evalMIS(1, ls.pdf, 1, scatterPdf);
        }

        // Match the original path RNG stream. Visibility does not affect later RNG.
        mi.eval(sd, ls.dir, path.sg);
        return true;
    }

    /** Replay a source suffix and evaluate the fixed-x2 reconnection shift.

        This is the deeper-event path for Y=(y0,y1,x2,...). It rebuilds the
        source path from pathID/pathSeed, verifies replayed lobe consistency,
        reconnects y1 to x2, and returns the shifted p-hat, contribution, and
        Jacobian used by spatial GRIS.
    */
    ShiftMappingResult evaluateReplayReconnection(PathReservoir candidate, ShadingData dstSd, HitInfo dstHit, uint2 dstPixel)
    {
        ShiftMappingResult result = {};
        result.originalTarget = candidate.target;

        PathSample sample = candidate.sample;
        const bool selectedIsNEE = sample.selectedKind == kPathSampleKindNEE;
        const bool selectedIsBSDFTerminal = sample.selectedKind == kPathSampleKindHitLight || sample.selectedKind == kPathSampleKindEnv;
        if ((sample.flags & kPathSampleFlagNeedsReplay) == 0u || (!selectedIsNEE && !selectedIsBSDFTerminal) || sample.selectedVertex < 2u)
        {
            result.rejectionReason = 8u;
            return result;
        }
        if ((sample.flags & kPathSampleFlagAlreadyShifted) != 0u)
        {
            // A shifted replay sample no longer has a canonical primary prefix
            // recoverable from pathSeed/pathID. Treat this as a replay
            // self-consistency failure until temporal/replay state is expanded.
            result.rejectionReason = 15u;
            return result;
        }
        if (sample.reconnectVertex != 2u || sample.selectedVertex > kMaxStoredReplayLobes + 1u)
        {
            result.rejectionReason = 14u;
            return result;
        }

        PathState path = {};
        generatePath(sample.pathID, sample.pathSeed, path);
        if (!path.isHit())
        {
            result.rejectionReason = 3u;
            return result;
        }

        const uint maxReplayVertex = min(sample.selectedVertex, kMaxBounces + 1u);
        float3 sourcePrimaryPosW = sample.primaryPosW;
        float3 sourceFirstFactor = float3(0.f);
        float3 shiftedFirstFactor = float3(0.f);
        float3 sourceX2Factor = float3(1.f);
        float3 shiftedX2Factor = float3(1.f);
        float3 reconnectPosW = float3(0.f);
        float3 reconnectNormalW = float3(0.f, 1.f, 0.f);
        bool haveFirstConnection = false;
        bool haveX2Ratio = sample.selectedVertex == 2u;
        bool haveVisibility = false;
        float sourceFirstPdf = 1.f;
        float shiftedFirstPdf = 1.f;
        float sourceX2Pdf = 1.f;
        float shiftedX2Pdf = 1.f;

        for (uint replayVertex = 1u; replayVertex <= maxReplayVertex; replayVertex++)
        {
            if (!path.isHit() || path.getVertexIndex() != replayVertex)
            {
                result.rejectionReason = 15u;
                return result;
            }

            const bool isPrimaryHit = replayVertex == 1u;
            const HitType hitType = path.hit.getType();
            const bool isTriangleHit = hitType == HitType::Triangle;
            const bool isCurveHit = kUseCurves && (hitType == HitType::Curve);
            if (!isTriangleHit || isCurveHit || !path.interiorList.isEmpty() || path.isInsideDielectricVolume())
            {
                result.rejectionReason = 14u;
                return result;
            }

            ShadingData sd = loadShadingData(path.hit, path.origin, path.dir);
            if (!handleNestedDielectrics(sd, path))
            {
                result.rejectionReason = 14u;
                return result;
            }

            let lod = createTextureSampler(path, isPrimaryHit);
            let hints = getMaterialInstanceHints(path.hit, isPrimaryHit);
            const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);
            BSDFProperties bsdfProperties = mi.getProperties(sd);
            bool isSpecular = bsdfProperties.roughness <= params.specularRoughnessThreshold;

            if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && isSpecular)
            {
                sd.mtl.setActiveLobes((uint)LobeType::Diffuse);
            }
            if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume())
            {
                bsdfProperties.emission = float3(0.f);
            }
            const uint lobeTypes = mi.getLobeTypes(sd);
            const bool sourceConnectible = isFixedX2ReconnectibleSurface(path, bsdfProperties, lobeTypes, isTriangleHit, isCurveHit);
            if (replayVertex == 2u && !sourceConnectible)
            {
                result.rejectionReason = 14u;
                return result;
            }

            if (replayVertex == sample.selectedVertex)
            {
                if (!haveFirstConnection || !haveX2Ratio || !haveVisibility)
                {
                    result.rejectionReason = 15u;
                    return result;
                }

                float3 selectedRatio = float3(1.f);
                if (selectedIsNEE && sample.selectedVertex == 2u)
                {
                    const float3 dstOrigin = dstSd.computeRayOrigin();
                    const float3 dstToX2 = sd.posW - dstOrigin;
                    const float dstDist = length(dstToX2);
                    if (dstDist <= 1e-6f)
                    {
                        result.rejectionReason = 13u;
                        return result;
                    }

                    ShadingData shiftedSd = loadShadingData(path.hit, dstOrigin, dstToX2 / dstDist);

                    uint sourceNEEReason = 0u;
                    uint shiftedNEEReason = 0u;
                    float3 sourceNEE;
                    float3 shiftedNEE;
                    const uint2 sourcePixel = uint2(sample.pathID, sample.pathID >> 12) & 0xfff;
                    if (!evaluateStoredNEEAtVertex(sample, sd, path.hit, sourcePixel, sourceNEE, sourceNEEReason) ||
                        !evaluateStoredNEEAtVertex(sample, shiftedSd, path.hit, dstPixel, shiftedNEE, shiftedNEEReason))
                    {
                        result.rejectionReason = shiftedNEEReason != 0u ? shiftedNEEReason : sourceNEEReason;
                        return result;
                    }
                    selectedRatio = safeColorRatio(shiftedNEE, sourceNEE);
                }

                const float3 shiftedContribution = candidate.sample.contribution *
                    safeColorRatio(shiftedFirstFactor, sourceFirstFactor) *
                    safeColorRatio(shiftedX2Factor, sourceX2Factor) *
                    selectedRatio;
                float3 finalShiftedContribution = shiftedContribution;
                if (selectedIsBSDFTerminal)
                {
                    if (sourceFirstPdf <= 0.f || shiftedFirstPdf <= 0.f || sourceX2Pdf <= 0.f || shiftedX2Pdf <= 0.f)
                    {
                        result.rejectionReason = 13u;
                        return result;
                    }
                    // The stored terminal contribution was sampled in the
                    // source BSDF measure. We move the reciprocal PDF ratios
                    // into the shifted contribution and the matching forward
                    // PDF ratios into the GRIS Jacobian below so their product
                    // represents the fixed-x2 mapped contribution measure.
                    finalShiftedContribution *= (sourceFirstPdf / shiftedFirstPdf) * (sourceX2Pdf / shiftedX2Pdf);
                }
                const float shiftedTarget = max(0.f, luminance(finalShiftedContribution));
                if (shiftedTarget <= 0.f)
                {
                    result.rejectionReason = 10u;
                    return result;
                }

                const float jacobian = computeVertexReconnectJacobian(sourcePrimaryPosW, dstSd.posW, reconnectPosW, reconnectNormalW);
                float finalJacobian = jacobian;
                if (selectedIsBSDFTerminal)
                {
                    // See the paired contribution-side PDF ratio above.
                    finalJacobian *= (shiftedFirstPdf / sourceFirstPdf) * (shiftedX2Pdf / sourceX2Pdf);
                }
                if (finalJacobian <= 0.f || isnan(finalJacobian) || isinf(finalJacobian) || finalJacobian > params.maxReconnectJacobian)
                {
                    result.rejectionReason = 13u;
                    return result;
                }

                result.valid = true;
                result.shiftedContribution = finalShiftedContribution;
                result.shiftedTarget = shiftedTarget;
                result.targetRatio = candidate.target > 0.f ? shiftedTarget / candidate.target : 0.f;
                result.jacobian = finalJacobian;
                result.rejectionReason = 0u;
                return result;
            }

            if (hasFinishedSurfaceBounces(path))
            {
                result.rejectionReason = 15u;
                return result;
            }

            path.origin = sd.computeRayOrigin();
            if (!consumeReplayNEE(sd, mi, lobeTypes, false, false, path))
            {
                result.rejectionReason = 15u;
                return result;
            }

            if (kUseRussianRoulette)
            {
                if (terminatePathByRussianRoulette(path, sampleNext1D(path.sg)))
                {
                    result.rejectionReason = 15u;
                    return result;
                }
            }

            BSDFSample bs;
            bool validScatter = mi.sample(sd, path.sg, bs, kUseBSDFSampling);
            if (!validScatter || bs.pdf <= 0.f || bs.isLobe(LobeType::Delta) || bs.isLobe(LobeType::Transmission))
            {
                result.rejectionReason = 14u;
                return result;
            }
            const uint expectedLobe = getSampleReplayLobeTag(sample, replayVertex);
            if (!isNonDeltaReflectionLobe(expectedLobe) || bs.lobeType != expectedLobe)
            {
                result.rejectionReason = 14u;
                return result;
            }

            const float3 sampledDir = bs.wo;
            const float3 sourceScatterFactor = bs.weight * bs.pdf;
            const float sourceScatterPdf = bs.pdf;
            if (!any(sourceScatterFactor > 0.f))
            {
                result.rejectionReason = 10u;
                return result;
            }

            if (!generateScatterRay(bs, sd, mi, path) || path.isSpecular() || path.isDelta() || path.isTransmission())
            {
                result.rejectionReason = 14u;
                return result;
            }

            if (replayVertex == 1u)
            {
                sourceFirstFactor = sourceScatterFactor;
                sourceFirstPdf = sourceScatterPdf;
            }
            if (replayVertex == 2u && sample.selectedVertex > 2u)
            {
                sourceX2Factor = sourceScatterFactor;
                sourceX2Pdf = sourceScatterPdf;

                const float3 dstOrigin = dstSd.computeRayOrigin();
                const float3 dstToX2 = sd.posW - dstOrigin;
                const float dstDist = length(dstToX2);
                if (dstDist <= 1e-6f)
                {
                    result.rejectionReason = 13u;
                    return result;
                }

                ShadingData shiftedX2Sd = loadShadingData(path.hit, dstOrigin, dstToX2 / dstDist);
                let x2Lod = ExplicitLodTextureSampler(0.f);
                let x2Hints = getMaterialInstanceHints(path.hit, false);
                const IMaterialInstance shiftedX2Mi = gScene.materials.getMaterialInstance(shiftedX2Sd, x2Lod, x2Hints);
                TinyUniformSampleGenerator x2EvalSg = TinyUniformSampleGenerator(dstPixel, params.frameCount);
                BSDFProperties shiftedX2Properties = shiftedX2Mi.getProperties(shiftedX2Sd);
                const uint x2Lobe = getSampleReplayLobeTag(sample, 2u);
                if (shiftedX2Properties.roughness <= params.specularRoughnessThreshold ||
                    !isNonDeltaReflectionLobe(x2Lobe) ||
                    !canRealizeReplayLobe(shiftedX2Mi.getLobeTypes(shiftedX2Sd), x2Lobe))
                {
                    result.rejectionReason = 14u;
                    return result;
                }
                shiftedX2Factor = shiftedX2Mi.eval(shiftedX2Sd, sampledDir, x2EvalSg);
                shiftedX2Pdf = shiftedX2Mi.evalPdf(shiftedX2Sd, sampledDir, kUseBSDFSampling);
                if (!any(shiftedX2Factor > 0.f))
                {
                    result.rejectionReason = 10u;
                    return result;
                }
                if (selectedIsBSDFTerminal && shiftedX2Pdf <= 0.f)
                {
                    result.rejectionReason = 13u;
                    return result;
                }
                haveX2Ratio = true;
            }

            float hitT;
            if (!traceReplayNextHit(path, hitT))
            {
                if (selectedIsBSDFTerminal && sample.selectedKind == kPathSampleKindEnv && replayVertex + 1u == sample.selectedVertex)
                {
                    if (!haveFirstConnection || !haveX2Ratio || !haveVisibility || sourceFirstPdf <= 0.f || shiftedFirstPdf <= 0.f || sourceX2Pdf <= 0.f || shiftedX2Pdf <= 0.f)
                    {
                        result.rejectionReason = 15u;
                        return result;
                    }

                    const float3 shiftedContribution = candidate.sample.contribution *
                        safeColorRatio(shiftedFirstFactor, sourceFirstFactor) *
                        safeColorRatio(shiftedX2Factor, sourceX2Factor) *
                        (sourceFirstPdf / shiftedFirstPdf) *
                        (sourceX2Pdf / shiftedX2Pdf);
                    const float shiftedTarget = max(0.f, luminance(shiftedContribution));
                    if (shiftedTarget <= 0.f)
                    {
                        result.rejectionReason = 10u;
                        return result;
                    }

                    float jacobian = computeVertexReconnectJacobian(sourcePrimaryPosW, dstSd.posW, reconnectPosW, reconnectNormalW);
                    // Keep the same terminal PDF/Jacobian convention as the
                    // finite-hit terminal branch above.
                    jacobian *= (shiftedFirstPdf / sourceFirstPdf) * (shiftedX2Pdf / sourceX2Pdf);
                    if (jacobian <= 0.f || isnan(jacobian) || isinf(jacobian) || jacobian > params.maxReconnectJacobian)
                    {
                        result.rejectionReason = 13u;
                        return result;
                    }

                    result.valid = true;
                    result.shiftedContribution = shiftedContribution;
                    result.shiftedTarget = shiftedTarget;
                    result.targetRatio = candidate.target > 0.f ? shiftedTarget / candidate.target : 0.f;
                    result.jacobian = jacobian;
                    result.rejectionReason = 0u;
                    return result;
                }
                result.rejectionReason = 15u;
                return result;
            }

            if (replayVertex == 1u)
            {
                ShadingData x2SourceSd = loadShadingData(path.hit, path.origin, path.dir);
                reconnectPosW = x2SourceSd.posW;
                reconnectNormalW = x2SourceSd.faceN;

                const float3 dstOrigin = dstSd.computeRayOrigin();
                const float3 dstToX2 = reconnectPosW - dstOrigin;
                const float dstDist = length(dstToX2);
                if (dstDist <= 1e-6f)
                {
                    result.rejectionReason = 13u;
                    return result;
                }

                SceneRayQuery<kUseAlphaTest> visibilityQuery;
                if (!visibilityQuery.traceVisibilityRay(Ray(dstOrigin, dstToX2 / dstDist, 0.f, dstDist)))
                {
                    result.rejectionReason = 11u;
                    return result;
                }

                let dstLod = ExplicitLodTextureSampler(0.f);
                let dstHints = getMaterialInstanceHints(dstHit, true);
                const IMaterialInstance dstMi = gScene.materials.getMaterialInstance(dstSd, dstLod, dstHints);
                TinyUniformSampleGenerator dstEvalSg = TinyUniformSampleGenerator(dstPixel, params.frameCount);
                BSDFProperties dstBsdfProperties = dstMi.getProperties(dstSd);
                const uint primaryLobe = getSampleReplayLobeTag(sample, 1u);
                if (dstBsdfProperties.roughness <= params.specularRoughnessThreshold ||
                    !isNonDeltaReflectionLobe(primaryLobe) ||
                    !canRealizeReplayLobe(dstMi.getLobeTypes(dstSd), primaryLobe))
                {
                    result.rejectionReason = 14u;
                    return result;
                }
                const float3 dstConnectionDir = dstToX2 / dstDist;
                shiftedFirstFactor = dstMi.eval(dstSd, dstConnectionDir, dstEvalSg);
                shiftedFirstPdf = dstMi.evalPdf(dstSd, dstConnectionDir, kUseBSDFSampling);
                if (!any(shiftedFirstFactor > 0.f))
                {
                    result.rejectionReason = 10u;
                    return result;
                }
                if (selectedIsBSDFTerminal && shiftedFirstPdf <= 0.f)
                {
                    result.rejectionReason = 13u;
                    return result;
                }

                haveFirstConnection = true;
                haveVisibility = true;
            }
        }

        result.rejectionReason = 15u;
        return result;
    }

    /** Handle hits on dielectrics.
        \return True if this is an valid intersection, false if it is rejected.
    */
    bool handleNestedDielectrics(inout ShadingData sd, inout PathState path)
    {
        // Check for false intersections.
        uint nestedPriority = sd.mtl.getNestedPriority();
        if (!path.interiorList.isTrueIntersection(nestedPriority))
        {
            // If it is a false intersection, we reject the hit and continue the path
            // on the other side of the interface.
            // If the offset position is not quite large enough, due to self-intersections
            // it is possible we repeatedly hit the same surface and try to reject it.
            // This has happened in a few occasions with large transmissive triangles.
            // As a workaround, count number of rejected hits and terminate the path if too many.
            if (path.rejectedHits < kMaxRejectedHits)
            {
                path.rejectedHits++;
                path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                path.origin = sd.computeRayOrigin(false);
                path.decrementVertexIndex();
            }
            else
            {
                path.terminate();
            }
            return false;
        }

        // Compute index of refraction for medium on the outside.
        sd.IoR = computeOutsideIoR(path.interiorList, sd.materialID, sd.frontFacing);

        return true;
    }

    /** Apply russian roulette to terminate paths early.
        \param[in,out] path Path.
        \param[in] u Uniform random number in [0,1).
        \return Returns true if path was terminated.
    */
    bool terminatePathByRussianRoulette(inout PathState path, float u)
    {
        const float rrVal = luminance(path.thp);
        const float prob = max(0.f, 1.f - rrVal);
        if (u < prob)
        {
            path.terminate();
            return true;
        }
        path.thp /= 1.f - prob;
        return false;
    }

    /** Helper to create a texture sampler instance.
        The method for computing texture level-of-detail depends on the configuration.
        \param[in] path Path state.
        \param[in] isPrimaryTriangleHit True if primary hit on a triangle.
        \return Texture sampler instance.
    */
    ITextureSampler createTextureSampler(const PathState path, bool isPrimaryHit)
    {
        const bool isTriangleHit = (path.hit.getType() == HitType::Triangle);
        if (kPrimaryLodMode == TexLODMode::RayDiffs && isPrimaryHit && isTriangleHit)
        {
            // Filtered lookups at primary hit on triangle.
            float2 ddx, ddy;
            computeDerivativesAtPrimaryTriangleHit(path.hit.getTriangleHit(), path.getPixel(), params.frameDim, ddx, ddy);
            return ExplicitGradientTextureSampler(ddx, ddy);
        }
        else
        {
            float lod = isPrimaryHit ? 0.f : params.lodBias;
            return ExplicitLodTextureSampler(lod);
        }
    }

    /** Handle the case when a scatter ray hits a surface.
        After handling the hit, a new scatter ray is generated or the path is terminated.
        \param[in,out] path The path state.
        \param[in,out] vq Visibility query.
    */
    void handleHit<VisibilityQuery : IVisibilityQuery>(inout PathState path, inout VisibilityQuery vq)
    {
        // Upon hit:
        // - Load vertex/material data
        // - Compute MIS weight if path.getVertexIndex() > 1 and emissive hit
        // - Add emitted radiance
        // - Sample light(s) using shadow rays
        // - Sample scatter ray or terminate

        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const HitType hitType = path.hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;
        const bool isCurveHit = kUseCurves && (hitType == HitType::Curve);

        // Load shading data. This is a long latency operation.
        ShadingData sd = loadShadingData(path.hit, path.origin, path.dir);

        const bool isHairMaterial = kUseHairMaterial && (sd.mtl.getMaterialType() == MaterialType::Hair);
        // TODO: Decouple geometry from the material.
        const bool isCurvePolyTubeHit = isTriangleHit && isHairMaterial;

        // Account for volume absorption.
        if (!path.interiorList.isEmpty())
        {
            const uint materialID = path.interiorList.getTopMaterialID();
            let material = gScene.materials.getMaterial(materialID);
            let vp = material.getHomogeneousVolumeProperties(gScene.materials, materialID);
            updatePathThroughput(path, HomogeneousVolumeSampler::evalTransmittance(vp, length(sd.posW - path.origin)));
        }

        // Reject false hits in nested dielectrics.
        if (!handleNestedDielectrics(sd, path)) return;

        logPathVertex();

        // Create texture sampler.
        let lod = createTextureSampler(path, isPrimaryHit);

        // Create BSDF instance and query its properties.
        let hints = getMaterialInstanceHints(path.hit, isPrimaryHit);
        const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        BSDFProperties bsdfProperties = mi.getProperties(sd);

        // Disable specular lobes if caustics are disabled and path already contains a diffuse vertex.
        bool isSpecular = bsdfProperties.roughness <= params.specularRoughnessThreshold;
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && isSpecular)
        {
            sd.mtl.setActiveLobes((uint)LobeType::Diffuse);
        }

        // Optionally disable emission inside volumes.
        if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume())
        {
            bsdfProperties.emission = float3(0.f);
        }

        const uint lobeTypes = mi.getLobeTypes(sd);
        const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0;
        updateFixedX2ReconnectVertex(path, bsdfProperties, lobeTypes, isTriangleHit, isCurveHit);

        // Check if the scatter event is samplable by the light sampling technique.
        const bool isLightSamplable = path.isLightSamplable();

        // Add emitted radiance.
        // The primary hit is always included, secondary hits only if emissive lights are enabled and the full light contribution hasn't been sampled elsewhere.
        bool computeEmissive = isPrimaryHit || kUseEmissiveLights && (!kUseNEE || kUseMIS || !path.isLightSampled() || !isLightSamplable);

        if (computeEmissive && any(bsdfProperties.emission > 0.f))
        {
            float misWeight = 1.f;
            float lightPdf = 0.f;
            float scatterPdf = path.pdf;
            uint triangleIndex = 0xffffffffu;
            uint reusableFlags = select(path.isLightSampledUpper(), kReusableLightFlagSampleUpper, 0u) |
                                 select(path.isLightSampledLower(), kReusableLightFlagSampleLower, 0u);

            if (kUseEmissiveLights && isTriangleHit && !isPrimaryHit)
            {
                TriangleLightHit hit;
                TriangleHit triangleHit = path.hit.getTriangleHit();
                triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
                hit.triangleIndex = triangleIndex;
                hit.posW = sd.posW;
                hit.normalW = sd.getOrientedFaceNormal();

                bool upperHemisphere = path.isLightSampledUpper() && !path.isLightSampledLower();
                lightPdf = triangleIndex != 0xffffffffu ? getEmissiveSelectionProbability() * emissiveSampler.evalPdf(path.origin, path.normal, upperHemisphere, hit) : 0.f;
            }

            if (kUseEmissiveLights && kUseNEE && kUseMIS && isTriangleHit && !isPrimaryHit && path.isLightSampled() && isLightSamplable)
            {
                // If NEE and MIS are enabled, and we've already sampled emissive lights,
                // then we need to evaluate the MIS weight here to account for the remaining contribution.
                // Note that MIS is only applied for hits on emissive triangles (other emissive geometry is not supported).

                // Compute MIS weight by combining this with BSDF sampling.
                // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
                misWeight = evalMIS(1, path.pdf, 1, lightPdf);
                reusableFlags |= kReusableLightFlagUseMIS;
            }

            // Accumulate emitted radiance weighted by path throughput and MIS weight.
            addHitLightPathContribution(path, misWeight * bsdfProperties.emission, bsdfProperties.emission, sd, triangleIndex, lightPdf, scatterPdf, reusableFlags);
        }

        // Terminate after scatter ray on last vertex has been processed.
        if (hasFinishedSurfaceBounces(path))
        {
            path.terminate();
            return;
        }

        // Compute origin for rays traced from this path vertex.
        if (isCurveHit)
        {
            // For curves, we set the new origin at the sphere center.
            path.origin = sd.posW - sd.frame.N * sd.curveRadius;
        }
        else if (isCurvePolyTubeHit)
        {
            // For curves tessellated into poly-tubes, we offset the new origin away from the curve center.
            path.origin = sd.posW + sd.frame.N * sd.curveRadius * 0.1f;
        }
        else
        {
            path.origin = sd.computeRayOrigin();
        }

        // Check if we should apply NEE.
        const bool applyNEE = kUseNEE && hasNonDeltaLobes;

        // TODO: Support multiple shadow rays.
        path.setLightSampled(false, false);
        if (applyNEE)
        {
            LightSample ls = {};

            // Setup path vertex.
            PathVertex vertex = PathVertex(path.getVertexIndex(), sd.posW, sd.faceN, sd.frontFacing);

            const bool canNeeSampleSphere = isCurveHit || isCurvePolyTubeHit;
            // Determine if upper/lower hemispheres need to be sampled.
            bool sampleUpperHemisphere = canNeeSampleSphere || ((lobeTypes & (uint)LobeType::NonDeltaReflection) != 0);
            bool sampleLowerHemisphere = canNeeSampleSphere || ((lobeTypes & (uint)LobeType::NonDeltaTransmission) != 0);
            if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume()) sampleUpperHemisphere = false;

            // Sample a light.
            bool validSample = generateLightSample(vertex, sampleUpperHemisphere, sampleLowerHemisphere, path.sg, ls);
            path.setLightSampled(sampleUpperHemisphere, sampleLowerHemisphere);

            if (validSample)
            {
                // Apply MIS weight.
                if (kUseMIS && ls.lightType != (uint)SampledLightType::Analytic)
                {
                    float scatterPdf = mi.evalPdf(sd, ls.dir, kUseBSDFSampling);
                    ls.scatterPdf = scatterPdf;
                    ls.flags |= kReusableLightFlagUseMIS;
                    ls.Li *= evalMIS(1, ls.pdf, 1, scatterPdf);
                }

                float3 weight = mi.eval(sd, ls.dir, path.sg);
                float3 Lr = weight * ls.Li;
                if (any(Lr > 0.f))
                {
                    Ray ray = ls.getVisibilityRay();


                    if (isCurvePolyTubeHit)
                    {
                        // For curves tessellated into poly-tubes, we make sure that the origin is on the same side as light
                        // so there is no self-shadowing (transmission lobe of hair BSDF takes care of that).
                        if (dot(sd.frame.N, ray.dir) < 0.f)
                        {
                            ray.origin = ray.origin - sd.frame.N * sd.curveRadius * 2.1f;
                        }
                    }

                    logTraceRay(PixelStatsRayType::Visibility);
                    bool visible = vq.traceVisibilityRay(ray);
                    if (visible) addNEEPathContribution(path, Lr, ls);
                }
            }
        }

        // Russian roulette to terminate paths early.
        if (kUseRussianRoulette)
        {
            if (terminatePathByRussianRoulette(path, sampleNext1D(path.sg))) return;
        }

        // Generate the next path segment or terminate.
        bool valid = generateScatterRay(sd, mi, path);

        // Output guide data.
        if (path.getVertexIndex() == 1)
        {
            setPrimarySurfaceGuideData(path.guideData, sd, bsdfProperties);
        }
        if (path.getVertexIndex() == 2 && (path.getBounces(BounceType::Specular) == 1 || path.getBounces(BounceType::Transmission) == 1))
        {
            setIndirectSurfaceGuideData(path.guideData, sd, bsdfProperties);
        }

        // Check if this is the last path vertex.
        const bool isLastVertex = hasFinishedSurfaceBounces(path);

        // Terminate if this is the last path vertex and light sampling already completely sampled incident radiance.
        if (kUseNEE && !kUseMIS && isLastVertex && path.isLightSamplable()) valid = false;

        // Terminate caustics paths.
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && path.isSpecular()) valid = false;

        if (!valid)
        {
            path.terminate();
        }
    }

    /** Handle the case when a scatter ray misses the scene.
        \param[in,out] path The path state.
    */
    void handleMiss(inout PathState path)
    {
        // Upon miss:
        // - Compute MIS weight if previous path vertex sampled a light
        // - Evaluate environment map
        // - Write guiding data
        // - Terminate the path

        // Check if the scatter event is samplable by the light sampling technique.
        const bool isLightSamplable = path.isLightSamplable();

        // Add env radiance.
        bool computeEnv = kUseEnvLight && (!kUseNEE || kUseMIS || !path.isLightSampled() || !isLightSamplable);

        if (computeEnv)
        {
            logPathVertex();

            float misWeight = 1.f;
            float lightPdf = getEnvMapSelectionProbability() * envMapSampler.evalPdf(path.dir);
            uint reusableFlags = 0u;
            if (kUseNEE && kUseMIS && path.isLightSampled() && isLightSamplable)
            {
                // If NEE and MIS are enabled, and we've already sampled the env map,
                // then we need to evaluate the MIS weight here to account for the remaining contribution.

                // Evaluate PDF, had it been generated with light sampling.

                // Compute MIS weight by combining this with BSDF sampling.
                // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
                misWeight = evalMIS(1, path.pdf, 1, lightPdf);
                reusableFlags |= kReusableLightFlagUseMIS;
            }

            float3 Le = envMapSampler.eval(path.dir);
            addEnvPathContribution(path, misWeight * Le, Le, lightPdf, path.pdf, reusableFlags);

            if (kOutputGuideData && path.getVertexIndex() == 2
                && (path.getBounces(BounceType::Specular) == 1
                || path.getBounces(BounceType::Transmission) == 1))
            {
                // Compress dynamic range similar to UE4.
                float3 compressedColor = pow(Le / (Le + 1.0f), 0.454545f);
                path.guideData.setIndirectAlbedo(compressedColor);
                path.guideData.setReflectionPos(path.dir * kEnvMapDepth);
            }
        }

        if (kOutputGuideData && path.getVertexIndex() == 1)
        {
            path.guideData.setGuideNormal(-path.dir);
        }

        path.terminate();
    }

    /** Write per-sample outputs used by guide/debug resolve paths.
    */
    void writeSampleOutputs(const PathState path, const PathReservoir reservoir)
    {
        // Log path length.
        logPathLength(reservoir.sample.pathLength);

        const uint2 pixel = path.getPixel();
        const uint outIdx = params.getSampleOffset(pixel) + path.getSampleIdx();

        const float3 color = getDebugReservoirColor(reservoir);

        if (kSamplesPerPixel == 1)
        {
            // Write color directly to frame buffer.
            outputColor[pixel] = float4(color, 1.f);
        }
        else
        {
            // Write color to per-sample buffer.
            sampleColor[outIdx].set(color);
        }

        if (kOutputGuideData)
        {
            sampleGuideData[outIdx] = path.guideData;
        }

    }

    /** Publish the selected per-pixel reservoir and its debug/beauty output.
    */
    void writeReservoirOutput(const uint2 pixel, PathReservoir reservoir)
    {
        reservoir = applyIdentityShift(reservoir, pixel);
        currentReservoirs[pixel.y * params.frameDim.x + pixel.x] = reservoir;
        outputColor[pixel] = float4(getDebugReservoirColor(reservoir), 1.f);
    }
};
