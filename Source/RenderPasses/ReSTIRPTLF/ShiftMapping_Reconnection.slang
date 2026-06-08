/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

import ReSTIRPTLF;
import LoadShadingData;
import Rendering.Lights.EmissiveLightSamplerInterface;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Rendering.Materials.TexLODHelpers;
import Scene.RaytracingInline;
import Utils.Color.ColorHelpers;
import Utils.Sampling.TinyUniformSampleGenerator;

// Per-sub-shift enables (production default: both on). Flip one off only to bisect which
// reconnection sub-shift a bias originates from -- turning either off changes render output.
static const bool kEnableDirectNEEShift = true;
static const bool kEnableReplayShift = true;

/** Fixed-x2 pure reconnection shift entry point.

    This file owns the terminal and replay reconnection implementation for the
    current production shift map.
*/
extension ReSTIRPTLF
{
    /** Geometry-only Jacobian for direct terminal-at-x2 reconnection.

        Finite emissive hits change the solid-angle/area conversion at the
        terminal light point. Environment/directional samples use unit geometry
        Jacobian because there is no finite light-point area measure.
    */
    float computeTerminalReconnectGeometryJacobian(PathSample sample, ShadingData dstSd)
    {
        if (sample.reusableLightDistance >= kRayTMax) return 1.f;
        if (dot(sample.reusableLightNormalW, sample.reusableLightNormalW) <= 0.25f) return 1.f;

        const float3 oldEdge = sample.reusableLightPosW - sample.primaryPosW;
        const float3 newEdge = sample.reusableLightPosW - dstSd.posW;
        const float oldDistSqr = dot(oldEdge, oldEdge);
        const float newDistSqr = dot(newEdge, newEdge);
        const float minDistSqr = params.minReconnectDistance * params.minReconnectDistance;
        if (oldDistSqr <= minDistSqr || newDistSqr <= minDistSqr) return 0.f;

        const float3 lightNormal = normalize(sample.reusableLightNormalW);
        const float oldCos = abs(dot(lightNormal, -oldEdge * rsqrt(oldDistSqr)));
        const float newCos = abs(dot(lightNormal, -newEdge * rsqrt(newDistSqr)));
        if (oldCos <= kMinCosTheta || newCos <= kMinCosTheta) return 0.f;

        const float jacobian = (newCos / oldCos) * (oldDistSqr / newDistSqr);
        return !isnan(jacobian) && !isinf(jacobian) && jacobian > 0.f && jacobian <= params.maxReconnectJacobian ? jacobian : 0.f;
    }

    /** Recover terminal light geometry for a BSDF-sampled event at source x2.

        This handles only fixed-x2 terminal samples: directly hit emissive
        geometry or the environment after the source primary bounce.
    */
    bool evaluateTerminalBSDFGeometry(PathSample sample, ShadingData dstSd, out float3 origin, out float3 dir, out float distance, out float3 emission, out float lightPdf, out float geometryJacobian, out uint rejectionReason)
    {
        origin = dstSd.computeRayOrigin();
        dir = float3(0.f, 1.f, 0.f);
        distance = 0.f;
        emission = float3(0.f);
        lightPdf = 0.f;
        geometryJacobian = 1.f;
        rejectionReason = 0u;

        if ((sample.flags & kPathSampleFlagCanReconnectAtX2) == 0u || sample.reusableLightValid == 0u || sample.selectedVertex != 2u)
        {
            rejectionReason = 8u;
            return false;
        }

        if (sample.selectedKind == kPathSampleKindEnv)
        {
#if USE_ENV_LIGHT
            dir = normalize(sample.reusableLightDir);
            origin = dstSd.computeRayOrigin();
            distance = kRayTMax;
            emission = sample.reusableLightLi;

            const float selectionPdf = getEnvMapSelectionProbability();
            lightPdf = selectionPdf > 0.f ? selectionPdf * envMapSampler.evalPdf(dir) : 0.f;
            return any(emission > 0.f);
#else
            rejectionReason = 8u;
            return false;
#endif
        }

        if (sample.selectedKind == kPathSampleKindHitLight)
        {
#if USE_EMISSIVE_LIGHTS
            const float3 toLightPoint = sample.reusableLightPosW - dstSd.posW;
            const float distSqr = dot(toLightPoint, toLightPoint);
            if (distSqr <= 1e-8f)
            {
                rejectionReason = 9u;
                return false;
            }

            distance = sqrt(distSqr);
            dir = toLightPoint / distance;
            origin = dstSd.computeRayOrigin();
            if (dot(sample.reusableLightNormalW, -dir) <= 0.f)
            {
                rejectionReason = 9u;
                return false;
            }

            TriangleLightHit hit;
            hit.triangleIndex = sample.reusableLightTriangleIndex;
            hit.posW = sample.reusableLightPosW;
            hit.normalW = sample.reusableLightNormalW;
            const bool sampleUpperHemisphere = (sample.reusableLightFlags & kReusableLightFlagSampleUpper) != 0u;
            const bool sampleLowerHemisphere = (sample.reusableLightFlags & kReusableLightFlagSampleLower) != 0u;
            const bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;

            const float selectionPdf = getEmissiveSelectionProbability();
            lightPdf = selectionPdf > 0.f ? selectionPdf * emissiveSampler.evalPdf(dstSd.posW, dstSd.getOrientedFaceNormal(), upperHemisphere, hit) : 0.f;
            geometryJacobian = computeTerminalReconnectGeometryJacobian(sample, dstSd);
            if (geometryJacobian <= 0.f)
            {
                rejectionReason = 13u;
                return false;
            }

            emission = sample.reusableLightLi;
            return any(emission > 0.f);
#else
            rejectionReason = 8u;
            return false;
#endif
        }

        rejectionReason = 8u;
        return false;
    }

    /** Evaluate a non-replayed terminal candidate after replacing x1 by y1.
    */
    ShiftMappingResult evaluateTerminalBSDFCandidate(PathReservoir candidate, ShadingData dstSd, HitInfo dstHit, uint2 pixel)
    {
        ShiftMappingResult result = {};
        result.originalTarget = candidate.target;

        float3 origin;
        float3 dir;
        float distance;
        float3 emission;
        float lightPdf;
        float geometryJacobian;
        if (!evaluateTerminalBSDFGeometry(candidate.sample, dstSd, origin, dir, distance, emission, lightPdf, geometryJacobian, result.rejectionReason))
            return result;

        const float sourceScatterPdf = candidate.sample.reusableScatterPdf;
        if (sourceScatterPdf <= 0.f)
        {
            result.rejectionReason = 13u;
            return result;
        }

        let lod = ExplicitLodTextureSampler(0.f);
        let hints = getMaterialInstanceHints(dstHit, true);
        const IMaterialInstance mi = gScene.materials.getMaterialInstance(dstSd, lod, hints);
        TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, params.frameCount);
        BSDFProperties dstBsdfProperties = mi.getProperties(dstSd);

        const uint sourceLobe = getSampleReplayLobeTag(candidate.sample, 1u);
        if (dstBsdfProperties.roughness <= params.specularRoughnessThreshold ||
            !isNonDeltaReflectionLobe(sourceLobe) ||
            !canRealizeReplayLobe(mi.getLobeTypes(dstSd), sourceLobe))
        {
            result.rejectionReason = 14u;
            return result;
        }

        const float dstScatterPdf = mi.evalPdf(dstSd, dir, kUseBSDFSampling);
        if (dstScatterPdf <= 0.f)
        {
            result.rejectionReason = 13u;
            return result;
        }

        float3 bsdfWeight = mi.eval(dstSd, dir, sg) / dstScatterPdf;
        if ((candidate.sample.reusableLightFlags & kReusableLightFlagUseMIS) != 0u)
        {
            if (lightPdf <= 0.f)
            {
                result.rejectionReason = 9u;
                return result;
            }
            bsdfWeight *= evalMIS(1.f, dstScatterPdf, 1.f, lightPdf);
        }

        const float3 shiftedContribution = bsdfWeight * emission;
        const float shiftedTarget = max(0.f, luminance(shiftedContribution));
        if (shiftedTarget <= 0.f)
        {
            result.rejectionReason = 10u;
            return result;
        }

        SceneRayQuery<kUseAlphaTest> rayQuery;
        const Ray ray = Ray(origin, dir, 0.f, distance);
        if (!rayQuery.traceVisibilityRay(ray))
        {
            result.rejectionReason = 11u;
            return result;
        }

        // Terminal hit-light/env samples store BSDF-sampled estimator
        // contributions (f/p). The geometric reconnection determinant is paired
        // with the BSDF density ratio so shiftedTarget * jacobian is evaluated in
        // the same fixed-x2 source-to-destination shift measure used by GRIS.
        const float jacobian = geometryJacobian * dstScatterPdf / sourceScatterPdf;
        if (jacobian <= 0.f || isnan(jacobian) || isinf(jacobian) || jacobian > params.maxReconnectJacobian)
        {
            result.rejectionReason = 13u;
            return result;
        }

        PathSample shiftedSample = candidate.sample;
        shiftedSample.contribution = shiftedContribution;
        shiftedSample.target = shiftedTarget;
        shiftedSample.reusableScatterPdf = dstScatterPdf;
        shiftedSample.reusableLightPdf = lightPdf;
        shiftedSample.reusableLightDistance = distance;

        result.valid = true;
        result.shiftedContribution = shiftedContribution;
        result.shiftedSample = shiftedSample;
        result.shiftedTarget = shiftedTarget;
        result.targetRatio = candidate.target > 0.f ? shiftedTarget / candidate.target : 0.f;
        result.jacobian = jacobian;
        result.rejectionReason = 0u;
        return result;
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

    /** Verify that source-path replay recovers the stored primary surface.

        Fixed-x2 reconnection treats pathID/pathSeed as the canonical source
        domain. If replay cannot recover the original y=(x0,x1,...) primary
        state, the mapping is not self-consistent enough for GRIS reuse.
    */
    bool validateReplayPrimarySurface(PathSample sample, HitInfo hit, float3 rayOrigin, float3 rayDir, ShadingData sd, out uint rejectionReason)
    {
        rejectionReason = 0u;
        if (sample.primaryDepth <= 0.f)
        {
            rejectionReason = 4u;
            return false;
        }

        uint materialID = 0u;
        loadVertexData(hit, rayOrigin, rayDir, materialID);
        if (sample.primaryMaterialID != 0xffffffffu && materialID != sample.primaryMaterialID)
        {
            rejectionReason = 15u;
            return false;
        }

        const float posTolerance = max(1e-3f, 4.f * params.minReconnectDistance);
        const float3 delta = sd.posW - sample.primaryPosW;
        if (dot(delta, delta) > posTolerance * posTolerance)
        {
            rejectionReason = 15u;
            return false;
        }

        return true;
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
            const float lightSelectionPdf = selectionPdf / float(lightCount);
            if (lightSelectionPdf <= 0.f)
            {
                rejectionReason = 9u;
                return false;
            }

            if (light.type == LightType::Directional)
            {
                dir = normalize(sample.reusableLightDir);
                origin = sd.computeRayOrigin();
                distance = kRayTMax;
                lightPdf = lightSelectionPdf;
                lightFactor = light.intensity / lightSelectionPdf;
                return any(lightFactor > 0.f);
            }

            if (light.type == LightType::Distant)
            {
                const float solidAngle = 6.28318530718f * (1.f - light.cosSubtendedAngle);
                if (solidAngle <= 0.f)
                {
                    rejectionReason = 9u;
                    return false;
                }

                dir = normalize(sample.reusableLightDir);
                origin = sd.computeRayOrigin();
                distance = kRayTMax;
                lightPdf = lightSelectionPdf / solidAngle;
                lightFactor = light.intensity / lightSelectionPdf;
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
                lightPdf = lightSelectionPdf;
                lightFactor = light.intensity * falloff / distSqr / lightSelectionPdf;
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
                const float localLightPdf = distSqr / (cosLight * light.surfaceArea);
                lightPdf = lightSelectionPdf * localLightPdf;
                if (lightPdf <= 0.f)
                {
                    rejectionReason = 9u;
                    return false;
                }

                lightFactor = light.intensity / lightPdf;
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
            if (isPrimaryHit)
            {
                uint replayReason = 0u;
                if (!validateReplayPrimarySurface(sample, path.hit, path.origin, path.dir, sd, replayReason))
                {
                    result.rejectionReason = replayReason;
                    return result;
                }
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

                // candidate.sample.contribution is an estimator value carrying the SOURCE primary/x2
                // BSDF sampling pdfs. Move the forward pdf ratios into the jacobian so target*jacobian
                // stays in the same solid-angle measure the resampling-MIS uses for every shift.
                // Omitting it for NEE leaves a geometry-only jacobian whose cross-domain density
                // (rcp(jacobian)*source.target in getPairwiseMISWeight) does not cancel the source
                // primary pdf -> the partition over-counts. x2 ratio is 1 for selectedVertex==2.
                if (sourceFirstPdf <= 0.f || shiftedFirstPdf <= 0.f || sourceX2Pdf <= 0.f || shiftedX2Pdf <= 0.f)
                {
                    result.rejectionReason = 13u;
                    return result;
                }
                const float3 finalShiftedContribution = shiftedContribution * (sourceFirstPdf / shiftedFirstPdf) * (sourceX2Pdf / shiftedX2Pdf);
                const float shiftedTarget = max(0.f, luminance(finalShiftedContribution));
                if (shiftedTarget <= 0.f)
                {
                    result.rejectionReason = 10u;
                    return result;
                }

                const float jacobian = computeVertexReconnectJacobian(sourcePrimaryPosW, dstSd.posW, reconnectPosW, reconnectNormalW);
                const float finalJacobian = jacobian * (shiftedFirstPdf / sourceFirstPdf) * (shiftedX2Pdf / sourceX2Pdf);
                if (finalJacobian <= 0.f || isnan(finalJacobian) || isinf(finalJacobian) || finalJacobian > params.maxReconnectJacobian)
                {
                    result.rejectionReason = 13u;
                    return result;
                }

                PathSample shiftedSample = sample;
                shiftedSample.contribution = finalShiftedContribution;
                shiftedSample.target = shiftedTarget;

                result.valid = true;
                result.shiftedContribution = finalShiftedContribution;
                result.shiftedSample = shiftedSample;
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

                    PathSample shiftedSample = sample;
                    shiftedSample.contribution = shiftedContribution;
                    shiftedSample.target = shiftedTarget;

                    result.valid = true;
                    result.shiftedContribution = shiftedContribution;
                    result.shiftedSample = shiftedSample;
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

    /** Reconnect a single-bounce direct NEE sample to the destination primary surface.

        The selected candidate sampled a light directly from the source primary vertex x1. The shift
        reuses the SAME light sample's irradiance (Le / sourceLightPdf) and re-evaluates the BSDF, MIS,
        and visibility for the new edge y1 -> light at the destination primary.

        Measure convention (shared with the other reconnection shifts): the contribution keeps the
        SOURCE light-sampling pdf (the stored irradiance is not re-divided by a destination pdf), and
        the Jacobian is the geometric reconnection determinant of the light edge,
        computeTerminalReconnectGeometryJacobian = (cos_dst/cos_src) * (dist_src^2/dist_dst^2) -- the
        same solid-angle measure getPairwiseMISWeight assumes. A unit Jacobian would be internally
        consistent for an area-measure target but inconsistent with the geometry-based Jacobians of the
        other shifts, breaking the MIS partition. shiftedTarget * jacobian is the destination NEE
        estimator either way.
    */
    ShiftMappingResult evaluateDirectNEECandidate(PathReservoir candidate, ShadingData dstSd, HitInfo dstHit, uint2 dstPixel)
    {
        ShiftMappingResult result = {};
        result.originalTarget = candidate.target;

        const PathSample sample = candidate.sample;
        if ((sample.flags & kPathSampleFlagCanReconnectDirectNEE) == 0u ||
            sample.reusableLightValid == 0u ||
            sample.selectedKind != kPathSampleKindNEE ||
            sample.selectedVertex != 1u ||
            sample.reusableLightPdf <= 0.f)
        {
            result.rejectionReason = 8u;
            return result;
        }

        // Destination primary roughness gate (symmetric with the source primaryRoughHit gate). A
        // near-specular destination cannot realize the reconnected NEE edge with meaningful support.
        let lod = ExplicitLodTextureSampler(0.f);
        let hints = getMaterialInstanceHints(dstHit, true);
        const IMaterialInstance mi = gScene.materials.getMaterialInstance(dstSd, lod, hints);
        const BSDFProperties dstBsdfProperties = mi.getProperties(dstSd);
        if (dstBsdfProperties.roughness <= params.specularRoughnessThreshold)
        {
            result.rejectionReason = 14u;
            return result;
        }

        // Recover the destination edge geometry (direction/distance/visibility-ray origin) and validate
        // light-facing/hemisphere support at the destination. The dest-pdf lightFactor it returns is
        // discarded: this shift reuses the SOURCE light-sampling pdf (see measure convention above).
        float3 origin;
        float3 dir;
        float distance;
        float3 lightFactorDst;
        float lightPdfDst;
        if (!evaluateStoredLightGeometryAtVertex(sample, dstSd, origin, dir, distance, lightFactorDst, lightPdfDst, result.rejectionReason))
            return result;

        // Geometric reconnection Jacobian for the reused light edge (unit for env/directional). Paired
        // with the source-pdf irradiance below, shiftedTarget * jacobian equals the destination NEE
        // estimator while staying in the same measure as the other shifts.
        const float jacobian = computeTerminalReconnectGeometryJacobian(sample, dstSd);
        if (jacobian <= 0.f || isnan(jacobian) || isinf(jacobian) || jacobian > params.maxReconnectJacobian)
        {
            result.rejectionReason = 13u;
            return result;
        }

        // Reuse the source light sample's irradiance Le / sourceLightPdf (reusableLightLi is the pre-MIS
        // radiance Le), re-deriving the NEE MIS weight with the destination scatter pdf. Unlike
        // evaluateStoredNEEAtVertex, which re-divides by the DESTINATION light pdf, keeping the SOURCE pdf
        // here is what carries the cross-domain density into the Jacobian.
        TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(dstPixel, params.frameCount);
        const float dstScatterPdf = mi.evalPdf(dstSd, dir, kUseBSDFSampling);
        float3 lightFactor = sample.reusableLightLi / sample.reusableLightPdf;
        if ((sample.reusableLightFlags & kReusableLightFlagUseMIS) != 0u)
        {
            lightFactor *= evalMIS(1.f, sample.reusableLightPdf, 1.f, dstScatterPdf);
        }

        const float3 shiftedContribution = mi.eval(dstSd, dir, sg) * lightFactor;
        const float shiftedTarget = max(0.f, luminance(shiftedContribution));
        if (shiftedTarget <= 0.f)
        {
            result.rejectionReason = 10u;
            return result;
        }

        // Visibility from the destination primary surface to the stored light sample. Unlike the replay
        // helpers this destination edge is new and its occlusion is independent of the source path.
        SceneRayQuery<kUseAlphaTest> rayQuery;
        const Ray ray = Ray(origin, dir, 0.f, distance);
        if (!rayQuery.traceVisibilityRay(ray))
        {
            result.rejectionReason = 11u;
            return result;
        }

        // Copy the whole sample so the routing fields (CanReconnectDirectNEE, selectedKind/Vertex,
        // reusableLight*) survive: the spatial reverse (center->neighbor) shift re-dispatches on them and
        // must land back in this sub-shift for the forward/reverse pair to stay invertible.
        PathSample shiftedSample = sample;
        shiftedSample.contribution = shiftedContribution;
        shiftedSample.target = shiftedTarget;
        shiftedSample.reusableScatterPdf = dstScatterPdf;
        shiftedSample.reusableLightDistance = distance;

        result.valid = true;
        result.shiftedContribution = shiftedContribution;
        result.shiftedSample = shiftedSample;
        result.shiftedTarget = shiftedTarget;
        result.targetRatio = candidate.target > 0.f ? shiftedTarget / candidate.target : 0.f;
        result.jacobian = jacobian;
        result.rejectionReason = 0u;
        return result;
    }

    ShiftMappingResult applyReconnectionShift(PathReservoir source, ShadingData dstSd, HitInfo dstHit, uint2 dstPixel)
    {
        // Routed to the first matching sub-shift. Direct-NEE (NEE at vertex 1) and replay (NeedsReplay,
        // vertex >= 2) are disjoint by selectedVertex; the terminal-BSDF branch takes the remaining
        // hitLight/env terminals. Replay is checked before terminal so a hitLight that needs a replayed
        // prefix is not mis-routed to the plain terminal shift.
        if (kEnableDirectNEEShift && (source.sample.flags & kPathSampleFlagCanReconnectDirectNEE) != 0u)
            return evaluateDirectNEECandidate(source, dstSd, dstHit, dstPixel);

        if (kEnableReplayShift && (source.sample.flags & kPathSampleFlagNeedsReplay) != 0u)
            return evaluateReplayReconnection(source, dstSd, dstHit, dstPixel);

        if (source.sample.selectedKind == kPathSampleKindHitLight || source.sample.selectedKind == kPathSampleKindEnv)
            return evaluateTerminalBSDFCandidate(source, dstSd, dstHit, dstPixel);

        ShiftMappingResult result = {};
        result.originalTarget = source.target;
        result.rejectionReason = 8u;
        return result;
    }
}
