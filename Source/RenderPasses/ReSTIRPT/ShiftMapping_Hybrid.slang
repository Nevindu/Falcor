/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

import ReSTIRPT;
import LoadShadingData;
import ShiftMapping_Reconnection;
import Rendering.Lights.EmissiveLightSamplerInterface;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Rendering.Materials.TexLODHelpers;
import Scene.RaytracingInline;
import Utils.Color.ColorHelpers;
import Utils.Sampling.TinyUniformSampleGenerator;

/** Simple no-lobe hybrid shift mapping.

    This is the first complete hybrid implementation for this branch. It keeps
    the reference ReSTIR PT structure at the algorithmic level, but skips the
    lobe-index extended path-space mapping for now:

        source:      x0, x1, ..., x{k-1}, xk, ...
        destination: y0, y1, ..., y{k-1}, xk, ...

    Random replay generates y1..y{k-1} using the source random stream. The local
    edge y{k-1}->xk is then evaluated as a reconnection. Prefix replay has unit
    Jacobian; the returned Jacobian is the local reconnection determinant in the
    same convention as ShiftMapping_Reconnection.slang.
*/
extension ReSTIRPT
{
    /** Evaluate the finite-light PDF used by hit-light MIS when the local
        reconnect edge lands on the selected emissive vertex x_k.
    */
    bool evaluateSimpleHybridHitLightPdf(PathSample sample, ShadingData prevSd, out float lightPdf, out uint rejectionReason)
    {
        lightPdf = 0.f;
        rejectionReason = 0u;

        if (sample.selectedKind != kPathSampleKindHitLight || sample.reusableLightValid == 0u)
        {
            rejectionReason = 8u;
            return false;
        }
        if (sample.reusableLightType != uint(SampledLightType::Emissive))
        {
            rejectionReason = 8u;
            return false;
        }

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
        lightPdf = selectionPdf * emissiveSampler.evalPdf(prevSd.posW, prevSd.getOrientedFaceNormal(), upperHemisphere, hit);
        if (lightPdf <= 0.f)
        {
            rejectionReason = 9u;
            return false;
        }
        return true;
#else
        rejectionReason = 8u;
        return false;
#endif
    }

    /** Replay the source path up to x_k and collect the source-side factors
        needed by the simple hybrid map.
    */
    bool replaySimpleHybridSource(
        PathSample sample,
        out PathState sourceRcPath,
        out HitInfo sourceRcHit,
        out ShadingData sourceRcSd,
        out float3 sourcePrefixThp,
        out float3 sourcePrevPosW,
        out float3 sourcePrevFactor,
        out float sourcePrevPdf,
        out float3 sourceRcFactor,
        out float sourceRcPdf,
        out float3 sourceRcOutDir,
        out uint rejectionReason)
    {
        sourceRcPath = {};
        sourceRcHit = {};
        sourceRcSd = {};
        sourcePrefixThp = float3(0.f);
        sourcePrevPosW = float3(0.f);
        sourcePrevFactor = float3(0.f);
        sourcePrevPdf = 0.f;
        sourceRcFactor = float3(1.f);
        sourceRcPdf = 1.f;
        sourceRcOutDir = float3(0.f, 1.f, 0.f);
        rejectionReason = 0u;

        const uint k = sample.reconnectVertex;
        const bool needRcScatter = sample.selectedVertex > k;
        if (k > kMaxStoredReplayLobes)
        {
            rejectionReason = 14u;
            return false;
        }

        PathState path = {};
        generatePath(sample.pathID, sample.pathSeed, path);
        if (!path.isHit())
        {
            rejectionReason = 3u;
            return false;
        }

        bool prevConnectible = false;
        float3 prevPosW = float3(0.f);
        bool foundSourceK = false;

        [loop]
        for (uint replayVertex = 1u; replayVertex <= kMaxStoredReplayLobes; replayVertex++)
        {
            if (replayVertex > k) break;

            if (!path.isHit() || path.getVertexIndex() != replayVertex)
            {
                rejectionReason = 15u;
                return false;
            }

            const bool isPrimaryHit = replayVertex == 1u;
            const HitType hitType = path.hit.getType();
            const bool isTriangleHit = hitType == HitType::Triangle;
            const bool isCurveHit = kUseCurves && (hitType == HitType::Curve);

            ShadingData sd = loadShadingData(path.hit, path.origin, path.dir);
            if (!handleNestedDielectrics(sd, path))
            {
                rejectionReason = 14u;
                return false;
            }
            if (isPrimaryHit)
            {
                uint replayReason = 0u;
                if (!validateReplayPrimarySurface(sample, path.hit, path.origin, path.dir, sd, replayReason))
                {
                    rejectionReason = replayReason;
                    return false;
                }
            }

            let lod = createTextureSampler(path, isPrimaryHit);
            let hints = getMaterialInstanceHints(path.hit, isPrimaryHit);
            const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);
            BSDFProperties bsdfProperties = mi.getProperties(sd);
            bool isSpecular = bsdfProperties.roughness <= params.specularRoughnessThreshold;
            if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && isSpecular)
                sd.mtl.setActiveLobes((uint)LobeType::Diffuse);
            if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume())
                bsdfProperties.emission = float3(0.f);

            const uint lobeTypes = mi.getLobeTypes(sd);
            const bool currentConnectible = isSimpleHybridReconnectibleSurface(path, bsdfProperties, lobeTypes, isTriangleHit, isCurveHit);
            if (replayVertex >= 2u && prevConnectible && currentConnectible)
            {
                const float edgeDistance = distance(prevPosW, sd.posW);
                if (edgeDistance >= params.minReconnectDistance)
                {
                    if (replayVertex < k)
                    {
                        rejectionReason = 14u;
                        return false;
                    }
                    if (replayVertex == k) foundSourceK = true;
                }
            }

            if (replayVertex == k)
            {
                if (!foundSourceK || !currentConnectible)
                {
                    rejectionReason = 14u;
                    return false;
                }

                sourceRcPath = path;
                sourceRcHit = path.hit;
                sourceRcSd = sd;
            }

            if (replayVertex == k && !needRcScatter)
                return true;

            if (hasFinishedSurfaceBounces(path))
            {
                rejectionReason = 15u;
                return false;
            }

            path.origin = sd.computeRayOrigin();
            if (!consumeReplayNEE(sd, mi, lobeTypes, isCurveHit, false, path))
            {
                rejectionReason = 15u;
                return false;
            }

            if (kUseRussianRoulette)
            {
                if (terminatePathByRussianRoulette(path, sampleNext1D(path.sg)))
                {
                    rejectionReason = 15u;
                    return false;
                }
            }

            BSDFSample bs = {};
            bool validScatter = mi.sample(sd, path.sg, bs, kUseBSDFSampling);
            if (!validScatter || bs.pdf <= 0.f)
            {
                rejectionReason = 14u;
                return false;
            }

            if (replayVertex == k - 1u)
            {
                if (bs.isLobe(LobeType::Delta) || bs.isLobe(LobeType::Transmission))
                {
                    rejectionReason = 14u;
                    return false;
                }
                sourcePrefixThp = path.thp;
                sourcePrevPosW = sd.posW;
                sourcePrevFactor = bs.weight * bs.pdf;
                sourcePrevPdf = bs.pdf;
                if (!any(sourcePrevFactor > 0.f))
                {
                    rejectionReason = 10u;
                    return false;
                }
            }

            if (replayVertex == k)
            {
                if (bs.isLobe(LobeType::Delta) || bs.isLobe(LobeType::Transmission))
                {
                    rejectionReason = 14u;
                    return false;
                }
                sourceRcFactor = bs.weight * bs.pdf;
                sourceRcPdf = bs.pdf;
                sourceRcOutDir = bs.wo;
                if (!any(sourceRcFactor > 0.f))
                {
                    rejectionReason = 10u;
                    return false;
                }
                return true;
            }

            if (!generateScatterRay(bs, sd, mi, path))
            {
                rejectionReason = 14u;
                return false;
            }

            float hitT;
            if (!traceReplayNextHit(path, hitT))
            {
                rejectionReason = 15u;
                return false;
            }

            prevConnectible = currentConnectible;
            prevPosW = sd.posW;
        }

        rejectionReason = 15u;
        return false;
    }

    /** Random-replay the destination prefix through y{k-1}. */
    bool replaySimpleHybridDestinationPrefix(
        PathSample sample,
        ShadingData dstPrimarySd,
        HitInfo dstPrimaryHit,
        uint2 dstPixel,
        out PathState dstPrevPath,
        out HitInfo dstPrevHit,
        out ShadingData dstPrevSd,
        out float3 dstPrefixThp,
        out uint rejectionReason)
    {
        dstPrevPath = {};
        dstPrevHit = {};
        dstPrevSd = {};
        dstPrefixThp = float3(0.f);
        rejectionReason = 0u;

        const uint k = sample.reconnectVertex;
        const uint stopVertex = k - 1u;
        const uint2 sourcePixel = uint2(sample.pathID, sample.pathID >> 12) & 0xfff;
        if (k > kMaxStoredReplayLobes)
        {
            rejectionReason = 14u;
            return false;
        }

        PathState path = {};
        path.setActive();
        path.id = dstPixel.x | (dstPixel.y << 12) | (sample.pathID & 0xff000000u);
        path.pathSeed = sample.pathSeed;
        path.thp = float3(1.f);
        path.reconnectVertex = kInvalidReconnectVertex;

        Ray cameraRay = gScene.camera.computeRayPinhole(dstPixel, params.frameDim);
        if (kUseViewDir) cameraRay.dir = -viewDir[dstPixel];
        path.origin = cameraRay.origin;
        path.dir = cameraRay.dir;
        path.sg = SampleGenerator(sourcePixel, sample.pathSeed);
        path.risState = hashPathState(sample.pathSeed ^ 0xa511e9b3u);
        path.selectedKind = kPathSampleKindInvalid;
        path.setHit(dstPrimaryHit);
        path.setVertexIndex(1);

        bool prevConnectible = false;
        float3 prevPosW = float3(0.f);

        [loop]
        for (uint replayVertex = 1u; replayVertex <= kMaxStoredReplayLobes; replayVertex++)
        {
            if (replayVertex > stopVertex) break;

            if (!path.isHit() || path.getVertexIndex() != replayVertex)
            {
                rejectionReason = 15u;
                return false;
            }

            const bool isPrimaryHit = replayVertex == 1u;
            const HitType hitType = path.hit.getType();
            const bool isTriangleHit = hitType == HitType::Triangle;
            const bool isCurveHit = kUseCurves && (hitType == HitType::Curve);

            ShadingData sd = isPrimaryHit ? dstPrimarySd : loadShadingData(path.hit, path.origin, path.dir);
            if (!handleNestedDielectrics(sd, path))
            {
                rejectionReason = 14u;
                return false;
            }

            let lod = createTextureSampler(path, isPrimaryHit);
            let hints = getMaterialInstanceHints(path.hit, isPrimaryHit);
            const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);
            BSDFProperties bsdfProperties = mi.getProperties(sd);
            bool isSpecular = bsdfProperties.roughness <= params.specularRoughnessThreshold;
            if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && isSpecular)
                sd.mtl.setActiveLobes((uint)LobeType::Diffuse);
            if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume())
                bsdfProperties.emission = float3(0.f);

            const uint lobeTypes = mi.getLobeTypes(sd);
            const bool currentConnectible = isSimpleHybridReconnectibleSurface(path, bsdfProperties, lobeTypes, isTriangleHit, isCurveHit);
            if (replayVertex >= 2u && prevConnectible && currentConnectible)
            {
                const float edgeDistance = distance(prevPosW, sd.posW);
                if (edgeDistance >= params.minReconnectDistance)
                {
                    // First-valid-k invertibility: the shifted prefix must not
                    // contain an earlier valid reconnect vertex.
                    rejectionReason = 14u;
                    return false;
                }
            }

            if (replayVertex == stopVertex)
            {
                if (!currentConnectible)
                {
                    rejectionReason = 14u;
                    return false;
                }

                dstPrevPath = path;
                dstPrevHit = path.hit;
                dstPrevSd = sd;
                dstPrefixThp = path.thp;
                return true;
            }

            if (hasFinishedSurfaceBounces(path))
            {
                rejectionReason = 15u;
                return false;
            }

            path.origin = sd.computeRayOrigin();
            if (!consumeReplayNEE(sd, mi, lobeTypes, isCurveHit, false, path))
            {
                rejectionReason = 15u;
                return false;
            }

            if (kUseRussianRoulette)
            {
                if (terminatePathByRussianRoulette(path, sampleNext1D(path.sg)))
                {
                    rejectionReason = 15u;
                    return false;
                }
            }

            BSDFSample bs = {};
            bool validScatter = mi.sample(sd, path.sg, bs, kUseBSDFSampling);
            if (!validScatter || bs.pdf <= 0.f)
            {
                rejectionReason = 14u;
                return false;
            }
            if (!generateScatterRay(bs, sd, mi, path))
            {
                rejectionReason = 14u;
                return false;
            }

            float hitT;
            if (!traceReplayNextHit(path, hitT))
            {
                rejectionReason = 15u;
                return false;
            }

            prevConnectible = currentConnectible;
            prevPosW = sd.posW;
        }

        rejectionReason = 15u;
        return false;
    }

    ShiftMappingResult evaluateSimpleHybridShift(PathReservoir candidate, ShadingData dstSd, HitInfo dstHit, uint2 dstPixel)
    {
        ShiftMappingResult result = {};
        result.originalTarget = candidate.target;

        PathSample sample = candidate.sample;
        const uint k = sample.reconnectVertex;
        const bool selectedIsNEE = sample.selectedKind == kPathSampleKindNEE;
        const bool selectedIsHitLight = sample.selectedKind == kPathSampleKindHitLight;
        const bool selectedIsEnv = sample.selectedKind == kPathSampleKindEnv;

        if ((sample.flags & kPathSampleFlagNeedsHybridReplay) == 0u || k <= 2u || k > kMaxStoredReplayLobes)
        {
            result.rejectionReason = 8u;
            return result;
        }
        if ((sample.flags & kPathSampleFlagAlreadyShifted) != 0u)
        {
            result.rejectionReason = 15u;
            return result;
        }
        if (sample.selectedVertex < k || (!selectedIsNEE && !selectedIsHitLight && !selectedIsEnv))
        {
            result.rejectionReason = 8u;
            return result;
        }
        if (selectedIsEnv && sample.selectedVertex <= k)
        {
            result.rejectionReason = 14u;
            return result;
        }

        PathState sourceRcPath;
        HitInfo sourceRcHit;
        ShadingData sourceRcSd;
        float3 sourcePrefixThp;
        float3 sourcePrevPosW;
        float3 sourcePrevFactor;
        float sourcePrevPdf;
        float3 sourceRcFactor;
        float sourceRcPdf;
        float3 sourceRcOutDir;
        if (!replaySimpleHybridSource(
            sample,
            sourceRcPath,
            sourceRcHit,
            sourceRcSd,
            sourcePrefixThp,
            sourcePrevPosW,
            sourcePrevFactor,
            sourcePrevPdf,
            sourceRcFactor,
            sourceRcPdf,
            sourceRcOutDir,
            result.rejectionReason))
        {
            return result;
        }

        PathState dstPrevPath;
        HitInfo dstPrevHit;
        ShadingData dstPrevSd;
        float3 dstPrefixThp;
        if (!replaySimpleHybridDestinationPrefix(sample, dstSd, dstHit, dstPixel, dstPrevPath, dstPrevHit, dstPrevSd, dstPrefixThp, result.rejectionReason))
            return result;

        const float3 dstOrigin = dstPrevSd.computeRayOrigin();
        const float3 toRc = sourceRcSd.posW - dstOrigin;
        const float dstDist = length(toRc);
        if (dstDist <= params.minReconnectDistance)
        {
            result.rejectionReason = 13u;
            return result;
        }
        const float3 dstConnectDir = toRc / dstDist;

        SceneRayQuery<kUseAlphaTest> visibilityQuery;
        if (!visibilityQuery.traceVisibilityRay(Ray(dstOrigin, dstConnectDir, 0.f, dstDist)))
        {
            result.rejectionReason = 11u;
            return result;
        }

        const float geometryJacobian = computeVertexReconnectJacobian(sourcePrevPosW, dstPrevSd.posW, sourceRcSd.posW, sourceRcSd.faceN);
        if (geometryJacobian <= 0.f)
        {
            result.rejectionReason = 13u;
            return result;
        }

        let dstPrevLod = createTextureSampler(dstPrevPath, dstPrevPath.getVertexIndex() == 1u);
        let dstPrevHints = getMaterialInstanceHints(dstPrevHit, dstPrevPath.getVertexIndex() == 1u);
        const IMaterialInstance dstPrevMi = gScene.materials.getMaterialInstance(dstPrevSd, dstPrevLod, dstPrevHints);
        TinyUniformSampleGenerator dstPrevEvalSg = TinyUniformSampleGenerator(dstPixel, params.frameCount);

        const float3 shiftedPrevFactor = dstPrevMi.eval(dstPrevSd, dstConnectDir, dstPrevEvalSg);
        const float shiftedPrevPdf = dstPrevMi.evalPdf(dstPrevSd, dstConnectDir, kUseBSDFSampling);
        if (!any(shiftedPrevFactor > 0.f) || shiftedPrevPdf <= 0.f || sourcePrevPdf <= 0.f)
        {
            result.rejectionReason = shiftedPrevPdf <= 0.f || sourcePrevPdf <= 0.f ? 13u : 10u;
            return result;
        }

        ShadingData shiftedRcSd = loadShadingData(sourceRcHit, dstOrigin, dstConnectDir);
        let shiftedRcLod = ExplicitLodTextureSampler(params.lodBias);
        let shiftedRcHints = getMaterialInstanceHints(sourceRcHit, false);
        const IMaterialInstance shiftedRcMi = gScene.materials.getMaterialInstance(shiftedRcSd, shiftedRcLod, shiftedRcHints);
        const BSDFProperties shiftedRcProperties = shiftedRcMi.getProperties(shiftedRcSd);
        const uint shiftedRcLobes = shiftedRcMi.getLobeTypes(shiftedRcSd);
        const bool shiftedRcConnectible = isSimpleHybridReconnectibleSurface(sourceRcPath, shiftedRcProperties, shiftedRcLobes, sourceRcHit.getType() == HitType::Triangle, kUseCurves && sourceRcHit.getType() == HitType::Curve);
        if (!shiftedRcConnectible)
        {
            result.rejectionReason = 14u;
            return result;
        }

        float3 shiftedContribution = float3(0.f);
        float finalJacobian = 0.f;

        if (sample.selectedVertex == k)
        {
            finalJacobian = geometryJacobian * (shiftedPrevPdf / sourcePrevPdf);
            if (selectedIsNEE)
            {
                uint sourceNEEReason = 0u;
                uint shiftedNEEReason = 0u;
                float3 sourceNEE;
                float3 shiftedNEE;
                const uint2 sourcePixel = uint2(sample.pathID, sample.pathID >> 12) & 0xfff;
                if (!evaluateStoredNEEAtVertex(sample, sourceRcSd, sourceRcHit, sourcePixel, sourceNEE, sourceNEEReason) ||
                    !evaluateStoredNEEAtVertex(sample, shiftedRcSd, sourceRcHit, dstPixel, shiftedNEE, shiftedNEEReason))
                {
                    result.rejectionReason = shiftedNEEReason != 0u ? shiftedNEEReason : sourceNEEReason;
                    return result;
                }

                shiftedContribution = dstPrefixThp * (shiftedPrevFactor / shiftedPrevPdf) * shiftedNEE;
            }
            else if (selectedIsHitLight)
            {
                float lightPdf = 0.f;
                if (!evaluateSimpleHybridHitLightPdf(sample, dstPrevSd, lightPdf, result.rejectionReason))
                    return result;

                float3 bsdfWeight = shiftedPrevFactor / shiftedPrevPdf;
                if ((sample.reusableLightFlags & kReusableLightFlagUseMIS) != 0u)
                    bsdfWeight *= evalMIS(1.f, shiftedPrevPdf, 1.f, lightPdf);

                shiftedContribution = dstPrefixThp * bsdfWeight * sample.reusableLightLi;
            }
            else
            {
                result.rejectionReason = 14u;
                return result;
            }
        }
        else
        {
            TinyUniformSampleGenerator shiftedRcEvalSg = TinyUniformSampleGenerator(dstPixel, params.frameCount);
            const float3 shiftedRcFactor = shiftedRcMi.eval(shiftedRcSd, sourceRcOutDir, shiftedRcEvalSg);
            const float shiftedRcPdf = shiftedRcMi.evalPdf(shiftedRcSd, sourceRcOutDir, kUseBSDFSampling);
            if (!any(shiftedRcFactor > 0.f) || shiftedRcPdf <= 0.f || sourceRcPdf <= 0.f)
            {
                result.rejectionReason = shiftedRcPdf <= 0.f || sourceRcPdf <= 0.f ? 13u : 10u;
                return result;
            }

            shiftedContribution = candidate.sample.contribution *
                safeColorRatio(dstPrefixThp, sourcePrefixThp) *
                safeColorRatio(shiftedPrevFactor, sourcePrevFactor) *
                safeColorRatio(shiftedRcFactor, sourceRcFactor) *
                (sourcePrevPdf / shiftedPrevPdf) *
                (sourceRcPdf / shiftedRcPdf);

            finalJacobian = geometryJacobian *
                (shiftedPrevPdf / sourcePrevPdf) *
                (shiftedRcPdf / sourceRcPdf);
        }

        const float shiftedTarget = max(0.f, luminance(shiftedContribution));
        if (shiftedTarget <= 0.f || isnan(shiftedTarget) || isinf(shiftedTarget) || !any(shiftedContribution > 0.f))
        {
            result.rejectionReason = 10u;
            return result;
        }
        if (finalJacobian <= 0.f || isnan(finalJacobian) || isinf(finalJacobian) || finalJacobian > params.maxReconnectJacobian)
        {
            result.rejectionReason = 13u;
            return result;
        }

        PathSample shiftedSample = sample;
        shiftedSample.contribution = shiftedContribution;
        shiftedSample.target = shiftedTarget;
        shiftedSample.reusableScatterPdf = sample.selectedVertex == k ? shiftedPrevPdf : shiftedSample.reusableScatterPdf;

        result.valid = true;
        result.shiftedContribution = shiftedContribution;
        result.shiftedSample = shiftedSample;
        result.shiftedTarget = shiftedTarget;
        result.targetRatio = candidate.target > 0.f ? shiftedTarget / candidate.target : 0.f;
        result.jacobian = finalJacobian;
        result.rejectionReason = 0u;
        return result;
    }

    ShiftMappingResult applyHybridShift(PathReservoir source, ShadingData dstSd, HitInfo dstHit, uint2 dstPixel)
    {
        if ((source.sample.flags & kPathSampleFlagNeedsHybridReplay) != 0u && source.sample.reconnectVertex > 2u)
            return evaluateSimpleHybridShift(source, dstSd, dstHit, dstPixel);

        // The fixed-x2 case and direct-NEE samples remain exactly the production
        // reconnection implementation. Hybrid is a generalization, not a
        // replacement for those shorter maps.
        return applyReconnectionShift(source, dstSd, dstHit, dstPixel);
    }
}
