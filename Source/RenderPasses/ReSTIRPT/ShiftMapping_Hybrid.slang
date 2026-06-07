/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

import ReSTIRPT;
import LoadShadingData;
#ifndef RESTIRPT_ENABLE_HYBRID_K3
#define RESTIRPT_ENABLE_HYBRID_K3 0
#endif
#ifndef RESTIRPT_ENABLE_PREPARED_HYBRID_K3
#define RESTIRPT_ENABLE_PREPARED_HYBRID_K3 0
#endif
#ifndef RESTIRPT_USE_CACHED_PREPARED_HYBRID_K3
#define RESTIRPT_USE_CACHED_PREPARED_HYBRID_K3 0
#endif
#ifndef RESTIRPT_HYBRID_SHIFT_MODE
#define RESTIRPT_HYBRID_SHIFT_MODE 0xffffffff
#endif

#define RESTIRPT_HYBRID_SHIFT_MODE_FIXED_X2 0
#define RESTIRPT_HYBRID_SHIFT_MODE_K3_PREFIX 1
#define RESTIRPT_HYBRID_SHIFT_MODE_K3_FALLBACK 2
#define RESTIRPT_HYBRID_SHIFT_MODE_RUNTIME 0xffffffff

#if RESTIRPT_ENABLE_HYBRID_K3 || (RESTIRPT_ENABLE_PREPARED_HYBRID_K3 && !RESTIRPT_USE_CACHED_PREPARED_HYBRID_K3)
import Rendering.Materials.TexLODHelpers;
import Scene.RaytracingInline;
#endif
import ShiftMapping_Reconnection;
#if RESTIRPT_ENABLE_HYBRID_K3 || (RESTIRPT_ENABLE_PREPARED_HYBRID_K3 && !RESTIRPT_USE_CACHED_PREPARED_HYBRID_K3)
import Utils.Color.ColorHelpers;
import Utils.Sampling.TinyUniformSampleGenerator;
#endif

static const uint kHybridStageAFixedReconnectVertex = 2u;
static const uint kHybridPreparedK3ReconnectVertex = 3u;
#if RESTIRPT_ENABLE_HYBRID_K3
static const uint kHybridStageDK3ReconnectVertex = 3u;
#endif

/** Hybrid random-replay + reconnection shift mapping.

    Current status:
      Stage A is implemented: hybrid evaluates the fixed-x2 reconnection map as
      the k = 2 base case. This keeps the shifted-domain GRIS plumbing live while
      deeper random-replay prefixes are added in later stages.

    Goal:
      Replay a destination prefix with random replay, reconnect that prefix to a
      selected source vertex x_k, then reuse the source suffix:

          X = (x0, x1, x2, ..., x_{k-1}, x_k, x_{k+1}, ...)
          Y = (y0, y1, y2, ..., y_{k-1}, x_k, x_{k+1}, ...)

      Vanilla reconnection is the simple k = 2 case, where the destination
      primary vertex y1 connects directly to x2. Hybrid generalizes the same
      idea by replaying y2..y_{k-1} before reconnecting.

    GRIS requirements:
      - Implement a deterministic mapping Y = T(X) on the subset satisfying the
        hybrid validity conditions.
      - Return invalid/zero weight if any replay, connectibility, visibility,
        support, or invertibility condition fails.
      - The reuse code expects:
            weight = m(Y) * pHat(Y) * source.W * |dT/dX|
        Therefore shiftedContribution is the destination-domain estimator for Y,
        shiftedTarget is pHat(Y), and jacobian is the remaining determinant not
        already folded into shiftedContribution.

    Required source metadata:
      - pathSeed/rngState sufficient to reconstruct the source path.
      - reconnectVertex = k.
      - selectedKind, selectedVertex, selectedCandidate/path-tree identity.
      - sampled BSDF lobe tags along the replayed prefix and reconnection region.
      - NEE/light technique tags if the prefix or selected event uses NEE.
      - source suffix data, or the ability to regenerate the full source path
        exactly from replay state.

    Source-path precomputation once hybrid is implemented:
      1. Generate the source path normally.
      2. Determine the canonical earliest valid reconnect vertex k >= 2.
      3. Store k, replay state, and sampled lobe/technique tags.
      4. Use sampled-lobe roughness for connectibility decisions, not only the
         material's aggregate roughness.

    Connectibility and invertibility checks:
      - Distance condition:
            min(|x_k - x_{k-1}|, |x_k - y_{k-1}|) >= d_min
      - Roughness/support condition:
            the sampled lobe at x_{k-1}, the replayed lobe at y_{k-1}, and the
            sampled lobe at x_k are rough enough and available in both domains.
      - Visibility condition:
            the new edge y_{k-1} -> x_k is visible and geometrically valid.
      - Canonical earliest-k condition:
            no earlier k' < k may satisfy the same hybrid connectibility rules.
      - Symmetry condition:
            the inverse shift from Y must select the same reconnection structure
            and recover X on accepted paths.
      - Reject on failure; do not silently fall back to the unshifted source
        contribution.

    Implementation recipe:
      1. Load k = source.sample.reconnectVertex and reject if k < 2 or missing.
      2. Reconstruct the source path/replay state from pathSeed/rngState.
      3. Start the destination path at dstPixel's primary hit y1.
      4. Random-replay the prefix up to y_{k-1}, consuming the same random
         dimensions and copying required technique/lobe decisions.
      5. Reject if the destination BSDF/medium cannot realize a replayed event.
      6. Reconnect y_{k-1} to x_k: evaluate BSDFs/PDFs for the new edge, trace
         visibility, and apply distance/roughness/support checks.
      7. Reuse the source suffix x_k, x_{k+1}, ... while re-evaluating factors
         that depend on the changed incoming direction at x_k.
      8. Compute shiftedContribution and shiftedTarget in the same convention as
         initial path/RIS samples.
      9. Compute jacobian consistently with shiftedContribution. For the PSS
         hybrid derivation this is the local reconnection determinant expressed
         through the random-number-to-direction mappings at y_{k-1} and x_k,
         including the needed BSDF PDF ratios and directional/area geometry term.
         Do not double-count PDF ratios in both contribution and jacobian.
     10. Reject, rather than clamp, if the jacobian or contribution is zero, NaN,
         Inf, or above a chosen robustness threshold.
     11. Fill shiftedSample with destination-domain metadata and mark it shifted.

    Suggested staged implementation:
      - Stage A: k fixed to 2, numerically matching vanilla reconnection. Done.
      - Stage B: random-replay prefix for k > 2 on diffuse/rough lobes only.
      - Stage C: sampled-lobe roughness and earliest-k selection.
      - Stage D: NEE/light technique tags.
      - Stage E: glossy/refraction/medium support after reference validation.

    Validation:
      - Same-pixel shift must be identity with jacobian = 1.
      - Forward then inverse shift should recover the original path on accepted
        paths.
      - Compare spatial-only accumulated results against a normal path tracer
        before adding temporal reuse.
*/
extension ReSTIRPT
{
    /** Stage-A hybrid implementation.

        Fixed-x2 reconnection is the k = 2 specialization of the hybrid map:

            X = (x0,x1,x2,...)
            Y = (y0,y1,x2,...)

        Keeping this wrapper separate gives the deeper random-replay
        implementation a narrow replacement point without changing the shared
        shift-mapping dispatch or spatial GRIS code again.
    */
    ShiftMappingResult applyHybridFixedX2BaseCase(PathReservoir source, ShadingData dstSd, HitInfo dstHit, uint2 dstPixel)
    {
        return applyReconnectionShift(source, dstSd, dstHit, dstPixel);
    }

    ShiftMappingResult applyHybridFixedX2Fallback(PathReservoir source, ShadingData dstSd, HitInfo dstHit, uint2 dstPixel)
    {
        if ((source.sample.flags & kPathSampleFlagHybridK3) != 0u)
        {
            source.sample.reconnectVertex = kHybridStageAFixedReconnectVertex;
            source.sample.flags &= ~kPathSampleFlagHybridK3;
        }

        return applyHybridFixedX2BaseCase(source, dstSd, dstHit, dstPixel);
    }

#if RESTIRPT_ENABLE_PREPARED_HYBRID_K3 && !RESTIRPT_USE_CACHED_PREPARED_HYBRID_K3
    /** Prepared-prefix hybrid path for general k > 2.

        SpatialPathRetrace random-replays the destination prefix and stores
        y_{k-1}. This evaluator replays the source path to x_k, reconnects
        y_{k-1}->x_k, and reuses the source suffix.
    */
    ShiftMappingResult applyPreparedHybridK3Shift(PathReservoir source, ShadingData dstSd, HitInfo dstHit, uint2 dstPixel, HybridReconnectionData hybridData)
    {
        ShiftMappingResult result = {};
        result.originalTarget = source.target;

        PathSample sample = source.sample;
        const bool selectedIsNEE = sample.selectedKind == kPathSampleKindNEE;
        const bool selectedIsHitLight = sample.selectedKind == kPathSampleKindHitLight;
        const uint reconnectVertex = sample.reconnectVertex;
        if ((sample.flags & kPathSampleFlagHybridK3) == 0u ||
            (sample.flags & kPathSampleFlagNeedsReplay) == 0u ||
            reconnectVertex < kHybridPreparedK3ReconnectVertex ||
            reconnectVertex > kMaxStoredReplayLobes ||
            sample.selectedVertex < reconnectVertex ||
            (!selectedIsNEE && !selectedIsHitLight))
        {
            result.rejectionReason = 8u;
            return result;
        }
        if ((sample.flags & kPathSampleFlagAlreadyShifted) != 0u)
        {
            result.rejectionReason = 15u;
            return result;
        }
        if (!isHybridReconnectionDataValid(hybridData) || getHybridReconnectionDataVertex(hybridData) != reconnectVertex)
        {
            result.rejectionReason = max(18u, getHybridReconnectionDataReason(hybridData));
            return result;
        }
        if (kUseRussianRoulette)
        {
            result.rejectionReason = 14u;
            return result;
        }

        HitInfo rcPrevHit = unpackHitInfo(hybridData.rcPrevHit);
        if (!rcPrevHit.isValid() || !any(hybridData.prefixThroughput > 0.f))
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

        float3 sourcePrefixThroughput = float3(1.f);
        float3 sourcePrevFactor = float3(0.f);
        float sourcePrevPdf = 1.f;
        float3 sourcePrevPosW = float3(0.f);

        for (uint replayVertex = 1u; replayVertex < reconnectVertex; replayVertex++)
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

            ShadingData sourceSd = loadShadingData(path.hit, path.origin, path.dir);
            if (!handleNestedDielectrics(sourceSd, path))
            {
                result.rejectionReason = 14u;
                return result;
            }
            if (isPrimaryHit)
            {
                uint primaryReplayReason = 0u;
                if (!validateReplayPrimarySurface(sample, path.hit, path.origin, path.dir, sourceSd, primaryReplayReason))
                {
                    result.rejectionReason = primaryReplayReason;
                    return result;
                }
            }

            path.origin = sourceSd.computeRayOrigin();
            let sourceLod = createTextureSampler(path, isPrimaryHit);
            let sourceHints = getMaterialInstanceHints(path.hit, isPrimaryHit);
            const IMaterialInstance sourceMi = gScene.materials.getMaterialInstance(sourceSd, sourceLod, sourceHints);
            BSDFProperties sourceProps = sourceMi.getProperties(sourceSd);
            const uint sourceLobes = sourceMi.getLobeTypes(sourceSd);
            if (sourceProps.roughness <= params.specularRoughnessThreshold)
            {
                result.rejectionReason = 14u;
                return result;
            }

            if (!consumeReplayNEE(sourceSd, sourceMi, sourceLobes, false, false, path))
            {
                result.rejectionReason = 15u;
                return result;
            }

            BSDFSample sourceBs = {};
            const bool validSourceScatter = sourceMi.sample(sourceSd, path.sg, sourceBs, kUseBSDFSampling);
            const uint expectedLobe = getSampleReplayLobeTag(sample, replayVertex);
            if (!validSourceScatter ||
                sourceBs.pdf <= 0.f ||
                sourceBs.lobeType != expectedLobe ||
                sourceBs.isLobe(LobeType::Delta) ||
                sourceBs.isLobe(LobeType::Transmission) ||
                !isNonDeltaReflectionLobe(expectedLobe))
            {
                result.rejectionReason = 14u;
                return result;
            }

            if (!any(sourceBs.weight > 0.f))
            {
                result.rejectionReason = 10u;
                return result;
            }

            if (replayVertex + 1u < reconnectVertex)
            {
                sourcePrefixThroughput *= sourceBs.weight;
            }
            else
            {
                sourcePrevPosW = sourceSd.posW;
                sourcePrevFactor = sourceBs.weight * sourceBs.pdf;
                sourcePrevPdf = sourceBs.pdf;
                if (!any(sourcePrevFactor > 0.f))
                {
                    result.rejectionReason = 10u;
                    return result;
                }
            }

            if (!generateScatterRay(sourceBs, sourceSd, sourceMi, path))
            {
                result.rejectionReason = 14u;
                return result;
            }

            float sourceDistance = 0.f;
            if (!traceReplayNextHit(path, sourceDistance))
            {
                result.rejectionReason = 15u;
                return result;
            }
        }

        if (!path.isHit() || path.getVertexIndex() != reconnectVertex)
        {
            result.rejectionReason = 15u;
            return result;
        }

        const HitType rcHitType = path.hit.getType();
        const bool rcIsTriangleHit = rcHitType == HitType::Triangle;
        const bool rcIsCurveHit = kUseCurves && (rcHitType == HitType::Curve);
        const HitType rcPrevHitType = rcPrevHit.getType();
        const bool rcPrevIsTriangleHit = rcPrevHitType == HitType::Triangle;
        const bool rcPrevIsCurveHit = kUseCurves && (rcPrevHitType == HitType::Curve);
        if (!rcIsTriangleHit || rcIsCurveHit || !rcPrevIsTriangleHit || rcPrevIsCurveHit || !path.interiorList.isEmpty() || path.isInsideDielectricVolume())
        {
            result.rejectionReason = 14u;
            return result;
        }

        ShadingData sourceRcSd = loadShadingData(path.hit, path.origin, path.dir);
        if (!handleNestedDielectrics(sourceRcSd, path))
        {
            result.rejectionReason = 14u;
            return result;
        }

        let sourceRcLod = createTextureSampler(path, false);
        let sourceRcHints = getMaterialInstanceHints(path.hit, false);
        const IMaterialInstance sourceRcMi = gScene.materials.getMaterialInstance(sourceRcSd, sourceRcLod, sourceRcHints);
        BSDFProperties sourceRcProps = sourceRcMi.getProperties(sourceRcSd);
        const uint sourceRcLobes = sourceRcMi.getLobeTypes(sourceRcSd);
        if (!isHybridReconnectibleSurface(path, sourceRcProps, sourceRcLobes, rcIsTriangleHit, rcIsCurveHit))
        {
            result.rejectionReason = 14u;
            return result;
        }

        ShadingData rcPrevSd = loadShadingData(rcPrevHit, hybridData.rcPrevOrigin, hybridData.rcPrevWo);
        let rcPrevLod = ExplicitLodTextureSampler(0.f);
        let rcPrevHints = getMaterialInstanceHints(rcPrevHit, false);
        const IMaterialInstance rcPrevMi = gScene.materials.getMaterialInstance(rcPrevSd, rcPrevLod, rcPrevHints);
        BSDFProperties rcPrevProps = rcPrevMi.getProperties(rcPrevSd);
        const uint prevLobe = getSampleReplayLobeTag(sample, reconnectVertex - 1u);
        if (rcPrevProps.roughness <= params.specularRoughnessThreshold ||
            !isNonDeltaReflectionLobe(prevLobe) ||
            !canRealizeReplayLobe(rcPrevMi.getLobeTypes(rcPrevSd), prevLobe))
        {
            result.rejectionReason = 14u;
            return result;
        }

        const float3 rcPrevOrigin = rcPrevSd.computeRayOrigin();
        const float3 rcPrevToRc = sourceRcSd.posW - rcPrevOrigin;
        const float rcPrevToRcDistance = length(rcPrevToRc);
        if (rcPrevToRcDistance <= params.minReconnectDistance)
        {
            result.rejectionReason = 13u;
            return result;
        }
        const float3 rcPrevToRcDir = rcPrevToRc / rcPrevToRcDistance;

        TinyUniformSampleGenerator rcPrevEvalSg = TinyUniformSampleGenerator(dstPixel, params.frameCount ^ (0x9e3779b9u * reconnectVertex));
        float3 shiftedPrevFactor = rcPrevMi.eval(rcPrevSd, rcPrevToRcDir, rcPrevEvalSg);
        float shiftedPrevPdf = rcPrevMi.evalPdf(rcPrevSd, rcPrevToRcDir, kUseBSDFSampling);
        if (!any(shiftedPrevFactor > 0.f) || shiftedPrevPdf <= 0.f)
        {
            result.rejectionReason = 10u;
            return result;
        }

        SceneRayQuery<kUseAlphaTest> visibilityQuery;
        if (!visibilityQuery.traceVisibilityRay(Ray(rcPrevOrigin, rcPrevToRcDir, 0.f, rcPrevToRcDistance)))
        {
            result.rejectionReason = 11u;
            return result;
        }

        ShadingData shiftedRcSd = loadShadingData(path.hit, rcPrevOrigin, rcPrevToRcDir);
        float3 selectedRatio = float3(1.f);
        float3 sourceRcFactor = float3(1.f);
        float3 shiftedRcFactor = float3(1.f);
        float sourceRcPdf = 1.f;
        float shiftedRcPdf = 1.f;

        if (sample.selectedVertex == reconnectVertex && selectedIsNEE)
        {
            uint sourceNEEReason = 0u;
            uint shiftedNEEReason = 0u;
            float3 sourceNEE;
            float3 shiftedNEE;
            const uint2 sourcePixel = uint2(sample.pathID, sample.pathID >> 12) & 0xfff;
            if (!evaluateStoredNEEAtVertex(sample, sourceRcSd, path.hit, sourcePixel, sourceNEE, sourceNEEReason) ||
                !evaluateStoredNEEAtVertex(sample, shiftedRcSd, path.hit, dstPixel, shiftedNEE, shiftedNEEReason))
            {
                result.rejectionReason = shiftedNEEReason != 0u ? shiftedNEEReason : sourceNEEReason;
                return result;
            }
            selectedRatio = safeColorRatio(shiftedNEE, sourceNEE);
        }
        else if (sample.selectedVertex > reconnectVertex)
        {
            path.origin = sourceRcSd.computeRayOrigin();
            if (!consumeReplayNEE(sourceRcSd, sourceRcMi, sourceRcLobes, false, false, path))
            {
                result.rejectionReason = 15u;
                return result;
            }

            BSDFSample sourceRcBs = {};
            const bool validSourceRcScatter = sourceRcMi.sample(sourceRcSd, path.sg, sourceRcBs, kUseBSDFSampling);
            const uint rcLobe = getSampleReplayLobeTag(sample, reconnectVertex);
            if (!validSourceRcScatter ||
                sourceRcBs.pdf <= 0.f ||
                sourceRcBs.lobeType != rcLobe ||
                sourceRcBs.isLobe(LobeType::Delta) ||
                sourceRcBs.isLobe(LobeType::Transmission) ||
                !isNonDeltaReflectionLobe(rcLobe) ||
                !canRealizeReplayLobe(sourceRcLobes, rcLobe))
            {
                result.rejectionReason = 14u;
                return result;
            }

            let shiftedRcLod = ExplicitLodTextureSampler(0.f);
            let shiftedRcHints = getMaterialInstanceHints(path.hit, false);
            const IMaterialInstance shiftedRcMi = gScene.materials.getMaterialInstance(shiftedRcSd, shiftedRcLod, shiftedRcHints);
            BSDFProperties shiftedRcProps = shiftedRcMi.getProperties(shiftedRcSd);
            if (shiftedRcProps.roughness <= params.specularRoughnessThreshold ||
                !canRealizeReplayLobe(shiftedRcMi.getLobeTypes(shiftedRcSd), rcLobe))
            {
                result.rejectionReason = 14u;
                return result;
            }

            TinyUniformSampleGenerator rcEvalSg = TinyUniformSampleGenerator(dstPixel, params.frameCount ^ 0x45d9f3bu ^ reconnectVertex);
            sourceRcFactor = sourceRcBs.weight * sourceRcBs.pdf;
            shiftedRcFactor = shiftedRcMi.eval(shiftedRcSd, sourceRcBs.wo, rcEvalSg);
            sourceRcPdf = sourceRcBs.pdf;
            shiftedRcPdf = shiftedRcMi.evalPdf(shiftedRcSd, sourceRcBs.wo, kUseBSDFSampling);
            if (!any(sourceRcFactor > 0.f) || !any(shiftedRcFactor > 0.f) || shiftedRcPdf <= 0.f)
            {
                result.rejectionReason = 10u;
                return result;
            }
        }

        float3 shiftedContribution = sample.contribution *
            safeColorRatio(hybridData.prefixThroughput, sourcePrefixThroughput) *
            safeColorRatio(shiftedPrevFactor, sourcePrevFactor) *
            safeColorRatio(shiftedRcFactor, sourceRcFactor) *
            selectedRatio;

        if (selectedIsHitLight)
        {
            shiftedContribution *= sourcePrevPdf / shiftedPrevPdf;
            if (sample.selectedVertex > reconnectVertex)
                shiftedContribution *= sourceRcPdf / shiftedRcPdf;
        }

        const float shiftedTarget = max(0.f, luminance(shiftedContribution));
        if (shiftedTarget <= 0.f || isnan(shiftedTarget) || isinf(shiftedTarget))
        {
            result.rejectionReason = 10u;
            return result;
        }

        float jacobian = computeVertexReconnectJacobian(sourcePrevPosW, rcPrevSd.posW, sourceRcSd.posW, sourceRcSd.faceN);
        if (selectedIsHitLight)
        {
            jacobian *= shiftedPrevPdf / sourcePrevPdf;
            if (sample.selectedVertex > reconnectVertex)
                jacobian *= shiftedRcPdf / sourceRcPdf;
        }
        if (jacobian <= 0.f || isnan(jacobian) || isinf(jacobian) || jacobian > params.maxReconnectJacobian)
        {
            result.rejectionReason = 13u;
            return result;
        }

        PathSample shiftedSample = sample;
        shiftedSample.contribution = shiftedContribution;
        shiftedSample.target = shiftedTarget;
        shiftedSample.reconnectVertex = reconnectVertex;
        shiftedSample.reusableScatterPdf = shiftedPrevPdf;

        result.valid = true;
        result.shiftedContribution = shiftedContribution;
        result.shiftedSample = shiftedSample;
        result.shiftedTarget = shiftedTarget;
        result.targetRatio = source.target > 0.f ? shiftedTarget / source.target : 0.f;
        result.jacobian = jacobian;
        result.rejectionReason = 0u;
        return result;
    }
#endif

#if RESTIRPT_ENABLE_PREPARED_HYBRID_K3 && RESTIRPT_USE_CACHED_PREPARED_HYBRID_K3
    ShiftMappingResult applyCachedPreparedHybridK3Shift(PathReservoir source, HybridReconnectionData hybridData)
    {
        ShiftMappingResult result = {};
        result.originalTarget = source.target;

        const uint reconnectVertex = source.sample.reconnectVertex;
        if (reconnectVertex < kHybridPreparedK3ReconnectVertex ||
            !isHybridReconnectionDataValid(hybridData) ||
            getHybridReconnectionDataVertex(hybridData) != reconnectVertex)
        {
            result.rejectionReason = max(18u, getHybridReconnectionDataReason(hybridData));
            return result;
        }

        if (!isHybridCachedShiftValid(hybridData))
        {
            result.rejectionReason = max(18u, getHybridCachedShiftReason(hybridData));
            return result;
        }

        if (hybridData.cachedShiftedTarget <= 0.f || hybridData.cachedJacobian <= 0.f ||
            isnan(hybridData.cachedShiftedTarget) || isinf(hybridData.cachedShiftedTarget) ||
            isnan(hybridData.cachedJacobian) || isinf(hybridData.cachedJacobian))
        {
            result.rejectionReason = 13u;
            return result;
        }

        PathSample shiftedSample = source.sample;
        shiftedSample.contribution = hybridData.cachedShiftedContribution;
        shiftedSample.target = hybridData.cachedShiftedTarget;
        shiftedSample.reconnectVertex = reconnectVertex;
        shiftedSample.reusableScatterPdf = hybridData.cachedReusableScatterPdf;

        result.valid = true;
        result.shiftedContribution = hybridData.cachedShiftedContribution;
        result.shiftedSample = shiftedSample;
        result.shiftedTarget = hybridData.cachedShiftedTarget;
        result.targetRatio = hybridData.cachedTargetRatio;
        result.jacobian = hybridData.cachedJacobian;
        result.rejectionReason = 0u;
        return result;
    }
#endif

#if RESTIRPT_ENABLE_HYBRID_K3
    /** Stage-D trial: one destination random-replay bounce, then reconnect y2->x3.

        This deliberately supports only selected events at x3 on rough,
        non-delta reflection paths. It exercises the first nontrivial hybrid
        structure without hiding deeper-suffix or earliest-k bugs inside one
        large change.
    */
    ShiftMappingResult applyHybridK3ReplayPrefixShift(PathReservoir source, ShadingData dstSd, HitInfo dstHit, uint2 dstPixel)
    {
        ShiftMappingResult result = {};
        result.originalTarget = source.target;

        PathSample sample = source.sample;
        const bool selectedIsNEE = sample.selectedKind == kPathSampleKindNEE;
        const bool selectedIsHitLight = sample.selectedKind == kPathSampleKindHitLight;
        if ((sample.flags & kPathSampleFlagNeedsReplay) == 0u || (!selectedIsNEE && !selectedIsHitLight) || sample.selectedVertex != kHybridStageDK3ReconnectVertex)
        {
            result.rejectionReason = 8u;
            return result;
        }
        if ((sample.flags & kPathSampleFlagAlreadyShifted) != 0u)
        {
            result.rejectionReason = 15u;
            return result;
        }
        if (kUseRussianRoulette)
        {
            // Stage D keeps random-dimension alignment simple. RR support comes
            // with the generalized replay loop.
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

        float3 sourceFirstWeight = float3(0.f);
        float3 shiftedFirstWeight = float3(0.f);
        float3 sourceX2Factor = float3(0.f);
        float3 shiftedY2Factor = float3(0.f);
        float3 sourceX2PosW = float3(0.f);
        float sourceX2Pdf = 1.f;
        float shiftedY2Pdf = 1.f;

        HitInfo y2Hit = {};
        float3 y1Origin = dstSd.computeRayOrigin();
        float3 y1Dir = float3(0.f, 1.f, 0.f);
        float y2Distance = 0.f;

        // Replay source x1 and random-replay the same primary sample from y1.
        {
            if (!path.isHit() || path.getVertexIndex() != 1u)
            {
                result.rejectionReason = 15u;
                return result;
            }

            const HitType hitType = path.hit.getType();
            const bool isTriangleHit = hitType == HitType::Triangle;
            const bool isCurveHit = kUseCurves && (hitType == HitType::Curve);
            if (!isTriangleHit || isCurveHit || !path.interiorList.isEmpty() || path.isInsideDielectricVolume())
            {
                result.rejectionReason = 14u;
                return result;
            }

            ShadingData sourceSd = loadShadingData(path.hit, path.origin, path.dir);
            if (!handleNestedDielectrics(sourceSd, path))
            {
                result.rejectionReason = 14u;
                return result;
            }

            uint primaryReplayReason = 0u;
            if (!validateReplayPrimarySurface(sample, path.hit, path.origin, path.dir, sourceSd, primaryReplayReason))
            {
                result.rejectionReason = primaryReplayReason;
                return result;
            }

            let sourceLod = createTextureSampler(path, true);
            let sourceHints = getMaterialInstanceHints(path.hit, true);
            const IMaterialInstance sourceMi = gScene.materials.getMaterialInstance(sourceSd, sourceLod, sourceHints);
            BSDFProperties sourceProps = sourceMi.getProperties(sourceSd);
            const uint sourceLobes = sourceMi.getLobeTypes(sourceSd);
            if (sourceProps.roughness <= params.specularRoughnessThreshold)
            {
                result.rejectionReason = 14u;
                return result;
            }

            if (!consumeReplayNEE(sourceSd, sourceMi, sourceLobes, false, false, path))
            {
                result.rejectionReason = 15u;
                return result;
            }

            SampleGenerator sharedReplaySg = path.sg;
            BSDFSample sourceBs = {};
            const bool validSourceScatter = sourceMi.sample(sourceSd, path.sg, sourceBs, kUseBSDFSampling);

            let dstLod = ExplicitLodTextureSampler(0.f);
            let dstHints = getMaterialInstanceHints(dstHit, true);
            const IMaterialInstance dstMi = gScene.materials.getMaterialInstance(dstSd, dstLod, dstHints);
            BSDFProperties dstProps = dstMi.getProperties(dstSd);
            BSDFSample shiftedBs = {};
            const bool validShiftedScatter = dstMi.sample(dstSd, sharedReplaySg, shiftedBs, kUseBSDFSampling);

            const uint primaryLobe = getSampleReplayLobeTag(sample, 1u);
            if (!validSourceScatter || !validShiftedScatter)
            {
                result.rejectionReason = 14u;
                return result;
            }
            if (sourceBs.pdf <= 0.f || shiftedBs.pdf <= 0.f ||
                sourceBs.lobeType != primaryLobe || shiftedBs.lobeType != primaryLobe ||
                sourceBs.isLobe(LobeType::Delta) || shiftedBs.isLobe(LobeType::Delta) ||
                sourceBs.isLobe(LobeType::Transmission) || shiftedBs.isLobe(LobeType::Transmission) ||
                dstProps.roughness <= params.specularRoughnessThreshold ||
                !isNonDeltaReflectionLobe(primaryLobe) ||
                !canRealizeReplayLobe(dstMi.getLobeTypes(dstSd), primaryLobe))
            {
                result.rejectionReason = 14u;
                return result;
            }

            sourceFirstWeight = sourceBs.weight;
            shiftedFirstWeight = shiftedBs.weight;
            if (!any(sourceFirstWeight > 0.f) || !any(shiftedFirstWeight > 0.f))
            {
                result.rejectionReason = 10u;
                return result;
            }

            if (!generateScatterRay(sourceBs, sourceSd, sourceMi, path))
            {
                result.rejectionReason = 14u;
                return result;
            }

            float sourceX2Distance = 0.f;
            if (!traceReplayNextHit(path, sourceX2Distance))
            {
                result.rejectionReason = 15u;
                return result;
            }

            y1Dir = shiftedBs.wo;
            SceneRayQuery<kUseAlphaTest> y2Query;
            y2Hit = y2Query.traceRay(Ray(y1Origin, y1Dir, 0.f, kRayTMax), y2Distance);
            if (!y2Hit.isValid() || y2Distance <= params.minReconnectDistance)
            {
                result.rejectionReason = 14u;
                return result;
            }
        }

        // Replay source x2 to recover x3. Destination y2 will reconnect to x3.
        {
            if (!path.isHit() || path.getVertexIndex() != 2u)
            {
                result.rejectionReason = 15u;
                return result;
            }

            const HitType hitType = path.hit.getType();
            const bool isTriangleHit = hitType == HitType::Triangle;
            const bool isCurveHit = kUseCurves && (hitType == HitType::Curve);
            if (!isTriangleHit || isCurveHit || !path.interiorList.isEmpty() || path.isInsideDielectricVolume())
            {
                result.rejectionReason = 14u;
                return result;
            }

            ShadingData sourceX2Sd = loadShadingData(path.hit, path.origin, path.dir);
            sourceX2PosW = sourceX2Sd.posW;
            if (!handleNestedDielectrics(sourceX2Sd, path))
            {
                result.rejectionReason = 14u;
                return result;
            }

            let sourceX2Lod = createTextureSampler(path, false);
            let sourceX2Hints = getMaterialInstanceHints(path.hit, false);
            const IMaterialInstance sourceX2Mi = gScene.materials.getMaterialInstance(sourceX2Sd, sourceX2Lod, sourceX2Hints);
            BSDFProperties sourceX2Props = sourceX2Mi.getProperties(sourceX2Sd);
            const uint sourceX2Lobes = sourceX2Mi.getLobeTypes(sourceX2Sd);
            if (!isFixedX2ReconnectibleSurface(path, sourceX2Props, sourceX2Lobes, isTriangleHit, isCurveHit))
            {
                result.rejectionReason = 14u;
                return result;
            }

            if (!consumeReplayNEE(sourceX2Sd, sourceX2Mi, sourceX2Lobes, false, false, path))
            {
                result.rejectionReason = 15u;
                return result;
            }

            BSDFSample sourceX2Bs = {};
            const bool validSourceX2Scatter = sourceX2Mi.sample(sourceX2Sd, path.sg, sourceX2Bs, kUseBSDFSampling);
            const uint x2Lobe = getSampleReplayLobeTag(sample, 2u);
            if (!validSourceX2Scatter)
            {
                result.rejectionReason = 14u;
                return result;
            }
            if (sourceX2Bs.pdf <= 0.f ||
                sourceX2Bs.lobeType != x2Lobe ||
                sourceX2Bs.isLobe(LobeType::Delta) ||
                sourceX2Bs.isLobe(LobeType::Transmission) ||
                !isNonDeltaReflectionLobe(x2Lobe))
            {
                result.rejectionReason = 14u;
                return result;
            }

            sourceX2Factor = sourceX2Bs.weight * sourceX2Bs.pdf;
            sourceX2Pdf = sourceX2Bs.pdf;
            if (!any(sourceX2Factor > 0.f))
            {
                result.rejectionReason = 10u;
                return result;
            }

            if (!generateScatterRay(sourceX2Bs, sourceX2Sd, sourceX2Mi, path))
            {
                result.rejectionReason = 14u;
                return result;
            }

            float sourceX3Distance = 0.f;
            if (!traceReplayNextHit(path, sourceX3Distance))
            {
                result.rejectionReason = 15u;
                return result;
            }
        }

        if (!path.isHit() || path.getVertexIndex() != kHybridStageDK3ReconnectVertex)
        {
            result.rejectionReason = 15u;
            return result;
        }

        const HitType x3HitType = path.hit.getType();
        const bool x3IsTriangleHit = x3HitType == HitType::Triangle;
        const bool x3IsCurveHit = kUseCurves && (x3HitType == HitType::Curve);
        const HitType y2HitType = y2Hit.getType();
        const bool y2IsTriangleHit = y2HitType == HitType::Triangle;
        const bool y2IsCurveHit = kUseCurves && (y2HitType == HitType::Curve);
        if (!x3IsTriangleHit || x3IsCurveHit || !y2IsTriangleHit || y2IsCurveHit || !path.interiorList.isEmpty() || path.isInsideDielectricVolume())
        {
            result.rejectionReason = 14u;
            return result;
        }

        ShadingData sourceX3Sd = loadShadingData(path.hit, path.origin, path.dir);
        if (!handleNestedDielectrics(sourceX3Sd, path))
        {
            result.rejectionReason = 14u;
            return result;
        }

        ShadingData y2Sd = loadShadingData(y2Hit, y1Origin, y1Dir);
        let y2Lod = ExplicitLodTextureSampler(0.f);
        let y2Hints = getMaterialInstanceHints(y2Hit, false);
        const IMaterialInstance y2Mi = gScene.materials.getMaterialInstance(y2Sd, y2Lod, y2Hints);
        BSDFProperties y2Props = y2Mi.getProperties(y2Sd);
        const uint x2Lobe = getSampleReplayLobeTag(sample, 2u);
        if (y2Props.roughness <= params.specularRoughnessThreshold ||
            !canRealizeReplayLobe(y2Mi.getLobeTypes(y2Sd), x2Lobe))
        {
            result.rejectionReason = 14u;
            return result;
        }

        const float3 y2Origin = y2Sd.computeRayOrigin();
        const float3 y2ToX3 = sourceX3Sd.posW - y2Origin;
        const float y2ToX3Distance = length(y2ToX3);
        if (y2ToX3Distance <= params.minReconnectDistance)
        {
            result.rejectionReason = 13u;
            return result;
        }
        const float3 y2ToX3Dir = y2ToX3 / y2ToX3Distance;

        TinyUniformSampleGenerator y2EvalSg = TinyUniformSampleGenerator(dstPixel, params.frameCount);
        shiftedY2Factor = y2Mi.eval(y2Sd, y2ToX3Dir, y2EvalSg);
        shiftedY2Pdf = y2Mi.evalPdf(y2Sd, y2ToX3Dir, kUseBSDFSampling);
        if (!any(shiftedY2Factor > 0.f) || shiftedY2Pdf <= 0.f)
        {
            result.rejectionReason = 10u;
            return result;
        }

        SceneRayQuery<kUseAlphaTest> visibilityQuery;
        if (!visibilityQuery.traceVisibilityRay(Ray(y2Origin, y2ToX3Dir, 0.f, y2ToX3Distance)))
        {
            result.rejectionReason = 11u;
            return result;
        }

        float3 selectedRatio = float3(1.f);
        if (selectedIsNEE)
        {
            ShadingData shiftedX3Sd = loadShadingData(path.hit, y2Origin, y2ToX3Dir);

            uint sourceNEEReason = 0u;
            uint shiftedNEEReason = 0u;
            float3 sourceNEE;
            float3 shiftedNEE;
            const uint2 sourcePixel = uint2(sample.pathID, sample.pathID >> 12) & 0xfff;
            if (!evaluateStoredNEEAtVertex(sample, sourceX3Sd, path.hit, sourcePixel, sourceNEE, sourceNEEReason) ||
                !evaluateStoredNEEAtVertex(sample, shiftedX3Sd, path.hit, dstPixel, shiftedNEE, shiftedNEEReason))
            {
                result.rejectionReason = shiftedNEEReason != 0u ? shiftedNEEReason : sourceNEEReason;
                return result;
            }
            selectedRatio = safeColorRatio(shiftedNEE, sourceNEE);
        }

        float3 shiftedContribution = sample.contribution *
            safeColorRatio(shiftedFirstWeight, sourceFirstWeight) *
            safeColorRatio(shiftedY2Factor, sourceX2Factor) *
            selectedRatio;

        if (selectedIsHitLight)
        {
            shiftedContribution *= sourceX2Pdf / shiftedY2Pdf;
        }

        const float shiftedTarget = max(0.f, luminance(shiftedContribution));
        if (shiftedTarget <= 0.f || isnan(shiftedTarget) || isinf(shiftedTarget))
        {
            result.rejectionReason = 10u;
            return result;
        }

        float jacobian = computeVertexReconnectJacobian(sourceX2PosW, y2Sd.posW, sourceX3Sd.posW, sourceX3Sd.faceN);
        if (selectedIsHitLight)
        {
            jacobian *= shiftedY2Pdf / sourceX2Pdf;
        }
        if (jacobian <= 0.f || isnan(jacobian) || isinf(jacobian) || jacobian > params.maxReconnectJacobian)
        {
            result.rejectionReason = 13u;
            return result;
        }

        PathSample shiftedSample = sample;
        shiftedSample.contribution = shiftedContribution;
        shiftedSample.target = shiftedTarget;
        shiftedSample.reconnectVertex = kHybridStageDK3ReconnectVertex;
        shiftedSample.reusableScatterPdf = shiftedY2Pdf;

        result.valid = true;
        result.shiftedContribution = shiftedContribution;
        result.shiftedSample = shiftedSample;
        result.shiftedTarget = shiftedTarget;
        result.targetRatio = source.target > 0.f ? shiftedTarget / source.target : 0.f;
        result.jacobian = jacobian;
        result.rejectionReason = 0u;
        return result;
    }
#endif

    ShiftMappingResult applyHybridShift(PathReservoir source, ShadingData dstSd, HitInfo dstHit, uint2 dstPixel, HybridReconnectionData hybridData)
    {
#if RESTIRPT_HYBRID_SHIFT_MODE == RESTIRPT_HYBRID_SHIFT_MODE_FIXED_X2
        return applyHybridFixedX2Fallback(source, dstSd, dstHit, dstPixel);
#else
        const uint reconnectVertex = source.sample.reconnectVertex;
#if RESTIRPT_HYBRID_SHIFT_MODE == RESTIRPT_HYBRID_SHIFT_MODE_RUNTIME
        const HybridShiftMode mode = HybridShiftMode(params.hybridShiftMode);
        if (mode == HybridShiftMode::FixedX2)
            return applyHybridFixedX2Fallback(source, dstSd, dstHit, dstPixel);
#endif

        if (reconnectVertex >= kHybridPreparedK3ReconnectVertex)
        {
#if RESTIRPT_ENABLE_PREPARED_HYBRID_K3 && RESTIRPT_USE_CACHED_PREPARED_HYBRID_K3
            ShiftMappingResult k3Result = applyCachedPreparedHybridK3Shift(source, hybridData);
#if RESTIRPT_HYBRID_SHIFT_MODE == RESTIRPT_HYBRID_SHIFT_MODE_K3_PREFIX
            return k3Result;
#elif RESTIRPT_HYBRID_SHIFT_MODE == RESTIRPT_HYBRID_SHIFT_MODE_K3_FALLBACK
            if (k3Result.valid)
                return k3Result;
#else
            if (mode == HybridShiftMode::K3ReplayPrefix || k3Result.valid)
                return k3Result;
#endif

            return applyHybridFixedX2Fallback(source, dstSd, dstHit, dstPixel);
#elif RESTIRPT_ENABLE_PREPARED_HYBRID_K3
            ShiftMappingResult k3Result = applyPreparedHybridK3Shift(source, dstSd, dstHit, dstPixel, hybridData);
#if RESTIRPT_HYBRID_SHIFT_MODE == RESTIRPT_HYBRID_SHIFT_MODE_K3_PREFIX
            return k3Result;
#elif RESTIRPT_HYBRID_SHIFT_MODE == RESTIRPT_HYBRID_SHIFT_MODE_K3_FALLBACK
            if (k3Result.valid)
                return k3Result;
#else
            if (mode == HybridShiftMode::K3ReplayPrefix || k3Result.valid)
                return k3Result;
#endif

            return applyHybridFixedX2Fallback(source, dstSd, dstHit, dstPixel);
#else
            return applyHybridFixedX2Fallback(source, dstSd, dstHit, dstPixel);
#endif
        }

        if (reconnectVertex <= kHybridStageAFixedReconnectVertex)
            return applyHybridFixedX2BaseCase(source, dstSd, dstHit, dstPixel);

        ShiftMappingResult result = {};
        result.originalTarget = source.target;
        result.rejectionReason = isHybridReconnectionDataValid(hybridData) ? 18u : max(18u, getHybridReconnectionDataReason(hybridData));
        return result;
#endif
    }

    ShiftMappingResult applyHybridShift(PathReservoir source, ShadingData dstSd, HitInfo dstHit, uint2 dstPixel)
    {
        return applyHybridShift(source, dstSd, dstHit, dstPixel, makeInvalidHybridReconnectionData(0u));
    }
}
