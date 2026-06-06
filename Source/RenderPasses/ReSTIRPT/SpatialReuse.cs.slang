/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

import Params;
import PathState;
import LoadShadingData;
import Scene.RaytracingInline;
import Rendering.Lights.EmissiveLightSamplerInterface;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Rendering.Materials.TexLODHelpers;
import Utils.Math.MathHelpers;
import Utils.Sampling.TinyUniformSampleGenerator;
import ReSTIRPT;

ParameterBlock<ReSTIRPT> gReSTIRPT;

/** Spatial reuse for ReSTIR PT.

    The current pixel's reservoir is the defensive/canonical identity candidate.
    Neighbor reservoirs are shifted with a conservative path-space reconnection:
    direct import is limited to BSDF terminal samples at vertex 2, while deeper
    selected NEE/terminal events replay the source path and reconnect the
    destination primary hit to the source x2 tail. Direct primary NEE import is
    intentionally disabled here to avoid falling back to ReSTIR DI behavior.
*/
struct SpatialReusePass
{
    ReSTIRPTParams params;

    StructuredBuffer<PathReservoir> inputReservoirs;
    RWStructuredBuffer<PathReservoir> outputReservoirs;
    RWTexture2D<float4> outputColor;
    Texture2D<PackedHitInfo> vbuffer;
    Texture2D<float3> viewDir;

    uint neighborCount;
    uint radius;
    uint debugView;
    uint misStrategy;

    static const bool kUseViewDir = USE_VIEW_DIR;
    static const bool kUseAlphaTest = USE_ALPHA_TEST;
    static const uint kMaxSpatialCandidates = 33u;
    static const uint kSpatialMISConstant = 0u;
    static const uint kSpatialMISGeneralized = 1u;
    static const uint kSpatialMISPairwise = 2u;
    static const uint kReusableLightEnvMap = 0u;
    static const uint kReusableLightEmissive = 1u;
    static const uint kReusableLightAnalytic = 2u;
    static const uint kReusableLightFlagUseMIS = 1u << 0;
    static const uint kReusableLightFlagSampleUpper = 1u << 1;
    static const uint kReusableLightFlagSampleLower = 1u << 2;

    /** Small deterministic integer hash used for neighbor sampling.

        This is not meant to be a final sampling strategy. It is just enough to
        keep the spatial skeleton deterministic per pixel/frame without adding
        another sample-generator dependency to the compute pass.
    */
    uint hash(uint v)
    {
        v ^= v >> 16;
        v *= 0x7feb352du;
        v ^= v >> 15;
        v *= 0x846ca68bu;
        v ^= v >> 16;
        return v;
    }

    /** Advance the local spatial-pass RNG and return a uniform value in [0,1).
    */
    float nextRand(inout uint state)
    {
        state = hash(state);
        return float(state & 0x00ffffffu) / float(0x01000000u);
    }

    /** Luminance proxy used for scalar reservoir targets in this pass.
    */
    float localLuminance(float3 c)
    {
        return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
    }

    /** Convert a pixel coordinate to the flattened reservoir-buffer index.
    */
    uint reservoirIndex(uint2 pixel)
    {
        return pixel.y * params.frameDim.x + pixel.x;
    }

    /** Reconstruct the source pass's light-type selection probability.

        Stored light PDFs already include light-type selection, so shifted
        evaluation must use the same enabled-light normalization.
    */
    float getLightTypeSelectionProbability(uint lightType)
    {
        float p0 = 0.f;
        float p1 = 0.f;
        float p2 = 0.f;
#if USE_ENV_LIGHT
        p0 = 1.f;
#endif
#if USE_EMISSIVE_LIGHTS
        p1 = 1.f;
#endif
#if USE_ANALYTIC_LIGHTS
        p2 = 1.f;
#endif
        const float sum = p0 + p1 + p2;
        if (sum <= 0.f) return 0.f;

        if (lightType == kReusableLightEnvMap) return p0 / sum;
        if (lightType == kReusableLightEmissive) return p1 / sum;
        if (lightType == kReusableLightAnalytic) return p2 / sum;
        return 0.f;
    }

    /** Evaluate the same two-technique MIS heuristic used by the trace pass.
    */
    float evalMIS(float n0, float p0, float n1, float p1)
    {
        switch (MISHeuristic(kMISHeuristic))
        {
        case MISHeuristic::Balance:
        {
            float q0 = n0 * p0;
            float q1 = n1 * p1;
            return q0 / (q0 + q1);
        }
        case MISHeuristic::PowerTwo:
        {
            float q0 = (n0 * p0) * (n0 * p0);
            float q1 = (n1 * p1) * (n1 * p1);
            return q0 / (q0 + q1);
        }
        case MISHeuristic::PowerExp:
        {
            float q0 = pow(n0 * p0, kMISPowerExponent);
            float q1 = pow(n1 * p1, kMISPowerExponent);
            return q0 / (q0 + q1);
        }
        default:
            return 0.f;
        }
    }

    /** Conservative validity test for a stored reservoir.
    */
    bool isValid(PathReservoir reservoir)
    {
        return (reservoir.sample.flags & kPathSampleFlagValid) != 0u && reservoir.target > 0.f && reservoir.sample.target > 0.f && reservoir.W > 0.f;
    }

    /** Return the proposal-domain sample count used by GRIS MIS weights.
    */
    uint getProposalConfidence(PathReservoir reservoir)
    {
        return isValid(reservoir) ? max(1u, reservoir.M) : 0u;
    }

    /** True for BSDF lobes whose support survives changing the incident edge.
    */
    bool isNonDeltaReflectionLobe(uint lobeType)
    {
        return (lobeType & (uint)LobeType::NonDeltaReflection) != 0u && (lobeType & (uint)LobeType::Transmission) == 0u;
    }

    /** Check whether a destination material can realize a replayed source lobe.
    */
    bool canRealizeReplayLobe(uint availableLobes, uint lobeType)
    {
        return lobeType != 0u && (availableLobes & lobeType) == lobeType;
    }

    /** Fetch the stored sampled lobe at a source path vertex.
    */
    uint getSampleReplayLobeTag(PathSample sample, uint vertex)
    {
        return getReplayLobeTag(sample.scatterLobeTags0, sample.scatterLobeTags1, vertex);
    }

    /** Load destination visible surface data from the V-buffer.
    */
    bool loadPrimarySurface(uint2 pixel, out ShadingData sd, out HitInfo hit, out float3 posW, out float3 normalW, out float depth, out uint materialID)
    {
        sd = {};
        hit = {};
        posW = float3(0.f);
        normalW = float3(0.f, 1.f, 0.f);
        depth = 0.f;
        materialID = 0u;

        hit = unpackHitInfo(vbuffer[pixel]);
        if (!hit.isValid()) return false;

        Ray cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        if (kUseViewDir) cameraRay.dir = -viewDir[pixel];

        sd = loadShadingData(hit, cameraRay.origin, cameraRay.dir);
        uint loadedMaterialID = 0u;
        loadVertexData(hit, cameraRay.origin, cameraRay.dir, loadedMaterialID);

        posW = sd.posW;
        normalW = normalize(sd.faceN);
        depth = distance(cameraRay.origin, sd.posW);
        materialID = loadedMaterialID;
        return true;
    }

    /** Validity check for a source reservoir/domain.

        Rejection reasons:
        0 accepted
        3 destination has no visible surface
        4 source reservoir has no stored primary surface
    */
    uint getCompatibilityRejection(PathReservoir candidate, float3 dstPosW, float3 dstNormalW, float dstDepth, uint dstMaterialID)
    {
        if (!isValid(candidate)) return 1u;
        if (dstDepth <= 0.f) return 3u;
        if (candidate.sample.primaryDepth <= 0.f) return 4u;

        return 0u;
    }

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

            const float selectionPdf = getLightTypeSelectionProbability(kReusableLightEnvMap);
            lightPdf = selectionPdf > 0.f ? selectionPdf * gReSTIRPT.envMapSampler.evalPdf(dir) : 0.f;
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

            const float selectionPdf = getLightTypeSelectionProbability(kReusableLightEmissive);
            lightPdf = selectionPdf > 0.f ? selectionPdf * gReSTIRPT.emissiveSampler.evalPdf(dstSd.posW, dstSd.getOrientedFaceNormal(), upperHemisphere, hit) : 0.f;
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

        The source sample stores the terminal light event and source BSDF PDF.
        This recomputes the destination BSDF/PDF/visibility and returns the
        shifted contribution, target, and pure-reconnection Jacobian.
    */
    bool evaluateTerminalBSDFCandidate(
        PathReservoir candidate,
        ShadingData dstSd,
        HitInfo dstHit,
        uint2 pixel,
        out PathReservoir shifted,
        out float shiftedTarget,
        out uint rejectionReason)
    {
        shifted = candidate;
        shiftedTarget = 0.f;
        rejectionReason = 0u;

        float3 origin;
        float3 dir;
        float distance;
        float3 emission;
        float lightPdf;
        float geometryJacobian;
        if (!evaluateTerminalBSDFGeometry(candidate.sample, dstSd, origin, dir, distance, emission, lightPdf, geometryJacobian, rejectionReason))
            return false;

        const float sourceScatterPdf = candidate.sample.reusableScatterPdf;
        if (sourceScatterPdf <= 0.f)
        {
            rejectionReason = 13u;
            return false;
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
            rejectionReason = 14u;
            return false;
        }

        const float dstScatterPdf = mi.evalPdf(dstSd, dir, kUseBSDFSampling);
        if (dstScatterPdf <= 0.f)
        {
            rejectionReason = 13u;
            return false;
        }

        float3 bsdfWeight = mi.eval(dstSd, dir, sg) / dstScatterPdf;
        if ((candidate.sample.reusableLightFlags & kReusableLightFlagUseMIS) != 0u)
        {
            if (lightPdf <= 0.f)
            {
                rejectionReason = 9u;
                return false;
            }
            bsdfWeight *= evalMIS(1.f, dstScatterPdf, 1.f, lightPdf);
        }

        const float3 shiftedContribution = bsdfWeight * emission;
        shiftedTarget = max(0.f, localLuminance(shiftedContribution));
        if (shiftedTarget <= 0.f)
        {
            rejectionReason = 10u;
            return false;
        }

        SceneRayQuery<kUseAlphaTest> rayQuery;
        const Ray ray = Ray(origin, dir, 0.f, distance);
        if (!rayQuery.traceVisibilityRay(ray))
        {
            rejectionReason = 11u;
            return false;
        }

        shifted.sample.contribution = shiftedContribution;
        shifted.sample.reusableScatterPdf = dstScatterPdf;
        shifted.sample.reusableLightPdf = lightPdf;
        shifted.sample.reusableLightDistance = distance;
        shifted.target = shiftedTarget;
        shifted.shiftedTarget = shiftedTarget;
        shifted.targetRatio = candidate.target > 0.f ? shiftedTarget / candidate.target : 0.f;
        shifted.jacobian = geometryJacobian * dstScatterPdf / sourceScatterPdf;
        if (shifted.jacobian <= 0.f || isnan(shifted.jacobian) || isinf(shifted.jacobian) || shifted.jacobian > params.maxReconnectJacobian)
        {
            rejectionReason = 13u;
            return false;
        }
        shifted.rejectionReason = 0u;
        return true;
    }

    /** Validate the current pixel's canonical sample under the identity map.
    */
    bool evaluateIdentityCandidate(PathReservoir candidate, out PathReservoir shifted, out uint rejectionReason)
    {
        shifted = candidate;
        rejectionReason = 0u;
        if (!isValid(candidate))
        {
            rejectionReason = 1u;
            return false;
        }

        shifted.shiftedTarget = candidate.target;
        shifted.targetRatio = 1.f;
        shifted.jacobian = 1.f;
        shifted.rejectionReason = 0u;
        return true;
    }

    /** Update accepted neighbor metadata so the reservoir lives in dst domain.

        pathID/pathSeed intentionally stay as the source suffix generator; the
        AlreadyShifted flag prevents replaying that generator as a canonical
        primary-prefix sample in later reuse.
    */
    void localizeShiftedSampleMetadata(inout PathReservoir shifted, ShadingData dstSd, float dstDepth, uint dstMaterialID, float shiftedTarget)
    {
        shifted.sample.primaryPosW = dstSd.posW;
        shifted.sample.primaryNormalW = normalize(dstSd.faceN);
        shifted.sample.primaryDepth = dstDepth;
        shifted.sample.primaryMaterialID = dstMaterialID;
        shifted.sample.target = shiftedTarget;
        shifted.sample.flags |= kPathSampleFlagAlreadyShifted;
    }

    /** Evaluate a stored NEE light sample at the destination surface.

        This handles the reusable selected events generated by ReSTIRPT: NEE light
        samples, BSDF-sampled emissive hits, and BSDF-sampled environment hits.
        Unsupported path-tree candidates are rejected here.

        Rejection reasons:
        8 unsupported reusable sample
        9 invalid light geometry or destination light PDF
        10 destination BSDF evaluates to zero
        11 visibility ray blocked
        12 selected event requires source path replay/tail evaluation
        13 invalid reconnection Jacobian
        15 replay self-consistency/RNG mismatch, or an already-shifted replay sample
    */
    bool evaluateReconnectedCandidate(
        PathReservoir candidate,
        ShadingData dstSd,
        HitInfo dstHit,
        uint2 pixel,
        out PathReservoir shifted,
        out float shiftedTarget,
        out uint rejectionReason)
    {
        shifted = candidate;
        shiftedTarget = 0.f;
        rejectionReason = 0u;

        if ((candidate.sample.flags & kPathSampleFlagNeedsReplay) != 0u)
        {
            if ((candidate.sample.flags & kPathSampleFlagAlreadyShifted) != 0u)
            {
                rejectionReason = 15u;
                return false;
            }

            ShiftMappingResult replay = gReSTIRPT.evaluateReplayReconnection(candidate, dstSd, dstHit, pixel);
            rejectionReason = replay.rejectionReason;
            if (!replay.valid) return false;

            shifted.sample.contribution = replay.shiftedContribution;
            shifted.target = replay.shiftedTarget;
            shifted.shiftedTarget = replay.shiftedTarget;
            shifted.targetRatio = replay.targetRatio;
            shifted.jacobian = replay.jacobian;
            shifted.rejectionReason = 0u;
            shiftedTarget = replay.shiftedTarget;
            return true;
        }

        if (candidate.sample.selectedKind == kPathSampleKindHitLight || candidate.sample.selectedKind == kPathSampleKindEnv)
        {
            return evaluateTerminalBSDFCandidate(candidate, dstSd, dstHit, pixel, shifted, shiftedTarget, rejectionReason);
        }

        rejectionReason = 8u;
        return false;
    }

    /** GRIS reservoir update weight for an already shifted candidate.

        Constant mode preserves the baseline merge form. Generalized/pairwise
        modes fold their MIS weight into the shifted p-hat * W * M * |J| term.
    */
    float getReservoirCandidateWeight(PathReservoir reservoir, float misWeight = 1.f, uint totalCandidateCount = 0u)
    {
        if (reservoir.shiftedTarget <= 0.f || reservoir.jacobian <= 0.f || reservoir.W <= 0.f || misWeight <= 0.f) return 0.f;

        if (misStrategy == kSpatialMISConstant)
            return reservoir.shiftedTarget * reservoir.W * float(max(1u, reservoir.M)) * reservoir.jacobian;

        return reservoir.shiftedTarget * reservoir.W * float(max(1u, totalCandidateCount)) * reservoir.jacobian * misWeight;
    }

    /** Evaluate a source-domain candidate in another proposal domain.

        Used by MIS denominator terms. It returns a transient shifted reservoir
        and deliberately does not localize/store destination metadata.
    */
    bool evaluateCandidateAtPixel(PathReservoir sourceCandidate, uint2 sourcePixel, uint2 evalPixel, out PathReservoir shifted, out uint rejectionReason)
    {
        shifted = sourceCandidate;
        rejectionReason = 0u;

        if (all(sourcePixel == evalPixel))
        {
            return evaluateIdentityCandidate(sourceCandidate, shifted, rejectionReason);
        }

        ShadingData evalSd;
        HitInfo evalHit;
        float3 evalPosW;
        float3 evalNormalW;
        float evalDepth;
        uint evalMaterialID;
        if (!loadPrimarySurface(evalPixel, evalSd, evalHit, evalPosW, evalNormalW, evalDepth, evalMaterialID))
        {
            rejectionReason = 3u;
            return false;
        }

        rejectionReason = getCompatibilityRejection(sourceCandidate, evalPosW, evalNormalW, evalDepth, evalMaterialID);
        if (rejectionReason != 0u) return false;

        float shiftedTarget = 0.f;
        return evaluateReconnectedCandidate(sourceCandidate, evalSd, evalHit, evalPixel, shifted, shiftedTarget, rejectionReason);
    }

    /** Generalized balance-heuristic weight over all spatial proposal domains.

        Densities are compared in the current pixel domain by mapping the source
        candidate into each proposal pixel and accounting for the Jacobian ratio.
    */
    float getGeneralizedMISWeight(PathReservoir sourceCandidate, PathReservoir shiftedToCenter, uint2 sourcePixel, uint sourceM, uint proposalCount, uint2 proposalPixels[kMaxSpatialCandidates], PathReservoir proposalReservoirs[kMaxSpatialCandidates])
    {
        if (shiftedToCenter.jacobian <= 0.f || sourceCandidate.sample.target <= 0.f) return 0.f;

        const float centerJacobian = shiftedToCenter.jacobian;
        const float numerator = sourceCandidate.sample.target * rcp(centerJacobian) * float(max(1u, sourceM));
        float denominator = 0.f;

        for (uint j = 0u; j < proposalCount; j++)
        {
            PathReservoir domainShifted;
            uint reason = 0u;
            const uint domainM = getProposalConfidence(proposalReservoirs[j]);
            if (domainM == 0u) continue;

            if (all(proposalPixels[j] == sourcePixel))
            {
                denominator += sourceCandidate.sample.target * rcp(centerJacobian) * float(domainM);
                continue;
            }

            if (!evaluateCandidateAtPixel(sourceCandidate, sourcePixel, proposalPixels[j], domainShifted, reason))
                continue;

            const float centerToDomainJacobian = domainShifted.jacobian / centerJacobian;
            if (domainShifted.shiftedTarget > 0.f && centerToDomainJacobian > 0.f)
                denominator += domainShifted.shiftedTarget * centerToDomainJacobian * float(domainM);
        }

        return denominator > 0.f ? numerator / denominator : 0.f;
    }

    /** Pairwise MIS between the canonical proposal and one neighbor at a time.

        The canonical branch averages pairwise canonical weights over neighbors;
        non-canonical branches return their pair contribution divided by the
        number of pairs, not the total proposal count.
    */
    float getPairwiseMISWeight(PathReservoir sourceCandidate, PathReservoir shiftedToCenter, uint2 sourcePixel, uint sourceM, uint proposalCount, uint2 proposalPixels[kMaxSpatialCandidates], PathReservoir proposalReservoirs[kMaxSpatialCandidates])
    {
        if (proposalCount <= 1u) return 1.f;

        const uint canonicalM = getProposalConfidence(proposalReservoirs[0]);
        const float pairCount = float(max(1u, proposalCount - 1u));

        if (all(sourcePixel == proposalPixels[0]))
        {
            float canonicalWeight = 0.f;
            for (uint j = 1u; j < proposalCount; j++)
            {
                PathReservoir domainShifted;
                uint reason = 0u;
                if (!evaluateCandidateAtPixel(sourceCandidate, sourcePixel, proposalPixels[j], domainShifted, reason))
                {
                    canonicalWeight += 1.f;
                    continue;
                }

                const uint domainM = getProposalConfidence(proposalReservoirs[j]);
                const float neighborPdf = domainM > 0u ? domainShifted.shiftedTarget * domainShifted.jacobian * float(domainM) : 0.f;
                const float canonicalPdf = sourceCandidate.sample.target * float(canonicalM) / pairCount;
                canonicalWeight += 1.f - (neighborPdf > 0.f ? neighborPdf / (neighborPdf + canonicalPdf) : 0.f);
            }
            return canonicalWeight / pairCount;
        }

        if (shiftedToCenter.jacobian <= 0.f || sourceCandidate.sample.target <= 0.f) return 0.f;

        const float neighborPdf = sourceCandidate.sample.target * rcp(shiftedToCenter.jacobian) * float(sourceM);
        const float canonicalPdf = shiftedToCenter.shiftedTarget * float(canonicalM) / pairCount;
        return (neighborPdf + canonicalPdf) > 0.f ? (neighborPdf / (neighborPdf + canonicalPdf)) / pairCount : 0.f;
    }

    /** Beauty contribution from a finalized spatial reservoir.
    */
    float3 evaluateReservoirContribution(PathReservoir reservoir)
    {
        return isValid(reservoir) ? reservoir.sample.contribution * reservoir.W : float3(0.f);
    }

    /** Finish a combined spatial reservoir and attach debug counters.

        The accepted/rejected/source-distance values are stored in explicit
        reservoir diagnostics so replay metadata remains intact.
    */
    PathReservoir finalizeReservoir(PathReservoir reservoir, float weightSum, uint candidateCount, uint accepted, uint rejected, float sourceDistance, uint rejectionReason)
    {
        reservoir.wSum = weightSum;
        reservoir.M = max(1u, candidateCount);
        reservoir.shiftedTarget = reservoir.target;
        reservoir.rejectionReason = isValid(reservoir) ? rejectionReason : 1u;
        reservoir.acceptedCount = accepted;
        reservoir.rejectedCount = rejected;
        reservoir.selectedDistance = sourceDistance;
        reservoir.W = reservoir.target > 0.f ? weightSum / (reservoir.target * float(reservoir.M)) : 0.f;
        return reservoir;
    }

    /** Shared beauty/debug output for the spatial result.

        Debug modes 10-13 are spatial-specific and read diagnostics stored by
        finalizeReservoir().
    */
    float3 getDebugColor(PathReservoir reservoir)
    {
        switch (debugView)
        {
        case 1: return reservoir.target.xxx * 0.02f;
        case 2: return reservoir.W.xxx;
        case 3: return float(reservoir.sample.pathLength).xxx / 10.f;
        case 4: return isValid(reservoir) ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);
        case 5: return float(reservoir.sample.selectedCandidate).xxx / max(1.f, float(radius));
        case 6: return reservoir.shiftedTarget.xxx * 0.02f;
        case 7: return reservoir.targetRatio.xxx;
        case 8: return reservoir.rejectionReason == 0u ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);
        case 9: return float(reservoir.rejectionReason).xxx / 8.f;
        case 10: return float(reservoir.acceptedCount).xxx / max(1.f, float(neighborCount + 1u));
        case 11: return float(reservoir.rejectedCount).xxx / max(1.f, float(neighborCount + 1u));
        case 12: return float(reservoir.M).xxx / max(1.f, float(neighborCount + 1u));
        case 13: return reservoir.selectedDistance.xxx / max(1.f, float(radius));
        case 14: return float(reservoir.rejectionReason).xxx / 8.f;
        case 15: return reservoir.rejectionReason == 0u ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);
        case 16: return reservoir.rejectionReason == 15u ? float3(1.f, 0.75f, 0.f) : float3(0.f);
        default: return evaluateReservoirContribution(reservoir);
        }
    }

    /** Combine the current pixel reservoir with randomly sampled neighbor reservoirs.

        This is a strict connection-shift combine. The local reservoir and every
        imported neighbor reservoir are first reconnected/evaluated at the
        destination pixel. Rejected candidates do not contribute to the output.
    */
    void execute(uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;

        uint state = hash(pixel.x | (pixel.y << 16) | (params.frameCount * 0x9e3779b9u));
        PathReservoir selected = inputReservoirs[reservoirIndex(pixel)];

        // Zero-neighbor spatial reuse is a debugging passthrough for comparing
        // against initial RIS.
        if (neighborCount == 0u)
        {
            outputReservoirs[reservoirIndex(pixel)] = selected;
            outputColor[pixel] = float4(getDebugColor(selected), 1.f);
            return;
        }

        uint candidateCount = 0u;
        uint accepted = 0u;
        uint rejected = 0u;
        uint lastRejectionReason = 0u;
        float weightSum = 0.f;
        float selectedDistance = 0.f;
        PathReservoir proposalReservoirs[kMaxSpatialCandidates];
        PathReservoir shiftedCandidates[kMaxSpatialCandidates];
        uint2 proposalPixels[kMaxSpatialCandidates];
        float proposalDistances[kMaxSpatialCandidates];
        uint proposalAccepted[kMaxSpatialCandidates];
        uint proposalCount = 0u;

        ShadingData dstSd;
        HitInfo dstHit;
        float3 dstPosW;
        float3 dstNormalW;
        float dstDepth;
        uint dstMaterialID;
        const bool dstValid = loadPrimarySurface(pixel, dstSd, dstHit, dstPosW, dstNormalW, dstDepth, dstMaterialID);
        if (!dstValid)
        {
            outputReservoirs[reservoirIndex(pixel)] = selected;
            outputColor[pixel] = float4(getDebugColor(selected), 1.f);
            return;
        }

        // Seed the proposal-domain list with the current pixel's canonical
        // domain. It remains in MIS even if the current reservoir is invalid.
        uint selfShiftRejection = 0u;
        PathReservoir selfShifted;
        proposalReservoirs[proposalCount] = selected;
        shiftedCandidates[proposalCount] = {};
        proposalPixels[proposalCount] = pixel;
        proposalDistances[proposalCount] = 0.f;
        proposalAccepted[proposalCount] = 0u;
        candidateCount += getProposalConfidence(selected);
        if (evaluateIdentityCandidate(selected, selfShifted, selfShiftRejection))
        {
            shiftedCandidates[proposalCount] = selfShifted;
            proposalAccepted[proposalCount] = 1u;
            accepted++;
        }
        else
        {
            lastRejectionReason = selfShiftRejection;
            rejected++;
        }
        proposalCount++;

        // Draw neighbors in a disk around the current pixel and combine valid reservoirs.
        for (uint i = 0u; i < neighborCount; i++)
        {
            float2 u = float2(nextRand(state), nextRand(state));
            float angle = 6.28318530718f * u.x;
            float dist = max(1.f, float(radius) * sqrt(u.y));
            int2 offset = int2(round(float2(cos(angle), sin(angle)) * dist));
            int2 neighborPixelSigned = int2(pixel) + offset;

            if (any(neighborPixelSigned < int2(0, 0)) || any(neighborPixelSigned >= int2(params.frameDim)))
            {
                rejected++;
                continue;
            }

            uint2 neighborPixel = uint2(neighborPixelSigned);
            PathReservoir sourceCandidate = inputReservoirs[reservoirIndex(neighborPixel)];
            if (proposalCount >= kMaxSpatialCandidates)
            {
                rejected++;
                continue;
            }

            const uint proposalIndex = proposalCount;
            proposalReservoirs[proposalIndex] = sourceCandidate;
            shiftedCandidates[proposalIndex] = {};
            proposalPixels[proposalIndex] = neighborPixel;
            proposalDistances[proposalIndex] = length(float2(offset));
            proposalAccepted[proposalIndex] = 0u;
            proposalCount++;
            candidateCount += getProposalConfidence(sourceCandidate);

            const uint rejectionReason = getCompatibilityRejection(sourceCandidate, dstPosW, dstNormalW, dstDepth, dstMaterialID);
            if (rejectionReason != 0u)
            {
                lastRejectionReason = rejectionReason;
                rejected++;
                continue;
            }

            uint shiftRejectionReason = 0u;
            PathReservoir shiftedCandidate;
            float shiftedTarget = 0.f;
            if (!evaluateReconnectedCandidate(sourceCandidate, dstSd, dstHit, pixel, shiftedCandidate, shiftedTarget, shiftRejectionReason))
            {
                lastRejectionReason = shiftRejectionReason;
                rejected++;
                continue;
            }

            localizeShiftedSampleMetadata(shiftedCandidate, dstSd, dstDepth, dstMaterialID, shiftedTarget);
            shiftedCandidates[proposalIndex] = shiftedCandidate;
            proposalAccepted[proposalIndex] = 1u;
            accepted++;
        }

        for (uint i = 0u; i < proposalCount; i++)
        {
            if (proposalAccepted[i] == 0u) continue;
            PathReservoir candidate = shiftedCandidates[i];
            const uint sourceM = getProposalConfidence(proposalReservoirs[i]);
            if (sourceM == 0u) continue;
            float misWeight = 1.f;
            if (misStrategy == kSpatialMISGeneralized)
            {
                misWeight = getGeneralizedMISWeight(proposalReservoirs[i], candidate, proposalPixels[i], sourceM, proposalCount, proposalPixels, proposalReservoirs);
            }
            else if (misStrategy == kSpatialMISPairwise)
            {
                misWeight = getPairwiseMISWeight(proposalReservoirs[i], candidate, proposalPixels[i], sourceM, proposalCount, proposalPixels, proposalReservoirs);
            }

            float candidateWeight = getReservoirCandidateWeight(candidate, misWeight, candidateCount);
            float newWeightSum = weightSum + candidateWeight;

            // Standard weighted reservoir update over the local + neighbor candidates.
            if (newWeightSum > 0.f && nextRand(state) * newWeightSum < candidateWeight)
            {
                selected = candidate;
                selectedDistance = proposalDistances[i];
            }

            weightSum = newWeightSum;
        }

        if (accepted == 0u || weightSum <= 0.f)
        {
            selected = {};
        }

        selected = finalizeReservoir(selected, weightSum, candidateCount, accepted, rejected, selectedDistance, accepted > 0u ? 0u : lastRejectionReason);
        outputReservoirs[reservoirIndex(pixel)] = selected;
        outputColor[pixel] = float4(getDebugColor(selected), 1.f);
    }
};

cbuffer CB
{
    SpatialReusePass gSpatialReusePass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gSpatialReusePass.execute(dispatchThreadId.xy);
}
