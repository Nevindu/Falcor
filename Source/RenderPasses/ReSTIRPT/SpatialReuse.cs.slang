/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

import Params;
import PathState;
import LoadShadingData;
import Utils.Math.MathHelpers;
import ReSTIRPT;
import ShiftMappings;

ParameterBlock<ReSTIRPT> gReSTIRPT;

#ifndef RESTIRPT_SPATIAL_MIS_STRATEGY
#define RESTIRPT_SPATIAL_MIS_STRATEGY 0xffffffff
#endif

/** Spatial reuse for ReSTIR PT.

    The current pixel's reservoir is the defensive/canonical identity candidate.
    Neighbor reservoirs are evaluated through the runtime-selected shift mapping.
    This pass owns only reservoir/MIS plumbing; the mapping-specific path-space
    evaluation lives on ReSTIRPT.applyShiftMapping().
*/
struct SpatialReusePass
{
    ReSTIRPTParams params;

    StructuredBuffer<PathReservoir> inputReservoirs;
    RWStructuredBuffer<PathReservoir> outputReservoirs;
    RWTexture2D<float4> outputColor;
    Texture2D<PackedHitInfo> vbuffer;
    Texture2D<float3> viewDir;
    StructuredBuffer<HybridReconnectionData> hybridReconnectionData;

    uint neighborCount;
    uint radius;
    uint debugView;
    uint misStrategy;
    uint hybridDataStride;

    static const bool kUseViewDir = USE_VIEW_DIR;
    static const uint kMaxSpatialCandidates = 33u;
    static const uint kSpatialMISConstant = 0u;
    static const uint kSpatialMISGeneralized = 1u;
    static const uint kSpatialMISPairwise = 2u;

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

    /** Convert a pixel coordinate to the flattened reservoir-buffer index.
    */
    uint reservoirIndex(uint2 pixel)
    {
        return pixel.y * params.frameDim.x + pixel.x;
    }

    uint hybridDataIndex(uint pixelIndex, uint entry)
    {
        return pixelIndex * hybridDataStride + entry;
    }

    HybridReconnectionData getHybridReconnectionData(uint pixelIndex, uint entry)
    {
        if (ShiftMapping(params.shiftMapping) != ShiftMapping::Hybrid)
            return makeInvalidHybridReconnectionData(0u);
        return hybridReconnectionData[hybridDataIndex(pixelIndex, entry)];
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

    /** Convert a shift-mapping result into reservoir state for this proposal.

        The shift implementation owns contribution/target/Jacobian evaluation.
        Spatial reuse only installs the result and localizes metadata so a
        selected neighbor reservoir becomes self-consistent in the destination
        primary domain.
    */
    bool materializeShiftMappingResult(
        PathReservoir sourceCandidate,
        ShiftMappingResult shift,
        ShadingData dstSd,
        float dstDepth,
        uint dstMaterialID,
        out PathReservoir shifted,
        out float shiftedTarget,
        out uint rejectionReason)
    {
        shifted = sourceCandidate;
        shiftedTarget = 0.f;
        rejectionReason = shift.rejectionReason;

        if (!shift.valid) return false;
        if (shift.shiftedTarget <= 0.f || shift.jacobian <= 0.f || isnan(shift.shiftedTarget) || isinf(shift.shiftedTarget) || isnan(shift.jacobian) || isinf(shift.jacobian))
        {
            rejectionReason = 13u;
            return false;
        }

        shifted.sample = shift.shiftedSample;
        shifted.sample.contribution = shift.shiftedContribution;
        shifted.target = shift.shiftedTarget;
        shifted.shiftedTarget = shift.shiftedTarget;
        shifted.targetRatio = shift.targetRatio;
        shifted.jacobian = shift.jacobian;
        shifted.rejectionReason = 0u;
        shiftedTarget = shift.shiftedTarget;

        localizeShiftedSampleMetadata(shifted, dstSd, dstDepth, dstMaterialID, shiftedTarget);
        return true;
    }

    /** GRIS reservoir update weight for an already shifted candidate.

        Constant mode uses the source reservoir confidence M_i directly.

        Generalized/pairwise modes receive a normalized domain MIS weight m_i(y)
        that already includes proposal confidence terms. Because finalizeReservoir()
        divides by the total confidence sum, the update uses totalM * m_i(y).
        When all proposal domains are identical, totalM * m_i(y) reduces to M_i.
    */
    float getReservoirCandidateWeight(PathReservoir reservoir, float misWeight = 1.f, uint totalCandidateCount = 0u)
    {
        if (reservoir.shiftedTarget <= 0.f || reservoir.jacobian <= 0.f || reservoir.W <= 0.f || misWeight <= 0.f) return 0.f;

        if (misStrategy == kSpatialMISConstant)
            return reservoir.shiftedTarget * reservoir.W * float(max(1u, reservoir.M)) * reservoir.jacobian;

        return reservoir.shiftedTarget * reservoir.W * float(max(1u, totalCandidateCount)) * reservoir.jacobian * misWeight;
    }

    /** Estimated normalizer for a reservoir proposal distribution.

        For a RIS reservoir, W = wSum / (pHat(selected) * M), so
        Z ~= wSum / M = pHat(selected) * W. Dividing target terms by this
        normalizer gives the proposal-density proxy used by spatial MIS.
    */
    float getReservoirNormalizer(PathReservoir reservoir)
    {
        const uint reservoirM = getProposalConfidence(reservoir);
        if (reservoirM == 0u || reservoir.wSum <= 0.f) return 0.f;
        return max(1e-8f, reservoir.wSum / float(reservoirM));
    }

    /** Evaluate a source candidate in another proposal domain for MIS terms.

        This uses the same selected shift mapping as the accepted-candidate path,
        so generalized/pairwise denominators see the same proposal support.
    */
    bool evaluateCandidateAtPixel(PathReservoir sourceCandidate, uint2 sourcePixel, uint2 evalPixel, HybridReconnectionData hybridData, out PathReservoir shifted, out uint rejectionReason)
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

        ShiftMappingResult shift = gReSTIRPT.applyShiftMapping(sourceCandidate, evalSd, evalHit, evalPixel, hybridData);
        float shiftedTarget = 0.f;
        return materializeShiftMappingResult(sourceCandidate, shift, evalSd, evalDepth, evalMaterialID, shifted, shiftedTarget, rejectionReason);
    }

    /** Check that an imported sample has reverse support under the selected map.

        Spatial reuse compares proposal densities across domains. If a sample can
        be shifted from neighbor to center but not back into the neighbor domain,
        the domains do not share support and the imported sample is not eligible.
    */
    bool hasReverseShiftSupport(PathReservoir shiftedCandidate, uint2 centerPixel, uint2 sourcePixel, HybridReconnectionData hybridData)
    {
        if (all(centerPixel == sourcePixel)) return true;

        PathReservoir reverseShifted;
        uint reverseReason = 0u;
        return evaluateCandidateAtPixel(shiftedCandidate, centerPixel, sourcePixel, hybridData, reverseShifted, reverseReason);
    }

    /** Generalized balance-heuristic weight over all spatial proposal domains.

        Densities are compared in the current pixel domain by mapping the source
        candidate into each proposal pixel and accounting for the Jacobian ratio.
        The returned value is m_i(y), not an extra M multiplier.
    */
    float getGeneralizedMISWeight(PathReservoir sourceCandidate, PathReservoir shiftedToCenter, uint2 sourcePixel, uint sourceM, uint proposalCount, uint2 proposalPixels[kMaxSpatialCandidates], PathReservoir proposalReservoirs[kMaxSpatialCandidates])
    {
        if (shiftedToCenter.jacobian <= 0.f || sourceCandidate.sample.target <= 0.f) return 0.f;

        const float centerJacobian = shiftedToCenter.jacobian;
        const float sourceZ = getReservoirNormalizer(sourceCandidate);
        const float numerator = sourceZ > 0.f ? (sourceCandidate.sample.target / sourceZ) * rcp(centerJacobian) * float(max(1u, sourceM)) : 0.f;
        if (numerator <= 0.f) return 0.f;
        float denominator = 0.f;

        for (uint j = 0u; j < proposalCount; j++)
        {
            PathReservoir domainShifted;
            uint reason = 0u;
            const uint domainM = getProposalConfidence(proposalReservoirs[j]);
            if (domainM == 0u) continue;

            const float domainZ = getReservoirNormalizer(proposalReservoirs[j]);
            if (domainZ <= 0.f) continue;

            if (all(proposalPixels[j] == sourcePixel))
            {
                denominator += (sourceCandidate.sample.target / domainZ) * rcp(centerJacobian) * float(domainM);
                continue;
            }

            if (!evaluateCandidateAtPixel(sourceCandidate, sourcePixel, proposalPixels[j], makeInvalidHybridReconnectionData(0u), domainShifted, reason))
                continue;

            const float centerToDomainJacobian = domainShifted.jacobian / centerJacobian;
            if (domainShifted.shiftedTarget > 0.f && centerToDomainJacobian > 0.f)
                denominator += (domainShifted.shiftedTarget / domainZ) * centerToDomainJacobian * float(domainM);
        }

        return denominator > 0.f ? numerator / denominator : 0.f;
    }

    /** Pairwise MIS between the canonical proposal and one neighbor at a time.

        The canonical branch averages pairwise canonical weights over neighbors;
        non-canonical branches return their pair contribution divided by the
        number of pairs, not the total proposal count. As with generalized MIS,
        the returned value is m_i(y); getReservoirCandidateWeight() supplies the
        final totalM factor.
    */
    float getPairwiseMISWeight(PathReservoir sourceCandidate, PathReservoir shiftedToCenter, uint2 sourcePixel, uint sourceM, uint proposalCount, uint2 proposalPixels[kMaxSpatialCandidates], PathReservoir proposalReservoirs[kMaxSpatialCandidates])
    {
        if (proposalCount <= 1u) return 1.f;

        const uint canonicalM = getProposalConfidence(proposalReservoirs[0]);
        const float canonicalZ = getReservoirNormalizer(proposalReservoirs[0]);
        const float pairCount = float(max(1u, proposalCount - 1u));

        if (all(sourcePixel == proposalPixels[0]))
        {
            if (canonicalM == 0u || canonicalZ <= 0.f || sourceCandidate.sample.target <= 0.f) return 0.f;

            float canonicalWeight = 0.f;
            const float canonicalPdf = (sourceCandidate.sample.target / canonicalZ) * float(canonicalM);
            for (uint j = 1u; j < proposalCount; j++)
            {
                PathReservoir domainShifted;
                uint reason = 0u;
                if (!evaluateCandidateAtPixel(sourceCandidate, sourcePixel, proposalPixels[j], makeInvalidHybridReconnectionData(0u), domainShifted, reason))
                {
                    canonicalWeight += 1.f;
                    continue;
                }

                const uint domainM = getProposalConfidence(proposalReservoirs[j]);
                const float domainZ = getReservoirNormalizer(proposalReservoirs[j]);
                const float neighborPdf = domainM > 0u && domainZ > 0.f ? (domainShifted.shiftedTarget / domainZ) * domainShifted.jacobian * float(domainM) : 0.f;
                const float pairNormalization = neighborPdf + canonicalPdf;
                canonicalWeight += 1.f - (neighborPdf > 0.f && pairNormalization > 0.f ? neighborPdf / pairNormalization : 0.f);
            }
            return canonicalWeight / pairCount;
        }

        if (shiftedToCenter.jacobian <= 0.f || sourceCandidate.sample.target <= 0.f) return 0.f;

        const float sourceZ = getReservoirNormalizer(sourceCandidate);
        if (sourceZ <= 0.f) return 0.f;

        const float neighborPdf = (sourceCandidate.sample.target / sourceZ) * rcp(shiftedToCenter.jacobian) * float(sourceM);
        const float canonicalPdf = canonicalZ > 0.f ? (shiftedToCenter.shiftedTarget / canonicalZ) * float(canonicalM) : 0.f;
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

        The local reservoir is the canonical self proposal. Neighbor reservoirs
        are evaluated at the destination pixel by the selected shift mapping.
        Rejected candidates do not contribute to the output.
    */
    void execute(uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;

        uint state = hash(pixel.x | (pixel.y << 16) | (params.frameCount * 0x9e3779b9u));
        const uint centerIndex = reservoirIndex(pixel);
        PathReservoir selected = inputReservoirs[centerIndex];

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
        if (evaluateIdentityCandidate(selected, selfShifted, selfShiftRejection))
        {
            candidateCount += getProposalConfidence(selected);
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
            HybridReconnectionData neighborToCenterData = getHybridReconnectionData(centerIndex, 2u * i + 1u);
            ShiftMappingResult shift = gReSTIRPT.applyShiftMapping(sourceCandidate, dstSd, dstHit, pixel, neighborToCenterData);
            if (!materializeShiftMappingResult(sourceCandidate, shift, dstSd, dstDepth, dstMaterialID, shiftedCandidate, shiftedTarget, shiftRejectionReason))
            {
                lastRejectionReason = shiftRejectionReason;
                rejected++;
                continue;
            }

            HybridReconnectionData centerToNeighborData = getHybridReconnectionData(centerIndex, 2u * i);
            if (!hasReverseShiftSupport(shiftedCandidate, pixel, neighborPixel, centerToNeighborData))
            {
                lastRejectionReason = 17u;
                rejected++;
                continue;
            }

            candidateCount += getProposalConfidence(sourceCandidate);
            shiftedCandidates[proposalIndex] = shiftedCandidate;
            proposalAccepted[proposalIndex] = 1u;
            accepted++;
        }

        {
            uint compactCount = 0u;
            uint compactCandidateCount = 0u;
            for (uint i = 0u; i < proposalCount; i++)
            {
                if (proposalAccepted[i] == 0u) continue;

                if (compactCount != i)
                {
                    proposalReservoirs[compactCount] = proposalReservoirs[i];
                    shiftedCandidates[compactCount] = shiftedCandidates[i];
                    proposalPixels[compactCount] = proposalPixels[i];
                    proposalDistances[compactCount] = proposalDistances[i];
                    proposalAccepted[compactCount] = proposalAccepted[i];
                }

                compactCandidateCount += getProposalConfidence(proposalReservoirs[compactCount]);
                compactCount++;
            }

            proposalCount = compactCount;
            candidateCount = compactCandidateCount;
        }

        for (uint i = 0u; i < proposalCount; i++)
        {
            if (proposalAccepted[i] == 0u) continue;
            PathReservoir candidate = shiftedCandidates[i];
            const uint sourceM = getProposalConfidence(proposalReservoirs[i]);
            if (sourceM == 0u) continue;
            float misWeight = 1.f;
#if RESTIRPT_SPATIAL_MIS_STRATEGY == 1
            misWeight = getGeneralizedMISWeight(proposalReservoirs[i], candidate, proposalPixels[i], sourceM, proposalCount, proposalPixels, proposalReservoirs);
#elif RESTIRPT_SPATIAL_MIS_STRATEGY == 2
            misWeight = getPairwiseMISWeight(proposalReservoirs[i], candidate, proposalPixels[i], sourceM, proposalCount, proposalPixels, proposalReservoirs);
#elif RESTIRPT_SPATIAL_MIS_STRATEGY == 0xffffffff
            if (misStrategy == kSpatialMISGeneralized)
            {
                misWeight = getGeneralizedMISWeight(proposalReservoirs[i], candidate, proposalPixels[i], sourceM, proposalCount, proposalPixels, proposalReservoirs);
            }
            else if (misStrategy == kSpatialMISPairwise)
            {
                misWeight = getPairwiseMISWeight(proposalReservoirs[i], candidate, proposalPixels[i], sourceM, proposalCount, proposalPixels, proposalReservoirs);
            }
#endif

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
