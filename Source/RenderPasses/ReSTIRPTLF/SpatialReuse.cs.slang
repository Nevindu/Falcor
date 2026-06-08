/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

import Params;
import PathState;
import LoadShadingData;
import Utils.Math.MathHelpers;
import ReSTIRPTLF;
import ShiftMappings;
import LightField;

ParameterBlock<ReSTIRPTLF> gReSTIRPTLF;

/** Spatial reuse for ReSTIR PT.

    The current pixel's reservoir is the defensive/canonical identity candidate.
    Neighbor reservoirs are evaluated through the runtime-selected shift mapping.
    This pass owns only reservoir/MIS plumbing; the mapping-specific path-space
    evaluation lives on ReSTIRPTLF.applyShiftMapping().
*/
struct SpatialReusePass
{
    ReSTIRPTLFParams params;

    StructuredBuffer<PathReservoir> inputReservoirs;
    RWStructuredBuffer<PathReservoir> outputReservoirs;
    RWTexture2D<float4> outputColor;
    Texture2D<PackedHitInfo> vbuffer;
    Texture2D<float3> viewDir;

    uint neighborCount;
    uint radius;
    uint debugView;
    uint misStrategy;
    uint featureBasedRejection;
    uint angularReuse;

    static const bool kUseViewDir = USE_VIEW_DIR;
    // Geometry-similarity thresholds for feature-based neighbor rejection
    // (normal dot >= 0.5, relative depth within 10%).
    static const float kFeatureNormalThreshold = 0.5f;
    static const float kFeatureRelativeDepthThreshold = 0.1f;
    static const uint kMaxSpatialCandidates = 33u;
    static const uint kSpatialMISConstant = 0u;
    static const uint kSpatialMISGeneralized = 1u;
    static const uint kSpatialMISPairwise = 2u;

    // Admit a neighbor only if BOTH the forward (neighbor->center) and reverse
    // (center->neighbor) shifts succeed. Asymmetric support (one direction fails) unbalances
    // the defensive pairwise -> Sum m_i != 1 -> systematic bias. The reverse target/jacobian
    // is cached per neighbor so the canonical MIS term does not re-trace it.
    static const bool kEnforceShiftInvertibility = true;

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

        Ray cameraRay = computeLightFieldRay(params, pixel);
        if (params.lfEnabled == 0u && kUseViewDir) cameraRay.dir = -viewDir[pixel];

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
        5 neighbor surface is too geometrically dissimilar (feature-based rejection)
    */
    uint getCompatibilityRejection(PathReservoir candidate, float3 dstPosW, float3 dstNormalW, float dstDepth, uint dstMaterialID)
    {
        if (!isValid(candidate)) return 1u;
        if (dstDepth <= 0.f) return 3u;
        if (candidate.sample.primaryDepth <= 0.f) return 4u;

        // Skip neighbors whose primary surface differs too much in orientation or camera
        // distance: keeps high-variance, low-support shifts (a firefly source) out of the
        // reuse set. Unbiased -- a rejected domain simply does not participate.
        if (angularReuse != 0u && params.lfEnabled != 0u)
        {
            if (dot(dstNormalW, candidate.sample.primaryNormalW) < params.lfNormalThreshold) return 5u;
            if (abs(dstDepth - candidate.sample.primaryDepth) >= params.lfDepthRelThreshold * max(dstDepth, 1e-6f)) return 5u;
        }
        else if (featureBasedRejection != 0u)
        {
            if (dot(dstNormalW, candidate.sample.primaryNormalW) < kFeatureNormalThreshold) return 5u;
            if (abs(dstDepth - candidate.sample.primaryDepth) >= kFeatureRelativeDepthThreshold * dstDepth) return 5u;
        }

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

        ShiftMappingResult shift = gReSTIRPTLF.applyShiftMapping(sourceCandidate, evalSd, evalHit, evalPixel);
        float shiftedTarget = 0.f;
        return materializeShiftMappingResult(sourceCandidate, shift, evalSd, evalDepth, evalMaterialID, shifted, shiftedTarget, rejectionReason);
    }

    /** Check that an imported sample has reverse support under the selected map.

        Spatial reuse compares proposal densities across domains. If a sample can
        be shifted from neighbor to center but not back into the neighbor domain,
        the domains do not share support and the imported sample is not eligible.
    */
    bool hasReverseShiftSupport(PathReservoir shiftedCandidate, uint2 centerPixel, uint2 sourcePixel)
    {
        if (all(centerPixel == sourcePixel)) return true;

        PathReservoir reverseShifted;
        uint reverseReason = 0u;
        return evaluateCandidateAtPixel(shiftedCandidate, centerPixel, sourcePixel, reverseShifted, reverseReason);
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

            if (!evaluateCandidateAtPixel(sourceCandidate, sourcePixel, proposalPixels[j], domainShifted, reason))
                continue;

            const float centerToDomainJacobian = domainShifted.jacobian / centerJacobian;
            if (domainShifted.shiftedTarget > 0.f && centerToDomainJacobian > 0.f)
                denominator += (domainShifted.shiftedTarget / domainZ) * centerToDomainJacobian * float(domainM);
        }

        return denominator > 0.f ? numerator / denominator : 0.f;
    }

    /** Defensive pairwise MIS between the canonical proposal and one neighbor at a time.

        The canonical weight carries a defensive self-pair term of 1 and both branches
        normalize by the total proposal count (k + 1), k = neighbor count. The defensive
        term keeps the estimator unbiased under asymmetric shift support: when the
        center->neighbor shift fails the canonical absorbs the missing pair weight instead
        of the partition over-counting. A plain /pairCount without the defensive +1
        over-brightens whenever shifts fail one-directionally. The per-pair canonical
        confidence is still split by pairCount inside each share.

        Returns m_i(y); getReservoirCandidateWeight() supplies the final totalM factor
        (which cancels the /M in finalizeReservoir()).
    */
    float getPairwiseMISWeight(PathReservoir sourceCandidate, PathReservoir shiftedToCenter, uint2 sourcePixel, uint sourceM, uint proposalCount, uint2 proposalPixels[kMaxSpatialCandidates], PathReservoir proposalReservoirs[kMaxSpatialCandidates], float reverseShiftTarget[kMaxSpatialCandidates], float reverseShiftJacobian[kMaxSpatialCandidates])
    {
        if (proposalCount <= 1u) return 1.f;

        const uint canonicalM = getProposalConfidence(proposalReservoirs[0]);
        const float canonicalZ = getReservoirNormalizer(proposalReservoirs[0]);
        const float pairCount = float(max(1u, proposalCount - 1u));

        if (all(sourcePixel == proposalPixels[0]))
        {
            // Defensive canonical self-pair term. Each pair then subtracts the
            // neighbor's share; a failed center->neighbor shift leaves the full 1.
            float canonicalWeight = 1.f;
            for (uint j = 1u; j < proposalCount; j++)
            {
                // Canonical (=sourceCandidate) shifted into domain j is the reverse center->j
                // shift, cached at admission for every geometry-valid neighbor. A cached
                // target of 0 means the reverse failed -> neighborPdf 0 -> the canonical keeps
                // its full defensive +1 for this pair (no re-trace here).
                const uint domainM = getProposalConfidence(proposalReservoirs[j]);
                const float neighborPdf = domainM > 0u ? reverseShiftTarget[j] * reverseShiftJacobian[j] * float(domainM) : 0.f;
                const float canonicalPdf = sourceCandidate.sample.target * float(canonicalM) / pairCount;
                canonicalWeight += 1.f - (neighborPdf > 0.f ? neighborPdf / (neighborPdf + canonicalPdf) : 0.f);
            }
            return canonicalWeight / float(proposalCount);
        }

        if (shiftedToCenter.jacobian <= 0.f || sourceCandidate.sample.target <= 0.f) return 0.f;

        const float neighborPdf = sourceCandidate.sample.target * rcp(shiftedToCenter.jacobian) * float(sourceM);
        const float canonicalPdf = shiftedToCenter.shiftedTarget * float(canonicalM) / pairCount;
        return (neighborPdf + canonicalPdf) > 0.f ? (neighborPdf / (neighborPdf + canonicalPdf)) / float(proposalCount) : 0.f;
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
        PathReservoir selected = inputReservoirs[reservoirIndex(pixel)];

        const uint activeNeighborCount = angularReuse != 0u ? uint(getLightFieldAngularProposalCount(params)) : neighborCount;

        // Zero-neighbor spatial reuse is a debugging passthrough for comparing
        // against initial RIS.
        if (activeNeighborCount == 0u)
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
        // Diagnostics: sum of the effective MIS weights actually used over accepted
        // candidates (debugView 17). Constant uses M_i/candidateCount, which sums to
        // exactly 1 by construction. Generalized/pairwise accumulate m_i(y_i), each
        // evaluated at a DIFFERENT selected sample y_i, so this sum is NOT expected to be
        // 1 (the partition of unity Sum_i m_i(y)=1 holds per-fixed-y, not across the
        // per-candidate samples). Do not treat a non-1 value here as a bug for those modes.
        float misWeightSum = 0.f;
        PathReservoir proposalReservoirs[kMaxSpatialCandidates];
        PathReservoir shiftedCandidates[kMaxSpatialCandidates];
        uint2 proposalPixels[kMaxSpatialCandidates];
        float proposalDistances[kMaxSpatialCandidates];
        uint proposalAccepted[kMaxSpatialCandidates];
        // Cached reverse shift (canonical -> proposal pixel), filled at admission time so the
        // pairwise canonical MIS term does not re-trace it. Only valid for accepted neighbor
        // proposals (index > 0); the canonical's self entry is unused.
        float reverseShiftTarget[kMaxSpatialCandidates];
        float reverseShiftJacobian[kMaxSpatialCandidates];
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
        // domain. Confidence is only added once a domain produces a positive-density
        // shift, so the GRIS normalization counts only the candidates that can
        // actually contribute. Counting failed shifts here biases the
        // Constant estimator dark.
        uint selfShiftRejection = 0u;
        PathReservoir selfShifted;
        proposalReservoirs[proposalCount] = selected;
        shiftedCandidates[proposalCount] = {};
        proposalPixels[proposalCount] = pixel;
        proposalDistances[proposalCount] = 0.f;
        proposalAccepted[proposalCount] = 0u;
        if (evaluateIdentityCandidate(selected, selfShifted, selfShiftRejection))
        {
            shiftedCandidates[proposalCount] = selfShifted;
            proposalAccepted[proposalCount] = 1u;
            candidateCount += getProposalConfidence(selected);
            accepted++;
        }
        else
        {
            lastRejectionReason = selfShiftRejection;
            rejected++;
        }
        proposalCount++;

        // Draw spatial neighbors from the current view, or angular neighbors by
        // projecting this primary point into adjacent light-field views.
        for (uint i = 0u; i < activeNeighborCount; i++)
        {
            int2 offset = int2(0, 0);
            int2 neighborPixelSigned = int2(pixel);
            float proposalDistance = 0.f;

            if (angularReuse != 0u)
            {
                if (!getAngularNeighborPixel(params, pixel, dstPosW, int(i), neighborPixelSigned))
                {
                    rejected++;
                    continue;
                }
                offset = neighborPixelSigned - int2(pixel);
                proposalDistance = length(float2(offset));
            }
            else
            {
                float2 u = float2(nextRand(state), nextRand(state));
                float angle = 6.28318530718f * u.x;
                float dist = max(1.f, float(radius) * sqrt(u.y));
                offset = int2(round(float2(cos(angle), sin(angle)) * dist));
                neighborPixelSigned = int2(pixel) + offset;
                proposalDistance = length(float2(offset));
            }

            if (any(neighborPixelSigned < int2(0, 0)) || any(neighborPixelSigned >= int2(params.frameDim)))
            {
                rejected++;
                continue;
            }

            if (angularReuse == 0u && !isSameLightFieldViewPixel(params, pixel, neighborPixelSigned))
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

            const uint rejectionReason = getCompatibilityRejection(sourceCandidate, dstPosW, dstNormalW, dstDepth, dstMaterialID);
            if (rejectionReason != 0u)
            {
                lastRejectionReason = rejectionReason;
                rejected++;
                continue;
            }

            // Geometry-valid neighbor: joins the proposal set (counted in proposalCount and
            // the canonical's defensive accounting) regardless of whether its shifts succeed,
            // keeping the /(k+1) normalization correct. Its CONTRIBUTION is admitted
            // separately, only when the shift is invertible.
            const uint proposalIndex = proposalCount;
            proposalReservoirs[proposalIndex] = sourceCandidate;
            shiftedCandidates[proposalIndex] = {};
            proposalPixels[proposalIndex] = neighborPixel;
            proposalDistances[proposalIndex] = proposalDistance;
            proposalAccepted[proposalIndex] = 0u;
            reverseShiftTarget[proposalIndex] = 0.f;
            reverseShiftJacobian[proposalIndex] = 0.f;
            proposalCount++;

            // Reverse shift center -> neighbor, cached for the pairwise canonical MIS term
            // (a cached target of 0 means the reverse failed -> the canonical keeps its full
            // defensive weight for this pair). Computed once for every geometry-valid neighbor
            // so getPairwiseMISWeight never re-traces it.
            bool reverseValid = false;
            {
                ShadingData nSd; HitInfo nHit; float3 nP; float3 nN; float nD; uint nMat;
                if (loadPrimarySurface(neighborPixel, nSd, nHit, nP, nN, nD, nMat))
                {
                    ShiftMappingResult rev = gReSTIRPTLF.applyShiftMapping(selected, nSd, nHit, neighborPixel);
                    if (rev.valid && rev.shiftedTarget > 0.f && rev.jacobian > 0.f)
                    {
                        reverseShiftTarget[proposalIndex] = rev.shiftedTarget;
                        reverseShiftJacobian[proposalIndex] = rev.jacobian;
                        reverseValid = true;
                    }
                }
            }

            // Forward shift: neighbor -> center (the reused contribution).
            uint shiftRejectionReason = 0u;
            PathReservoir shiftedCandidate;
            float shiftedTarget = 0.f;
            ShiftMappingResult shift = gReSTIRPTLF.applyShiftMapping(sourceCandidate, dstSd, dstHit, pixel);
            const bool forwardValid = materializeShiftMappingResult(sourceCandidate, shift, dstSd, dstDepth, dstMaterialID, shiftedCandidate, shiftedTarget, shiftRejectionReason);

            // Admit the CONTRIBUTION only if the shift is invertible (both directions succeed).
            // A forward-ok/reverse-fail neighbor stays in the defensive accounting above but
            // must not contribute, else the canonical keeps full weight AND the neighbor adds
            // its own -> Sum m_i > 1 -> over-bright.
            // (Set kEnforceShiftInvertibility = false to reproduce that bug.)
            const bool contribute = forwardValid && (!kEnforceShiftInvertibility || reverseValid);
            if (contribute)
            {
                shiftedCandidates[proposalIndex] = shiftedCandidate;
                proposalAccepted[proposalIndex] = 1u;
                candidateCount += getProposalConfidence(sourceCandidate);
                accepted++;
            }
            else
            {
                // Contribution not admitted: clear the cached reverse shift so the canonical's
                // defensive term does not subtract this pair's share either. The mirror case --
                // reverse (center->neighbor) ok but forward (neighbor->center) failed, e.g. a
                // direct-NEE neighbor reconnection cannot import -- would otherwise drain
                // canonical weight with no contribution -> Sum m_i < 1 -> too dark. Keeps the
                // subtraction symmetric with contribution admission.
                reverseShiftTarget[proposalIndex] = 0.f;
                reverseShiftJacobian[proposalIndex] = 0.f;
                lastRejectionReason = forwardValid ? 16u : shiftRejectionReason;
                rejected++;
            }
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
                misWeight = getPairwiseMISWeight(proposalReservoirs[i], candidate, proposalPixels[i], sourceM, proposalCount, proposalPixels, proposalReservoirs, reverseShiftTarget, reverseShiftJacobian);
            }

            // Track the effective MIS weight contributed by this candidate so the
            // debug view can confirm the per-pixel weights sum to 1.
            misWeightSum += (misStrategy == kSpatialMISConstant)
                ? float(sourceM) / float(max(1u, candidateCount))
                : misWeight;

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

        float3 debugColor;
        if (debugView == 17u)
        {
            // Sum of effective MIS weights. Meaningful only for Constant, where it must
            // read 1 (green); >1 red, <1 blue. For generalized/pairwise the sum is over
            // distinct per-candidate samples and is not expected to be 1 (see note above).
            const float s = misWeightSum;
            debugColor = float3(max(0.f, s - 1.f), max(0.f, 1.f - abs(s - 1.f)), max(0.f, 1.f - s));
        }
        else if (debugView == 18u)
        {
            // Fraction of considered proposal domains whose shift succeeded.
            debugColor = (float(accepted) / max(1.f, float(proposalCount))).xxx;
        }
        else if (debugView == 19u)
        {
            // DIAGNOSTIC: same-pixel identity test for the shift mapping. Shift the
            // canonical reservoir into its OWN primary domain through the real shift
            // (not the trivial identity short-circuit) and output shiftedTarget/target.
            // A faithful unit-Jacobian identity yields exactly 1 everywhere; a reject/
            // invalid yields 0. This isolates replay-mapping faithfulness from cross-
            // pixel Jacobian and resampling-MIS effects, and separates it from the
            // rejection rate (the 0-valued pixels).
            PathReservoir canon = inputReservoirs[reservoirIndex(pixel)];
            ShiftMappingResult selfShift = gReSTIRPTLF.applyShiftMapping(canon, dstSd, dstHit, pixel);
            if (isValid(canon) && selfShift.valid && selfShift.shiftedTarget > 0.f && canon.target > 0.f)
                debugColor = float3(selfShift.shiftedTarget / canon.target, float(canon.sample.selectedVertex), float(canon.sample.selectedKind));
            else
                debugColor = float3(0.f, -1.f, -1.f);
        }
        else
        {
            debugColor = getDebugColor(selected);
        }
        outputColor[pixel] = float4(debugColor, 1.f);
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
