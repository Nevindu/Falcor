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

/** Temporal reuse for ReSTIR PT.

    This pass combines the current pixel's canonical reservoir with the previous
    frame's final reservoir reprojected by motion vectors. The previous sample is
    shifted into the current primary domain through Reconnection or RandomReplay.
    A debug mode can force same-pixel identity reuse, bypassing reprojection and
    the temporal shift map.
*/
struct TemporalReusePass
{
    ReSTIRPTParams params;

    StructuredBuffer<PathReservoir> currentReservoirs;
    StructuredBuffer<PathReservoir> previousReservoirs;
    RWStructuredBuffer<PathReservoir> outputReservoirs;
    RWTexture2D<float4> outputColor;
    Texture2D<PackedHitInfo> vbuffer;
    Texture2D<PackedHitInfo> previousVBuffer;
    Texture2D<float3> viewDir;
    Texture2D<float2> motionVectors;

    uint historyLength;
    uint debugView;
    uint featureBasedRejection;
    uint forceSamePixel;

    static const bool kUseViewDir = USE_VIEW_DIR;
    static const float kFeatureNormalThreshold = 0.5f;
    static const float kFeatureRelativeDepthThreshold = 0.1f;

    uint hash(uint v)
    {
        v ^= v >> 16;
        v *= 0x7feb352du;
        v ^= v >> 15;
        v *= 0x846ca68bu;
        v ^= v >> 16;
        return v;
    }

    float nextRand(inout uint state)
    {
        state = hash(state);
        return float(state & 0x00ffffffu) / float(0x01000000u);
    }

    uint reservoirIndex(uint2 pixel)
    {
        return pixel.y * params.frameDim.x + pixel.x;
    }

    bool isValid(PathReservoir reservoir)
    {
        return (reservoir.sample.flags & kPathSampleFlagValid) != 0u && reservoir.target > 0.f && reservoir.sample.target > 0.f && reservoir.W > 0.f;
    }

    uint getProposalConfidence(PathReservoir reservoir)
    {
        return isValid(reservoir) ? max(1u, reservoir.M) : 0u;
    }

    bool loadPrimarySurfaceFromHit(uint2 pixel, HitInfo hit, out ShadingData sd, out float3 posW, out float3 normalW, out float depth, out uint materialID)
    {
        sd = {};
        posW = float3(0.f);
        normalW = float3(0.f, 1.f, 0.f);
        depth = 0.f;
        materialID = 0u;

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

    bool loadPrimarySurface(uint2 pixel, out ShadingData sd, out HitInfo hit, out float3 posW, out float3 normalW, out float depth, out uint materialID)
    {
        hit = unpackHitInfo(vbuffer[pixel]);
        return loadPrimarySurfaceFromHit(pixel, hit, sd, posW, normalW, depth, materialID);
    }

    bool loadPreviousPrimarySurface(uint2 pixel, out ShadingData sd, out HitInfo hit, out float3 posW, out float3 normalW, out float depth, out uint materialID)
    {
        hit = unpackHitInfo(previousVBuffer[pixel]);
        return loadPrimarySurfaceFromHit(pixel, hit, sd, posW, normalW, depth, materialID);
    }

    uint getCompatibilityRejection(PathReservoir candidate, float3 dstNormalW, float dstDepth)
    {
        if (!isValid(candidate)) return 1u;
        if (dstDepth <= 0.f) return 3u;
        if (candidate.sample.primaryDepth <= 0.f) return 4u;

        if (featureBasedRejection != 0u)
        {
            if (dot(dstNormalW, candidate.sample.primaryNormalW) < kFeatureNormalThreshold) return 5u;
            if (abs(dstDepth - candidate.sample.primaryDepth) >= kFeatureRelativeDepthThreshold * dstDepth) return 5u;
        }

        return 0u;
    }

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

    void localizeIdentityTemporalSample(inout PathReservoir shifted, ShadingData dstSd, float dstDepth, uint dstMaterialID)
    {
        shifted.sample.primaryPosW = dstSd.posW;
        shifted.sample.primaryNormalW = normalize(dstSd.faceN);
        shifted.sample.primaryDepth = dstDepth;
        shifted.sample.primaryMaterialID = dstMaterialID;
        shifted.sample.target = shifted.target;
    }

    void localizeShiftedSampleMetadata(inout PathReservoir shifted, ShadingData dstSd, float dstDepth, uint dstMaterialID, float shiftedTarget)
    {
        shifted.sample.primaryPosW = dstSd.posW;
        shifted.sample.primaryNormalW = normalize(dstSd.faceN);
        shifted.sample.primaryDepth = dstDepth;
        shifted.sample.primaryMaterialID = dstMaterialID;
        shifted.sample.target = shiftedTarget;
        shifted.sample.flags |= kPathSampleFlagAlreadyShifted;
    }

    bool materializeShiftMappingResult(
        PathReservoir sourceCandidate,
        ShiftMappingResult shift,
        ShadingData dstSd,
        float dstDepth,
        uint dstMaterialID,
        out PathReservoir shifted,
        out uint rejectionReason)
    {
        shifted = sourceCandidate;
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

        localizeShiftedSampleMetadata(shifted, dstSd, dstDepth, dstMaterialID, shift.shiftedTarget);
        return true;
    }

    uint getCappedTemporalConfidence(uint previousM, uint currentM)
    {
        if (historyLength == 0u || previousM == 0u) return 0u;

        const uint baseM = max(1u, currentM);
        const float capped = min(float(previousM), float(baseM) * float(historyLength));
        return max(1u, uint(capped));
    }

    float getTemporalCandidateWeight(PathReservoir reservoir, uint totalM, float misWeight)
    {
        if (reservoir.shiftedTarget <= 0.f || reservoir.jacobian <= 0.f || reservoir.W <= 0.f || totalM == 0u || misWeight <= 0.f) return 0.f;
        return reservoir.shiftedTarget * reservoir.W * float(totalM) * reservoir.jacobian * misWeight;
    }

    float getCurrentTalbotMISWeight(PathReservoir current, uint currentM, float temporalTargetAtCurrent, float temporalJacobianAtCurrent, uint previousM)
    {
        const float currentPdf = current.target > 0.f && currentM > 0u ? current.target * float(currentM) : 0.f;
        const float temporalPdf = previousM > 0u && temporalTargetAtCurrent > 0.f && temporalJacobianAtCurrent > 0.f
            ? temporalTargetAtCurrent * temporalJacobianAtCurrent * float(previousM)
            : 0.f;
        return (currentPdf + temporalPdf) > 0.f ? currentPdf / (currentPdf + temporalPdf) : 0.f;
    }

    float getPreviousTalbotMISWeight(PathReservoir previous, PathReservoir shiftedPrevious, uint previousM, uint currentM)
    {
        if (shiftedPrevious.jacobian <= 0.f || previous.sample.target <= 0.f || shiftedPrevious.shiftedTarget <= 0.f) return 0.f;

        const float temporalPdf = previous.sample.target * rcp(shiftedPrevious.jacobian) * float(previousM);
        const float canonicalPdf = shiftedPrevious.shiftedTarget * float(currentM);
        return (temporalPdf + canonicalPdf) > 0.f ? temporalPdf / (temporalPdf + canonicalPdf) : 0.f;
    }

    float3 evaluateReservoirContribution(PathReservoir reservoir)
    {
        return isValid(reservoir) ? reservoir.sample.contribution * reservoir.W : float3(0.f);
    }

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

    float3 getDebugColor(PathReservoir reservoir)
    {
        switch (debugView)
        {
        case 1: return reservoir.target.xxx * 0.02f;
        case 2: return reservoir.W.xxx;
        case 3: return float(reservoir.sample.pathLength).xxx / 10.f;
        case 4: return isValid(reservoir) ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);
        case 5: return float(reservoir.sample.selectedCandidate).xxx;
        case 6: return reservoir.shiftedTarget.xxx * 0.02f;
        case 7: return reservoir.targetRatio.xxx;
        case 8: return reservoir.rejectionReason == 0u ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);
        case 9: return float(reservoir.rejectionReason).xxx / 8.f;
        case 10: return float(reservoir.acceptedCount).xxx * 0.5f;
        case 11: return float(reservoir.rejectedCount).xxx * 0.5f;
        case 12: return float(reservoir.M).xxx / max(1.f, float(historyLength + 1u));
        case 13: return reservoir.selectedDistance.xxx / max(1.f, length(float2(params.frameDim)));
        case 14: return float(reservoir.rejectionReason).xxx / 8.f;
        case 15: return reservoir.rejectionReason == 0u ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);
        case 16: return reservoir.rejectionReason == 15u ? float3(1.f, 0.75f, 0.f) : float3(0.f);
        default: return evaluateReservoirContribution(reservoir);
        }
    }

    void execute(uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;

        const uint outputIndex = reservoirIndex(pixel);
        uint state = hash(pixel.x | (pixel.y << 16) | (params.frameCount * 0x9e3779b9u) | 0x3c6ef35fu);

        PathReservoir current = currentReservoirs[outputIndex];
        PathReservoir selected = current;

        ShadingData dstSd;
        HitInfo dstHit;
        float3 dstPosW;
        float3 dstNormalW;
        float dstDepth;
        uint dstMaterialID;
        const bool dstValid = loadPrimarySurface(pixel, dstSd, dstHit, dstPosW, dstNormalW, dstDepth, dstMaterialID);
        if (!dstValid || historyLength == 0u || (forceSamePixel == 0u && ShiftMapping(params.shiftMapping) == ShiftMapping::Hybrid))
        {
            outputReservoirs[outputIndex] = selected;
            outputColor[pixel] = float4(getDebugColor(selected), 1.f);
            return;
        }

        PathReservoir candidates[2];
        uint candidateM[2];
        float candidateDistance[2];
        uint candidateRole[2];
        uint candidateCount = 0u;
        uint proposalCount = 0u;
        uint accepted = 0u;
        uint rejected = 0u;
        uint lastRejectionReason = 0u;
        float temporalTargetAtCurrent = 0.f;
        float temporalJacobianAtCurrent = 0.f;
        uint temporalProposalM = 0u;
        PathReservoir temporalProposal = {};

        PathReservoir currentShifted;
        uint currentRejection = 0u;
        const uint currentM = getProposalConfidence(current);
        if (evaluateIdentityCandidate(current, currentShifted, currentRejection))
        {
            candidates[candidateCount] = currentShifted;
            candidateM[candidateCount] = currentM;
            candidateDistance[candidateCount] = 0.f;
            candidateRole[candidateCount] = 0u;
            candidateCount++;
            proposalCount++;
            accepted++;
        }
        else
        {
            lastRejectionReason = currentRejection;
            rejected++;
        }

        int2 previousPixelSigned = int2(pixel);
        if (forceSamePixel == 0u)
        {
            const float2 motionVector = motionVectors[pixel];
            previousPixelSigned = int2(round(float2(pixel) + motionVector * float2(params.frameDim)));
        }
        if (any(previousPixelSigned < int2(0, 0)) || any(previousPixelSigned >= int2(params.frameDim)))
        {
            lastRejectionReason = 3u;
            rejected++;
        }
        else
        {
            const uint2 previousPixel = uint2(previousPixelSigned);
            const bool sameTemporalDomain = forceSamePixel != 0u || all(previousPixel == pixel);
            PathReservoir previous = previousReservoirs[reservoirIndex(previousPixel)];
            temporalProposal = previous;

            uint rejectionReason = getCompatibilityRejection(previous, dstNormalW, dstDepth);

            if (rejectionReason != 0u)
            {
                lastRejectionReason = rejectionReason;
                rejected++;
            }
            else
            {
                uint previousM = getCappedTemporalConfidence(getProposalConfidence(previous), getProposalConfidence(current));
                if (previousM == 0u)
                {
                    lastRejectionReason = 1u;
                    rejected++;
                }
                else
                {
                    ShadingData previousSd;
                    HitInfo previousHit;
                    float3 previousPosW;
                    float3 previousNormalW;
                    float previousDepth;
                    uint previousMaterialID;
                    if (!sameTemporalDomain && !loadPreviousPrimarySurface(previousPixel, previousSd, previousHit, previousPosW, previousNormalW, previousDepth, previousMaterialID))
                    {
                        lastRejectionReason = 3u;
                        rejected++;
                    }
                    else
                    {
                        previous.M = previousM;
                        temporalProposal = previous;
                        temporalProposalM = previousM;
                        proposalCount = currentM > 0u ? 2u : 1u;
                        bool reverseValid = currentM == 0u || sameTemporalDomain;

                        if (currentM > 0u)
                        {
                            if (sameTemporalDomain)
                            {
                                temporalTargetAtCurrent = current.target;
                                temporalJacobianAtCurrent = 1.f;
                                reverseValid = true;
                            }
                            else
                            {
                                ShiftMappingResult reverseShift = gReSTIRPT.applyShiftMapping(current, previousSd, previousHit, previousPixel);
                                if (reverseShift.valid && reverseShift.shiftedTarget > 0.f && reverseShift.jacobian > 0.f &&
                                    !isnan(reverseShift.shiftedTarget) && !isinf(reverseShift.shiftedTarget) &&
                                    !isnan(reverseShift.jacobian) && !isinf(reverseShift.jacobian))
                                {
                                    temporalTargetAtCurrent = reverseShift.shiftedTarget;
                                    temporalJacobianAtCurrent = reverseShift.jacobian;
                                    reverseValid = true;
                                }
                            }
                        }

                        uint shiftRejectionReason = 0u;
                        PathReservoir shiftedPrevious;
                        bool shiftValid = false;
                        if (sameTemporalDomain)
                        {
                            shiftValid = evaluateIdentityCandidate(previous, shiftedPrevious, shiftRejectionReason);
                            if (shiftValid) localizeIdentityTemporalSample(shiftedPrevious, dstSd, dstDepth, dstMaterialID);
                        }
                        else
                        {
                            ShiftMappingResult shift = gReSTIRPT.applyShiftMapping(previous, dstSd, dstHit, pixel);
                            shiftValid = materializeShiftMappingResult(previous, shift, dstSd, dstDepth, dstMaterialID, shiftedPrevious, shiftRejectionReason);
                            if (shiftValid && !reverseValid)
                            {
                                shiftValid = false;
                                shiftRejectionReason = 16u;
                            }
                        }
                        if (shiftValid)
                        {
                            candidates[candidateCount] = shiftedPrevious;
                            candidateM[candidateCount] = previousM;
                            candidateDistance[candidateCount] = length(float2(previousPixelSigned) - float2(pixel));
                            candidateRole[candidateCount] = 1u;
                            candidateCount++;
                            accepted++;
                        }
                        else
                        {
                            lastRejectionReason = shiftRejectionReason;
                            rejected++;
                        }
                    }
                }
            }
        }

        uint totalM = 0u;
        [unroll]
        for (uint i = 0u; i < 2u; i++)
        {
            if (i < candidateCount) totalM += candidateM[i];
        }

        float weightSum = 0.f;
        float misWeightSum = 0.f;
        float selectedDistance = 0.f;
        [unroll]
        for (uint i = 0u; i < 2u; i++)
        {
            if (i >= candidateCount) continue;

            float misWeight = 1.f;
            if (candidateRole[i] == 0u)
            {
                misWeight = getCurrentTalbotMISWeight(current, candidateM[i], temporalTargetAtCurrent, temporalJacobianAtCurrent, temporalProposalM);
            }
            else
            {
                misWeight = getPreviousTalbotMISWeight(temporalProposal, candidates[i], candidateM[i], currentM);
            }
            misWeightSum += misWeight;

            const float candidateWeight = getTemporalCandidateWeight(candidates[i], totalM, misWeight);
            const float newWeightSum = weightSum + candidateWeight;
            if (newWeightSum > 0.f && nextRand(state) * newWeightSum < candidateWeight)
            {
                selected = candidates[i];
                selectedDistance = candidateDistance[i];
            }
            weightSum = newWeightSum;
        }

        if (accepted == 0u || weightSum <= 0.f || totalM == 0u)
        {
            selected = {};
        }

        selected = finalizeReservoir(selected, weightSum, totalM, accepted, rejected, selectedDistance, accepted > 0u ? 0u : lastRejectionReason);
        outputReservoirs[outputIndex] = selected;

        float3 debugColor;
        if (debugView == 17u)
        {
            const float s = misWeightSum;
            debugColor = float3(max(0.f, s - 1.f), max(0.f, 1.f - abs(s - 1.f)), max(0.f, 1.f - s));
        }
        else if (debugView == 18u)
        {
            debugColor = (float(accepted) / max(1.f, float(max(1u, proposalCount)))).xxx;
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
    TemporalReusePass gTemporalReusePass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gTemporalReusePass.execute(dispatchThreadId.xy);
}
