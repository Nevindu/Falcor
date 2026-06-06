/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

import Params;
import PathState;

/** Spatial reuse skeleton for ReSTIR PT.

    This pass deliberately does not implement path reconnection yet. It samples
    nearby reservoirs, accepts only reservoirs that are already valid under the
    identity-style stored target, and combines them with a simple reservoir update.
    The goal is to validate buffer plumbing, neighbor sampling, and GRIS-style
    combine infrastructure before adding real shift mappings and Jacobians.
*/
struct SpatialReusePass
{
    ReSTIRPTParams params;

    StructuredBuffer<PathReservoir> inputReservoirs;
    RWStructuredBuffer<PathReservoir> outputReservoirs;
    RWTexture2D<float4> outputColor;

    uint neighborCount;
    uint radius;
    uint debugView;

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

    float nextRand(inout uint state)
    {
        state = hash(state);
        return float(state & 0x00ffffffu) / float(0x01000000u);
    }

    uint reservoirIndex(uint2 pixel)
    {
        return pixel.y * params.frameDim.x + pixel.x;
    }

    /** Conservative validity test for the current skeleton.

        A future shift mapping will also check geometry, material compatibility,
        visibility, and Jacobian terms. For now we only reject reservoirs that
        cannot produce a finite stored-target contribution.
    */
    bool isValid(PathReservoir reservoir)
    {
        return reservoir.sample.flags != 0u && reservoir.target > 0.f && reservoir.sample.target > 0.f;
    }

    float3 evaluateReservoirContribution(PathReservoir reservoir)
    {
        return isValid(reservoir) ? reservoir.sample.contribution * reservoir.W * float(max(1u, reservoir.M)) : float3(0.f);
    }

    /** Finish a combined spatial reservoir and attach debug counters.

        The accepted/rejected/source-distance values are temporarily packed into
        existing PathSample fields so the current debug-view machinery can display
        them without introducing a separate statistics buffer.
    */
    PathReservoir finalizeReservoir(PathReservoir reservoir, float weightSum, uint candidateCount, uint accepted, uint rejected, float sourceDistance)
    {
        reservoir.wSum = weightSum;
        reservoir.M = max(1u, candidateCount);
        reservoir.shiftedTarget = reservoir.target;
        reservoir.targetRatio = reservoir.target > 0.f ? 1.f : 0.f;
        reservoir.jacobian = 1.f;
        reservoir.rejectionReason = isValid(reservoir) ? 0u : 1u;

        // Reuse existing diagnostic fields until we split out a dedicated spatial stats buffer.
        reservoir.sample.pathLength = accepted;
        reservoir.sample.rngState = rejected;
        reservoir.sample.selectedCandidate = uint(round(sourceDistance));
        reservoir.W = reservoir.target > 0.f ? weightSum / (reservoir.target * float(reservoir.M)) : 0.f;
        return reservoir;
    }

    /** Shared beauty/debug output for the spatial result.

        Debug modes 10-13 are spatial-specific and interpret the temporary stats
        packed by finalizeReservoir().
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
        case 10: return float(reservoir.sample.pathLength).xxx / max(1.f, float(neighborCount + 1u));
        case 11: return float(reservoir.sample.rngState).xxx / max(1.f, float(neighborCount + 1u));
        case 12: return float(reservoir.M).xxx / max(1.f, float(neighborCount + 1u));
        case 13: return float(reservoir.sample.selectedCandidate).xxx / max(1.f, float(radius));
        default: return evaluateReservoirContribution(reservoir);
        }
    }

    /** Combine the current pixel reservoir with randomly sampled neighbor reservoirs.

        This is a reservoir-combine test, not a physically correct spatial shift.
        Neighbor samples are currently evaluated as if their stored target were
        valid at the destination pixel, which is why this milestone may smooth
        aggressively or look biased. The next milestone should replace this with
        an actual reconnection/shift evaluator.
    */
    void execute(uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;

        uint state = hash(pixel.x | (pixel.y << 16) | (params.frameCount * 0x9e3779b9u));
        PathReservoir selected = inputReservoirs[reservoirIndex(pixel)];

        uint candidateCount = 0u;
        uint accepted = 0u;
        uint rejected = 0u;
        float weightSum = 0.f;
        float selectedDistance = 0.f;

        // Seed the combined reservoir with the current pixel's own initial sample.
        if (isValid(selected))
        {
            candidateCount += max(1u, selected.M);
            accepted++;
            weightSum += selected.target;
        }
        else
        {
            rejected++;
        }

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
            PathReservoir candidate = inputReservoirs[reservoirIndex(neighborPixel)];
            if (!isValid(candidate))
            {
                rejected++;
                continue;
            }

            accepted++;
            candidateCount += max(1u, candidate.M);
            float candidateWeight = candidate.target;
            float newWeightSum = weightSum + candidateWeight;

            // Standard weighted reservoir update over the local + neighbor candidates.
            if (newWeightSum > 0.f && nextRand(state) * newWeightSum < candidateWeight)
            {
                selected = candidate;
                selectedDistance = length(float2(offset));
            }

            weightSum = newWeightSum;
        }

        if (accepted == 0u)
        {
            selected = {};
        }

        selected = finalizeReservoir(selected, weightSum, candidateCount, accepted, rejected, selectedDistance);
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
