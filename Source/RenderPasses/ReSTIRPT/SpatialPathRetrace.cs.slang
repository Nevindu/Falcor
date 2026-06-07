/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

import Params;
import PathState;
import LoadShadingData;
#ifndef RESTIRPT_ENABLE_PREPARED_HYBRID_K3
#define RESTIRPT_ENABLE_PREPARED_HYBRID_K3 0
#endif

#if RESTIRPT_ENABLE_PREPARED_HYBRID_K3
import ReSTIRPT;
import ShiftMapping_Reconnection;
import ShiftMapping_Hybrid;
import Rendering.Materials.TexLODHelpers;
import Scene.RaytracingInline;

ParameterBlock<ReSTIRPT> gReSTIRPT;
#endif

/** Spatial prefix retrace prepass for hybrid ReSTIR PT.

    This mirrors the GRIS reference architecture: spatial reuse owns reservoir
    merging, while this pass owns per-neighbor destination prefix preparation for
    hybrid shifts. For k=2 the prefix is the identity fixed-reconnection case.
    For k>2 this pass random-replays y1..y_{k-2} and stores y_{k-1},
    the ray that hit y_{k-1}, and the replay prefix throughput.
*/
struct SpatialPathRetracePass
{
    ReSTIRPTParams params;

    StructuredBuffer<PathReservoir> inputReservoirs;
    RWStructuredBuffer<HybridReconnectionData> hybridReconnectionData;
    Texture2D<PackedHitInfo> vbuffer;
    Texture2D<float3> viewDir;

    uint neighborCount;
    uint radius;
    uint hybridDataStride;

    static const bool kUseViewDir = USE_VIEW_DIR;
    static const bool kUseCurves = USE_CURVES;

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

    uint reconnectionDataIndex(uint pixelIndex, uint entry)
    {
        return pixelIndex * hybridDataStride + entry;
    }

    bool isValidScreenRegion(int2 pixel)
    {
        return all(pixel >= int2(0, 0)) && all(pixel < int2(params.frameDim));
    }

    bool isReservoirValid(PathReservoir reservoir)
    {
        return (reservoir.sample.flags & kPathSampleFlagValid) != 0u && reservoir.target > 0.f && reservoir.sample.target > 0.f && reservoir.W > 0.f;
    }

    bool loadPrimarySurface(uint2 pixel, out HitInfo hit, out ShadingData sd)
    {
        hit = unpackHitInfo(vbuffer[pixel]);
        if (!hit.isValid())
        {
            sd = {};
            return false;
        }

        Ray cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        if (kUseViewDir) cameraRay.dir = -viewDir[pixel];
        sd = loadShadingData(hit, cameraRay.origin, cameraRay.dir);
        return true;
    }

#if RESTIRPT_ENABLE_PREPARED_HYBRID_K3
    HybridReconnectionData tracePreparedHybridPrefix(uint2 dstPixel, HitInfo dstHit, ShadingData dstSd, PathReservoir source)
    {
        PathSample sample = source.sample;
        const uint reconnectVertex = sample.reconnectVertex;
        if ((sample.flags & kPathSampleFlagHybridK3) == 0u ||
            (sample.flags & kPathSampleFlagNeedsReplay) == 0u ||
            reconnectVertex < 3u ||
            reconnectVertex > kMaxStoredReplayLobes ||
            sample.selectedVertex < reconnectVertex)
            return makeInvalidHybridReconnectionData(8u, sample.reconnectVertex);
        if ((sample.flags & kPathSampleFlagAlreadyShifted) != 0u)
            return makeInvalidHybridReconnectionData(15u, sample.reconnectVertex);
        if (kUseRussianRoulette)
            return makeInvalidHybridReconnectionData(14u, sample.reconnectVertex);

        PathState path = {};
        gReSTIRPT.generatePath(sample.pathID, sample.pathSeed, path);
        HitInfo dstCurrentHit = dstHit;
        ShadingData dstCurrentSd = dstSd;
        float3 dstCurrentParentOrigin = gScene.camera.computeRayPinhole(dstPixel, params.frameDim).origin;
        float3 dstCurrentRayDir = kUseViewDir ? -viewDir[dstPixel] : gScene.camera.computeRayPinhole(dstPixel, params.frameDim).dir;
        float3 shiftedPrefixThroughput = float3(1.f);

        for (uint replayVertex = 1u; replayVertex < reconnectVertex - 1u; replayVertex++)
        {
            if (!path.isHit() || path.getVertexIndex() != replayVertex)
                return makeInvalidHybridReconnectionData(15u, sample.reconnectVertex);

            const bool isPrimaryHit = replayVertex == 1u;
            const HitType sourceHitType = path.hit.getType();
            const bool sourceIsTriangleHit = sourceHitType == HitType::Triangle;
            const bool sourceIsCurveHit = kUseCurves && (sourceHitType == HitType::Curve);
            if (!sourceIsTriangleHit || sourceIsCurveHit || !path.interiorList.isEmpty() || path.isInsideDielectricVolume())
                return makeInvalidHybridReconnectionData(14u, sample.reconnectVertex);

            ShadingData sourceSd = loadShadingData(path.hit, path.origin, path.dir);
            if (!gReSTIRPT.handleNestedDielectrics(sourceSd, path))
                return makeInvalidHybridReconnectionData(14u, sample.reconnectVertex);
            path.origin = sourceSd.computeRayOrigin();
            if (isPrimaryHit)
            {
                uint primaryReplayReason = 0u;
                if (!gReSTIRPT.validateReplayPrimarySurface(sample, path.hit, path.origin, path.dir, sourceSd, primaryReplayReason))
                    return makeInvalidHybridReconnectionData(primaryReplayReason, sample.reconnectVertex);
            }

            let sourceLod = gReSTIRPT.createTextureSampler(path, isPrimaryHit);
            let sourceHints = getMaterialInstanceHints(path.hit, isPrimaryHit);
            const IMaterialInstance sourceMi = gScene.materials.getMaterialInstance(sourceSd, sourceLod, sourceHints);
            BSDFProperties sourceProps = sourceMi.getProperties(sourceSd);
            const uint sourceLobes = sourceMi.getLobeTypes(sourceSd);
            if (sourceProps.roughness <= params.specularRoughnessThreshold)
                return makeInvalidHybridReconnectionData(14u, sample.reconnectVertex);

            if (!gReSTIRPT.consumeReplayNEE(sourceSd, sourceMi, sourceLobes, false, false, path))
                return makeInvalidHybridReconnectionData(15u, sample.reconnectVertex);

            SampleGenerator sharedReplaySg = path.sg;
            BSDFSample sourceBs = {};
            const bool validSourceScatter = sourceMi.sample(sourceSd, path.sg, sourceBs, kUseBSDFSampling);

            let dstLod = ExplicitLodTextureSampler(0.f);
            let dstHints = getMaterialInstanceHints(dstCurrentHit, isPrimaryHit);
            const IMaterialInstance dstMi = gScene.materials.getMaterialInstance(dstCurrentSd, dstLod, dstHints);
            BSDFProperties dstProps = dstMi.getProperties(dstCurrentSd);
            BSDFSample shiftedBs = {};
            const bool validShiftedScatter = dstMi.sample(dstCurrentSd, sharedReplaySg, shiftedBs, kUseBSDFSampling);

            const uint expectedLobe = gReSTIRPT.getSampleReplayLobeTag(sample, replayVertex);
            if (!validSourceScatter || !validShiftedScatter ||
                sourceBs.pdf <= 0.f || shiftedBs.pdf <= 0.f ||
                sourceBs.lobeType != expectedLobe || shiftedBs.lobeType != expectedLobe ||
                sourceBs.isLobe(LobeType::Delta) || shiftedBs.isLobe(LobeType::Delta) ||
                sourceBs.isLobe(LobeType::Transmission) || shiftedBs.isLobe(LobeType::Transmission) ||
                dstProps.roughness <= params.specularRoughnessThreshold ||
                !gReSTIRPT.isNonDeltaReflectionLobe(expectedLobe) ||
                !gReSTIRPT.canRealizeReplayLobe(dstMi.getLobeTypes(dstCurrentSd), expectedLobe))
            {
                return makeInvalidHybridReconnectionData(14u, sample.reconnectVertex);
            }

            if (!any(sourceBs.weight > 0.f) || !any(shiftedBs.weight > 0.f))
                return makeInvalidHybridReconnectionData(10u, sample.reconnectVertex);

            shiftedPrefixThroughput *= shiftedBs.weight;

            if (!gReSTIRPT.generateScatterRay(sourceBs, sourceSd, sourceMi, path))
                return makeInvalidHybridReconnectionData(14u, sample.reconnectVertex);

            float sourceDistance = 0.f;
            if (!gReSTIRPT.traceReplayNextHit(path, sourceDistance))
                return makeInvalidHybridReconnectionData(15u, sample.reconnectVertex);

            const float3 dstRayOrigin = dstCurrentSd.computeRayOrigin();
            const float3 dstRayDir = shiftedBs.wo;
            SceneRayQuery<kUseAlphaTest> dstQuery;
            float dstDistance = 0.f;
            HitInfo dstNextHit = dstQuery.traceRay(Ray(dstRayOrigin, dstRayDir, 0.f, kRayTMax), dstDistance);
            if (!dstNextHit.isValid() || dstDistance <= params.minReconnectDistance)
                return makeInvalidHybridReconnectionData(14u, sample.reconnectVertex);

            dstCurrentParentOrigin = dstRayOrigin;
            dstCurrentRayDir = dstRayDir;
            dstCurrentHit = dstNextHit;
            dstCurrentSd = loadShadingData(dstCurrentHit, dstCurrentParentOrigin, dstCurrentRayDir);
        }

        HybridReconnectionData data = makeValidHybridReconnectionData(
            dstCurrentHit.pack(),
            dstCurrentRayDir,
            shiftedPrefixThroughput,
            sample.reconnectVertex,
            dstCurrentParentOrigin);
        ShiftMappingResult shift = gReSTIRPT.applyPreparedHybridK3Shift(source, dstSd, dstHit, dstPixel, data);
        return storeCachedHybridShiftResult(data, shift);
    }
#endif

    HybridReconnectionData tracePreparedPrefix(uint2 dstPixel, PathReservoir source)
    {
        if (!isReservoirValid(source))
            return makeInvalidHybridReconnectionData(1u, source.sample.reconnectVertex);

        HitInfo dstHit;
        ShadingData dstSd;
        if (!loadPrimarySurface(dstPixel, dstHit, dstSd))
            return makeInvalidHybridReconnectionData(3u, source.sample.reconnectVertex);

        const uint reconnectVertex = source.sample.reconnectVertex;
        if (reconnectVertex == kInvalidReconnectVertex)
            return makeInvalidHybridReconnectionData(8u, reconnectVertex);

        if (reconnectVertex <= 2u)
        {
            // k=2 is the fixed-reconnection specialization. Prefix replay has
            // identity throughput and no destination predecessor beyond y1.
            return makeValidHybridReconnectionData(vbuffer[dstPixel], float3(0.f, 0.f, 1.f), float3(1.f), reconnectVertex);
        }

        if (reconnectVertex > 2u)
        {
#if RESTIRPT_ENABLE_PREPARED_HYBRID_K3
            return tracePreparedHybridPrefix(dstPixel, dstHit, dstSd, source);
#else
            return makeInvalidHybridReconnectionData(18u, reconnectVertex);
#endif
        }

        return makeInvalidHybridReconnectionData(18u, reconnectVertex);
    }

    void execute(uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;
        if (neighborCount == 0u) return;

        const uint centerIndex = reservoirIndex(pixel);
        PathReservoir centralReservoir = inputReservoirs[centerIndex];

        uint state = hash(pixel.x | (pixel.y << 16) | (params.frameCount * 0x9e3779b9u));

        for (uint i = 0u; i < neighborCount; i++)
        {
            const uint centralToNeighborEntry = 2u * i;
            const uint neighborToCentralEntry = centralToNeighborEntry + 1u;
            hybridReconnectionData[reconnectionDataIndex(centerIndex, centralToNeighborEntry)] = makeInvalidHybridReconnectionData(3u);
            hybridReconnectionData[reconnectionDataIndex(centerIndex, neighborToCentralEntry)] = makeInvalidHybridReconnectionData(3u);

            float2 u = float2(nextRand(state), nextRand(state));
            float angle = 6.28318530718f * u.x;
            float dist = max(1.f, float(radius) * sqrt(u.y));
            int2 offset = int2(round(float2(cos(angle), sin(angle)) * dist));
            int2 neighborPixelSigned = int2(pixel) + offset;
            if (!isValidScreenRegion(neighborPixelSigned)) continue;

            uint2 neighborPixel = uint2(neighborPixelSigned);
            PathReservoir neighborReservoir = inputReservoirs[reservoirIndex(neighborPixel)];

            hybridReconnectionData[reconnectionDataIndex(centerIndex, centralToNeighborEntry)] = tracePreparedPrefix(neighborPixel, centralReservoir);
            hybridReconnectionData[reconnectionDataIndex(centerIndex, neighborToCentralEntry)] = tracePreparedPrefix(pixel, neighborReservoir);
        }
    }
};

cbuffer CB
{
    SpatialPathRetracePass gSpatialPathRetracePass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gSpatialPathRetracePass.execute(dispatchThreadId.xy);
}
