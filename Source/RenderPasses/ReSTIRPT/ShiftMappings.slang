/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

import ReSTIRPT;
import LoadShadingData;

#ifndef RESTIRPT_SHIFT_MAPPING
#define RESTIRPT_SHIFT_MAPPING 0xffffffff
#endif

#define RESTIRPT_SHIFT_MAPPING_RECONNECTION 0
#define RESTIRPT_SHIFT_MAPPING_RANDOM_REPLAY 1
#define RESTIRPT_SHIFT_MAPPING_HYBRID 2
#define RESTIRPT_SHIFT_MAPPING_RUNTIME 0xffffffff

#if RESTIRPT_SHIFT_MAPPING == RESTIRPT_SHIFT_MAPPING_RECONNECTION || RESTIRPT_SHIFT_MAPPING == RESTIRPT_SHIFT_MAPPING_RUNTIME
import ShiftMapping_Reconnection;
#endif
#if RESTIRPT_SHIFT_MAPPING == RESTIRPT_SHIFT_MAPPING_RANDOM_REPLAY || RESTIRPT_SHIFT_MAPPING == RESTIRPT_SHIFT_MAPPING_RUNTIME
import ShiftMapping_RandomReplay;
#endif
#if RESTIRPT_SHIFT_MAPPING == RESTIRPT_SHIFT_MAPPING_HYBRID || RESTIRPT_SHIFT_MAPPING == RESTIRPT_SHIFT_MAPPING_RUNTIME
import ShiftMapping_Hybrid;
#endif

/** Shared shift-mapping dispatch for ReSTIR PT reuse passes.

    Spatial reuse and future temporal reuse should call only this function, so
    new maps do not leak shift-specific logic into reservoir plumbing.
*/
extension ReSTIRPT
{
    ShiftMappingResult applyShiftMapping(PathReservoir source, ShadingData dstSd, HitInfo dstHit, uint2 dstPixel, HybridReconnectionData hybridData)
    {
#if RESTIRPT_SHIFT_MAPPING == RESTIRPT_SHIFT_MAPPING_RECONNECTION
        return applyReconnectionShift(source, dstSd, dstHit, dstPixel);
#elif RESTIRPT_SHIFT_MAPPING == RESTIRPT_SHIFT_MAPPING_RANDOM_REPLAY
        return applyRandomReplayShift(source, dstSd, dstHit, dstPixel);
#elif RESTIRPT_SHIFT_MAPPING == RESTIRPT_SHIFT_MAPPING_HYBRID
        return applyHybridShift(source, dstSd, dstHit, dstPixel, hybridData);
#else
        switch (ShiftMapping(params.shiftMapping))
        {
        case ShiftMapping::Reconnection:
            return applyReconnectionShift(source, dstSd, dstHit, dstPixel);
        case ShiftMapping::RandomReplay:
            return applyRandomReplayShift(source, dstSd, dstHit, dstPixel);
        case ShiftMapping::Hybrid:
            return applyHybridShift(source, dstSd, dstHit, dstPixel, hybridData);
        default:
            break;
        }

        ShiftMappingResult result = {};
        result.originalTarget = source.target;
        result.rejectionReason = 8u;
        return result;
#endif
    }

    ShiftMappingResult applyShiftMapping(PathReservoir source, ShadingData dstSd, HitInfo dstHit, uint2 dstPixel)
    {
        return applyShiftMapping(source, dstSd, dstHit, dstPixel, makeInvalidHybridReconnectionData(0u));
    }
}
