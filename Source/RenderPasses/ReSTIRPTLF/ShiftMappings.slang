/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

import ReSTIRPTLF;
import LoadShadingData;
import ShiftMapping_Reconnection;
import ShiftMapping_RandomReplay;

/** Shared shift-mapping dispatch for ReSTIR PT reuse passes.

    Spatial reuse and future temporal reuse should call only this function, so
    new maps do not leak shift-specific logic into reservoir plumbing.
*/
extension ReSTIRPTLF
{
    ShiftMappingResult applyShiftMapping(PathReservoir source, ShadingData dstSd, HitInfo dstHit, uint2 dstPixel)
    {
        switch (ShiftMapping(params.shiftMapping))
        {
        case ShiftMapping::Reconnection:
            return applyReconnectionShift(source, dstSd, dstHit, dstPixel);
        case ShiftMapping::RandomReplay:
            return applyRandomReplayShift(source, dstSd, dstHit, dstPixel);
        default:
            break;
        }

        ShiftMappingResult result = {};
        result.originalTarget = source.target;
        result.rejectionReason = 8u;
        return result;
    }
}
