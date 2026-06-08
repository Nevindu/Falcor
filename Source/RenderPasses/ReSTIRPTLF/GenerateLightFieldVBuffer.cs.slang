/***************************************************************************
 # Copyright (c) 2026.  All rights reserved.
 **************************************************************************/
import Scene.RaytracingInline;
import Params;
import LightField;

static const bool kUseAlphaTest = USE_ALPHA_TEST;

cbuffer CB
{
    ReSTIRPTLFParams params;
}

RWTexture2D<PackedHitInfo> outputVBuffer;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= params.frameDim)) return;

    if (!isLightFieldPixelActive(params, pixel))
    {
        outputVBuffer[pixel] = {};
        return;
    }

    const Ray ray = computeLightFieldRay(params, pixel);
    SceneRayQuery<kUseAlphaTest> sceneRayQuery;
    float hitT;
    const HitInfo hit = sceneRayQuery.traceRay(ray, hitT, RAY_FLAG_NONE);
    PackedHitInfo packed = {};
    if (hit.isValid()) packed = hit.pack();
    outputVBuffer[pixel] = packed;
}
