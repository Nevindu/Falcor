/***************************************************************************
 # Copyright (c) 2026.  All rights reserved.
 **************************************************************************/
import Params;
import Scene.Scene;
__exported import Utils.Math.Ray;

struct LFAddress
{
    uint view;
    uint2 localPixel;
    uint2 tile;
    bool valid;
};

uint2 getLightFieldViewDim(const ReSTIRPTLFParams params)
{
    if (params.lfEnabled == 0u) return params.frameDim;
    if (params.lfViewDim.x == 0u || params.lfViewDim.y == 0u) return params.frameDim;
    return params.lfViewDim;
}

LFAddress decodeLightFieldPixel(const ReSTIRPTLFParams params, const uint2 quiltPixel)
{
    LFAddress a;
    a.view = 0u;
    a.localPixel = quiltPixel;
    a.tile = uint2(0u, 0u);
    a.valid = all(quiltPixel < params.frameDim);

    if (params.lfEnabled == 0u) return a;

    const uint2 viewDim = max(getLightFieldViewDim(params), uint2(1u, 1u));
    const uint viewsPerRow = max(params.lfViewsPerRow, 1u);

    a.tile = quiltPixel / viewDim;
    a.localPixel = quiltPixel - a.tile * viewDim;
    a.view = a.tile.y * viewsPerRow + a.tile.x;
    a.valid = all(quiltPixel < params.frameDim) && a.view < max(params.lfViewCount, 1u);
    return a;
}

uint2 encodeLightFieldPixel(const ReSTIRPTLFParams params, const uint view, const uint2 localPixel)
{
    if (params.lfEnabled == 0u) return localPixel;

    const uint viewsPerRow = max(params.lfViewsPerRow, 1u);
    const uint2 tile = uint2(view % viewsPerRow, view / viewsPerRow);
    return tile * getLightFieldViewDim(params) + localPixel;
}

bool isLightFieldPixelActive(const ReSTIRPTLFParams params, const uint2 quiltPixel)
{
    return decodeLightFieldPixel(params, quiltPixel).valid;
}

bool isSameLightFieldViewPixel(const ReSTIRPTLFParams params, const uint2 centerPixel, const int2 candidatePixel)
{
    if (params.lfEnabled == 0u) return true;
    if (any(candidatePixel < int2(0, 0))) return false;

    const uint2 candidate = uint2(candidatePixel);
    if (any(candidate >= params.frameDim)) return false;

    const LFAddress center = decodeLightFieldPixel(params, centerPixel);
    const LFAddress neighbor = decodeLightFieldPixel(params, candidate);
    return center.valid && neighbor.valid && center.view == neighbor.view;
}

float getLightFieldViewOffset(const ReSTIRPTLFParams params, const uint view)
{
    const uint viewCount = max(params.lfViewCount, 1u);
    if (viewCount <= 1u) return 0.f;
    return (float(view) - 0.5f * float(viewCount - 1u)) * params.lfViewSpacing;
}

float3 getLightFieldCameraPositionForView(const ReSTIRPTLFParams params, const uint view)
{
    if (params.lfEnabled == 0u) return gScene.camera.data.posW;
    const float3 right = normalize(gScene.camera.data.cameraU);
    return gScene.camera.data.posW + right * getLightFieldViewOffset(params, view);
}

float3 getLightFieldCameraPosition(const ReSTIRPTLFParams params, const uint2 quiltPixel)
{
    if (params.lfEnabled == 0u) return gScene.camera.data.posW;
    const LFAddress a = decodeLightFieldPixel(params, quiltPixel);
    return getLightFieldCameraPositionForView(params, a.view);
}

float2 projectWorldToLightFieldLocalPixel(const ReSTIRPTLFParams params, const float3 posW, const uint view)
{
    const uint2 viewDim = getLightFieldViewDim(params);
    const float3 center = gScene.camera.data.posW;
    const float3 viewCenter = getLightFieldCameraPositionForView(params, view);
    const float3 forward = normalize(gScene.camera.data.cameraW);

    const float denom = dot(posW - viewCenter, forward);
    if (denom <= 1e-6f) return float2(-1.f, -1.f);

    const float focalDistance = max(params.lfFocalDistance, 1e-4f);
    const float3 focalPoint = viewCenter + (posW - viewCenter) * (focalDistance / denom);
    const float3 centerToFocal = focalPoint - center;

    const float3 right = normalize(gScene.camera.data.cameraU);
    const float3 up = normalize(gScene.camera.data.cameraV);
    const float cameraULength = max(length(gScene.camera.data.cameraU), 1e-6f);
    const float cameraVLength = max(length(gScene.camera.data.cameraV), 1e-6f);
    const float cameraWLength = max(length(gScene.camera.data.cameraW), 1e-6f);
    const float z = max(1e-6f, dot(centerToFocal, forward));

    const float2 ndc = float2(
        dot(centerToFocal, right) / z * (cameraWLength / cameraULength),
        dot(centerToFocal, up) / z * (cameraWLength / cameraVLength));
    const float2 pWithJitter = float2(0.5f * (ndc.x + 1.f), 0.5f * (1.f - ndc.y));
    const float2 p = pWithJitter - float2(-gScene.camera.data.jitterX, gScene.camera.data.jitterY);
    return p * float2(viewDim) - float2(0.5f, 0.5f);
}

int getLightFieldAngularProposalCount(const ReSTIRPTLFParams params)
{
    if (params.lfEnabled == 0u) return 0;
    return 2 * int(max(params.lfAngularNeighborRadius, 0u));
}

bool getAngularNeighborPixel(
    const ReSTIRPTLFParams params,
    const uint2 centerPixel,
    const float3 centerPosW,
    const int candidateId,
    out int2 neighborPixel)
{
    neighborPixel = int2(-1, -1);

    if (params.lfEnabled == 0u) return false;

    const LFAddress c = decodeLightFieldPixel(params, centerPixel);
    if (!c.valid) return false;

    const int radius = int(max(params.lfAngularNeighborRadius, 0u));
    if (radius <= 0) return false;

    int dv = candidateId - radius;
    if (dv >= 0) dv += 1;

    const int neighborView = int(c.view) + dv;
    if (neighborView < 0 || neighborView >= int(params.lfViewCount)) return false;

    const float2 projected = projectWorldToLightFieldLocalPixel(params, centerPosW, uint(neighborView));
    const int2 local = int2(round(projected));
    const int2 viewDim = int2(getLightFieldViewDim(params));

    if (any(local < int2(0, 0)) || local.x >= viewDim.x || local.y >= viewDim.y) return false;
    if (abs(projected.y - float(c.localPixel.y)) > params.lfMaxReprojectionErrorPx) return false;

    neighborPixel = int2(encodeLightFieldPixel(params, uint(neighborView), uint2(local)));
    return true;
}

Ray computeLightFieldRay(const ReSTIRPTLFParams params, const uint2 quiltPixel)
{
    if (params.lfEnabled == 0u)
    {
        return gScene.camera.computeRayPinhole(quiltPixel, params.frameDim);
    }

    const LFAddress a = decodeLightFieldPixel(params, quiltPixel);
    if (!a.valid)
    {
        return Ray(gScene.camera.data.posW, normalize(gScene.camera.data.cameraW), 0.f, 0.f);
    }

    const uint2 viewDim = getLightFieldViewDim(params);
    const Ray baseRay = gScene.camera.computeRayPinhole(a.localPixel, viewDim);

    const float3 center = gScene.camera.data.posW;
    const float3 forward = normalize(gScene.camera.data.cameraW);
    const float3 shiftedCenter = getLightFieldCameraPosition(params, quiltPixel);

    const float focalDistance = max(params.lfFocalDistance, 1e-4f);
    const float focalDenom = max(1e-6f, dot(baseRay.dir, forward));
    const float3 focalPoint = center + baseRay.dir * (focalDistance / focalDenom);

    Ray ray;
    ray.origin = shiftedCenter;
    ray.dir = normalize(focalPoint - shiftedCenter);

    const float invCos = 1.f / max(1e-6f, dot(forward, ray.dir));
    ray.tMin = gScene.camera.data.nearZ * invCos;
    ray.tMax = gScene.camera.data.farZ * invCos;
    return ray;
}
