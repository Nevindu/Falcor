/***************************************************************************
 LightFieldReprojPathTracer spatial reprojection pass.

 This pass assumes the path tracer has shaded only the source tile of a 2D
 light-field quilt. It fills the other quilt tiles by backward reprojecting
 target G-buffer/V-buffer hits into the source camera and fetching the source
 tile radiance. Invalid target pixels are visualized according to display mode.
***************************************************************************/
#include "Scene/SceneDefines.slangh"
import Scene.Shading;
import Utils.Math.MathHelpers;
import LightFieldReprojPathTracer;

ParameterBlock<LightFieldReprojPathTracer> gLightFieldReprojPathTracer;

bool getSurfaceData(uint2 atlasPixel, out float3 posW, out float3 normalW)
{
    posW = 0.f;
    normalW = float3(0.f, 1.f, 0.f);
    const HitInfo hit = unpackHitInfo(gLightFieldReprojPathTracer.vbuffer[atlasPixel]);
    if (!hit.isValid()) return false;

    VertexData v;
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit h = hit.getTriangleHit();
        v = gScene.getVertexData(h);
    }
    else if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit h = hit.getDisplacedTriangleHit();
        v = gScene.getVertexData(h, float3(0,0,1));
    }
    else if (hit.getType() == HitType::Curve)
    {
        const CurveHit h = hit.getCurveHit();
        v = gScene.getVertexDataFromCurve(h);
    }
    else if (hit.getType() == HitType::SDFGrid)
    {
        // Keep SDF grids conservative in this starter pass.
        return false;
    }
    else return false;

    posW = v.posW;
    normalW = normalize(v.normalW);
    return true;
}

float4 debugViewID(uint2 viewXY, uint2 grid)
{
    uint viewID = viewXY.y * grid.x + viewXY.x;
    float f = (float)viewID / max(1.0, (float)(grid.x * grid.y - 1));
    return float4(f, frac(f * 7.0), frac(f * 13.0), 1.0);
}

void setFallback(uint2 pixel, bool value)
{
    gLightFieldReprojPathTracer.lfFallbackMask[pixel] = value ? 1u : 0u;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint2 atlasPixel = tid.xy;
    let params = gLightFieldReprojPathTracer.params;
    if (any(atlasPixel >= params.frameDim)) return;

    uint2 viewXY = params.getLFViewXY(atlasPixel);
    uint2 localPixel = params.getLFLocalPixel(atlasPixel);
    uint2 sourceOrigin = params.lfSourceView * params.lfViewDim;

    if (params.lfDisplayMode == 4)
    {
        setFallback(atlasPixel, false);
        gLightFieldReprojPathTracer.outputColor[atlasPixel] = debugViewID(viewXY, params.lfViewGrid);
        return;
    }

    // Source tile already contains path-traced radiance.
    if (all(viewXY == params.lfSourceView))
    {
        setFallback(atlasPixel, false);
        return;
    }

    if (params.lfDisplayMode == 1)
    {
        setFallback(atlasPixel, false);
        gLightFieldReprojPathTracer.outputColor[atlasPixel] = float4(0, 0, 0, 1);
        return;
    }

    float3 Pt, Nt;
    bool targetValid = getSurfaceData(atlasPixel, Pt, Nt);
    if (!targetValid)
    {
        setFallback(atlasPixel, true);
        gLightFieldReprojPathTracer.outputColor[atlasPixel] = (params.lfDisplayMode == 2) ? float4(0,0,0,1) : float4(1,0,0,1);
        return;
    }

    // Backward reprojection: target world point -> source/base camera pixel.
    float4 clip = mul(gScene.camera.data.viewProjMatNoJitter, float4(Pt, 1.0));
    if (clip.w <= 0.0)
    {
        setFallback(atlasPixel, true);
        gLightFieldReprojPathTracer.outputColor[atlasPixel] = (params.lfDisplayMode == 2) ? float4(0,0,0,1) : float4(1,0,0,1);
        return;
    }

    float3 ndc = clip.xyz / clip.w;
    float2 sourceUV = float2(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
    bool inside = all(sourceUV >= 0.0.xx) && all(sourceUV <= 1.0.xx) && ndc.z >= 0.0 && ndc.z <= 1.0;
    uint2 srcLocal = uint2(clamp(sourceUV * float2(params.lfViewDim), 0.0.xx, float2(params.lfViewDim - uint2(1, 1))));
    uint2 srcAtlas = sourceOrigin + srcLocal;

    float3 Ps, Ns;
    bool sourceValid = inside && getSurfaceData(srcAtlas, Ps, Ns);

    bool valid = sourceValid;
    valid = valid && (distance(Ps, Pt) < params.lfPositionThreshold);
    valid = valid && (dot(Ns, Nt) > params.lfNormalDotThreshold);

    if (params.lfDisplayMode == 2)
    {
        setFallback(atlasPixel, !valid);
        gLightFieldReprojPathTracer.outputColor[atlasPixel] = valid ? float4(1,1,1,1) : float4(0,0,0,1);
    }
    else if (params.lfDisplayMode == 3)
    {
        setFallback(atlasPixel, !valid);
        gLightFieldReprojPathTracer.outputColor[atlasPixel] = valid ? gLightFieldReprojPathTracer.outputColor[srcAtlas] : float4(1,0,0,1);
    }
    else
    {
        setFallback(atlasPixel, !valid);
        gLightFieldReprojPathTracer.outputColor[atlasPixel] = valid ? gLightFieldReprojPathTracer.outputColor[srcAtlas] : float4(1,0,0,1);
    }
}
