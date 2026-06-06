/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

cbuffer CB
{
    uint2 gAtlasDim;
    uint2 gTileDim;
    uint2 gViewGrid;
    uint2 gSourceView;
}

Texture2D<float4> gSrcColor;
Texture2D<float4> gSrcAlbedo;
Texture2D<float4> gSrcEmission;
Texture2D<float4> gSrcPosW;
Texture2D<float4> gSrcGuideNormalW;
Texture2D<float2> gSrcPNFwidth;
Texture2D<float2> gSrcLinearZ;
Texture2D<float2> gSrcMvec;

RWTexture2D<float4> gColor;
RWTexture2D<float4> gAlbedo;
RWTexture2D<float4> gEmission;
RWTexture2D<float4> gPosW;
RWTexture2D<float4> gGuideNormalW;
RWTexture2D<float2> gPNFwidth;
RWTexture2D<float2> gLinearZ;
RWTexture2D<float2> gMvec;
RWTexture2D<float4> gDstColor;

[numthreads(16, 16, 1)]
void extractMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gTileDim))
        return;

    const uint2 atlasPixel = gSourceView * gTileDim + pixel;
    if (any(atlasPixel >= gAtlasDim))
        return;

    gColor[pixel] = gSrcColor[atlasPixel];
    gAlbedo[pixel] = gSrcAlbedo[atlasPixel];
    gEmission[pixel] = gSrcEmission[atlasPixel];
    gPosW[pixel] = gSrcPosW[atlasPixel];
    gGuideNormalW[pixel] = gSrcGuideNormalW[atlasPixel];
    gPNFwidth[pixel] = gSrcPNFwidth[atlasPixel];
    gLinearZ[pixel] = gSrcLinearZ[atlasPixel];
    gMvec[pixel] = gSrcMvec[atlasPixel];
}

[numthreads(16, 16, 1)]
void insertMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gAtlasDim))
        return;

    const uint2 sourceTileMin = gSourceView * gTileDim;
    const uint2 sourceTileMax = sourceTileMin + gTileDim;

    if (all(pixel >= sourceTileMin) && all(pixel < sourceTileMax))
    {
        gDstColor[pixel] = gSrcColor[pixel - sourceTileMin];
    }
    else
    {
        gDstColor[pixel] = float4(0.f);
    }
}
