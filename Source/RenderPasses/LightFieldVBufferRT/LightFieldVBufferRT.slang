/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
__exported import Utils.Timing.GpuTimer;
__exported import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
__exported import Scene.Shading;
import Rendering.Materials.IMaterialInstance;

RWTexture2D<PackedHitInfo> gVBuffer;
RWTexture2D<float4> gPosW;
RWTexture2D<float4> gGuideNormalW;
RWTexture2D<float2> gPosNormalFwidth;
RWTexture2D<float> gDepth;
RWTexture2D<float2> gLinearZ;
RWTexture2D<float4> gNormalWRoughnessMaterialID;
RWTexture2D<float2> gMotionVector;
RWTexture2D<float4> gViewW;
RWTexture2D<uint> gTime;
RWTexture2D<float> gMask;

#define is_valid(name) (is_valid_##name != 0)

#if !defined(COMPUTE_DEPTH_OF_FIELD) || !defined(USE_ALPHA_TEST) || !defined(RAY_FLAGS)
#error "Not all defines are set!"
#endif

struct LightFieldVBufferRT
{
    static const bool kComputeDepthOfField = COMPUTE_DEPTH_OF_FIELD;
    static const bool kUseAlphaTest = USE_ALPHA_TEST;
    static const uint kRayFlags = RAY_FLAGS;

    uint2 frameDim;          ///< Atlas/quilt dimension.
    uint frameCount;
    uint2 lfViewGrid;        ///< Number of views in the atlas. Default {6,6}.
    uint2 lfViewDim;         ///< Per-view tile dimension, computed on host as frameDim / lfViewGrid.
    uint2 lfSourceView;      ///< Source/anchor view. Default {2,2} for a 6x6 grid.
    float2 lfBaseline;       ///< World-space baseline step along camera right/up.

    uint2 getViewXY(uint2 pixel)
    {
        return min(pixel / max(lfViewDim, uint2(1, 1)), lfViewGrid - uint2(1, 1));
    }

    uint getViewID(uint2 pixel)
    {
        uint2 viewXY = getViewXY(pixel);
        return viewXY.y * lfViewGrid.x + viewXY.x;
    }

    uint2 getLocalPixel(uint2 pixel)
    {
        return pixel - getViewXY(pixel) * lfViewDim;
    }

    float3 getViewOffsetW(uint2 viewXY)
    {
        // Parallel-camera light-field model. We intentionally do not toe-in.
        // cameraU/cameraV are Falcor camera frame vectors; normalize before using as a pure baseline direction.
        float2 centered = float2(viewXY) - float2(lfSourceView);
        float3 rightW = normalize(gScene.camera.data.cameraU);
        float3 upW = normalize(gScene.camera.data.cameraV);
        return centered.x * lfBaseline.x * rightW + centered.y * lfBaseline.y * upW;
    }

    Ray generateRay(uint2 pixel)
    {
        uint2 viewXY = getViewXY(pixel);
        uint2 localPixel = getLocalPixel(pixel);

        // Generate a normal pinhole ray for the local per-view pixel and then translate its origin.
        // This gives a rectified/parallel light-field camera grid.
        Ray ray = gScene.camera.computeRayPinhole(localPixel, lfViewDim);
        ray.origin += getViewOffsetW(viewXY);
        return ray;
    }

    float3 computeDdxPosW(float3 posW, float3 normW)
    {
        float3 projRight = normalize(cross(normW, cross(normW, gScene.camera.data.cameraU)));
        float distanceToHit = length(posW - gScene.camera.data.posW);
        float2 invFrameDim = 1.f / float2(max(frameDim, uint2(1, 1)));
        float2 ddNdc = float2(2.f, -2.f) * invFrameDim;
        float denom = dot(normalize(gScene.camera.data.cameraU), projRight);
        if (abs(denom) < 1e-4f) return 0.f;
        return (distanceToHit * ddNdc.x / denom) * projRight;
    }

    float3 computeDdyPosW(float3 posW, float3 normW)
    {
        float3 projUp = normalize(cross(normW, cross(normW, gScene.camera.data.cameraV)));
        float distanceToHit = length(posW - gScene.camera.data.posW);
        float2 invFrameDim = 1.f / float2(max(frameDim, uint2(1, 1)));
        float2 ddNdc = float2(2.f, -2.f) * invFrameDim;
        float denom = dot(normalize(gScene.camera.data.cameraV), projUp);
        if (abs(denom) < 1e-4f) return 0.f;
        return (distanceToHit * ddNdc.y / denom) * projUp;
    }

    void writeHit(uint2 pixel, float3 rayOrigin, float3 rayDir, const HitInfo hit)
    {
        gVBuffer[pixel] = hit.pack();

        VertexData v;
        float depth = 1.f;
        float2 linearZ = float2(0.f);
        float4 normalRoughnessMaterialID = float4(0.f);
        float4 posW = float4(0.f);
        float4 guideNormalW = float4(0.f);
        float2 posNormalFwidth = float2(0.f);
        float3 prevPosW = float3(0.f);
        float2 motionVector = float2(0.f);
        uint materialID = 0u;

        if (hit.getType() == HitType::Triangle)
        {
            const TriangleHit triangleHit = hit.getTriangleHit();
            materialID = gScene.getMaterialID(triangleHit.instanceID);
            v = gScene.getVertexData(triangleHit);
            prevPosW = gScene.getPrevPosW(triangleHit);
        }
        else if (hit.getType() == HitType::DisplacedTriangle)
        {
            const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
            materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
            v = gScene.getVertexData(displacedTriangleHit, -rayDir);
            prevPosW = gScene.getPrevPosW(displacedTriangleHit);
        }
        else if (hit.getType() == HitType::Curve)
        {
            const CurveHit curveHit = hit.getCurveHit();
            materialID = gScene.getMaterialID(curveHit.instanceID);
            v = gScene.getVertexDataFromCurve(curveHit);
            prevPosW = gScene.getPrevPosWFromCurve(curveHit);
        }
        else if (hit.getType() == HitType::SDFGrid)
        {
            const SDFGridHit sdfGridHit = hit.getSDFGridHit();
            materialID = gScene.getMaterialID(sdfGridHit.instanceID);
            v = gScene.getVertexDataFromSDFGrid(sdfGridHit, rayOrigin, rayDir);

            prevPosW = gScene.getPrevPosWFromSDFGrid(sdfGridHit, v.posW);
        }

        if (hit.getType() == HitType::Triangle || hit.getType() == HitType::DisplacedTriangle || hit.getType() == HitType::Curve ||
            hit.getType() == HitType::SDFGrid)
        {
            ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);
            let lod = ExplicitLodTextureSampler(0.f);
            let mi = gScene.materials.getMaterialInstance(sd, lod);
            let bsdfProperties = mi.getProperties(sd);

            // Compute depth similar to raster (NDC).
            float4 curPosH = mul(gScene.camera.data.viewProjMatNoJitter, float4(sd.posW, 1.f));
            depth = curPosH.z / curPosH.w;
            float curLinearZ = curPosH.w;

            float3 ddxPosW = computeDdxPosW(sd.posW, sd.faceN);
            float3 ddyPosW = computeDdyPosW(sd.posW, sd.faceN);
            float4 curPosH_dx = mul(gScene.camera.data.viewProjMatNoJitter, float4(sd.posW + ddxPosW, 1.f));
            float4 curPosH_dy = mul(gScene.camera.data.viewProjMatNoJitter, float4(sd.posW + ddyPosW, 1.f));
            linearZ = float2(curLinearZ, max(abs(curPosH_dx.w - curLinearZ), abs(curPosH_dy.w - curLinearZ)));

            float3 normal = normalize(bsdfProperties.guideNormal);
            if (sd.mtl.getMaterialType() == MaterialType::Hair) normal = sd.frame.T;
            float2 octNormal = ndir_to_oct_unorm(normal);
            normalRoughnessMaterialID = float4(octNormal, bsdfProperties.roughness, 0.f);
            posW = float4(sd.posW, 1.f);
            guideNormalW = float4(normal, 0.f);
            posNormalFwidth = float2(length(ddxPosW) + length(ddyPosW), 0.f);

            // Compute motion vector.
            float2 pixelPos = pixel + float2(0.5f, 0.5f);
            float4 prevPosH = mul(gScene.camera.data.prevViewProjMatNoJitter, float4(prevPosW, 1.f));
            // Remove camera jitter from motion vector
            motionVector = calcMotionVector(pixelPos, prevPosH, frameDim) + float2(gScene.camera.data.jitterX, -gScene.camera.data.jitterY);
        }

        if (is_valid(gPosW))
            gPosW[pixel] = posW;
        if (is_valid(gGuideNormalW))
            gGuideNormalW[pixel] = guideNormalW;
        if (is_valid(gPosNormalFwidth))
            gPosNormalFwidth[pixel] = posNormalFwidth;
        if (is_valid(gDepth))
            gDepth[pixel] = depth;
        if (is_valid(gLinearZ))
            gLinearZ[pixel] = linearZ;
        if (is_valid(gNormalWRoughnessMaterialID))
            gNormalWRoughnessMaterialID[pixel] = normalRoughnessMaterialID;
        if (is_valid(gMotionVector))
            gMotionVector[pixel] = motionVector;
        if (is_valid(gMask))
            gMask[pixel] = 1.0f;
    }

    void writeMiss(uint2 pixel, float3 rayOrigin, float3 rayDir)
    {
        gVBuffer[pixel] = {};

        if (is_valid(gPosW))
            gPosW[pixel] = {};
        if (is_valid(gGuideNormalW))
            gGuideNormalW[pixel] = {};
        if (is_valid(gPosNormalFwidth))
            gPosNormalFwidth[pixel] = {};
        if (is_valid(gDepth))
            gDepth[pixel] = 1.f;
        if (is_valid(gLinearZ))
            gLinearZ[pixel] = float2(1e38f, 0.f);
        if (is_valid(gNormalWRoughnessMaterialID))
        {
            const float2 octNormal = ndir_to_oct_unorm(float3(0.2215f, 0.2215f, 0.2215f));
            gNormalWRoughnessMaterialID[pixel] = float4(octNormal, 1.f, 0.f);
        }
        if (is_valid(gMotionVector))
            gMotionVector[pixel] = {};
        if (is_valid(gMask))
            gMask[pixel] = 0.0f;
    }

    void writeAux(uint2 pixel, const Ray ray)
    {
        // Write view direction.
        if (is_valid(gViewW))
            gViewW[pixel] = float4(-ray.dir, 0.f);
    }

    void beginTime(inout GpuTimer timer)
    {
        if (is_valid(gTime))
            timer.start();
    }

    void endTime(uint2 pixel, inout GpuTimer timer)
    {
        if (is_valid(gTime))
            gTime[pixel] = timer.getElapsed();
    }
};
