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

#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.Intersection;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;

cbuffer CB
{
    uint gFrameCount;
    uint gPRNGDimension;
}

Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW;

RWTexture2D<float4> gOutputColor;

struct DirectLightingResources
{
    EnvMapSampler envMapSampler;
    EmissiveLightSampler emissiveSampler;
};

ParameterBlock<DirectLightingResources> gDirectLighting;

#define is_valid(name) (is_valid_##name != 0)
static const uint kSamplesPerPixel = SAMPLES_PER_PIXEL;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEmissiveMaterials = USE_EMISSIVE_MATERIALS;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const bool kUseMIS = USE_MIS;
static const bool kUseBRDFSampling = USE_BRDF_SAMPLING;
static const float kRayTMax = FLT_MAX;

enum class LightType
{
    EnvMap,
    Emissive,
    Analytic
};

struct LightSample
{
    float3 Li;
    float pdf;
    float3 origin;
    float distance;
    float3 dir;
    uint lightType;
};

struct PathVertex
{
    float3 pos;
    float3 faceNormal;
    bool frontFacing;

    __init(float3 pos, float3 faceNormal, bool frontFacing)
    {
        this.pos = pos;
        this.faceNormal = faceNormal;
        this.frontFacing = frontFacing;
    }

    float3 getRayOrigin(float3 rayDir)
    {
        return computeRayOrigin(pos, dot(faceNormal, rayDir) >= 0.f ? faceNormal : -faceNormal);
    }

    float3 getOrientedFaceNormal()
    {
        return frontFacing ? faceNormal : -faceNormal;
    }
};

struct ShadowRayData
{
    bool visible;
};

struct BRDFRayData
{
    PackedHitInfo hit;
    float t;
};

ShadingData loadShadingData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir)
{
    VertexData v = {};
    uint materialID = {};

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        v = gScene.getVertexData(triangleHit);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
        v = gScene.getVertexData(displacedTriangleHit, -rayDir);
        materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)
    if (hit.getType() == HitType::Curve)
    {
        const CurveHit curveHit = hit.getCurveHit();
        v = gScene.getVertexDataFromCurve(curveHit);
        materialID = gScene.getMaterialID(curveHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)
    if (hit.getType() == HitType::SDFGrid)
    {
        const SDFGridHit sdfGridHit = hit.getSDFGridHit();
        v = gScene.getVertexDataFromSDFGrid(sdfGridHit, rayOrigin, rayDir);
        materialID = gScene.getMaterialID(sdfGridHit.instanceID);
    }
#endif

    return gScene.materials.prepareShadingData(v, materialID, -rayDir);
}

float3 getPrimaryRayDir(const uint2 pixel, const uint2 frameDim)
{
    if (is_valid(gViewW))
        return -gViewW[pixel].xyz;

    return gScene.camera.computeRayPinhole(pixel, frameDim).dir;
}

bool traceShadowRay(const float3 origin, const float3 dir, const float distance)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.f;
    ray.TMax = distance;

    ShadowRayData rayData;
    rayData.visible = false;

    TraceRay(
        gScene.rtAccel,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff,
        0,
        getRayTypeCount(),
        0,
        ray,
        rayData
    );

    return rayData.visible;
}

bool traceBRDFRay(const float3 origin, const float3 dir, out HitInfo hit, out float hitT)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.f;
    ray.TMax = kRayTMax;

    BRDFRayData rayData;
    rayData.hit = makeInvalidHit().pack();
    rayData.t = kRayTMax;

    TraceRay(
        gScene.rtAccel,
        RAY_FLAG_NONE,
        0xff,
        1,
        getRayTypeCount(),
        1,
        ray,
        rayData
    );

    hit = HitInfo(rayData.hit);
    hitT = rayData.t;
    return hit.isValid();
}

float evalLightMIS(float lightPdf, float bsdfPdf)
{
    if (!kUseMIS || !kUseBRDFSampling)
        return 1.f;

    return lightPdf > 0.f ? lightPdf / (lightPdf + bsdfPdf) : 0.f;
}

float evalBRDFMIS(float bsdfPdf, float lightPdf)
{
    if (!kUseMIS)
        return 1.f;

    return bsdfPdf > 0.f ? bsdfPdf / (bsdfPdf + lightPdf) : 0.f;
}

void getLightTypeSelectionProbabilities(out float p[3])
{
    p[0] = kUseEnvLight ? 1.f : 0.f;
    p[1] = kUseEmissiveLights ? 1.f : 0.f;
    p[2] = kUseAnalyticLights ? 1.f : 0.f;

    float sum = p[0] + p[1] + p[2];
    if (sum == 0.f) return;

    float invSum = 1.f / sum;
    p[0] *= invSum;
    p[1] *= invSum;
    p[2] *= invSum;
}

float getLightTypeSelectionProbability(const uint lightType)
{
    float p[3];
    getLightTypeSelectionProbabilities(p);
    return lightType < 3 ? p[lightType] : 0.f;
}

bool selectLightType(out uint lightType, out float pdf, inout SampleGenerator sg)
{
    float p[3];
    getLightTypeSelectionProbabilities(p);

    float u = sampleNext1D(sg);
    [unroll]
    for (lightType = 0; lightType < 3; ++lightType)
    {
        if (u < p[lightType])
        {
            pdf = p[lightType];
            return true;
        }
        u -= p[lightType];
    }

    lightType = 0;
    pdf = 0.f;
    return false;
}

bool generateEnvMapSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
{
    ls = {};

#if USE_ENV_LIGHT
    EnvMapSample lightSample;
    if (!gDirectLighting.envMapSampler.sample(sampleNext2D(sg), lightSample)) return false;

    ls.Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0.f);
    ls.pdf = lightSample.pdf;
    ls.origin = vertex.getRayOrigin(lightSample.dir);
    ls.distance = kRayTMax;
    ls.dir = lightSample.dir;
    ls.lightType = (uint)LightType::EnvMap;

    return any(ls.Li > 0.f);
#else
    return false;
#endif
}

bool generateEmissiveSample(const PathVertex vertex, const bool upperHemisphere, inout SampleGenerator sg, out LightSample ls)
{
    ls = {};

#if USE_EMISSIVE_LIGHTS
    TriangleLightSample tls;
    if (!gDirectLighting.emissiveSampler.sampleLight(vertex.pos, vertex.getOrientedFaceNormal(), upperHemisphere, sg, tls)) return false;

    ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0.f);
    ls.pdf = tls.pdf;
    float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
    ls.origin = vertex.getRayOrigin(lightPos - vertex.pos);
    float3 toLight = lightPos - ls.origin;
    ls.distance = length(toLight);
    ls.dir = normalize(toLight);
    ls.lightType = (uint)LightType::Emissive;

    return any(ls.Li > 0.f);
#else
    return false;
#endif
}

bool generateAnalyticSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
{
    ls = {};
    if (!kUseAnalyticLights)
        return false;

    const uint lightCount = gScene.getLightCount();
    if (lightCount == 0)
        return false;

    const uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);

    AnalyticLightSample lightSample;
    if (!sampleLight(vertex.pos, gScene.getLight(lightIndex), sg, lightSample))
        return false;

    ls.pdf = lightSample.pdf / lightCount;
    ls.Li = lightSample.Li * lightCount;
    ls.origin = vertex.getRayOrigin(lightSample.dir);
    ls.distance = lightSample.distance;
    ls.dir = lightSample.dir;
    ls.lightType = (uint)LightType::Analytic;

    return any(ls.Li > 0.f);
}

bool generateLightSample(const PathVertex vertex, const bool sampleUpperHemisphere, const bool sampleLowerHemisphere, inout SampleGenerator sg, out LightSample ls)
{
    ls = {};

    uint lightType;
    float selectionPdf;
    if (!selectLightType(lightType, selectionPdf, sg)) return false;

    bool valid = false;
    if (lightType == (uint)LightType::EnvMap) valid = generateEnvMapSample(vertex, sg, ls);
    if (lightType == (uint)LightType::Emissive)
    {
        bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;
        valid = generateEmissiveSample(vertex, upperHemisphere, sg, ls);
    }
    if (lightType == (uint)LightType::Analytic) valid = generateAnalyticSample(vertex, sg, ls);
    if (!valid) return false;

    float NdotL = dot(vertex.getOrientedFaceNormal(), ls.dir);
    if ((!sampleUpperHemisphere && NdotL >= -kMinCosTheta) || (!sampleLowerHemisphere && NdotL <= kMinCosTheta))
        return false;

    ls.pdf *= selectionPdf;
    ls.Li /= selectionPdf;
    return ls.pdf > 0.f && any(ls.Li > 0.f);
}

float3 evalDirectLightSample(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg)
{
    const uint lobeTypes = mi.getLobeTypes(sd);
    const bool sampleUpperHemisphere = (lobeTypes & uint(LobeType::NonDeltaReflection)) != 0;
    const bool sampleLowerHemisphere = (lobeTypes & uint(LobeType::NonDeltaTransmission)) != 0;
    if (!sampleUpperHemisphere && !sampleLowerHemisphere)
        return float3(0.f);

    PathVertex vertex = PathVertex(sd.posW, sd.faceN, sd.frontFacing);
    LightSample lightSample;
    if (!generateLightSample(vertex, sampleUpperHemisphere, sampleLowerHemisphere, sg, lightSample))
        return float3(0.f);

    if (!traceShadowRay(lightSample.origin, lightSample.dir, lightSample.distance))
        return float3(0.f);

    float misWeight = 1.f;
    if (lightSample.lightType != (uint)LightType::Analytic)
    {
        const float bsdfPdf = mi.evalPdf(sd, lightSample.dir, true);
        misWeight = evalLightMIS(lightSample.pdf, bsdfPdf);
    }

    return misWeight * mi.eval(sd, lightSample.dir, sg) * lightSample.Li;
}

float evalEnvMapPdf(const float3 dir)
{
#if USE_ENV_LIGHT
    return getLightTypeSelectionProbability((uint)LightType::EnvMap) * gDirectLighting.envMapSampler.evalPdf(dir);
#else
    return 0.f;
#endif
}

float evalEmissivePdf(const ShadingData sd, const IMaterialInstance mi, const HitInfo hit, const ShadingData hitSd)
{
#if USE_EMISSIVE_LIGHTS && SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() != HitType::Triangle)
        return 0.f;

    const uint lobeTypes = mi.getLobeTypes(sd);
    const bool sampleUpperHemisphere = (lobeTypes & uint(LobeType::NonDeltaReflection)) != 0;
    const bool sampleLowerHemisphere = (lobeTypes & uint(LobeType::NonDeltaTransmission)) != 0;
    const bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;

    TriangleHit triangleHit = hit.getTriangleHit();
    TriangleLightHit lightHit;
    lightHit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
    lightHit.posW = hitSd.posW;
    lightHit.normalW = hitSd.getOrientedFaceNormal();

    PathVertex vertex = PathVertex(sd.posW, sd.faceN, sd.frontFacing);
    return getLightTypeSelectionProbability((uint)LightType::Emissive) *
        gDirectLighting.emissiveSampler.evalPdf(vertex.pos, vertex.getOrientedFaceNormal(), upperHemisphere, lightHit);
#else
    return 0.f;
#endif
}

float3 evalDirectBRDFSample(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg)
{
    if (!kUseBRDFSampling)
        return float3(0.f);

    BSDFSample bsdfSample = {};
    if (!mi.sample(sd, sg, bsdfSample, true) || bsdfSample.pdf <= 0.f || !any(bsdfSample.weight > 0.f))
        return float3(0.f);

    const float3 dir = normalize(bsdfSample.wo);
    PathVertex vertex = PathVertex(sd.posW, sd.faceN, sd.frontFacing);
    const float3 origin = vertex.getRayOrigin(dir);

    HitInfo hit;
    float hitT;
    const bool hitScene = traceBRDFRay(origin, dir, hit, hitT);
    if (!hitScene)
    {
#if USE_ENV_LIGHT
        const float3 Le = gDirectLighting.envMapSampler.eval(dir);
        const float lightPdf = evalEnvMapPdf(dir);
        return evalBRDFMIS(bsdfSample.pdf, lightPdf) * bsdfSample.weight * Le;
#else
        return float3(0.f);
#endif
    }

#if USE_EMISSIVE_LIGHTS
    const ShadingData hitSd = loadShadingData(hit, origin, dir);
    let lod = ExplicitLodTextureSampler(0.f);
    let hitMi = gScene.materials.getMaterialInstance(hitSd, lod);
    const float3 Le = hitMi.getProperties(hitSd).emission;
    if (!any(Le > 0.f))
        return float3(0.f);

    const float lightPdf = evalEmissivePdf(sd, mi, hit, hitSd);
    return evalBRDFMIS(bsdfSample.pdf, lightPdf) * bsdfSample.weight * Le;
#else
    return float3(0.f);
#endif
}

float3 shadePrimaryHit(const uint2 pixel, const uint2 frameDim, const HitInfo hit)
{
    const float3 rayOrigin = gScene.camera.getPosition();
    const float3 rayDir = getPrimaryRayDir(pixel, frameDim);

    ShadingData sd = loadShadingData(hit, rayOrigin, rayDir);

    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    for (uint i = 0; i < gPRNGDimension; i++)
        sampleNext1D(sg);

    float3 emission = float3(0.f);

    if (kUseEmissiveMaterials)
        emission = mi.getProperties(sd).emission;

    float3 direct = float3(0.f);
    for (uint i = 0; i < kSamplesPerPixel; ++i)
    {
        direct += evalDirectLightSample(sd, mi, sg);
        direct += evalDirectBRDFSample(sd, mi, sg);
    }

    return emission + direct / float(kSamplesPerPixel);
}

[shader("miss")]
void shadowMiss(inout ShadowRayData rayData)
{
    rayData.visible = true;
}

[shader("miss")]
void brdfMiss(inout BRDFRayData rayData)
{}

[shader("anyhit")]
void shadowTriangleMeshAnyHit(inout ShadowRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}

[shader("closesthit")]
void shadowTriangleMeshClosestHit(inout ShadowRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{}

[shader("anyhit")]
void brdfTriangleMeshAnyHit(inout BRDFRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}

[shader("closesthit")]
void brdfTriangleMeshClosestHit(inout BRDFRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;

    HitInfo hitInfo = HitInfo(triangleHit);
    rayData.hit = hitInfo.pack();
    rayData.t = RayTCurrent();
}

[shader("intersection")]
void displacedTriangleMeshIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    DisplacedTriangleMeshIntersector::Attribs attribs;
    float t;
    if (DisplacedTriangleMeshIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
        ReportHit(t, 0, attribs);
}

[shader("closesthit")]
void shadowDisplacedTriangleMeshClosestHit(inout ShadowRayData rayData, DisplacedTriangleMeshIntersector::Attribs attribs)
{}

[shader("closesthit")]
void brdfDisplacedTriangleMeshClosestHit(inout BRDFRayData rayData, DisplacedTriangleMeshIntersector::Attribs attribs)
{
    DisplacedTriangleHit displacedTriangleHit;
    displacedTriangleHit.instanceID = getGeometryInstanceID();
    displacedTriangleHit.primitiveIndex = PrimitiveIndex();
    displacedTriangleHit.barycentrics = attribs.barycentrics;
    displacedTriangleHit.displacement = attribs.displacement;

    HitInfo hitInfo = HitInfo(displacedTriangleHit);
    rayData.hit = hitInfo.pack();
    rayData.t = RayTCurrent();
}

[shader("intersection")]
void curveIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    CurveIntersector::Attribs attribs;
    float t;
    if (CurveIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
        ReportHit(t, 0, attribs);
}

[shader("closesthit")]
void shadowCurveClosestHit(inout ShadowRayData rayData, CurveIntersector::Attribs attribs)
{}

[shader("closesthit")]
void brdfCurveClosestHit(inout BRDFRayData rayData, CurveIntersector::Attribs attribs)
{
    CurveHit curveHit;
    curveHit.instanceID = getGeometryInstanceID();
    curveHit.primitiveIndex = PrimitiveIndex();
    curveHit.barycentrics = attribs.barycentrics;

    HitInfo hitInfo = HitInfo(curveHit);
    rayData.hit = hitInfo.pack();
    rayData.t = RayTCurrent();
}

[shader("intersection")]
void sdfGridIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    SDFGridHitData sdfGridHitData;
    float t;
    if (SDFGridIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), sdfGridHitData, t))
        ReportHit(t, 0, sdfGridHitData);
}

[shader("closesthit")]
void shadowSdfGridClosestHit(inout ShadowRayData rayData, SDFGridHitData sdfGridHitData)
{}

[shader("closesthit")]
void brdfSdfGridClosestHit(inout BRDFRayData rayData, SDFGridHitData sdfGridHitData)
{
    SDFGridHit sdfGridHit;
    sdfGridHit.instanceID = getGeometryInstanceID();
    sdfGridHit.hitData = sdfGridHitData;

    HitInfo hitInfo = HitInfo(sdfGridHit);
    rayData.hit = hitInfo.pack();
    rayData.t = RayTCurrent();
}

[shader("raygeneration")]
void rayGen()
{
    const uint2 pixel = DispatchRaysIndex().xy;
    const uint2 frameDim = DispatchRaysDimensions().xy;
    if (any(pixel >= frameDim))
        return;

    const HitInfo hit = HitInfo(gVBuffer[pixel]);
    float3 color = float3(0.f);

    if (hit.isValid())
    {
        color = shadePrimaryHit(pixel, frameDim, hit);
    }
    else if (kUseEnvBackground)
    {
        color = gScene.envMap.eval(getPrimaryRayDir(pixel, frameDim));
    }

    gOutputColor[pixel] = float4(color, 1.f);
}
