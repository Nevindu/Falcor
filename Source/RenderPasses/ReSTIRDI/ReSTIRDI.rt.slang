/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
Texture2D<float2> gMotionVectors;

RWTexture2D<float4> gOutputColor;

struct ReSTIRDIResources
{
    EnvMapSampler envMapSampler;
    EmissiveLightSampler emissiveSampler;
};

ParameterBlock<ReSTIRDIResources> gReSTIRDI;

#define is_valid(name) (is_valid_##name != 0)
static const uint kLightCandidateCount = LIGHT_CANDIDATE_COUNT;
static const uint kSpatialNeighborCount = SPATIAL_NEIGHBOR_COUNT;
static const uint kSpatialRadius = SPATIAL_RADIUS;
static const uint kSpatialMCap = SPATIAL_M_CAP;
static const uint kTemporalMCap = TEMPORAL_M_CAP;
static const uint kMaxSpatialSources = SPATIAL_NEIGHBOR_COUNT + 1;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEmissiveMaterials = USE_EMISSIVE_MATERIALS;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const bool kUseTemporalReuse = USE_TEMPORAL_REUSE;
static const bool kUseBRDFSampling = USE_BRDF_SAMPLING;
static const uint kSpatialMISStrategy = SPATIAL_MIS_STRATEGY;
static const bool kUseVisibilityInTarget = USE_VISIBILITY_IN_TARGET;
static const bool kUseVisibilityInSpatialGBH = USE_VISIBILITY_IN_SPATIAL_GBH;
static const bool kUseVisibilityInTemporalMIS = USE_VISIBILITY_IN_TEMPORAL_MIS;
static const float kRayTMax = FLT_MAX;

#if USE_BRDF_SAMPLING
static const uint kBRDFCandidateCount = BRDF_CANDIDATE_COUNT;
#else
static const uint kBRDFCandidateCount = 0;
#endif

static const uint kSpatialMISNone = 0;
static const uint kSpatialMISContribution = 1;
static const uint kSpatialMISGBH = 2;
static const uint kSpatialMISPairwise = 3;

enum class SampledLightType
{
    EnvMap,
    Emissive,
    Analytic
};

struct ReusableLightSample
{
    uint lightType;
    uint valid;
    uint lightIndex;
    uint triangleIndex;

    float3 envDir;
    float candidatePdf;

    float3 lightPosW;
    float lightDistance;

    float3 lightNormalW;
    float pad0;

    float3 Le;
    float pad1;
};

struct DIReservoir
{
    ReusableLightSample y;
    float wSum;
    float target;
    float W;
    uint M;
    uint valid;
    uint selectedInputIndex;
    uint2 pad;
};

StructuredBuffer<DIReservoir> gInputReservoirs;
RWStructuredBuffer<DIReservoir> gOutputReservoirs;
StructuredBuffer<DIReservoir> gFinalReservoirs;
StructuredBuffer<DIReservoir> gPrevReservoirs;

struct TemporalSurface
{
    float3 normalW;
    float linearDepth;
    uint valid;
    uint3 pad;
};

RWStructuredBuffer<TemporalSurface> gCurrentSurfaces;
StructuredBuffer<TemporalSurface> gPrevSurfaces;

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

/// Reconstruct shading data for any hit type encoded in a packed VBuffer hit.
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

/// Return the primary ray direction used to reconstruct the surface stored in the VBuffer.
float3 getPrimaryRayDir(const uint2 pixel, const uint2 frameDim)
{
    if (is_valid(gViewW))
        return -gViewW[pixel].xyz;

    return gScene.camera.computeRayPinhole(pixel, frameDim).dir;
}

/// Load the visible surface for a pixel and return the data needed by reuse filters.
bool loadSurface(const uint2 pixel, const uint2 frameDim, out ShadingData sd, out float linearDepth, out float3 normalW)
{
    sd = {};
    linearDepth = 0.f;
    normalW = float3(0.f);

    const HitInfo hit = HitInfo(gVBuffer[pixel]);
    if (!hit.isValid())
        return false;

    const float3 rayOrigin = gScene.camera.getPosition();
    const float3 rayDir = getPrimaryRayDir(pixel, frameDim);
    sd = loadShadingData(hit, rayOrigin, rayDir);
    linearDepth = distance(rayOrigin, sd.posW);
    normalW = normalize(sd.faceN);
    return true;
}

/// Trace the visibility segment from the shaded point to a sampled light.
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

/// Trace a BSDF-sampled ray and return the first emissive/environment candidate it reaches.
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

/// Cheap luminance target used by RIS/ReSTIR selection.
float luminance(float3 c)
{
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

/// Reject directions unsupported by the material's non-delta reflection/transmission lobes.
bool acceptsDirection(const ShadingData sd, const IMaterialInstance mi, const float3 dir)
{
    const uint lobeTypes = mi.getLobeTypes(sd);
    const bool sampleUpperHemisphere = (lobeTypes & uint(LobeType::NonDeltaReflection)) != 0;
    const bool sampleLowerHemisphere = (lobeTypes & uint(LobeType::NonDeltaTransmission)) != 0;
    if (!sampleUpperHemisphere && !sampleLowerHemisphere)
        return false;

    const float NdotL = dot(sd.frontFacing ? sd.faceN : -sd.faceN, dir);
    if ((!sampleUpperHemisphere && NdotL >= -kMinCosTheta) || (!sampleLowerHemisphere && NdotL <= kMinCosTheta))
        return false;

    return true;
}

/// Convert a reusable light sample into a shadow ray segment and emitted radiance.
bool evalSampleGeometry(const ShadingData sd, const ReusableLightSample y, out float3 origin, out float3 dir, out float distance, out float3 Le)
{
    origin = float3(0.f);
    dir = float3(0.f);
    distance = 0.f;
    Le = float3(0.f);

    if (y.valid == 0)
        return false;

    PathVertex vertex = PathVertex(sd.posW, sd.faceN, sd.frontFacing);

    if (y.lightType == (uint)SampledLightType::EnvMap)
    {
        dir = normalize(y.envDir);
        distance = kRayTMax;
        origin = vertex.getRayOrigin(dir);
        Le = y.Le;
        return any(Le > 0.f);
    }

    if (y.lightType == (uint)SampledLightType::Emissive)
    {
        origin = vertex.getRayOrigin(y.lightPosW - sd.posW);
        float3 toLight = y.lightPosW - origin;
        distance = length(toLight);
        if (distance <= 0.f)
            return false;

        dir = toLight / distance;
        if (dot(y.lightNormalW, -dir) <= 0.f)
            return false;

        Le = y.Le;
        return any(Le > 0.f);
    }

    if (y.lightType == (uint)SampledLightType::Analytic)
    {
        if (y.lightIndex >= gScene.getLightCount())
            return false;

        const LightData light = gScene.getLight(y.lightIndex);
        if (light.type == LightType::Distant)
        {
            dir = normalize(y.envDir);
            distance = kRayTMax;
            origin = vertex.getRayOrigin(dir);
            Le = light.intensity;
            return any(Le > 0.f);
        }

        origin = vertex.getRayOrigin(y.lightPosW - sd.posW);
        float3 toLight = y.lightPosW - origin;
        float distSqr = max(dot(toLight, toLight), kMinLightDistSqr);
        distance = sqrt(distSqr);
        dir = toLight / distance;

        if (light.type == LightType::Rect || light.type == LightType::Sphere || light.type == LightType::Disc)
        {
            if (dot(y.lightNormalW, -dir) <= 0.f)
                return false;
            Le = light.intensity;
            return any(Le > 0.f);
        }
    }

    return false;
}

/// Evaluate the vector direct-lighting contribution for a reusable sample.
float3 evalRawContribution(const ShadingData sd, const IMaterialInstance mi, const ReusableLightSample y, const bool includeVisibility, inout SampleGenerator sg)
{
    float3 origin;
    float3 dir;
    float distance;
    float3 Le;
    if (!evalSampleGeometry(sd, y, origin, dir, distance, Le))
        return float3(0.f);

    if (!acceptsDirection(sd, mi, dir))
        return float3(0.f);

    if (includeVisibility && !traceShadowRay(origin, dir, distance))
        return float3(0.f);

    return mi.eval(sd, dir, sg) * Le;
}

/// Scalar target p_hat used by reservoir updates and MIS balance terms.
float evalTarget(const ShadingData sd, const IMaterialInstance mi, const ReusableLightSample y, const bool includeVisibility, inout SampleGenerator sg)
{
    return max(0.f, luminance(evalRawContribution(sd, mi, y, includeVisibility, sg)));
}

/// Add one weighted candidate stream to a reservoir.
/// M is counted even when the candidate weight is zero; this preserves the final W normalization.
void updateReservoir(
    inout DIReservoir r,
    const ReusableLightSample y,
    const float target,
    const float weight,
    const uint M,
    const uint selectedInputIndex,
    const float u)
{
    r.M += M;

    if (weight <= 0.f || target <= 0.f || y.valid == 0)
        return;

    r.wSum += weight;

    if (u * r.wSum < weight)
    {
        r.y = y;
        r.target = target;
        r.valid = 1;
        r.selectedInputIndex = selectedInputIndex;
    }
}

/// Convert the accumulated weight sum into the selected sample's final reservoir weight.
void finalizeReservoir(inout DIReservoir r)
{
    if (r.valid != 0 && r.target > 0.f && r.M > 0)
        r.W = r.wSum / (float(r.M) * r.target);
    else
    {
        r.W = 0.f;
        r.valid = 0;
    }
}

/// Clamp the effective sample count while preserving the reservoir's expected contribution.
void capReservoirM(inout DIReservoir r, const uint maxM)
{
    if (r.M > maxM && r.M > 0)
    {
        const float scale = float(maxM) / float(r.M);
        r.wSum *= scale;
        r.M = maxM;
    }
}

/// Effective source confidence used by spatial reuse.
uint reservoirConfidence(const DIReservoir r)
{
    return min(r.M, kSpatialMCap);
}

/// Effective source confidence used by temporal reuse.
uint temporalReservoirConfidence(const DIReservoir r)
{
    return min(r.M, kTemporalMCap);
}

/// Compute the probability of choosing env, emissive, or analytic light sampling.
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

/// Return the selection probability for a stored reusable sample's light type.
float getLightTypeSelectionProbability(const uint lightType)
{
    float p[3];
    getLightTypeSelectionProbabilities(p);
    return lightType < 3 ? p[lightType] : 0.f;
}

/// Randomly select which light domain will generate the next light-sampled candidate.
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

/// Balance-heuristic weight for two proposal families.
float evalBalanceWeight(const uint thisCandidateCount, const float thisPdf, const uint otherCandidateCount, const float otherPdf)
{
    const float thisTerm = float(thisCandidateCount) * thisPdf;
    const float otherTerm = float(otherCandidateCount) * otherPdf;
    const float denom = thisTerm + otherTerm;
    return denom > 0.f ? thisTerm / denom : 0.f;
}

/// Convert balance MIS into a per-candidate multiplier for the mixed light/BRDF proposal.
float evalMixtureWeight(const uint thisCandidateCount, const float thisPdf, const uint otherCandidateCount, const float otherPdf)
{
    if (thisCandidateCount == 0 || thisPdf <= 0.f)
        return 0.f;

    const float mis = evalBalanceWeight(thisCandidateCount, thisPdf, otherCandidateCount, otherPdf);
    return float(thisCandidateCount + otherCandidateCount) * mis / float(thisCandidateCount);
}

/// Generate a reusable candidate from the environment-map sampler.
bool generateReusableEnvSample(inout SampleGenerator sg, out ReusableLightSample y, out float candidatePdf)
{
    y = {};
    candidatePdf = 0.f;

#if USE_ENV_LIGHT
    EnvMapSample lightSample;
    if (!gReSTIRDI.envMapSampler.sample(sampleNext2D(sg), lightSample) || lightSample.pdf <= 0.f || !any(lightSample.Le > 0.f))
        return false;

    y.valid = 1;
    y.lightType = (uint)SampledLightType::EnvMap;
    y.envDir = lightSample.dir;
    y.Le = lightSample.Le;
    candidatePdf = lightSample.pdf;
    return true;
#else
    return false;
#endif
}

/// Generate a reusable candidate from the emissive-triangle light sampler.
bool generateReusableEmissiveSample(
    const ShadingData sd,
    const bool sampleUpperHemisphere,
    const bool sampleLowerHemisphere,
    inout SampleGenerator sg,
    out ReusableLightSample y,
    out float candidatePdf)
{
    y = {};
    candidatePdf = 0.f;

#if USE_EMISSIVE_LIGHTS
    TriangleLightSample tls;
    const bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;
    PathVertex vertex = PathVertex(sd.posW, sd.faceN, sd.frontFacing);
    if (!gReSTIRDI.emissiveSampler.sampleLight(vertex.pos, vertex.getOrientedFaceNormal(), upperHemisphere, sg, tls) ||
        tls.pdf <= 0.f ||
        !any(tls.Le > 0.f))
    {
        return false;
    }

    y.valid = 1;
    y.lightType = (uint)SampledLightType::Emissive;
    y.triangleIndex = tls.triangleIndex;
    y.lightPosW = computeRayOrigin(tls.posW, tls.normalW);
    y.lightNormalW = tls.normalW;
    y.Le = tls.Le;
    candidatePdf = tls.pdf;
    return true;
#else
    return false;
#endif
}

/// Generate a reusable candidate from a uniformly selected analytic light.
bool generateReusableAnalyticSample(const ShadingData sd, inout SampleGenerator sg, out ReusableLightSample y, out float candidatePdf)
{
    y = {};
    candidatePdf = 0.f;

    if (!kUseAnalyticLights)
        return false;

    const uint lightCount = gScene.getLightCount();
    if (lightCount == 0)
        return false;

    const uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);
    const LightData light = gScene.getLight(lightIndex);

    AnalyticLightSample lightSample;
    if (!sampleLight(sd.posW, light, sg, lightSample) || lightSample.pdf <= 0.f)
        return false;

    y.valid = 1;
    y.lightType = (uint)SampledLightType::Analytic;
    y.lightIndex = lightIndex;
    y.lightPosW = lightSample.posW;
    y.lightNormalW = lightSample.normalW;
    y.envDir = lightSample.dir;
    y.Le = light.intensity;
    candidatePdf = lightSample.pdf / float(lightCount);
    return any(y.Le > 0.f);
}

/// Generate one light-sampled candidate, including light-domain selection probability.
bool generateReusableLightSample(
    const ShadingData sd,
    const IMaterialInstance mi,
    inout SampleGenerator sg,
    out ReusableLightSample y,
    out float candidatePdf)
{
    y = {};
    candidatePdf = 0.f;

    const uint lobeTypes = mi.getLobeTypes(sd);
    const bool sampleUpperHemisphere = (lobeTypes & uint(LobeType::NonDeltaReflection)) != 0;
    const bool sampleLowerHemisphere = (lobeTypes & uint(LobeType::NonDeltaTransmission)) != 0;
    if (!sampleUpperHemisphere && !sampleLowerHemisphere)
        return false;

    uint lightType;
    float selectionPdf;
    if (!selectLightType(lightType, selectionPdf, sg) || selectionPdf <= 0.f)
        return false;

    bool valid = false;
    if (lightType == (uint)SampledLightType::EnvMap)
        valid = generateReusableEnvSample(sg, y, candidatePdf);
    if (lightType == (uint)SampledLightType::Emissive)
        valid = generateReusableEmissiveSample(sd, sampleUpperHemisphere, sampleLowerHemisphere, sg, y, candidatePdf);
    if (lightType == (uint)SampledLightType::Analytic)
        valid = generateReusableAnalyticSample(sd, sg, y, candidatePdf);
    if (!valid || candidatePdf <= 0.f)
        return false;

    float3 origin;
    float3 dir;
    float distance;
    float3 Le;
    if (!evalSampleGeometry(sd, y, origin, dir, distance, Le) || !acceptsDirection(sd, mi, dir))
        return false;

    candidatePdf *= selectionPdf;
    y.candidatePdf = candidatePdf;
    return candidatePdf > 0.f;
}

/// Evaluate the BSDF proposal PDF for a reusable light sample at the current surface.
float evalBRDFSamplingPdf(const ShadingData sd, const IMaterialInstance mi, const ReusableLightSample y)
{
    float3 origin;
    float3 dir;
    float distance;
    float3 Le;
    if (!evalSampleGeometry(sd, y, origin, dir, distance, Le) || !acceptsDirection(sd, mi, dir))
        return 0.f;

    return max(0.f, mi.evalPdf(sd, dir, true));
}

/// Evaluate the light-sampling proposal PDF for a reusable sample.
/// Analytic lights currently return zero here because the BRDF candidate path does not create analytic-light samples.
float evalLightSamplingPdf(const ShadingData sd, const IMaterialInstance mi, const ReusableLightSample y)
{
    if (y.valid == 0)
        return 0.f;

    const float selectionPdf = getLightTypeSelectionProbability(y.lightType);
    if (selectionPdf <= 0.f)
        return 0.f;

    if (y.lightType == (uint)SampledLightType::EnvMap)
    {
#if USE_ENV_LIGHT
        return selectionPdf * gReSTIRDI.envMapSampler.evalPdf(normalize(y.envDir));
#else
        return 0.f;
#endif
    }

    if (y.lightType == (uint)SampledLightType::Emissive)
    {
#if USE_EMISSIVE_LIGHTS
        const uint lobeTypes = mi.getLobeTypes(sd);
        const bool sampleUpperHemisphere = (lobeTypes & uint(LobeType::NonDeltaReflection)) != 0;
        const bool sampleLowerHemisphere = (lobeTypes & uint(LobeType::NonDeltaTransmission)) != 0;
        const bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;

        TriangleLightHit hit;
        hit.triangleIndex = y.triangleIndex;
        hit.posW = y.lightPosW;
        hit.normalW = y.lightNormalW;

        PathVertex vertex = PathVertex(sd.posW, sd.faceN, sd.frontFacing);
        return selectionPdf * gReSTIRDI.emissiveSampler.evalPdf(vertex.pos, vertex.getOrientedFaceNormal(), upperHemisphere, hit);
#else
        return 0.f;
#endif
    }

    return 0.f;
}

/// Generate one BSDF-sampled candidate by tracing the sampled direction to an emissive hit or environment.
bool generateReusableBRDFSample(
    const ShadingData sd,
    const IMaterialInstance mi,
    inout SampleGenerator sg,
    out ReusableLightSample y,
    out float candidatePdf)
{
    y = {};
    candidatePdf = 0.f;

#if USE_BRDF_SAMPLING
    BSDFSample bsdfSample = {};
    if (!mi.sample(sd, sg, bsdfSample, true) || bsdfSample.pdf <= 0.f || !any(bsdfSample.weight > 0.f))
        return false;

    const float3 dir = normalize(bsdfSample.wo);
    PathVertex vertex = PathVertex(sd.posW, sd.faceN, sd.frontFacing);
    const float3 origin = vertex.getRayOrigin(dir);

    HitInfo hit;
    float hitT;
    const bool hitScene = traceBRDFRay(origin, dir, hit, hitT);
    if (!hitScene)
    {
#if USE_ENV_LIGHT
        const float3 Le = gReSTIRDI.envMapSampler.eval(dir);
        if (!any(Le > 0.f))
            return false;

        y.valid = 1;
        y.lightType = (uint)SampledLightType::EnvMap;
        y.envDir = dir;
        y.Le = Le;
        candidatePdf = bsdfSample.pdf;
        y.candidatePdf = candidatePdf;
        return true;
#else
        return false;
#endif
    }

#if USE_EMISSIVE_LIGHTS && SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() != HitType::Triangle)
        return false;

    const TriangleHit triangleHit = hit.getTriangleHit();
    const ShadingData hitSd = loadShadingData(hit, origin, dir);
    let lod = ExplicitLodTextureSampler(0.f);
    let hitMi = gScene.materials.getMaterialInstance(hitSd, lod);
    const float3 Le = hitMi.getProperties(hitSd).emission;
    if (!any(Le > 0.f))
        return false;

    y.valid = 1;
    y.lightType = (uint)SampledLightType::Emissive;
    y.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
    y.lightPosW = computeRayOrigin(hitSd.posW, hitSd.faceN);
    y.lightNormalW = hitSd.getOrientedFaceNormal();
    y.Le = Le;
    y.lightDistance = hitT;
    candidatePdf = bsdfSample.pdf;
    y.candidatePdf = candidatePdf;
    return true;
#else
    return false;
#endif
#else
    return false;
#endif
}

/// Build the initial per-pixel RIS reservoir from light-sampled and optional BSDF-sampled candidates.
DIReservoir generateInitialReservoir(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg)
{
    DIReservoir r = {};

    for (uint i = 0; i < kLightCandidateCount; ++i)
    {
        ReusableLightSample y = {};
        float candidatePdf = 0.f;
        float target = 0.f;
        float weight = 0.f;

        if (generateReusableLightSample(sd, mi, sg, y, candidatePdf))
        {
            target = evalTarget(sd, mi, y, kUseVisibilityInTarget, sg);
            const bool matchableByBRDF = y.lightType != (uint)SampledLightType::Analytic;
            const float brdfPdf = matchableByBRDF && kBRDFCandidateCount > 0 ? evalBRDFSamplingPdf(sd, mi, y) : 0.f;
            const float misWeight = evalMixtureWeight(kLightCandidateCount, candidatePdf, kBRDFCandidateCount, brdfPdf);
            weight = candidatePdf > 0.f ? misWeight * target / candidatePdf : 0.f;
        }

        updateReservoir(r, y, target, weight, 1, 0, sampleNext1D(sg));
    }

    for (uint i = 0; i < kBRDFCandidateCount; ++i)
    {
        ReusableLightSample y = {};
        float candidatePdf = 0.f;
        float target = 0.f;
        float weight = 0.f;

        if (generateReusableBRDFSample(sd, mi, sg, y, candidatePdf))
        {
            target = evalTarget(sd, mi, y, kUseVisibilityInTarget, sg);
            const float lightPdf = evalLightSamplingPdf(sd, mi, y);
            const float misWeight = evalMixtureWeight(kBRDFCandidateCount, candidatePdf, kLightCandidateCount, lightPdf);
            weight = candidatePdf > 0.f ? misWeight * target / candidatePdf : 0.f;
        }

        updateReservoir(r, y, target, weight, 1, 1, sampleNext1D(sg));
    }

    finalizeReservoir(r);
    return r;
}

/// Pick a uniformly distributed integer neighbor offset within the spatial reuse radius.
int2 sampleNeighborOffset(inout SampleGenerator sg, const int radius)
{
    float2 u = sampleNext2D(sg);
    int diameter = 2 * radius + 1;
    int dx = int(floor(u.x * float(diameter))) - radius;
    int dy = int(floor(u.y * float(diameter))) - radius;
    return int2(dx, dy);
}

/// Bounds check for integer pixel coordinates.
bool insideFrame(const int2 p, const uint2 frameDim)
{
    return p.x >= 0 && p.y >= 0 && p.x < int(frameDim.x) && p.y < int(frameDim.y);
}

/// G-buffer compatibility test for spatial neighbor reuse.
bool isCompatible(const bool aValid, const float aDepth, const float3 aNormal, const bool bValid, const float bDepth, const float3 bNormal)
{
    if (!aValid || !bValid)
        return false;

    if (dot(aNormal, bNormal) < 0.97f)
        return false;

    float relDepthDiff = abs(aDepth - bDepth) / max(aDepth, 1e-3f);
    if (relDepthDiff > 0.03f)
        return false;

    return true;
}

/// G-buffer compatibility test for accepting a reprojected temporal history reservoir.
bool isTemporalCompatible(const TemporalSurface currentSurface, const TemporalSurface prevSurface)
{
    if (currentSurface.valid == 0 || prevSurface.valid == 0)
        return false;

    if (dot(currentSurface.normalW, prevSurface.normalW) < 0.998f)
        return false;

    float relDepthDiff = abs(currentSurface.linearDepth - prevSurface.linearDepth) / max(currentSurface.linearDepth, 1e-3f);
    if (relDepthDiff > 0.005f)
        return false;

    return true;
}

/// Reject temporal samples whose shifted target changes too much between domains.
bool isTemporalTargetStable(const float pCurrent, const float pTemporal)
{
    if (pCurrent <= 0.f || pTemporal <= 0.f)
        return false;

    const float ratio = pCurrent / max(pTemporal, 1e-20f);
    return ratio >= 0.1f && ratio <= 10.f;
}

/// Pack the subset of surface data preserved for temporal history validation.
TemporalSurface makeTemporalSurface(const bool valid, const float linearDepth, const float3 normalW)
{
    TemporalSurface surface = {};
    surface.valid = valid ? 1 : 0;
    surface.linearDepth = valid ? linearDepth : 0.f;
    surface.normalW = valid ? normalW : float3(0.f);
    return surface;
}

/// Flatten a pixel coordinate into the reservoir buffer layout.
uint reservoirIndex(const uint2 pixel, const uint2 frameDim)
{
    return pixel.x + pixel.y * frameDim.x;
}

/// Evaluate p_hat for a reusable sample at another pixel's current-frame surface.
/// Temporal reuse uses this as an approximation because full previous-frame shading data is not stored.
float evalTargetAtPixel(const uint2 pixel, const uint2 frameDim, const ReusableLightSample y, const bool includeVisibility, inout SampleGenerator sg)
{
    ShadingData sd;
    float linearDepth;
    float3 normalW;
    if (!loadSurface(pixel, frameDim, sd, linearDepth, normalW))
        return 0.f;

    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);
    return evalTarget(sd, mi, y, includeVisibility, sg);
}

#if SPATIAL_MIS_STRATEGY == 2
/// Generalized balance heuristic for the selected sample across all accepted spatial sources.
float evalGeneralizedBalanceWeight(
    const uint sourceIdx,
    const ReusableLightSample y,
    const uint sourceCount,
    const uint2 sourcePixels[kMaxSpatialSources],
    const DIReservoir sourceReservoirs[kMaxSpatialSources],
    const uint2 frameDim,
    inout SampleGenerator sg)
{
    float numerator = 0.f;
    float denominator = 0.f;

    for (uint j = 0; j < sourceCount; ++j)
    {
        DIReservoir rj = sourceReservoirs[j];
        if (rj.valid == 0 || rj.M == 0)
            continue;

        const float confidence = float(reservoirConfidence(rj));
        const float pHatJ = evalTargetAtPixel(sourcePixels[j], frameDim, y, kUseVisibilityInSpatialGBH, sg);
        const float term = confidence * pHatJ;
        denominator += term;

        if (j == sourceIdx)
            numerator = term;
    }

    return denominator > 0.f ? numerator / denominator : 0.f;
}
#endif

/// Initial pass body: load the current surface, store temporal validation data, and write the initial RIS reservoir.
void runInitialReservoir(const uint2 pixel, const uint2 frameDim)
{
    const uint idx = reservoirIndex(pixel, frameDim);

    ShadingData sd;
    float linearDepth;
    float3 normalW;
    if (!loadSurface(pixel, frameDim, sd, linearDepth, normalW))
    {
        gOutputReservoirs[idx] = {};
        gCurrentSurfaces[idx] = makeTemporalSurface(false, 0.f, float3(0.f));
        return;
    }
    gCurrentSurfaces[idx] = makeTemporalSurface(true, linearDepth, normalW);

    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    for (uint i = 0; i < gPRNGDimension; i++)
        sampleNext1D(sg);

    gOutputReservoirs[idx] = generateInitialReservoir(sd, mi, sg);
}

/// Reproject the current pixel to the previous frame using VBufferRT motion vectors.
int2 reprojectToPreviousPixel(const uint2 pixel, const uint2 frameDim)
{
    if (!is_valid(gMotionVectors))
        return int2(-1);

    const float2 prevPixel = float2(pixel) + gMotionVectors[pixel] * float2(frameDim);
    return int2(round(prevPixel));
}

/// Temporal reuse pass body: merge the current initial reservoir with the reprojected previous reservoir.
/// Zero-weight source streams still count in M so the final reservoir W is normalized by all considered sources.
void runTemporalReuse(const uint2 pixel, const uint2 frameDim)
{
    const uint idx = reservoirIndex(pixel, frameDim);
    DIReservoir currentReservoir = gInputReservoirs[idx];

    if (!kUseTemporalReuse || !is_valid(gMotionVectors))
    {
        gOutputReservoirs[idx] = currentReservoir;
        return;
    }

    TemporalSurface currentSurface = gCurrentSurfaces[idx];
    if (currentSurface.valid == 0)
    {
        gOutputReservoirs[idx] = currentReservoir;
        return;
    }

    const int2 prevPixelI = reprojectToPreviousPixel(pixel, frameDim);
    if (!insideFrame(prevPixelI, frameDim))
    {
        gOutputReservoirs[idx] = currentReservoir;
        return;
    }

    const uint2 prevPixel = uint2(prevPixelI);
    const uint prevIdx = reservoirIndex(prevPixel, frameDim);
    TemporalSurface prevSurface = gPrevSurfaces[prevIdx];
    if (!isTemporalCompatible(currentSurface, prevSurface))
    {
        gOutputReservoirs[idx] = currentReservoir;
        return;
    }

    DIReservoir temporalReservoir = gPrevReservoirs[prevIdx];
    if (temporalReservoir.valid == 0 || temporalReservoir.M == 0 || temporalReservoir.W <= 0.f)
    {
        gOutputReservoirs[idx] = currentReservoir;
        return;
    }

    ShadingData sd;
    float linearDepth;
    float3 normalW;
    if (!loadSurface(pixel, frameDim, sd, linearDepth, normalW))
    {
        gOutputReservoirs[idx] = currentReservoir;
        return;
    }

    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    for (uint i = 0; i < gPRNGDimension + 50; i++)
        sampleNext1D(sg);

    DIReservoir outR = {};
    const uint currentConfidence = temporalReservoirConfidence(currentReservoir);
    const uint temporalConfidence = temporalReservoirConfidence(temporalReservoir);
    const uint totalConfidence = currentConfidence + temporalConfidence;
    if (totalConfidence == 0)
    {
        gOutputReservoirs[idx] = {};
        return;
    }

    if (currentConfidence > 0)
    {
        float pCurrent = 0.f;
        float weight = 0.f;

        if (currentReservoir.valid != 0 && currentReservoir.M > 0 && currentReservoir.W > 0.f)
        {
            pCurrent = evalTarget(sd, mi, currentReservoir.y, kUseVisibilityInTarget, sg);
            const float pTemporal = evalTargetAtPixel(prevPixel, frameDim, currentReservoir.y, kUseVisibilityInTarget || kUseVisibilityInTemporalMIS, sg);
            const float currentTerm = float(currentConfidence) * pCurrent;
            const float temporalTerm = float(temporalConfidence) * pTemporal;
            const float denominator = currentTerm + temporalTerm;

            if (pCurrent > 0.f && denominator > 0.f)
            {
                const float mis = currentTerm / denominator;
                weight = pCurrent * currentReservoir.W * float(totalConfidence) * mis;
            }
        }

        updateReservoir(outR, currentReservoir.y, pCurrent, weight, currentConfidence, 0, sampleNext1D(sg));
    }

    if (temporalConfidence > 0)
    {
        const float pCurrent = evalTarget(sd, mi, temporalReservoir.y, kUseVisibilityInTarget, sg);
        float weight = 0.f;
        const float pTemporal = evalTargetAtPixel(prevPixel, frameDim, temporalReservoir.y, kUseVisibilityInTarget || kUseVisibilityInTemporalMIS, sg);
        const float currentTerm = float(currentConfidence) * pCurrent;
        const float temporalTerm = float(temporalConfidence) * pTemporal;
        const float denominator = currentTerm + temporalTerm;

        if (isTemporalTargetStable(pCurrent, pTemporal) && denominator > 0.f)
        {
            const float mis = temporalTerm / denominator;
            weight = pCurrent * temporalReservoir.W * float(totalConfidence) * mis;
        }

        updateReservoir(outR, temporalReservoir.y, pCurrent, weight, temporalConfidence, 1, sampleNext1D(sg));
    }

    capReservoirM(outR, kTemporalMCap);
    finalizeReservoir(outR);
    gOutputReservoirs[idx] = outR;
}

/// Spatial reuse pass body: gather compatible neighbor reservoirs and combine them with the selected MIS strategy.
void runSpatialReuse(const uint2 pixel, const uint2 frameDim)
{
    const uint idx = reservoirIndex(pixel, frameDim);

    ShadingData sd;
    float linearDepth;
    float3 normalW;
    if (!loadSurface(pixel, frameDim, sd, linearDepth, normalW))
    {
        gOutputReservoirs[idx] = {};
        return;
    }

    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    for (uint i = 0; i < gPRNGDimension + 100; i++)
        sampleNext1D(sg);

#if SPATIAL_MIS_STRATEGY == 0
    // Cheap spatial reuse: uniform/implicit source weights, no selected-sample MIS correction.
    DIReservoir outR = {};
    DIReservoir self = gInputReservoirs[idx];
    if (self.valid != 0 && self.M > 0 && self.W > 0.f)
    {
        const float targetCurrent = evalTarget(sd, mi, self.y, kUseVisibilityInTarget, sg);
        const uint confidence = reservoirConfidence(self);
        updateReservoir(outR, self.y, targetCurrent, targetCurrent * self.W * float(confidence), confidence, 0, sampleNext1D(sg));
    }

    for (uint n = 0; n < kSpatialNeighborCount; ++n)
    {
        int2 q = int2(pixel) + sampleNeighborOffset(sg, int(kSpatialRadius));
        if (!insideFrame(q, frameDim))
            continue;
        if (q.x == int(pixel.x) && q.y == int(pixel.y))
            continue;

        ShadingData neighborSd;
        float neighborDepth;
        float3 neighborNormal;
        uint2 qPixel = uint2(q);
        if (!loadSurface(qPixel, frameDim, neighborSd, neighborDepth, neighborNormal))
            continue;

        if (!isCompatible(true, linearDepth, normalW, true, neighborDepth, neighborNormal))
            continue;

        DIReservoir rq = gInputReservoirs[reservoirIndex(qPixel, frameDim)];
        if (rq.valid == 0 || rq.M == 0 || rq.W <= 0.f)
            continue;

        const float targetCurrent = evalTarget(sd, mi, rq.y, kUseVisibilityInTarget, sg);
        if (targetCurrent <= 0.f)
            continue;

        const uint confidence = reservoirConfidence(rq);
        updateReservoir(outR, rq.y, targetCurrent, targetCurrent * rq.W * float(confidence), confidence, n + 1, sampleNext1D(sg));
    }

    capReservoirM(outR, kSpatialMCap);
    finalizeReservoir(outR);
    gOutputReservoirs[idx] = outR;
#else
#if SPATIAL_MIS_STRATEGY == 1
    // Contribution MIS: first combine cheaply, then correct the chosen sample's W by its balance weight.
    SampleGenerator neighborSg = SampleGenerator(pixel, gFrameCount);
    for (uint i = 0; i < gPRNGDimension + 100; i++)
        sampleNext1D(neighborSg);

    DIReservoir outR = {};
    uint totalConfidence = 0;

    DIReservoir self = gInputReservoirs[idx];
    if (self.valid != 0 && self.M > 0 && self.W > 0.f)
    {
        const uint confidence = reservoirConfidence(self);
        totalConfidence += confidence;

        const float targetCurrent = evalTarget(sd, mi, self.y, kUseVisibilityInTarget, sg);
        if (targetCurrent > 0.f)
            updateReservoir(outR, self.y, targetCurrent, targetCurrent * self.W * float(confidence), confidence, 0, sampleNext1D(sg));
    }

    for (uint n = 0; n < kSpatialNeighborCount; ++n)
    {
        int2 q = int2(pixel) + sampleNeighborOffset(neighborSg, int(kSpatialRadius));
        if (!insideFrame(q, frameDim))
            continue;
        if (q.x == int(pixel.x) && q.y == int(pixel.y))
            continue;

        ShadingData neighborSd;
        float neighborDepth;
        float3 neighborNormal;
        uint2 qPixel = uint2(q);
        if (!loadSurface(qPixel, frameDim, neighborSd, neighborDepth, neighborNormal))
            continue;

        if (!isCompatible(true, linearDepth, normalW, true, neighborDepth, neighborNormal))
            continue;

        DIReservoir rq = gInputReservoirs[reservoirIndex(qPixel, frameDim)];
        if (rq.valid == 0 || rq.M == 0 || rq.W <= 0.f)
            continue;

        const uint confidence = reservoirConfidence(rq);
        totalConfidence += confidence;

        const float targetCurrent = evalTarget(sd, mi, rq.y, kUseVisibilityInTarget, sg);
        if (targetCurrent <= 0.f)
            continue;

        updateReservoir(outR, rq.y, targetCurrent, targetCurrent * rq.W * float(confidence), confidence, n + 1, sampleNext1D(sg));
    }

    capReservoirM(outR, kSpatialMCap);
    finalizeReservoir(outR);

    if (outR.valid != 0 && outR.W > 0.f && totalConfidence > 0)
    {
        float selectedConfidence = 0.f;
        float contributionNumerator = 0.f;
        float contributionDenominator = 0.f;

        if (self.valid != 0 && self.M > 0 && self.W > 0.f)
        {
            const float confidence = float(reservoirConfidence(self));
            const float pHat = evalTargetAtPixel(pixel, frameDim, outR.y, kUseVisibilityInSpatialGBH, sg);
            const float term = confidence * pHat;
            contributionDenominator += term;

            if (outR.selectedInputIndex == 0)
            {
                selectedConfidence = confidence;
                contributionNumerator = term;
            }
        }

        SampleGenerator correctionNeighborSg = SampleGenerator(pixel, gFrameCount);
        for (uint i = 0; i < gPRNGDimension + 100; i++)
            sampleNext1D(correctionNeighborSg);

        for (uint n = 0; n < kSpatialNeighborCount; ++n)
        {
            int2 q = int2(pixel) + sampleNeighborOffset(correctionNeighborSg, int(kSpatialRadius));
            if (!insideFrame(q, frameDim))
                continue;
            if (q.x == int(pixel.x) && q.y == int(pixel.y))
                continue;

            ShadingData neighborSd;
            float neighborDepth;
            float3 neighborNormal;
            uint2 qPixel = uint2(q);
            if (!loadSurface(qPixel, frameDim, neighborSd, neighborDepth, neighborNormal))
                continue;

            if (!isCompatible(true, linearDepth, normalW, true, neighborDepth, neighborNormal))
                continue;

            DIReservoir rq = gInputReservoirs[reservoirIndex(qPixel, frameDim)];
            if (rq.valid == 0 || rq.M == 0 || rq.W <= 0.f)
                continue;

            const float confidence = float(reservoirConfidence(rq));
            const float pHat = evalTargetAtPixel(qPixel, frameDim, outR.y, kUseVisibilityInSpatialGBH, sg);
            const float term = confidence * pHat;
            contributionDenominator += term;

            if (outR.selectedInputIndex == n + 1)
            {
                selectedConfidence = confidence;
                contributionNumerator = term;
            }
        }

        const float cheapMIS = selectedConfidence / float(totalConfidence);
        const float contributionMIS = contributionDenominator > 0.f ? contributionNumerator / contributionDenominator : 0.f;
        outR.W *= cheapMIS > 0.f ? contributionMIS / cheapMIS : 0.f;
    }

    gOutputReservoirs[idx] = outR;
#elif SPATIAL_MIS_STRATEGY == 3
    // Pairwise MIS: approximate non-defensive pairwise weights against each accepted neighbor.
    DIReservoir self = gInputReservoirs[idx];
    if (self.valid == 0 || self.M == 0 || self.W <= 0.f)
    {
        gOutputReservoirs[idx] = {};
        return;
    }

    const uint selfConfidence = reservoirConfidence(self);
    if (selfConfidence == 0)
    {
        gOutputReservoirs[idx] = {};
        return;
    }

    uint pairCount = 0;
    uint totalConfidence = selfConfidence;

    SampleGenerator countNeighborSg = SampleGenerator(pixel, gFrameCount);
    for (uint i = 0; i < gPRNGDimension + 100; i++)
        sampleNext1D(countNeighborSg);

    for (uint n = 0; n < kSpatialNeighborCount; ++n)
    {
        int2 q = int2(pixel) + sampleNeighborOffset(countNeighborSg, int(kSpatialRadius));
        if (!insideFrame(q, frameDim))
            continue;
        if (q.x == int(pixel.x) && q.y == int(pixel.y))
            continue;

        ShadingData neighborSd;
        float neighborDepth;
        float3 neighborNormal;
        uint2 qPixel = uint2(q);
        if (!loadSurface(qPixel, frameDim, neighborSd, neighborDepth, neighborNormal))
            continue;

        if (!isCompatible(true, linearDepth, normalW, true, neighborDepth, neighborNormal))
            continue;

        DIReservoir rq = gInputReservoirs[reservoirIndex(qPixel, frameDim)];
        if (rq.valid == 0 || rq.M == 0 || rq.W <= 0.f)
            continue;

        pairCount++;
        totalConfidence += reservoirConfidence(rq);
    }

    if (pairCount == 0)
    {
        gOutputReservoirs[idx] = self;
        return;
    }

    DIReservoir outR = {};
    const float invPairCount = 1.f / float(pairCount);
    const float selfConfidenceF = float(selfConfidence);
    const float totalConfidenceF = float(totalConfidence);
    const float selfTargetCurrent = evalTarget(sd, mi, self.y, kUseVisibilityInTarget, sg);

    if (selfTargetCurrent > 0.f)
    {
        float selfMIS = 0.f;

        SampleGenerator selfMISNeighborSg = SampleGenerator(pixel, gFrameCount);
        for (uint i = 0; i < gPRNGDimension + 100; i++)
            sampleNext1D(selfMISNeighborSg);

        for (uint n = 0; n < kSpatialNeighborCount; ++n)
        {
            int2 q = int2(pixel) + sampleNeighborOffset(selfMISNeighborSg, int(kSpatialRadius));
            if (!insideFrame(q, frameDim))
                continue;
            if (q.x == int(pixel.x) && q.y == int(pixel.y))
                continue;

            ShadingData neighborSd;
            float neighborDepth;
            float3 neighborNormal;
            uint2 qPixel = uint2(q);
            if (!loadSurface(qPixel, frameDim, neighborSd, neighborDepth, neighborNormal))
                continue;

            if (!isCompatible(true, linearDepth, normalW, true, neighborDepth, neighborNormal))
                continue;

            DIReservoir rq = gInputReservoirs[reservoirIndex(qPixel, frameDim)];
            if (rq.valid == 0 || rq.M == 0 || rq.W <= 0.f)
                continue;

            const float neighborConfidence = float(reservoirConfidence(rq));
            const float pHatNeighbor = evalTargetAtPixel(qPixel, frameDim, self.y, kUseVisibilityInSpatialGBH, sg);
            const float canonicalTerm = selfConfidenceF * selfTargetCurrent * invPairCount;
            const float denom = neighborConfidence * pHatNeighbor + canonicalTerm;
            if (denom > 0.f)
                selfMIS += invPairCount * canonicalTerm / denom;
        }

        if (selfMIS > 0.f)
        {
            const float weight = selfTargetCurrent * self.W * totalConfidenceF * selfMIS;
            updateReservoir(outR, self.y, selfTargetCurrent, weight, selfConfidence, 0, sampleNext1D(sg));
        }
    }

    SampleGenerator neighborSg = SampleGenerator(pixel, gFrameCount);
    for (uint i = 0; i < gPRNGDimension + 100; i++)
        sampleNext1D(neighborSg);

    for (uint n = 0; n < kSpatialNeighborCount; ++n)
    {
        int2 q = int2(pixel) + sampleNeighborOffset(neighborSg, int(kSpatialRadius));
        if (!insideFrame(q, frameDim))
            continue;
        if (q.x == int(pixel.x) && q.y == int(pixel.y))
            continue;

        ShadingData neighborSd;
        float neighborDepth;
        float3 neighborNormal;
        uint2 qPixel = uint2(q);
        if (!loadSurface(qPixel, frameDim, neighborSd, neighborDepth, neighborNormal))
            continue;

        if (!isCompatible(true, linearDepth, normalW, true, neighborDepth, neighborNormal))
            continue;

        DIReservoir rq = gInputReservoirs[reservoirIndex(qPixel, frameDim)];
        if (rq.valid == 0 || rq.M == 0 || rq.W <= 0.f)
            continue;

        const uint neighborConfidence = reservoirConfidence(rq);
        const float targetCurrent = evalTarget(sd, mi, rq.y, kUseVisibilityInTarget, sg);
        if (targetCurrent <= 0.f)
            continue;

        const float pHatNeighbor = evalTargetAtPixel(qPixel, frameDim, rq.y, kUseVisibilityInSpatialGBH, sg);
        const float neighborTerm = float(neighborConfidence) * pHatNeighbor;
        const float canonicalTerm = selfConfidenceF * targetCurrent * invPairCount;
        const float denom = neighborTerm + canonicalTerm;
        if (denom <= 0.f)
            continue;

        const float pairwiseMIS = invPairCount * neighborTerm / denom;
        const float weight = targetCurrent * rq.W * totalConfidenceF * pairwiseMIS;
        updateReservoir(outR, rq.y, targetCurrent, weight, neighborConfidence, n + 1, sampleNext1D(sg));
    }

    capReservoirM(outR, kSpatialMCap);
    finalizeReservoir(outR);
    gOutputReservoirs[idx] = outR;
#else
    // Generalized balance path: collect all accepted sources before evaluating source balance weights.
    uint2 sourcePixels[kMaxSpatialSources];
    DIReservoir sourceReservoirs[kMaxSpatialSources];
    uint sourceCount = 0;

    DIReservoir self = gInputReservoirs[idx];
    if (self.valid != 0 && self.M > 0 && self.W > 0.f)
    {
        sourcePixels[sourceCount] = pixel;
        sourceReservoirs[sourceCount] = self;
        sourceCount++;
    }

    for (uint n = 0; n < kSpatialNeighborCount && sourceCount < kMaxSpatialSources; ++n)
    {
        int2 q = int2(pixel) + sampleNeighborOffset(sg, int(kSpatialRadius));
        if (!insideFrame(q, frameDim))
            continue;
        if (q.x == int(pixel.x) && q.y == int(pixel.y))
            continue;

        ShadingData neighborSd;
        float neighborDepth;
        float3 neighborNormal;
        uint2 qPixel = uint2(q);
        if (!loadSurface(qPixel, frameDim, neighborSd, neighborDepth, neighborNormal))
            continue;

        if (!isCompatible(true, linearDepth, normalW, true, neighborDepth, neighborNormal))
            continue;

        DIReservoir rq = gInputReservoirs[reservoirIndex(qPixel, frameDim)];
        if (rq.valid == 0 || rq.M == 0 || rq.W <= 0.f)
            continue;

        sourcePixels[sourceCount] = qPixel;
        sourceReservoirs[sourceCount] = rq;
        sourceCount++;
    }

    uint totalConfidence = 0;
    for (uint i = 0; i < sourceCount; ++i)
        totalConfidence += reservoirConfidence(sourceReservoirs[i]);

    if (totalConfidence == 0)
    {
        gOutputReservoirs[idx] = {};
        return;
    }

    DIReservoir outR = {};

    for (uint i = 0; i < sourceCount; ++i)
    {
        DIReservoir src = sourceReservoirs[i];
        if (src.valid == 0 || src.W <= 0.f || src.M == 0)
            continue;

        const float targetCurrent = evalTarget(sd, mi, src.y, kUseVisibilityInTarget, sg);
        if (targetCurrent <= 0.f)
            continue;

        const uint confidence = reservoirConfidence(src);
        float weight = targetCurrent * src.W * float(confidence);

#if SPATIAL_MIS_STRATEGY == 2
        {
            const float gbh = evalGeneralizedBalanceWeight(i, src.y, sourceCount, sourcePixels, sourceReservoirs, frameDim, sg);
            if (gbh <= 0.f)
                continue;

            weight = targetCurrent * src.W * float(totalConfidence) * gbh;
        }
#endif

        updateReservoir(outR, src.y, targetCurrent, weight, confidence, i, sampleNext1D(sg));
    }

    capReservoirM(outR, kSpatialMCap);
    finalizeReservoir(outR);

    gOutputReservoirs[idx] = outR;
#endif
#endif
}

/// Shade the selected reservoir sample with final visibility and reservoir weight W.
float3 shadeReservoir(const ShadingData sd, const IMaterialInstance mi, const DIReservoir r, inout SampleGenerator sg)
{
    if (r.valid == 0 || r.W <= 0.f)
        return float3(0.f);

    return evalRawContribution(sd, mi, r.y, true, sg) * r.W;
}

/// Shading pass body: evaluate emissive material contribution and the final selected reservoir sample.
void runShadeReservoir(const uint2 pixel, const uint2 frameDim)
{
    const uint idx = reservoirIndex(pixel, frameDim);

    ShadingData sd;
    float linearDepth;
    float3 normalW;
    if (!loadSurface(pixel, frameDim, sd, linearDepth, normalW))
    {
        float3 background = kUseEnvBackground ? gScene.envMap.eval(getPrimaryRayDir(pixel, frameDim)) : float3(0.f);
        gOutputColor[pixel] = float4(background, 1.f);
        return;
    }

    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    for (uint i = 0; i < gPRNGDimension + 200; i++)
        sampleNext1D(sg);

    float3 color = float3(0.f);
    if (kUseEmissiveMaterials)
        color += mi.getProperties(sd).emission;

    color += shadeReservoir(sd, mi, gFinalReservoirs[idx], sg);
    gOutputColor[pixel] = float4(color, 1.f);
}

/// Shadow miss means the tested light segment is unoccluded.
[shader("miss")]
void shadowMiss(inout ShadowRayData rayData)
{
    rayData.visible = true;
}

/// BRDF miss leaves the hit invalid, which generateReusableBRDFSample interprets as environment.
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

/// DXR entry point for the initial reservoir generation pass.
[shader("raygeneration")]
void initialRayGen()
{
    const uint2 pixel = DispatchRaysIndex().xy;
    const uint2 frameDim = DispatchRaysDimensions().xy;
    if (any(pixel >= frameDim))
        return;

    runInitialReservoir(pixel, frameDim);
}

/// DXR entry point for temporal reuse.
[shader("raygeneration")]
void temporalRayGen()
{
    const uint2 pixel = DispatchRaysIndex().xy;
    const uint2 frameDim = DispatchRaysDimensions().xy;
    if (any(pixel >= frameDim))
        return;

    runTemporalReuse(pixel, frameDim);
}

/// DXR entry point for spatial reuse.
[shader("raygeneration")]
void spatialRayGen()
{
    const uint2 pixel = DispatchRaysIndex().xy;
    const uint2 frameDim = DispatchRaysDimensions().xy;
    if (any(pixel >= frameDim))
        return;

    runSpatialReuse(pixel, frameDim);
}

/// DXR entry point for final reservoir shading.
[shader("raygeneration")]
void shadeRayGen()
{
    const uint2 pixel = DispatchRaysIndex().xy;
    const uint2 frameDim = DispatchRaysDimensions().xy;
    if (any(pixel >= frameDim))
        return;

    runShadeReservoir(pixel, frameDim);
}
