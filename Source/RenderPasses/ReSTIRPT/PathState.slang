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
import Utils.Math.PackedFormats;
__exported import Scene.HitInfo;
__exported import Utils.Math.Ray;
__exported import Utils.Sampling.SampleGenerator;
__exported import Rendering.Materials.InteriorList;
__exported import GuideData;

static const uint kMaxRejectedHits = 16; // Maximum number of rejected hits along a path. The path is terminated if the limit is reached to avoid getting stuck in pathological cases.

static const float kRayTMax = 1e30f;

static const uint kPathSampleFlagValid = 1u << 0;             ///< Sample carries a finite selected path contribution.
static const uint kPathSampleFlagCanReconnectAtX2 = 1u << 1;  ///< Sample can be shifted by replacing x1 with the destination primary hit and reconnecting to x2.
static const uint kPathSampleFlagNeedsReplay = 1u << 2;       ///< Sample requires source path replay/tail evaluation before the fixed-x2 reconnection shift.
static const uint kPathSampleFlagAlreadyShifted = 1u << 3;    ///< Sample has already been shifted into another primary domain; replayed suffix imports must not replay it as canonical.

static const uint kPathSampleKindInvalid = 0u;
static const uint kPathSampleKindHitLight = 1u;               ///< BSDF-sampled hit on an emissive surface.
static const uint kPathSampleKindNEE = 2u;                    ///< Next-event-estimation light sample.
static const uint kPathSampleKindEnv = 3u;                    ///< BSDF-sampled environment hit.

static const uint kInvalidReconnectVertex = 0u;
static const uint kMaxStoredReplayLobes = 8u;                 ///< One 8-bit sampled lobe tag for scatter vertices 1..8.

uint hashPathState(uint v)
{
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;
    return v;
}

uint makePathSeed(uint pathID, uint frameSeed)
{
    return hashPathState(pathID ^ hashPathState(frameSeed + 0x9e3779b9u));
}

float sampleNextRIS(inout uint state)
{
    state = hashPathState(state);
    return float(state & 0x00ffffffu) / float(0x01000000u);
}

void setReplayLobeTag(inout uint tags0, inout uint tags1, uint vertex, uint lobeType)
{
    if (vertex == 0u || vertex > kMaxStoredReplayLobes) return;

    const uint byteIndex = vertex - 1u;
    const uint shift = (byteIndex & 3u) << 3;
    const uint mask = 0xffu << shift;
    const uint value = (lobeType & 0xffu) << shift;
    if (byteIndex < 4u) tags0 = (tags0 & ~mask) | value;
    else tags1 = (tags1 & ~mask) | value;
}

uint getReplayLobeTag(uint tags0, uint tags1, uint vertex)
{
    if (vertex == 0u || vertex > kMaxStoredReplayLobes) return 0u;

    const uint byteIndex = vertex - 1u;
    const uint shift = (byteIndex & 3u) << 3;
    return byteIndex < 4u ? ((tags0 >> shift) & 0xffu) : ((tags1 >> shift) & 0xffu);
}

bool hasReplayLobeTag(uint tags0, uint tags1, uint vertex)
{
    return getReplayLobeTag(tags0, tags1, vertex) != 0u;
}

uint packPathRISMetadata(uint risCandidateCount, uint selectedCandidate, uint selectedKind, uint selectedVertex)
{
    return (min(risCandidateCount, 0xffu) & 0xffu) |
           ((min(selectedCandidate, 0xffu) & 0xffu) << 8) |
           ((selectedKind & 0xfu) << 16) |
           ((selectedVertex & 0x3ffu) << 20);
}

void unpackPathRISMetadata(uint packed, out uint risCandidateCount, out uint selectedCandidate, out uint selectedKind, out uint selectedVertex)
{
    risCandidateCount = packed & 0xffu;
    selectedCandidate = (packed >> 8) & 0xffu;
    selectedKind = (packed >> 16) & 0xfu;
    selectedVertex = (packed >> 20) & 0x3ffu;
}

// Be careful with changing these. PathFlags share 32-bit uint with vertexIndex. For now, we keep 10 bits for vertexIndex.
// PathFlags take higher bits, VertexIndex takes lower bits.
static const uint kVertexIndexBitCount = 10u;
static const uint kVertexIndexBitMask = (1u << kVertexIndexBitCount) - 1u;
static const uint kPathFlagsBitCount = 32u - kVertexIndexBitCount;
static const uint kPathFlagsBitMask = ((1u << kPathFlagsBitCount) - 1u) << kVertexIndexBitCount;

/** Path flags. The path flags are currently stored in kPathFlagsBitCount bits.
*/
enum class PathFlags
{
    active                      = 0x0001,   ///< Path is active/terminated.
    hit                         = 0x0002,   ///< Result of the scatter ray (0 = miss, 1 = hit).

    transmission                = 0x0004,   ///< Scatter ray went through a transmission event.
    specular                    = 0x0008,   ///< Scatter ray went through a specular event.
    delta                       = 0x0010,   ///< Scatter ray went through a delta event.

    insideDielectricVolume      = 0x0020,   ///< Path vertex is inside a dielectric volume.
    lightSampledUpper           = 0x0040,   ///< Last path vertex sampled lights using NEE (in upper hemisphere).
    lightSampledLower           = 0x0080,   ///< Last path vertex sampled lights using NEE (in lower hemisphere).

    diffusePrimaryHit           = 0x0100,   ///< Scatter ray went through a diffuse event on primary hit.
    specularPrimaryHit          = 0x0200,   ///< Scatter ray went through a specular event on primary hit.
    deltaReflectionPrimaryHit   = 0x0400,   ///< Primary hit was sampled as the delta reflection.
    deltaTransmissionPath       = 0x0800,   ///< Path started with and followed delta transmission events (whenever possible - TIR could be an exception) until it hit the first non-delta event.
    deltaOnlyPath               = 0x1000,   ///< There was no non-delta events along the path so far.

    // Bits 14 to kPathFlagsBitCount are still unused.
};

/** Bounce types. We keep separate counters for all of these.
*/
enum class BounceType
{
    Diffuse                 = 0,    ///< Diffuse reflection.
    Specular                = 1,    ///< Specular reflection (including delta).
    Transmission            = 2,    ///< Transmission (all kinds).
};

// TODO: Compact encoding to reduce live registers, e.g. packed HitInfo, packed normals.
/** Live state for the path tracer.
*/
struct PathState
{
    uint        id;                     ///< Path ID encodes (pixel, sampleIdx) with 12 bits each for pixel x|y and 8 bits for sample index.
    uint        pathSeed;               ///< Stable seed for replaying the path's sampling decisions.
    uint        risState;               ///< Independent RNG state for RIS decisions. This must not perturb path sampling/replay.
    uint        risCandidateCount;      ///< Number of positive path-tree candidates seen by initial RIS.
    uint        selectedCandidate;      ///< Index of the candidate selected by initial RIS.
    uint        selectedKind;           ///< kPathSampleKind* value for the selected candidate.
    uint        selectedVertex;         ///< Path vertex at which the selected candidate was generated.
    uint        reconnectVertex;        ///< Fixed pure-reconnection suffix vertex. Currently 2 for reusable x2 tails, or kInvalidReconnectVertex.
    uint        scatterLobeTags0;       ///< Packed sampled BSDF lobe tags for replay scatter vertices 1..4.
    uint        scatterLobeTags1;       ///< Packed sampled BSDF lobe tags for replay scatter vertices 5..8.

    uint        flagsAndVertexIndex;    ///< Higher kPathFlagsBitCount bits: Flags indicating the current status. This can be multiple PathFlags flags OR'ed together.
                                        ///< Lower kVertexIndexBitCount bits: Current vertex index (0 = camera, 1 = primary hit, 2 = secondary hit, etc.).
    uint16_t    rejectedHits;           ///< Number of false intersections rejected along the path. This is used as a safeguard to avoid deadlock in pathological cases.
    float16_t   sceneLength;            ///< Path length in scene units (0.f at primary hit).
    uint        bounceCounters;         ///< Packed counters for different types of bounces (see BounceType).

    // Scatter ray
    float3      origin;                 ///< Origin of the scatter ray.
    float3      dir;                    ///< Scatter ray normalized direction.
    float       pdf;                    ///< Pdf for generating the scatter ray.
    float3      normal;                 ///< Shading normal at the scatter ray origin.
    HitInfo     hit;                    ///< Hit information for the scatter ray. This is populated at committed triangle hits.

    float3      thp;                    ///< Path throughput.
    float3      L;                      ///< RIS-selected path-tree contribution.
    float       risTarget;              ///< Target value of the selected contribution.
    float       risWeightSum;           ///< Sum of path-tree RIS candidate weights.
    float3      reusableLightPosW;       ///< Selected finite NEE light point, if the light has one.
    float3      reusableLightNormalW;    ///< Selected finite/area light normal, if applicable.
    float3      reusableLightLi;         ///< Selected light radiance/value before destination PDF division.
    float3      reusableLightDir;        ///< Selected light direction for env/directional lights and diagnostics.
    float       reusableLightDistance;   ///< Selected light distance for visibility.
    float       reusableLightPdf;        ///< Full source light-sampling PDF including light-type selection.
    float       reusableScatterPdf;      ///< Source BSDF PDF used for MIS diagnostics.
    uint        reusableLightType;       ///< LightType of selected reusable NEE sample.
    uint        reusableLightValid;      ///< True if selected candidate can enter direct-light reconnection.
    uint        reusableLightIndex;      ///< Analytic light index, if applicable.
    uint        reusableLightTriangleIndex; ///< Emissive triangle index, if applicable.
    uint        reusableLightFlags;      ///< Bitfield for NEE sampling metadata used by shifted MIS.

    GuideData   guideData;              ///< Per-sample guide data.
    InteriorList interiorList;          ///< Interior list. Keeping track of a stack of materials with medium properties.
    SampleGenerator sg;                 ///< Sample generator state. Typically 4-16B.

    // Accessors

    bool isTerminated() { return !isActive(); }
    bool isActive() { return hasFlag(PathFlags::active); }
    bool isHit() { return hasFlag(PathFlags::hit); }
    bool isTransmission() { return hasFlag(PathFlags::transmission); }
    bool isSpecular() { return hasFlag(PathFlags::specular); }
    bool isDelta() { return hasFlag(PathFlags::delta); }
    bool isInsideDielectricVolume() { return hasFlag(PathFlags::insideDielectricVolume); }

    bool isLightSampled()
    {
        const uint bits = (uint(PathFlags::lightSampledUpper) | uint(PathFlags::lightSampledLower)) << kVertexIndexBitCount;
        return flagsAndVertexIndex & bits;
    }

    bool isLightSampledUpper() { return hasFlag(PathFlags::lightSampledUpper); }
    bool isLightSampledLower() { return hasFlag(PathFlags::lightSampledLower); }

    bool isDiffusePrimaryHit() { return hasFlag(PathFlags::diffusePrimaryHit); }
    bool isSpecularPrimaryHit() { return hasFlag(PathFlags::specularPrimaryHit); }
    bool isDeltaReflectionPrimaryHit() { return hasFlag(PathFlags::deltaReflectionPrimaryHit); }
    bool isDeltaTransmissionPath() { return hasFlag(PathFlags::deltaTransmissionPath); }
    bool isDeltaOnlyPath() { return hasFlag(PathFlags::deltaOnlyPath); }

    // Check if the scatter event is samplable by the light sampling technique.
    bool isLightSamplable() { return !isDelta(); }

    [mutating] void terminate() { setFlag(PathFlags::active, false); }
    [mutating] void setActive() { setFlag(PathFlags::active); }
    [mutating] void setHit(HitInfo hitInfo) { hit = hitInfo; setFlag(PathFlags::hit); }
    [mutating] void clearHit() { setFlag(PathFlags::hit, false); }

    [mutating] void clearEventFlags()
    {
        const uint bits = (uint(PathFlags::transmission) | uint(PathFlags::specular) | uint(PathFlags::delta)) << kVertexIndexBitCount;
        flagsAndVertexIndex &= ~bits;
    }

    [mutating] void setTransmission(bool value = true) { setFlag(PathFlags::transmission, value); }
    [mutating] void setSpecular(bool value = true) { setFlag(PathFlags::specular, value); }
    [mutating] void setDelta(bool value = true) { setFlag(PathFlags::delta, value); }
    [mutating] void setInsideDielectricVolume(bool value = true) { setFlag(PathFlags::insideDielectricVolume, value); }
    [mutating] void setLightSampled(bool upper, bool lower) { setFlag(PathFlags::lightSampledUpper, upper); setFlag(PathFlags::lightSampledLower, lower); }
    [mutating] void setDiffusePrimaryHit(bool value = true) { setFlag(PathFlags::diffusePrimaryHit, value); }
    [mutating] void setSpecularPrimaryHit(bool value = true) { setFlag(PathFlags::specularPrimaryHit, value); }
    [mutating] void setDeltaReflectionPrimaryHit(bool value = true) { setFlag(PathFlags::deltaReflectionPrimaryHit, value); }
    [mutating] void setDeltaTransmissionPath(bool value = true) { setFlag(PathFlags::deltaTransmissionPath, value); }
    [mutating] void setDeltaOnlyPath(bool value = true) { setFlag(PathFlags::deltaOnlyPath, value); }

    bool hasFlag(PathFlags flag)
    {
        const uint bit = uint(flag) << kVertexIndexBitCount;
        return (flagsAndVertexIndex & bit) != 0;
    }

    [mutating] void setFlag(PathFlags flag, bool value = true)
    {
        const uint bit = uint(flag) << kVertexIndexBitCount;
        if (value) flagsAndVertexIndex |= bit;
        else flagsAndVertexIndex &= ~bit;
    }

    uint getBounces(BounceType type)
    {
        const uint shift = (uint)type << 3;
        return (bounceCounters >> shift) & 0xff;
    }

    [mutating] void setBounces(BounceType type, uint bounces)
    {
        const uint shift = (uint)type << 3;
        bounceCounters = (bounceCounters & ~((uint)0xff << shift)) | ((bounces & 0xff) << shift);
    }

    [mutating] void incrementBounces(BounceType type)
    {
        const uint shift = (uint)type << 3;
        // We assume that bounce counters cannot overflow.
        bounceCounters += (1 << shift);
    }

    uint2 getPixel() { return uint2(id, id >> 12) & 0xfff; }
    uint getSampleIdx() { return id >> 24; }

    // Unsafe - assumes that index is small enough.
    [mutating] void setVertexIndex(uint index)
    {
        // Clear old vertex index.
        flagsAndVertexIndex &= kPathFlagsBitMask;
        // Set new vertex index (unsafe).
        flagsAndVertexIndex |= index;
    }

    uint getVertexIndex() { return flagsAndVertexIndex & kVertexIndexBitMask; }

    // Unsafe - assumes that vertex index never overflows.
    [mutating] void incrementVertexIndex() { flagsAndVertexIndex += 1; }
    // Unsafe - assumes that vertex index will never be decremented below zero.
    [mutating] void decrementVertexIndex() { flagsAndVertexIndex -= 1; }

    Ray getScatterRay()
    {
        return Ray(origin, dir, 0.f, kRayTMax);
    }
};

/** Compact reusable path sample selected by initial path-tree RIS.

    This is the state that temporal/spatial shift mappings replay or re-evaluate
    in a different pixel's path space. Current spatial reuse implements a fixed
    x2 pure reconnection shift: Y=(y0,y1,x2,...), where the destination pixel
    supplies y0/y1 and the source sample supplies the suffix beginning at x2.
*/
struct PathSample
{
    uint        pathID;                  ///< Encoded pixel/sample id of the candidate that generated this sample.
    uint        pathSeed;                ///< Stable replay seed for recovering the source path.
    uint        selectedCandidate;       ///< Candidate index within the initial path-tree RIS batch.
    uint        risCandidateCount;       ///< Number of positive path-tree RIS candidates in the source path.
    uint        selectedKind;            ///< kPathSampleKind* value for the selected candidate.
    uint        selectedVertex;          ///< Source path vertex where the selected event was generated.
    uint        reconnectVertex;         ///< Fixed pure-reconnection suffix vertex. Currently 2 for reusable x2 tails, or kInvalidReconnectVertex.
    uint        scatterLobeTags0;        ///< Packed sampled BSDF lobe tags for replay scatter vertices 1..4.
    uint        scatterLobeTags1;        ///< Packed sampled BSDF lobe tags for replay scatter vertices 5..8.
    uint        techniqueTags0;          ///< Reserved packed per-vertex technique tags for future random replay/hybrid shifts.
    uint        techniqueTags1;          ///< Reserved packed per-vertex technique tags for future random replay/hybrid shifts.
    uint        neeTags0;                ///< Reserved packed NEE/light-technique tags for future replay alignment.
    uint        neeTags1;                ///< Reserved packed NEE/light-technique tags for future replay alignment.
    uint        pathLength;              ///< Terminated path length used for debug and future compatibility checks.
    uint        flags;                   ///< Validity and future shift-mapping flags.
    uint        rngState;                ///< Compact replay seed placeholder for future path replay.
    uint        primaryMaterialID;       ///< Material ID at the source primary surface.
    float       target;                  ///< Unnormalized target p-hat for the selected candidate.
    float3      contribution;            ///< Full selected candidate contribution before reservoir normalization.
    float3      primaryPosW;             ///< World-space position of the source primary hit.
    float3      primaryNormalW;          ///< World-space geometric normal at the source primary hit.
    float       primaryDepth;            ///< Distance from camera to the source primary hit.
    uint        reusableLightType;       ///< LightType for the stored NEE reconnection sample.
    uint        reusableLightValid;      ///< True if direct-light reconnection data is populated.
    uint        reusableLightIndex;      ///< Analytic light index, if applicable.
    uint        reusableLightTriangleIndex; ///< Emissive triangle index, if applicable.
    float3      reusableLightPosW;       ///< Stored finite light point for direct-light reconnection.
    float3      reusableLightNormalW;    ///< Stored finite/area light normal for direct-light reconnection.
    float3      reusableLightLi;         ///< Stored light radiance/value before destination PDF division.
    float3      reusableLightDir;        ///< Stored env/directional light direction.
    float       reusableLightDistance;   ///< Stored light distance for visibility.
    float       reusableLightPdf;        ///< Full source light-sampling PDF including light-type selection.
    float       reusableScatterPdf;      ///< Source BSDF PDF used for MIS diagnostics.
    uint        reusableLightFlags;      ///< Bitfield for NEE sampling metadata used by shifted MIS.
};

/** Explicit ReSTIR PT reservoir for one pixel.
*/
struct PathReservoir
{
    PathSample sample;                   ///< Selected path sample.
    float      wSum;                     ///< Sum of candidate weights.
    float      target;                   ///< Target value of selected sample.
    uint       M;                        ///< Number of candidate samples seen by this reservoir.
    float      W;                        ///< Final reservoir contribution multiplier, wSum / (target * M) for current RIS.
    float      shiftedTarget;            ///< Target value after the current shift mapping.
    float      targetRatio;              ///< shiftedTarget / originalTarget for diagnostics.
    float      jacobian;                 ///< Shift mapping Jacobian. Identity shift uses 1.
    uint       rejectionReason;          ///< 0 = accepted, nonzero = rejected by shift mapping.
    uint       acceptedCount;            ///< Spatial/temporal combine diagnostics.
    uint       rejectedCount;            ///< Spatial/temporal combine diagnostics.
    float      selectedDistance;         ///< Distance to the selected spatial neighbor for diagnostics.
};

/** Result of evaluating a selected path sample through a shift mapping.

    Convention used by spatial/future temporal GRIS reuse:
    - shiftedContribution is the destination-domain estimator contribution for
      the selected path technique. It may contain technique PDF factors when the
      original path contribution did.
    - shiftedTarget is pHat(Y), currently luminance(shiftedContribution).
    - jacobian is the remaining measure conversion |dT/dX| not already folded
      into shiftedContribution. A shift implementation must keep this convention
      self-consistent because reuse weights multiply shiftedTarget * jacobian.
    - shiftedSample mirrors the accepted destination-domain metadata before the
      reuse pass localizes primary-surface fields to the destination pixel.
*/
struct ShiftMappingResult
{
    bool       valid;                     ///< True if the shifted sample is usable.
    float      shiftedTarget;             ///< Unnormalized shifted target p-hat.
    float      originalTarget;            ///< Original selected sample target.
    float      targetRatio;               ///< shiftedTarget / originalTarget.
    float      jacobian;                  ///< Shift mapping Jacobian.
    float3     shiftedContribution;        ///< Contribution after shift evaluation.
    PathSample shiftedSample;              ///< Sample metadata after the shift, before destination-domain localization.
    uint       rejectionReason;           ///< 0 = accepted, nonzero = rejected.
};
