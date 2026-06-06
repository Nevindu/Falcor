/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

import ReSTIRPT;
import LoadShadingData;

/** TODO: Random replay shift mapping.

    Current status:
      This is intentionally an identity stub. The UI, parameter plumbing, and
      shift dispatcher can select it, but it does not yet implement replay.

    Goal:
      Shift a source sample X into the destination pixel/domain by replaying the
      source path's random choices from the destination primary surface. Unlike
      reconnection shift, this map does not copy a geometric suffix. It reuses
      the source sample's random stream/replay state and lets the destination
      path tracer regenerate the path.

    GRIS requirements:
      - Implement a valid mapping Y = T(X) on the subset where replay succeeds.
      - Return invalid/zero weight when replay diverges or the destination path
        cannot realize the selected source event.
      - The reuse code expects the returned fields to satisfy:
            weight = m(Y) * pHat(Y) * source.W * |dT/dX|
        so shiftedTarget is pHat(Y), and jacobian is the remaining determinant
        not already folded into shiftedContribution.

    Expected convention:
      For a pure primary-sample-space replay that copies/reuses the same random
      coordinates and sampling decisions, jacobian = 1. Do not add geometric
      reconnection Jacobians here unless this function is changed to a mixed
      path-space/PSS convention.

    Required source metadata:
      - pathSeed/rngState sufficient to reconstruct the selected source path.
      - selectedKind, selectedVertex, selectedCandidate/path-tree identity.
      - sampled BSDF lobe tags for replayed vertices.
      - NEE/light technique tags if NEE candidates can be replayed.
      - random-dimension alignment, including dimensions consumed by branches
        that were generated but are not part of the selected reused path.

    Implementation recipe:
      1. Reject already-shifted samples at first, until composable replay has a
         derived and tested policy.
      2. Reconstruct the source sample's RNG stream from pathSeed/rngState.
      3. Start a new destination path at dstPixel's primary hit y1. The source
         camera vertices x0/x1 are not reused.
      4. At each replayed bounce, consume the same random dimensions and select
         the same technique where required: BSDF sample, NEE sample, light/env
         hit, and Russian roulette decision.
      5. For BSDF scattering, require that the destination BSDF can realize the
         stored source lobe. Reject unsupported delta/transmission/medium cases
         until they are explicitly implemented.
      6. Update throughput, PDFs, MIS factors, path length, and medium state
         from the destination path's actual geometry and BSDF values.
      7. Reject on unavailable lobes, path-length/RR mismatch, invalid surface
         or medium transitions, NEE/light technique mismatch, or zero/NaN/Inf
         PDFs/contributions.
      8. Compute shiftedContribution with the same estimator convention as the
         initial RIS sample, set shiftedTarget = pHat(Y), and set jacobian = 1.
      9. Fill shiftedSample with destination-domain metadata before returning so
         later spatial/temporal reuse evaluates the shifted sample consistently.

    Validation:
      - Same-pixel replay should reproduce the original sample with jacobian = 1.
      - With only self reuse enabled, random replay should match the base
        path/RIS estimator.
      - Bring up in stages: diffuse-only, glossy, NEE, RR, env lights, then
        transmission/medium support.
*/
extension ReSTIRPT
{
    ShiftMappingResult applyRandomReplayShift(PathReservoir source, ShadingData dstSd, HitInfo dstHit, uint2 dstPixel)
    {
        return evaluateIdentityShift(source, dstPixel);
    }
}
