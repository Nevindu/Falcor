/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

import ReSTIRPT;
import LoadShadingData;

/** TODO: Hybrid random-replay + reconnection shift mapping.

    Current status:
      This is intentionally an identity stub. The dispatcher can select hybrid
      shift today, but the actual hybrid mapping is not implemented yet.

    Goal:
      Replay a destination prefix with random replay, reconnect that prefix to a
      selected source vertex x_k, then reuse the source suffix:

          X = (x0, x1, x2, ..., x_{k-1}, x_k, x_{k+1}, ...)
          Y = (y0, y1, y2, ..., y_{k-1}, x_k, x_{k+1}, ...)

      Vanilla reconnection is the simple k = 2 case, where the destination
      primary vertex y1 connects directly to x2. Hybrid generalizes the same
      idea by replaying y2..y_{k-1} before reconnecting.

    GRIS requirements:
      - Implement a deterministic mapping Y = T(X) on the subset satisfying the
        hybrid validity conditions.
      - Return invalid/zero weight if any replay, connectibility, visibility,
        support, or invertibility condition fails.
      - The reuse code expects:
            weight = m(Y) * pHat(Y) * source.W * |dT/dX|
        Therefore shiftedContribution is the destination-domain estimator for Y,
        shiftedTarget is pHat(Y), and jacobian is the remaining determinant not
        already folded into shiftedContribution.

    Required source metadata:
      - pathSeed/rngState sufficient to reconstruct the source path.
      - reconnectVertex = k.
      - selectedKind, selectedVertex, selectedCandidate/path-tree identity.
      - sampled BSDF lobe tags along the replayed prefix and reconnection region.
      - NEE/light technique tags if the prefix or selected event uses NEE.
      - source suffix data, or the ability to regenerate the full source path
        exactly from replay state.

    Source-path precomputation once hybrid is implemented:
      1. Generate the source path normally.
      2. Determine the canonical earliest valid reconnect vertex k >= 2.
      3. Store k, replay state, and sampled lobe/technique tags.
      4. Use sampled-lobe roughness for connectibility decisions, not only the
         material's aggregate roughness.

    Connectibility and invertibility checks:
      - Distance condition:
            min(|x_k - x_{k-1}|, |x_k - y_{k-1}|) >= d_min
      - Roughness/support condition:
            the sampled lobe at x_{k-1}, the replayed lobe at y_{k-1}, and the
            sampled lobe at x_k are rough enough and available in both domains.
      - Visibility condition:
            the new edge y_{k-1} -> x_k is visible and geometrically valid.
      - Canonical earliest-k condition:
            no earlier k' < k may satisfy the same hybrid connectibility rules.
      - Symmetry condition:
            the inverse shift from Y must select the same reconnection structure
            and recover X on accepted paths.
      - Reject on failure; do not silently fall back to the unshifted source
        contribution.

    Implementation recipe:
      1. Load k = source.sample.reconnectVertex and reject if k < 2 or missing.
      2. Reconstruct the source path/replay state from pathSeed/rngState.
      3. Start the destination path at dstPixel's primary hit y1.
      4. Random-replay the prefix up to y_{k-1}, consuming the same random
         dimensions and copying required technique/lobe decisions.
      5. Reject if the destination BSDF/medium cannot realize a replayed event.
      6. Reconnect y_{k-1} to x_k: evaluate BSDFs/PDFs for the new edge, trace
         visibility, and apply distance/roughness/support checks.
      7. Reuse the source suffix x_k, x_{k+1}, ... while re-evaluating factors
         that depend on the changed incoming direction at x_k.
      8. Compute shiftedContribution and shiftedTarget in the same convention as
         initial path/RIS samples.
      9. Compute jacobian consistently with shiftedContribution. For the PSS
         hybrid derivation this is the local reconnection determinant expressed
         through the random-number-to-direction mappings at y_{k-1} and x_k,
         including the needed BSDF PDF ratios and directional/area geometry term.
         Do not double-count PDF ratios in both contribution and jacobian.
     10. Reject, rather than clamp, if the jacobian or contribution is zero, NaN,
         Inf, or above a chosen robustness threshold.
     11. Fill shiftedSample with destination-domain metadata and mark it shifted.

    Suggested staged implementation:
      - Stage A: k fixed to 2, numerically matching vanilla reconnection.
      - Stage B: random-replay prefix for k > 2 on diffuse/rough lobes only.
      - Stage C: sampled-lobe roughness and earliest-k selection.
      - Stage D: NEE/light technique tags.
      - Stage E: glossy/refraction/medium support after reference validation.

    Validation:
      - Same-pixel shift must be identity with jacobian = 1.
      - Forward then inverse shift should recover the original path on accepted
        paths.
      - Compare spatial-only accumulated results against a normal path tracer
        before adding temporal reuse.
*/
extension ReSTIRPT
{
    ShiftMappingResult applyHybridShift(PathReservoir source, ShadingData dstSd, HitInfo dstHit, uint2 dstPixel)
    {
        return evaluateIdentityShift(source, dstPixel);
    }
}
