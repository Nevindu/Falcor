# LFSubPixelPathTracer

This is a first-pass Falcor render pass cloned from `PathTracerN` and modified to render a lenticular/light-field encoded image directly.

## What changed

- The pass is renamed to `LFSubPixelPathTracer`.
- It does **not** require `VBufferRT` input. Primary rays are traced directly in the ray-generation shader.
- For each output LCD pixel, the ray-generation shader traces three independent subpixel paths:
  - R subpixel -> viewpoint from DirectL-style view matrix -> path trace -> use red channel.
  - G subpixel -> viewpoint from DirectL-style view matrix -> path trace -> use green channel.
  - B subpixel -> viewpoint from DirectL-style view matrix -> path trace -> use blue channel.
- The output is an `H x W` RGB lenticular/interleaved encoded image.
- This version implements the exact subpixel-aware path tracer path, **without subpixel repurposing**.

## Important files

- `LFSubPixelPathTracer.cpp/.h`: renamed render pass and UI/properties.
- `LFSubPixelPathTracer.slang`: DirectL-style subpixel-to-view mapping and primary ray generation.
- `TracePass.rt.slang`: traces three subpixel/channel paths per output pixel.
- `Params.slang`: added lenticular display parameters.
- `LFSubPixelPathTracer.py`: example render graph.

## Parameters

- `lfViewCount`: number of views/viewing zones.
- `lfLineCount`: DirectL `Lx`, lenticular pitch in subpixel-width units.
- `lfTiltDegrees`: slant angle in degrees.
- `lfOffset`: `K_offset`, horizontal display/lens offset in subpixel-width units.
- `lfBaseline`: total horizontal baseline of the virtual view camera array in scene units.
- `lfConvergenceDistance`: only used for toe-in/recenter mode.
- `lfCameraMode`: `0 = parallel`, `1 = toe-in/recenter`.
- `lfReverseViewOrder`: flips view index ordering if your display calibration is reversed.

## Notes

This is intended as an implementation starting point, not a polished production pass. The most likely local integration issue is the Falcor shader camera API: this code calls

```slang
gScene.camera.computeRayPinhole(float2 pixelCenter, params.frameDim)
```

If your Falcor version only exposes the `uint2` overload, replace `computeContinuousPinholeRay()` in `LFSubPixelPathTracer.slang` with your local camera's continuous-pixel ray generation function.

This version also intentionally avoids denoising and VBuffer reuse because the final encoded lenticular image should not be filtered like a normal image.
