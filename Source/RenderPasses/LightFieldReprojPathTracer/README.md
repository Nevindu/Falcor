# LightFieldReprojPathTracer experimental Falcor pass

This package contains an experimental implementation scaffold for the paper-style
light-field reprojection path tracer:

- `RenderPasses/LightFieldVBufferRT`: a VBufferRT variant that treats the output image as a quilt/atlas. Each tile is a parallel light-field camera translated along camera right/up.
- `RenderPasses/LightFieldReprojPathTracer`: a PathTracer clone that can trace only the selected source tile and then run a spatial backward-reprojection compute pass to synthesize the remaining tiles.
- `LightFieldReprojPathTracer.py`: a Mogwai render graph.

## What to display in Mogwai

Display/mark `ToneMapper.dst` from `LightFieldReprojPathTracer.py`.

That image is **not one camera view**. It is a **quilt atlas**:

```text
+-------+-------+-------+-------+-------+-------+
| view0 | view1 | view2 | view3 | view4 | view5 |
+-------+-------+-------+-------+-------+-------+
| view6 | ...                                   |
+-------+---------------------------------------+
```

The default is a 6x6 atlas, matching the paper. The tile resolution is the Mogwai/output resolution divided by 6 in each dimension. The exact paper resolution of 6x6 views at 1280x720 requires a 7680x4320 atlas. The path ID packing in this pass supports atlas dimensions up to 8192 pixels on each axis.

## Defaults from the paper

- 6x6 view grid.
- 1 spp source view.
- 8 max surface bounces.
- NEE enabled.
- MIS enabled.
- RTXDI disabled/ignored.
- Temporal alpha parameter exposed as 0.2.
- Source view default: `(2,2)`, one of the four central views in an even 6x6 grid.

## Debug modes

Set `lfDisplayMode` on `LightFieldReprojPathTracer`:

- `0`: final/reprojected quilt.
- `1`: source-only quilt.
- `2`: valid mask.
- `3`: discard overlay, invalid reprojection shown red.
- `4`: view ID visualization.

## Integration

Copy these folders into Falcor's render pass source tree and add them to the render-pass CMake/plugin list in the same way as your local `PathTracer` and `GBuffer` passes are registered.

```text
Source/RenderPasses/LightFieldVBufferRT/
Source/RenderPasses/LightFieldReprojPathTracer/
Source/Mogwai/Data/LightFieldReprojPathTracer.py   # or wherever you keep graph scripts
```

Then load `LightFieldReprojPathTracer.py` in Mogwai.

## Important limitations of this scaffold

This is an implementation starting point, not a fully production-quality paper reproduction.

- It uses a quilt/atlas instead of a `Texture2DArray` for simplicity.
- The reprojection pass uses world-position and normal consistency. It does not yet include a robust instance-ID test because Falcor's `GeometryInstanceID` packing differs across versions.
- Invalid/disoccluded pixels are currently visualized in red in the final mode. The paper path traces those invalid pixels as a fallback. Add a second masked path-tracing pass if you want that exact behavior.
- NRD outputs are preserved from the original PathTracer clone, but the paper-style denoise-before-reproject path is not wired as a separate NRD pass inside this package. In this starter version, the reprojection reads the source tile color output.
- For glossy/specular materials, final-radiance reprojection is still physically approximate.
