# Prototype F: Cached GAN And Autoencoder

E is frozen at `d135339`. F changes only these two splash renderers. All
content, placement, dimensions, tilt, opacity, interactions, and project-page
animations are preserved. The donut remains live and unchanged.

## Results

| Measurement | E | F |
| --- | ---: | ---: |
| Complete splash, short isolated-harness samples | 43-44 FPS | 60 FPS |
| Complete splash, 30-second playback | 49 FPS | 60 FPS |
| Long-run P95 frame interval | 33.4ms | 16.8ms |
| Hover window | 53 FPS | 60 FPS |
| Return-to-splash windows | 36-40 FPS | 54-56 FPS |
| Worst frame during return | 133-150ms | 133-150ms |
| Settled project pages | 60 FPS | 60 FPS |
| GAN alone, renderer task time | 212-236ms/s | 31-32ms/s |
| Autoencoder alone, renderer task time | 152-168ms/s | 38-39ms/s |

The 30-second F playback and hover samples had no long tasks. This is a useful
steady-playback improvement without changing the donut or reducing animation
speed. Startup/transition stalls are **not eliminated**. Opening the full
manifold had a worst frame of 217ms in this F run. Its original live WebGL asset
is unchanged. Future transition work should investigate initialization and
snapshot handoff separately, not silently lower the accepted detail quality.

## Fidelity And Cost

All 15 preview placements, dimensions, titles, tilts and opacity match E at
1024, 1440 and 1920px desktop widths. Pause/replay, fold/unfold, and the original
full SVG gallery pass checks. Sampled GAN frames match source pixels exactly;
autoencoder alpha matches exactly, with a maximum 1/255 color-rounding difference.
See [frame parity](prototype-f-parity.json) and [design notes](../design/prototype-f.md).

The cached loops use lossless WebP frame sheets and canvas blits, matching the
mouse-preview approach, not continuous video decoding. This avoids thousands
of SVG updates while retaining transparent overlap and original HTML titles.
The complete caches add about 8.32MiB of image data (most is the high-resolution
autoencoder); initial first/next sheets total about 1.07MiB. Old decoded sheets
are released; the explicit bitmap working set is about 30.5MiB at peak. This is
a CPU/bandwidth tradeoff, not a claim that cached previews are free or smaller
to download than the original sample data. Browser cache/GPU/transient storage
are additional. More aggressive video compression would require a separate
fidelity decision.

## Evidence And Limits

- [F per-asset CSV](prototype-f-costs.csv) and [raw samples](prototype-f-costs.json).
- [F playback, hover, six page transitions and CPU samples](prototype-f-interactions.json).
- [E baseline report](prototype-e-performance.md). E was measured before this
  change, using the same 1440x900 headless Chromium/SwiftShader environment.
- F's short harness used two 2.5-second samples versus E's two 1.5-second
  samples. Both also have a separate 30-second playback measurement. E has
  two transition rounds; F has one. These are comparative diagnostics, not
  statistically controlled device-performance guarantees.
- No screenshots; asset checks rasterize isolated SVG artwork only. No visual
  changes, production promotion, or sibling-repository changes.
- Local Vite development mode and software rendering; not production or the
  user's actual GPU. Main-thread task time is not total CPU/GPU utilization.
- F's per-asset samples preceded a lossless sheet-packing adjustment that removed
  one extra alpha-rounding step. Final frame checks and the 30-second interaction
  run use the final sheets; playback implementation and dimensions are identical.

Reproduce sequentially with Vite running:

```sh
node tests/prototype-f-browser.mjs
VARIANT=f REPETITIONS=2 SAMPLE_MS=2500 CASES=idle,neuraljax-gan,neuraljax-autoencoder,all,folded-about node tests/prototype-b-costs.mjs
VARIANT=f ROUNDS=1 node tests/prototype-e-performance.mjs
```
