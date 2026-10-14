# Prototype E Performance

## Scope

Measured E without changing layout, assets, playback speed, rendering quality,
or animation behavior. The only website change in this pass is the requested
first-person About biography. Earlier prototypes retain their previous copy.

Raw measurements: [per-asset CSV](prototype-e-costs.csv),
[per-asset samples](prototype-e-costs.json),
[interactions and CPU samples](prototype-e-interactions.json).

## Findings

| Scenario | Browser frame cadence | Notes |
| --- | ---: | --- |
| All previews frozen | 60 FPS | 24-27 main-thread ms/s |
| Frozen previews with scene motion | 60 FPS | 31-35 main-thread ms/s |
| All live, short samples | 43-44 FPS | 847-866 main-thread ms/s |
| All live, 30 seconds | 49 FPS | 878 main-thread ms/s; P95 33.4ms |
| Hovering NeuralJAXlib, 4 seconds | 53 FPS | Includes entering hover; not a controlled opacity comparison |
| Opening a page, 1.8-second windows | 47-55 FPS | Worst frame up to 150ms |
| Returning to splash, 1.8-second windows | 36-40 FPS | Worst frame 133-150ms |
| Settled project pages and About | 60 FPS | Six pages sampled for 2 seconds each |

Returning to the splash is the clearest interaction hitch. Opening and returning
were sampled twice for each of the five projects and About. These short windows
include the transition and some settled playback, not just the transition.
The frozen About case has all 15 inert previews and zero live preview canvases
or running scene-motion animations. Keeping the folded stack is not the issue.

### Per-Asset Cost

Each row has only that preview live, all other previews frozen and scene motion
paused. Values include baseline page work. These costs are not additive.

| Live preview | Renderer main-thread ms/s, two samples |
| --- | ---: |
| Manifold / donut | 365-372 |
| NeuralJAXlib GAN | 212-236 |
| NeuralJAXlib autoencoder | 152-168 |
| NeuralJAXlib RNN | 150-156 |
| NeuralJAXlib CNN | 101-103 |
| Steering conversation | 99-100 |
| Steering landscape | 91-92 |
| NeuralJAXlib transformer | 77 |
| pyLBA | 71-75 |
| LFP signal | 61-65 |
| Lag equations | 52-54 |
| Each cached mouse tile | 24-33 |

All isolated previews maintained 60 FPS. Their combined workload exhausts much
of the available main-thread budget. The mouse previews already use cached WebP
frame sheets, not four playing videos; reducing them further is low priority.

## Code Evidence And Next Steps

1. **Honor the preview update budget in E.** The scene passes `maxFps={30}`,
   but `SVGAnimation` does not accept it. Four NeuralJAXlib assets are configured
   for up to 60 updates/second; GAN is configured for 30. Add an E-scoped limit
   while retaining elapsed-time playback, full-size detail behavior, and frozen
   prototype behavior. This reduces redraw frequency, not playback speed.
2. **Use a bitmap surface for GAN pixels.** GAN currently creates 3,136 SVG
   rectangles (four 28x28 images) and changes their fills during interpolation.
   Its isolated style recalculation reaches 140ms/s. A reusable Canvas2D pixel
   buffer can represent the same samples/interpolation without thousands of DOM
   mutations. Verify exact frames and pixel-edge appearance before adopting it.
3. **Smooth preview restart.** `FreezePreview` unmounts the live renderers when
   folding; returning mounts them again, including SVG creation, sample decoding,
   and a fresh WebGL renderer. This lifecycle is a plausible cause of the measured
   return stalls, not isolated causal proof. Test staging/prewarming initialization
   and keeping snapshots visible until ready, while keeping folded previews cheap.
4. **Profile the donut on actual hardware before changing its quality.** Its
   thick-line WebGL rendering is costly in this software-rendered environment.
   Do not reduce the approved line width or geometry based on this result alone.

The separate 10-second CPU sample shows DOM `setAttribute` work, React development
JSX overhead, and RNN/autoencoder work. Most samples are the unspecific native
`(program)` bucket: this is not a GPU breakdown or attribution of native time to
a particular component. Renderer-provided paused RAF loops also continue ticking
under the shared SVG wrapper; removing redundant scheduling is secondary to the
larger drawing/style costs.

## Method And Limits

- Local Vite development server; Chromium headless with SwiftShader, 1440x900,
  default device pixel ratio. No screenshots or concurrent browser benchmarks.
- Per-asset isolation: two 1.5-second samples, reverse order on the second pass,
  900ms settling. CSV retains the existing harness's upper-middle aggregation
  for two samples; use JSON for both observations. Short samples are not a full
  language or animation-cycle benchmark.
- Live interaction test: 30 seconds of playback, hover/unhover, two rounds of
  opening/returning, plus a separate 10-second CPU profile.
- Main-thread ms/s is renderer task time, not total CPU/GPU usage. 878ms/s means
  roughly 88% of one renderer main thread was occupied during that interval.
- RAF cadence is not the frame rate of every individual asset. Native animation
  paint/presentation and hardware GPU behavior require a real-device trace.
- These figures reproduce local stutter but do not predict the user's GPU or a
  deployed production build. Prototype E is currently dev-only; the ordinary
  production build does not include it. Do not call these production benchmarks.
- Geometry, motion, rendering settings, and full-size animations remain untouched.
  The optimization ideas above have not yet been implemented or A/B tested.

## Reproduce

With the local server running:

```sh
VARIANT=e REPETITIONS=2 SAMPLE_MS=1500 \
CASES=idle,motion,nature-0,nature-1,nature-2,nature-3,signal,autoregressive,manifold,landscape,conversation,pylba,neuraljax-cnn,neuraljax-rnn,neuraljax-autoencoder,neuraljax-gan,neuraljax-transformer,all,folded-about \
node tests/prototype-b-costs.mjs
node tests/prototype-e-performance.mjs
```

Run sequentially and do not edit app source while sampling. The browser test
also checks the exact new biography, its retained institution link, navigation
destinations, and absence of page errors. Browser/Node paths follow the existing
local test setup.
