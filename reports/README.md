# Prototype B Performance

For the current E measurements, see [Prototype E Performance](prototype-e-performance.md).
Everything below describes historical B measurements, not the current splash.

Measured results live in [prototype-b-costs.csv](prototype-b-costs.csv).
[prototype-b-costs.json](prototype-b-costs.json) contains every sample, browser
version, viewport, preview sizes, source-frame counts, and the measurement date.

## Focus And Depth Iteration

[prototype-b-focus.json](prototype-b-focus.json) records the black-splash/depth
iteration before the white-background correction and 20% size/radius increase.
These are historical measurements: three 1500ms samples per condition, 1440x900, headless Chromium with
SwiftShader. Both conditions hold the same hovered scene, with its orbits
paused and internal figures running. The control disables only opacity dimming.

| Condition | FPS | P95 frame interval | Main-thread ms/s |
| --- | ---: | ---: | ---: |
| Hover, dimming disabled | 60.0 | 16.8ms | 476.19 |
| Hover, dimming enabled | 60.0 | 16.7ms | 451.35 |

No measurable frame-cadence penalty from opacity dimming in this local test.
The task-time difference is not evidence that dimming improves performance.
This measures settled hover, not the transition or the entire orbit cycle.
There is no blur filter. The thicker manifold uses triangle-based wide lines
and costs more to render than the previous native one-pixel lines; the preview
redraw cap is now 30 FPS while playback speed remains unchanged. Full diagrams
retain their original renderer settings and timing.

Reproduce: `node tests/prototype-b-motion.mjs`. This also checks depth ordering,
full-loop continuity, hover straightening, pointer lock during unfolding,
axis-label bounds and the hidden About stack, without screenshots.

## Earlier Per-Asset Baseline

The following measurements and the costs CSV/JSON predate the black splash,
full orbits and thicker manifold. They are historical, not current totals.

Three repetitions, 1000ms samples, after settling. All cases measured 60 FPS;
the 95th-percentile frame interval was 16.7-16.8ms. Values below are median total
renderer main-thread time, including the idle page overhead.

| Case | Main-thread ms/s |
| --- | ---: |
| Idle snapshots | 11.8 |
| Snapshots with orbital motion | 31.7 |
| Each cached footage tile, separately | 16.8-18.2 |
| Equation / lag axes | 46.6 |
| LFP waveform | 65.4 |
| Steering landscape | 81.5 |
| Steering conversation | 89.8 |
| Large WebGL manifold | 120.0 |
| Complete live collage | 234.9 |
| Folded About scene | 13.4 |

The small footage previews are no longer the main cost. The larger manifold
and the two React steering fragments deserve more attention as more assets are
added. The folded scene is near the idle baseline: **zero live preview canvases
and zero running orbit animations** were verified in every folded sample. A
project's full-size animation still runs separately when its reading view opens.

The four frame sheets total **157,448 bytes** (about 154KiB), with approximately
**18.9MiB** of raw decoded pixels. No claim about total process memory follows
from that number.

## What Is Being Measured

- `idle`: all nine previews replaced by inert snapshots; orbit motion paused.
- `motion`: the same snapshots, with compositor-driven orbital motion enabled.
- An asset name: only that asset is live; every other preview is a snapshot and
  the orbital motion is paused. This separates content animation from positioning.
- `all`: the complete live splash with orbit motion.
- `folded-about`: the reading state with nine frozen previews, the expanded
  portrait, zero live preview canvases, and zero running orbit animations.

`mainThreadMsPerSecond` measures Chromium renderer main-thread task time per
second of wall time. For example, 100ms/s is about 10% of one main thread. It is
not whole-process CPU or GPU utilization. Script, layout and style time are
subsets, not additional costs. FPS describes browser frame cadence; it is not
the source video's frame rate.

These are **SVG** animations, not CSV animations. The waveform uses cached LFP
samples to update SVG paths. The equation fragment updates an SVG point. The
manifold uses Three.js/WebGL. The steering landscape is React/SVG and its
conversation combines React text and SVG. Cached source geometry does not make
per-frame React updates or SVG painting free.

## Preview Policy

The full project assets retain their accepted resolution, geometry and timing.
Only the splash uses cheaper renderings:

- Footage: all original frames and 15 FPS timing, cached in WebP frame sheets.
  The dominant clip uses 320px frames; the three smaller clips use 160, 128 and
  96px. `assets` in the JSON records exact file bytes and raw decoded pixel sizes.
- Steering and neural-field fragments: at most 30 rendered updates per second. The clock
  and phase durations are unchanged. Full-size steering has no added frame cap.
- Folded previews: inert SVG/text clones and small bitmap copies of canvases.
  The live React components unmount, their observers/RAF loops stop, and the
  preview WebGL context is disposed. This is not a hidden running animation.
- The manifold has an optional current-frame capture hook for this conversion.
  It redraws the same frame once; it does not advance or change the simulation.

## Method And Limits

The harness runs in Chromium headless at 1440x900 with SwiftShader, using the
local Vite development server. Each case has a 900ms settling period before
sampling. Repeated runs alternate ordering; the CSV reports medians. The JSON
retains the variation between individual runs.

The samples cover initial playback, not every language or the entire steering
cycle. Increase `SAMPLE_MS` for a full-cycle stress test. Results are local
comparisons, not guarantees for another device. Actual GPU/compositor work,
decoder threads and native texture allocations are not included in the
main-thread metric. `decodedPixelBytes` is a raw frame-sheet estimate, not total
browser memory. Costs are not additive: several animations can share paint and
layout work. Small differences near idle are measurement noise.

The large original-video slowdown and the first frame-sheet experiment are
documented in `design/prototype-b.md`. The costs CSV records the earlier
implementation; it is not a controlled comparison with the new focus/depth iteration.

## Reproduce

Start the local development server, then run:

```sh
node tests/prototype-b-costs.mjs
```

Defaults: three repetitions, 2500ms per sample. Overrides:

```sh
REPETITIONS=3 SAMPLE_MS=1500 node tests/prototype-b-costs.mjs
CASES=all,folded-about SAMPLE_MS=45000 REPETITIONS=1 node tests/prototype-b-costs.mjs
```

The script uses the same locally installed Playwright/Chromium as the other
browser checks. Do not edit source files or run another browser benchmark while
it runs: Vite reloads invalidate the sample. No screenshots are taken.
