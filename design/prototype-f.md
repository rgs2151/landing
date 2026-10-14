# Prototype F

Route: `/prototype/f`. E is frozen at `d135339` and remains at `/prototype/e`.
F starts with the same content, 15 placements, sizing, tilt, depth, opacity,
motion, hover, navigation, project arrivals, centered stack, sidebar and controls.

Only GAN and autoencoder splash rendering changes. The complete NeuralJAXlib
gallery and all other full-size animations retain their live renderers. The
donut is unchanged: measure the two initial substitutions before doing more.

## Cached Previews

- `src/shared/neuraljaxlib/CachedPreviews.jsx` supplies F's two preview renderers
  through the existing `SVGAnimation` lifecycle. It respects pause/replay,
  visibility, reduced motion, and the existing freeze/unmount behavior.
- Frames come directly from the accepted vendor renderers and their data, at
  deterministic timestamps. No screenshots, rewritten simulations, altered
  easing, or changes to the source renderers are involved.
- The heading is the original live HTML text with its original inline style.
  The artwork becomes a transparent Canvas2D frame-sheet player. Cached frames
  are lossless WebP, like the mouse previews rather than opaque video files.
- GAN keeps its native 57x57 four-image pixel grid and crisp pixel scaling;
  138 cached frames span its exact 4590ms loop. Autoencoder uses 480x436 artwork,
  324 time samples spanning its exact 5400ms loop; repeated hold frames share
  one cached image. No loop stages are omitted.
- Only the current/next autoencoder sheets and shared first sheet are retained
  by the renderer; superseded ImageBitmaps are closed. Upcoming sheets preload.
  While a sheet is loading, the last valid frame remains visible.
- Source hashes and timings live in
  `public/assets/images/previews/neuraljaxlib/clips.json`. Rebuild with
  `node scripts/cache-neuraljax-previews.mjs` while Vite is running. The script
  uses the local test installation of Playwright and Sharp under
  `/tmp/landing-browser-check`; no runtime dependency was added.

## Fidelity

`tests/prototype-f-browser.mjs` checks all 15 assets' dimensions, placements,
tilts, opacity and titles against E at 1024, 1440 and 1920px desktop widths.
It also checks playback, pause/replay, folding, full-gallery SVG preservation,
source hashes, and sampled cached frames against the original SVG renderer.

The GAN sampled frames have zero pixel difference at native resolution.
Autoencoder alpha matches exactly; compositing over white differs by at most
1/255 per color channel, with mean difference under 0.041/255, from raster
premultiplication/rounding. This is not a claim of byte-identical rasterization
at every zoom, transform or display density. Source geometry and timing match;
titles stay vector text. Results: `reports/prototype-f-parity.json`.

## Tradeoff

GAN cache: 249,462 bytes. Autoencoder cache: 8,472,384 bytes across 16 sheets.
The initial GAN plus first/next autoencoder sheets total 1,117,556 bytes; later
sheets load as playback approaches them. This spends bandwidth/storage to avoid
per-frame SVG node updates. Native source data was smaller (about 611KB combined).
Explicit decoded bitmap storage is about 30.5MiB at peak for both previews,
excluding transient decoding, GPU copies, and the browser's HTTP/resource cache.

Do not quietly lower resolution or alter animation speed to reduce those bytes.
Keep the full project renderers and frozen E untouched in subsequent optimization.
