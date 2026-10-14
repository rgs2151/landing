# Project Animation Assets

High-quality sources stay shared. Landing previews are derived exports, never
independently rewritten animations.

```text
src/assets/<project>/
  fragments.js            # Named high-quality fragment entry points
  composite.js            # Full project animation entry point
public/assets/projects/<project>/
  videos/<fragment>.webm  # Small transparent VP9 preview
  posters/<fragment>.webp # Matching first frame
  previews.json          # Dimensions, loop, provenance and source hashes
```

Projects: `neuraljaxlib`, `neural-field-manifold`, `robust-steering`, `pylba`.
For example, `neuraljaxlib/fragments.js` exports `autoencoder`, paired with
`videos/autoencoder.webm` and `posters/autoencoder.webp`.

The entry points re-export existing implementations in `src/shared/`; they do
not duplicate source code. High-quality sources include SVG, React/HTML, and
WebGL for the manifold. Composites remain native, not rendered videos. Existing
project pages continue using these accepted sources without visual changes.

The four Nature previews already play prerecorded raster frames. They retain
their existing files/player under `public/assets/images/previews/`; their
original/full-width videos remain under `public/assets/videos/`. Frozen earlier
prototypes retain their original rendering paths.

## Export After Approval

1. Register a fragment in `exportCatalog.js`, using its complete loop and
   accepted compact framing. Add its source paths in `scripts/preview-sources.mjs`.
2. With Vite running on port 5173, run `node scripts/export-landing-videos.mjs`.
   To update selected clips only, use `ASSETS=landscape,conversation` (manifest IDs).
3. Run `node tests/prototype-f-browser.mjs` for functional checks, not benchmarks.

The exporter requires FFmpeg and the local Playwright/Sharp test tooling under
`/tmp/landing-browser-check`. It renders the actual components at deterministic
timestamps into isolated transparent frames, not page screenshots. The catalog
sets per-fragment resolution and frame rate. The manifest records source hashes,
encoded dimensions, original layout dimensions, duration, frame count and bytes.
Output files replace the same named exports; revision hashes invalidate caches.
For canvas-backed fragments, render the source drawing buffer at export quality
before encoding; increasing only the output video dimensions upscales blur.
The manifold uses a 4x source buffer, 960px output width and CRF 10. Other preview
settings remain unchanged. `VERIFY_ONLY=1 ASSETS=signal,autoregressive` checks
unchanged sibling first-frame pixels without regenerating their media.

`landingPreviews.jsx` is the shared native video player. F requires an exported
preview for every live-source fragment, with no live-renderer fallback. Original
layout aspect ratios are preserved independently of codec dimension rounding.
Videos respect composition pause/replay, reduced motion, visibility, and folding
into static snapshots. Steering includes all languages and loops its steady
cycle after the initial entrance. No speed or content changes are made.

These exports trade download/storage for avoiding live simulation and DOM
updates. They are compressed low-resolution previews, not pixel-identical SVGs.
Do not regenerate or change approved high-quality assets without permission.
