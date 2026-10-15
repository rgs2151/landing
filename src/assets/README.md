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
  videos/mobile/<fragment>.mp4 # H.264 RGB + alpha compatibility preview
  mobile-previews.json    # Compatibility dimensions and original video hashes
```

Projects: `neuraljaxlib`, `neural-field-manifold`, `robust-steering`, `pylba`.
For example, `neuraljaxlib/fragments.js` exports `autoencoder`, paired with
`videos/autoencoder.webm` and `posters/autoencoder.webp`.

NeuralJAXlib's current `cnn` and `gan` entries use the approved brain MRI
variants. The original MNIST CNN and Fashion-MNIST GAN remain in their original
vendor folders and are used within the abstract. All five NeuralJAXlib splash
videos omit the title pixels while preserving artwork geometry; detail figures
retain their titles. The source variant map is `neuraljaxlib/variants.js`.

The entry points re-export existing implementations in `src/shared/`; they do
not duplicate source code. High-quality sources include SVG, React/HTML, and
WebGL for the manifold. Composites remain native, not rendered videos. Existing
project pages continue using these accepted sources without visual changes.

The four Nature previews already play prerecorded raster frames. They retain
their existing files/player under `public/assets/images/previews/`; their
original/full-width videos remain under `public/assets/videos/`. Earlier
prototype rendering paths remain in Git history, not in the running website.

## Export After Approval

1. Register a fragment in `exportCatalog.js`, using its complete loop and
   accepted compact framing. Add its source paths in `scripts/preview-sources.mjs`.
2. With Vite running on port 5173, run `node scripts/export-landing-videos.mjs`.
   To update selected clips only, use `ASSETS=landscape,conversation` (manifest IDs).
3. Run `node scripts/export-mobile-videos.mjs` to derive phone-compatible H.264
   copies from approved WebMs. The same `ASSETS` filter selects individual clips.
4. Run `node tests/site-browser.mjs` and `node tests/mobile-video-compat.mjs`
   for functional checks, not benchmarks.

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

`landingPreviews.jsx` is the shared native video player. The main site requires an exported
preview for every live-source fragment, with no live-renderer fallback. Original
layout aspect ratios are preserved independently of codec dimension rounding.
Videos respect composition pause/replay, reduced motion, visibility, and folding
into static snapshots. Steering includes all languages and loops its steady
cycle after the initial entrance. No speed or content changes are made.

These exports trade download/storage for avoiding live simulation and DOM
updates. They are compressed low-resolution previews, not pixel-identical SVGs.
Do not regenerate or change approved high-quality assets without permission.

On phones that cannot decode WebM alpha, `src/site/MobileVideo.jsx` plays the
H.264 derivative, restoring the right-hand alpha plane over the left-hand RGB
frame in a canvas. These exports retain original timing and framing, at up to
480px width. They are used only after WebM alpha/decode failure, never on desktop.
The mobile manifest records the exact source WebM hash; no source animation is
rerendered or modified for this compatibility conversion.
