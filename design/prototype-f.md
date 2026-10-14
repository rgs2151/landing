# Prototype F

Route: `/prototype/f`. E remains frozen at `d135339` and `/prototype/e`.
Content, all 15 placements, dimensions, rotation, depth, opacity, motion, hover,
navigation, project arrivals, centered stack, sidebar and controls are unchanged.

## Prerecorded Landing

All eleven formerly live fragments now use transparent low-resolution WebM:
five NeuralJAXlib assets, three NeuralFieldManifold assets, both Robust Steering
panels and PyLBA. The four Nature previews retain their existing prerecorded
frame-sheet playback. No live SVG/WebGL simulation remains on F's splash.

Project pages retain their accepted live high-quality renderers and composites.
Videos are derived from those exact sources, preserving the complete timeline
and framing. Encoded pixels are compressed; they are not byte-identical to SVG.
The new files total approximately 23.8 MB, trading transfer/storage for avoiding
live simulation and SVG updates. No new performance measurements were requested
or performed, and no speedup is claimed here.

See [the asset guide](../src/assets/README.md) for project folders, manifests,
export commands and policy. `tests/prototype-f-browser.mjs` checks source hashes,
geometry against E, transparency, pause/replay, reduced motion, steering loops,
static folding snapshots and original full-gallery renderers. No screenshots.

## History

The initial F at `eb6eb7b` cached only GAN and autoencoder using WebP frame sheets.
The historical results in `reports/prototype-f-performance.md` and accompanying
JSON/CSV files apply to that commit only. Its superseded player, exporter and
generated sheets were removed after the all-video conversion.
