# Design Decisions

## Animation Controls

All animations must use the accepted NeuralFieldManifold control style when
created or revised:

- Pause/play toggle followed by replay, using Lucide Pause, Play and RotateCcw.
- 17px icons in 32px transparent, borderless buttons with a 4px gap.
- Dark-red icons (`#8b1e1e`), pale-red hover (`#f5ebeb`).
- Controls inside the animation at bottom-right, inset 10px horizontally and
  8px vertically; no separate control bar below the animation.
- Keep controls clear of the figure, with accessible names and hover titles.
- Pause/play controls playback; replay restores the animation's intended
  starting state. Preserve reduced-motion and offscreen playback handling.

This standard does not authorize unrelated changes to existing assets. Apply
it as each animation is worked on. The accepted compact collage preview remains
non-interactive; its full project view provides the controls.

## Frozen NeuralFieldManifold Asset

The asset is frozen after the final vertical-margin crop and inset-controls
adjustment. Do not redesign, reword, retime or regenerate it unless the user
explicitly requests reopening it.

- Preserve the figure, labels, emoji, arrows, fonts, colors and geometry.
- Preserve the recorded mouse LFP data, lag, eighth-speed playback, midpoint
  start and continuous history.
- Presentation uses SVG viewBox `0 50 1000 480` and aspect ratio `1000 / 480`.
  This trims whitespace without stretching or rearranging the figure. The
  overlaid manifold retains its original position and size in SVG coordinates.
- Reuse the shared asset across prototypes rather than making independent copies.

Implementation: `src/shared/NeuralFieldAnimation.jsx`,
`src/shared/neural-field-animation.css`, `src/shared/neuralFieldModel.js` and
`public/assets/data/mouse-lfp.json`. Provenance and playback details are in
`design/references/neural-field-manifold/README.md`.
