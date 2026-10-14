# Design Decisions

## Robust Steering Views

The full steering asset uses a 776 x 380 frame, trimming 20px of bottom space
without changing figure geometry. Standard inset controls move up with the frame.
`src/shared/RobustSteeringAnimation.jsx` also exports `RobustSteeringLandscape`
and `RobustSteeringConversation`, accepting the same `compact` and `active` props
as the neural-field fragments. Prototype A places these separately; either opens
the full project view. Standalone views retain pause/replay controls.

Unlike the native React/SVG neural-field assets, steering still uses an isolated
SVG/HTML iframe. Its two fragments are transparent viewport views of the same
renderer, timeline, content and cached geometry, not independently copied animations.
The unused side is hidden without changing layout measurements. Each placement
has its own playback state. A native React port is a separate task, not part of
this extraction. Do not describe these fragments as native SVG components.

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

## NeuralFieldManifold Fragments

The frozen full schematic also supplies three reusable, transparent rectangular
assets. They are named exports from the same `NeuralFieldAnimation.jsx` module:

- `NeuralFieldSignal`: waveform and moving lag dots only; no title or emoji.
- `NeuralFieldAutoregressive`: the existing equation, temporal axes, labels and
  moving dot; no title, border or tinted background.
- `NeuralFieldManifold`: the existing trajectory, plane, moving dot and coordinate
  equation; no heading.

These are views of the shared rendering and playback implementation, not copied
animations. All use the same cached recording, lag and playback settings. The
full schematic's appearance and behavior remain frozen. Preserve that output
when maintaining the shared implementation; do not propagate a fragment-only
style change into the full schematic.

Each fragment supports the existing `active` and `compact` props. Standalone
fragments have the standard inset controls. Compact clickable previews omit
controls and open the full project view, avoiding nested buttons. Each instance
has its own playback state; reuse does not require coupling unrelated placements.

Keep each project's assets together in code, not clustered together on the page.
Prototype A distributes the three fragments independently around the composition,
like the Nature panels, with individual positions and hover transforms. Their
white backing belongs to that prototype, not the transparent assets. Do not
change Nature media or the production homepage as part of this extraction.
