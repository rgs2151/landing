# D: Tools and Packages

Route: `/prototype/d`. C remains frozen at `/prototype/c`; nothing is promoted
to the production homepage.

- Reuse C's nine existing fragments, positions, sizes, rotations, motion,
  portrait travel, project pages and content. D imports the same scene layout.
- Add PyLBA at (1010,25), width 175, rotation 7 degrees, outside the central
  research composition. Idle opacity is 50%; hover/focus restores 100%.
  Other hover fading and folded snapshots retain C's behavior.
- Navigation labels in D are Research Publications and Tools & Packages.
  Both dropdowns share title typography, spacing and width, with metadata
  beneath each title. PyLBA's year is 2025.
  The tools dropdown and splash preview both open PyLBA within the existing
  slide-in page, including direct hash links and back navigation.
- PyLBA uses the README title: rocket emoji followed by
  "pyLBA: Linear Ballistic Accumulator Models in Python". Rudramani Singha is the
  sole author. Code then Documentation links come immediately below the author.
  The animation floats left at its original 300px maximum width, with the
  abstract wrapping on its right. Preserve figure size over forcing text below it.
- Shared content: `src/content/pylba.js` and `src/content/pylba.md`.
  Shared animation: `src/shared/pylba/`. The full view and splash use the same
  React SVG and original SMIL timeline. No iframe or video conversion.
- Controls follow the existing inset pause/play and replay standard. A small
  extension of the full viewBox reserves control space without hiding labels;
  the compact view retains the original 300 x 300 frame. Both are transparent.
- Playback pauses when inactive, offscreen, hidden, or reduced motion is enabled.
  Folded previews capture current opacity values and remove SMIL animation nodes
  so the stack stays inert.

## NeuralJAXlib Addition

- NeuralJAXlib (2024) uses its original SVG logo inline with the name in the
  heading and Tools & Packages dropdown. Links remain code, then documentation.
- Five independent splash fragments: CNN (795,640), RNN (20,55), Autoencoder
  (285,595), GAN (10,610), Transformer (1000,580). Existing placements are unchanged.
  New fragments sit above the state-space assets on the splash so their hover
  targets remain reachable. RNN leads in the upper-left; CNN sits below the waveform.
  All five share the tool's hover state and open `/prototype/d#neuraljaxlib`.
- The detail page uses the corrected five-panel composite in one row immediately
  below the links, with one shared pause/play and replay control. README/gallery
  text follows unchanged. These are source excerpts, not a new abstract.
  The Flax/PyTorch-informed abstract is a later, separate discussion.
- `src/shared/tools.js` defines each tool's ordered figures and associated text;
  `ToolDetail` and `ProjectTitle` render all tools uniformly. Neither the detail
  renderer nor the splash needs project-specific content conditionals.
- Native SVG renderers and cached samples live under `src/shared/neuraljaxlib/`.
  They share React lifecycle/control integration, not copied animations per view.
  RNN label sizing uses the untransformed SVG viewport, preventing inverse scaling
  of "now" when a preview is small or tilted.
- The composite preserves standalone SVGs in a 1248 x 300 HTML grid rather
  than transplanting them into nested SVG viewports. Its 240px panels share
  heading/plot/legend tracks. RNN, PCA and transformer labels match the 18px
  heading scale; the RNN legend is reused. The adapter uses the full 1248px
  design width to prevent inflated CNN/RNN strokes. Standalone assets, splash
  placements and content remain unchanged by this composite correction.
