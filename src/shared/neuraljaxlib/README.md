# NeuralJAXlib Assets

Source: `https://github.com/rgs2151/NeuralJAXlib`, imported from the sibling
`NeuralJAXwork` checkout. `vendor/` contains the five approved gallery renderers
and cached samples: CNN, RNN, Autoencoder, GAN and Transformer, plus the corrected
composite renderer that keeps those same five assets intact in an HTML grid row. `logo.svg` is
the original vector logo, used inline at text height rather than rasterized.
The source MIT license is included.

Run `node scripts/import-neuraljax-assets.mjs /path/to/NeuralJAXwork` only when
explicitly updating these assets. The importer changes module export boundaries,
not geometry, samples, labels, or animation timing. The shared source now includes
RNN's resize fix: `clientWidth` replaces `getBoundingClientRect().width`
so "now" scales with its SVG rather than compensating for ancestor transforms.
Source and imported hashes
are recorded in `provenance.json`. No models, training data, videos or checkpoints
are copied into the website.

`animations.jsx` registers the five lazy-loaded assets and the composite. The shared
`SVGAnimation.jsx` React adapter mounts native SVG, handles lifecycle and
standard pause/play/replay controls, and uses each asset's design width: 300px
for standalone figures, 1248px for the composite. This prevents double scaling
from inflating non-scaling SVG strokes. Only white outer SVG backings are made
transparent; meaningful white regions and black image grids are unchanged.
No iframe, injected script, canvas rendering or video conversion is used.

Each instance has an independent clock, with the source timing preserved.
Standalone GAN rendering retains its 30fps limit. Hidden, offscreen, inactive and
reduced-motion views do not advance. Folded previews retain inert SVG snapshots;
the running renderer is destroyed. A deterministic `elapsedMs` prop supports
source parity tests without screenshots.

Metadata and verbatim README/gallery excerpts live in `src/content/neuraljaxlib.*`.
The framework abstract is intentionally deferred for a separate copy iteration.
`src/shared/tools.js` pairs content with these assets for the common detail renderer.
Prototype D owns placement only; full pages and splash fragments use the same
asset components. There are no NeuralJAXlib-specific branches in the page renderer.
The optional tool `Gallery` slot places a composite above all copy without
repeating its individual figures. PyLBA keeps its existing wrapped-figure layout.

The corrected composite uses five 240px-wide panels with 12px gaps and a shared
heading/plot grid. Headings and labels share an 18px design scale; PCA
labels are normalized only within the row. The original RNN legend stays with
its panel, tucked directly below the circle. The website-only composite is
1248 x 270, ending at the CNN/GAN plots, with inset controls beneath Transformer.
No SVG is transplanted into another SVG viewport. This local spacing revision
has not been synced to the source repository; wait for user approval before
any upstream changes or export generation. Reimporting the composite would
overwrite this local layout adjustment.
