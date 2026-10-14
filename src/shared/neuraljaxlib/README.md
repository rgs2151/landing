# NeuralJAXlib Assets

Source: `https://github.com/rgs2151/NeuralJAXlib`, imported from the sibling
`NeuralJAXwork` checkout. `vendor/` contains the five approved gallery renderers
and cached samples: CNN, RNN, Autoencoder, GAN and Transformer. `logo.svg` is
the original vector logo, used inline at text height rather than rasterized.
The source MIT license is included.

Run `node scripts/import-neuraljax-assets.mjs /path/to/NeuralJAXwork` only when
explicitly updating these assets. The importer changes module export boundaries,
not geometry, samples, labels, or animation timing. Source and imported hashes
are recorded in `provenance.json`. No models, training data, videos or checkpoints
are copied into the website.

`animations.jsx` registers the five lazy-loaded assets. The shared
`SVGAnimation.jsx` React adapter mounts native SVG, handles lifecycle and
standard pause/play/replay controls, and keeps a consistent 300px design width
so text and artwork scale together. Only white outer SVG backings are made
transparent; meaningful white regions and black image grids are unchanged.
No iframe, injected script, canvas rendering or video conversion is used.

Each instance has an independent clock, with the source timing preserved.
GAN rendering retains its 30fps limit. Hidden, offscreen, inactive and
reduced-motion views do not advance. Folded previews retain inert SVG snapshots;
the running renderer is destroyed. A deterministic `elapsedMs` prop supports
source parity tests without screenshots.

Metadata and verbatim README/gallery excerpts live in `src/content/neuraljaxlib.*`.
The framework abstract is intentionally deferred for a separate copy iteration.
`src/shared/tools.js` pairs content with these assets for the common detail renderer.
Prototype D owns placement only; full pages and splash fragments use the same
asset components. There are no NeuralJAXlib-specific branches in the page renderer.
