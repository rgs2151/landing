# Robust Steering Animation

Initial animation supplied by the user through the controller project page.

- Repository: https://github.com/rgs2151/controller
- Source commit: `2f1fab6a59471b8a78d0d6e96a9f96b2241acc6c`
- Live source: `pages/index.html`, `.steering-demo` element
- Animation logic: `pages/steering-demo.js`
- Animation styles: `pages/steering-demo.css`
- Existing export: `assets/steering-demo.gif`
- Local export: `public/assets/videos/robust-steering.gif`
- Export properties: 776 x 549, 249 frames, 20.75 seconds, 326159 bytes.

The three source files here are unmodified snapshots. `source-page.html` is
reference markup from the full page, not a standalone local preview. The live
animation uses inline SVG and plain JavaScript. It also inherits the page's
box-sizing reset, `--muted` and `--teal` colors, and `.sr-only` utility.

The animation introduces an activation landscape and language-steering
direction, stores it in a controller, shifts the landscape, and transfers the
controller to an illustrative conversation. The response changes from Chinese
to Spanish while the mathematical answer is preserved. The upstream source
identifies its activations and question/response example as schematic, not
recorded benchmark outputs.

Preserve the current sequence, wording, colors, and timing for the first reuse.
The original GIF is retained. Prototype A uses `public/assets/videos/robust-steering.mp4`,
an H.264 conversion with the original timing and a single white padding row
(776 x 550). This allows native playback controls and pauses when hidden.
Its poster is `public/assets/images/robust-steering.webp`, extracted at 18 seconds.
The source supports later styling and interactive adaptation.
