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

The source files here began as snapshots and now include the approved local
animation edits. The upstream commit above identifies the original. `source-page.html` is
reference markup from the full page, not a standalone local preview. The live
animation uses inline SVG and plain JavaScript. It also inherits the page's
box-sizing reset, `--muted` and `--teal` colors, and `.sr-only` utility.

The upstream animation introduces an activation landscape and language-steering
direction, stores it in a controller, shifts the landscape, and transfers the
controller to an illustrative conversation. The response changes from Chinese
to Spanish while the mathematical answer is preserved. The upstream source
identifies its activations and question/response example as schematic, not
recorded benchmark outputs.

Prototype A runs the adapted JavaScript and SVG, not the GIF or MP4.
`src/shared/RobustSteeringAnimation.jsx` imports these local sources,
extracts only the animation markup using DOMParser, and runs it in a sandboxed
inline frame. There is no remote embed and no video request. The fixed internal
viewport preserves the source composition at thumbnail and reading sizes.

The wrapper supplies the inherited colors/reset and pauses the thumbnail while
a project is open. The full view uses the standard React/Lucide inset controls;
the source buttons are hidden and driven through the sandbox message bridge.
Replay returns to the initial state without changing the user's pause choice.
Reduced-motion starts paused, with explicit play available. Removing the frame
also disposes all its animation work.
Previous media exports remain archived but are no longer used by the prototype.

Approved local changes: omit the animation heading, controller box and both
metric readouts. The steering vector travels directly to the right-hand lightning
instead of stopping at a controller. The landscape and vector are visible at
the start, with no opening fade or white backing. Complementary clips replace
the opaque white landscape wipe. Preserve the question/response copy, seeded
landscapes, colors and remaining sequence for subsequent focused iterations.
