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
instead of stopping at a controller. The landscape is visible immediately, with
no opening fade or white backing. In English, the two direction points appear
before the connecting arrow draws and travels to the lightning. Complementary
clips replace the opaque white landscape wipe. Each language reuses a cached,
seeded synthetic field; these contours are not recorded model activations.

The current frame is 776 x 400, capped at its native width in the project view.
The language headings use the normal 20px text size in bold above the
plot. Question and model response share an aligned icon/text layout, with a
40px row gap. The model itself aligns with the first response line; lightning
sits above it, outside the row's height calculation, without vibration marks.
Both text rows reserve enough space
for every language and the corruption frames so the layout does not shift.

The cycle is English, Chinese, French, Japanese, Swahili, Telugu, then English
again. The initial English question and answer type while the direction appears.
Every later transition swipes the field and heading over 2200ms, quickly backspaces the old
question and types the new one, then backspaces and types the answer. The previous
answer remains visible while the question changes. Only the response uses the
Spanish corruption effect. Both text rows finish during the swipe; steering begins
immediately when the swipe completes and lasts 2800ms. There is no red state dot
or expanding pulse between these steps. Headings use the same left-to-right wipe
as the landscape. During steering, a teal line grows along a cached contour from
a red point to a distant teal point, without an arrowhead.
The gray axis lines are removed; axis labels remain. Swipes,
direction movement and contour drawing use a shared cubic ease-in/ease-out for
a snappier middle and soft arrival, with all phase timings unchanged. Lightning
uses a matching ease-in/ease-out rhythm without changing its cycle duration.
Text typing, deletion and response overwrite remain linear at the same timings.

`steering-landscapes.json` caches the unchanged seeded KDE landscapes and the
contour-following routes. Regenerate with `node scripts/cache-steering-landscapes.mjs`.
The generator selects seeded starts and distant endpoints on intermediate density
levels that are never drawn in the landscape, rather than tracing visible contours;
the browser only draws these stored paths. These are synthetic illustrations, not
measured model-state trajectories. The English direction introduction is unchanged.
English returns through the same transition, without a blank scene or hard reset.
Typing and deletion use Unicode grapheme clusters, including Telugu combining marks.

## Display Text

`mgsm-examples.json` replaces the earlier invented pencils example. It contains
complete recorded translations of problem 117 and short answer excerpts from sibling
`controller/code/benchmarks/mgsm/cache/` generation files. This display is illustrative:
the Swahili-to-Spanish answer is manually corrected and the Telugu-to-Spanish name
is corrected from "jhon" to "John". Other answers retain the existing excerpts.
Manual corrections are stored in `displayEdits`, separately from raw source excerpts,
model IDs and hashes. Edited display text must not be described as recorded output.
No disclaimer or illustration label is added to the animation; the user will decide
how to contextualize it elsewhere later.

English, Chinese, French and Japanese use Qwen3-4B problem 117. Telugu uses the
same problem from Phi-4-mini-instruct. Swahili now uses Llama-3.2-3B problem 117,
with the correct native baseline "Jibu ni 360." and the manually supplied Spanish
answer "Respuesta: 360 dólares." Every language asks about the same shoes purchase.
English remains an unsteered introduction.

Regenerate and verify every excerpt against its original completion:

```sh
node scripts/cache-mgsm-examples.mjs /home/dev/controller/code
node --test tests/steering-timeline.test.js
```

`steering-timeline.js` provides the deterministic, pausable sequence used by the
renderer and tests. Only the small selected cache is included in this website;
the controller repository and full experiment outputs remain outside it.
