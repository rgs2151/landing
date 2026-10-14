# B: Continuous Research Composition

A remains unchanged at `/prototype/a`. B lives at `/prototype/b`; the production
homepage is untouched. All project content and finalized animations are shared.

## A Findings

- `showModal()` and a dark backdrop replace the composition on selection. This
  contradicts the requested in-scene expansion and disconnects exploration from reading.
- All four Nature fragments are `<img>` elements. Browser inspection confirmed
  zero videos in the composition, unlike the live steering and manifold fragments.
- Layout coordinates accumulated independently. Title bars, individual figures,
  and the large Nature words lack a common hierarchy; the manifold title extends
  below the nominal 700px composition and crowds the footer at some sizes.
- The particle backbone is a static canvas sketch. Hover mostly nudges unrelated
  pieces rather than revealing a coherent arrangement around a selected project.
- Project discovery depends on scattered hit targets; there is no persistent
  project navigation or URL state for returning to a selected work.
- The desktop view has no H1. Dense figure text is shrunk inside fixed-size tilted
  previews, so reading and overview states need clearer separation.
- White figure rectangles, red title bars and a dark ground compete. Different
  surfaces were added for individual assets rather than composed as a whole.

No copy corrections are included. The email currently in shared content remains
unchanged; verifying its domain is a separate content decision.

## B Decisions

- The splash is black with white display lettering and white-backed scientific
  previews. Reading pages remain white. Display lettering uses the existing
  Impact stack for the large name; reading text and navigation use Arial, as
  confirmed by the user. Existing reading text sizes remain. Dark red stays the accent.
- Project figures and actual Nature footage form the composition. No new particle
  decoration, fabricated diagrams, stock imagery or rewritten project content.
- Hover/focus temporarily holds related orbital movement. Clicking opens an
  unframed reading area and folds the pieces into the lower-left profile group.
- The only large splash lettering is Rudramani Singha, a static heading. About
  opens only through its navigation link. No profile
  card, project headings or venue stamp appears on the splash. The nine fragments
  are interleaved, not grouped by project. Research, Tool Packages and About form
  the navigation; research expands to full titles and publication status. No
  packages have been supplied, so that dropdown has an honest empty state.
- On selection, the profile and navigation occupy the left rail. Artwork folds
  into a single bottom-left return stack, not separate project-switching cards.
  Canvas coordinates stay stable during folding. The 550ms reading transition
  enters from the right, not below.
- Project pages order title, authors, venue/status and code links, full animation,
  then the unchanged abstract. Abstracts use the full article width and justified
  text. NeuralFieldManifold is accepted at NeurIPS 2026; robust steering is under
  review, with no submitted-to-ICLR claim displayed.
- Category navigation stays available; URL hashes support deep links and browser
  Back/Forward. Escape and the return control restore the composition and focus.
- Mobile retains the agreed simple profile/contact view without mounting the
  desktop animations. Reduced motion suppresses scene movement; controls remain usable.
- A scene-level pause/replay pair controls the collage. Reading views keep each
  finalized animation's own controls. Hidden/background animations are suspended.
- All layouts and interactions must be finished. No placeholder-work exception.

## Orbital And Profile Iteration

- Keep the large name static and unadorned on hover. Artwork completes full
  elliptical loops in 48-66 seconds around the name, with subtle depth scaling.
  Hover pauses all orbital motion, straightens the related project's pieces,
  and reduces other pieces to 25% opacity while preserving the name's opacity.
  Leaving resumes the same paths. There is no latched hover selection.
- Equation and manifold previews sit above the name, with one dominant footage
  preview also in front. The conversation and three smaller footage previews
  sit behind. Overlap is intentional; the nav remains an unobstructed way to
  reach every project. Tilt has its own wrapper so it cannot displace the orbit
  or create a moving hover target. Fold/unfold transitions ignore pointer input.
- Preview widths: footage 270/125/95/65px, manifold 380px, waveform 150px,
  equation 310px, landscape 270px, conversation 350px. The user confirmed that
  the smaller "ARR" referred to the waveform, not the equation. These are current
  art-direction choices, not a freeze on future placement feedback.
- The selected state's categories remain at the top left. Open dropdowns paint
  above adjacent categories, so moving downward does not activate Tool Packages.
- Folded artwork sits above the lower-left identity/back row. A single persistent
  portrait lands on top of the stack, expands into the About photo slot, and
  folds back. About omits the small identity/contact row and repeated footer name.
  Entry from the splash About link starts the portrait transition from that click;
  entry from a project uses the already-visible portrait on the stack.
- About hides the complete artwork stack. Its return arrow stays available.
- Clicking the stacked portrait returns to the landing composition, never About.
  All return paths hide the portrait, including history, Escape and hash changes.
- B overrides the inherited body-font variable locally so paragraphs, author
  lists and links cannot pick up the production site's Garamond. Display lettering
  and finalized animation typography remain unchanged.
- Preview sheets carry content-hash revisions and checked tile dimensions. Each
  tile is one complete resized source frame, not a crop of the four-video strip.
  Mismatched sheets fall back to the poster instead of displaying adjacent frames.
- B's landscape preview adds 12px of framing padding to retain both axis labels.
  Its manifold preview uses a real 2px Three.js wide line, not the unsupported
  WebGL native-line width setting. Neural-field previews redraw at most 30 FPS
  without slowing recorded-data playback. These are opt-in preview settings;
  full diagrams and prototype A retain their finalized output.
- Frozen previews are inert DOM/bitmap snapshots, not hidden live components.
  Full project content and animation output remain unchanged. The native assets
  only gain opt-in preview sampling/capture support and unmount cleanup guards.
- Per-asset performance measurements and reproduction instructions live in
  `reports/`. The sizing-aware frame sheets supersede the uniform 256px previews
  described in the earlier experiment below.

## Splash Performance, October 14, 2026

Measured with Chromium headless, SwiftShader, 1440x900, four-second RAF samples
after warm-up (`tests/prototype-b-performance.mjs`). These are local comparison
results, not a guarantee for a visitor's hardware.

| Experiment | RAF FPS | 95th percentile frame interval |
| --- | ---: | ---: |
| Four original 1080px video previews | 30.0 | 33.4ms |
| Four 360px video previews | 30.0 | 33.4ms |
| One 360px video playing | 57.5 | 16.8ms |
| Cached frame-sheet previews | 60.0 | 16.8ms |
| Everything paused | 60.0 | 16.8ms |

There are no gradients in B. Lowering video resolution alone did not fix the
measured frame cadence. The final splash instead uses four cached WebP frame
sheets, approximately 276KB combined versus 1.25MB of original MP4s. All 26, 32,
39 and 28 frames remain, at their original 15 FPS and independent loop lengths.
Only preview resolution changes to 256px. One clock draws a new frame only when
needed, without React renders per frame or four continuous video decoders.
Decoded pixel data occupies roughly 35MiB before browser overhead; this is a bounded cache for these four short
clips, not a recommendation to cache arbitrary long videos as frame sheets.

Generate from the unchanged source MP4s with
`node scripts/build-splash-previews.mjs` (requires ffmpeg and ffprobe).
`NaturePreviews.jsx` is a B-only presentation adapter. Full project videos,
finalized steering/manifold animations and prototype A are not modified.
Hidden/offscreen playback stops, collapsed previews pause, and the existing
scene pause/replay controls also govern the cached clips. Reduced motion paints
one still frame. The phone view mounts none of these assets.
