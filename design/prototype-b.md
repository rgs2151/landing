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

- A white continuous stage integrates the accepted transparent/white assets
  without filtering or recoloring them. Black display lettering uses the existing
  Impact stack; body text retains the site's Garamond stack. Dark red stays the accent.
- Project figures and actual Nature footage form the composition. No new particle
  decoration, fabricated diagrams, stock imagery or rewritten project content.
- Hover/focus fans the relevant pieces outward. Clicking holds the selection:
  the pieces gather along the edge and the page opens into an unframed reading area.
- Project navigation stays available; URL hashes support deep links and browser
  Back/Forward. Escape and the return control restore the composition and focus.
- Mobile retains the agreed simple profile/contact view without mounting the
  desktop animations. Reduced motion suppresses scene movement; controls remain usable.
- A scene-level pause/replay pair controls the collage. Reading views keep each
  finalized animation's own controls. Hidden/background animations are suspended.
- All layouts and interactions must be finished. No placeholder-work exception.
