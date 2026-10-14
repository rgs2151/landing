# PyLBA Animation

Source: `/home/dev/pyLBA/logo/pylba.svg`, from the user's local pyLBA checkout.
Repository: https://github.com/rgs2151/pyLBA
Checkout HEAD at import: `eba0bb377f9c083cf18bbe02531e9c4ac0debfb9`.
The copied SVG is preserved byte-for-byte; the upstream MIT license is included.

`PyLBAAnimation.jsx` parses the local SVG once with DOMParser and creates React
SVG elements. Geometry, text, sample paths and one-second discrete SMIL timelines
are unchanged. Instance IDs are namespaced; the source's stylesheet is scoped in
`styles.css`; the white background rectangle is omitted for composition.

The wrapper adds the site's controls, visibility/reduced-motion handling and an
inert snapshot hook for folded previews. There is no simulation or frame-by-frame
React state update. Use `compact` for the original square preview, and `active`
for externally controlled playback. Full views add 38 SVG units at the bottom
for controls without covering the original diagram.

The blurb is based on the local README and model/fitting guides. In particular,
prediction uses posterior-mean parameters, not posterior predictive integration;
EM is not implemented and is not claimed in the website copy. The local README
is newer than the GitHub page inspected during integration.
