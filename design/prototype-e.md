# Prototype E

Route: `/prototype/e`. Starts as an exact copy of D at commit `ffe4525`.
D remains frozen at `/prototype/d`; no changes are promoted to the homepage.

E has independent placement and scoped styles in `src/prototypes/e/`.
Shared content, animation renderers, media, motion and scene interactions are
reused, not copied. Preserve D's output when changing shared modules; future
E-specific behavior must be isolated through E's files or explicit options.

The starting layout preserves all 15 previews, their positions, sizes, tilts,
stacking, opacity, hover behavior, navigation, detail views and controls.
No design or content changes are part of creating E.
