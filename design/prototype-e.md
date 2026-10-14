# Prototype E

Route: `/prototype/e`. Starts as an exact copy of D at commit `ffe4525`.
D remains frozen at `/prototype/d`; no changes are promoted to the homepage.

E has independent placement and scoped styles in `src/prototypes/e/`.
Shared content, animation renderers, media, motion and scene interactions are
reused, not copied. Preserve D's output when changing shared modules; future
E-specific behavior must be isolated through E's files or explicit options.

The starting layout preserves all 15 previews, their positions, sizes, tilts,
stacking, opacity, hover behavior, navigation, detail views and controls.
No design or content changes were part of creating E.

## E Changes

- Only the name's highlighted lines capture pointer input; empty space passes
  through to the assets behind it. The photo remains clickable.
- Tools navigation shows plain titles and years, without logos or emoji.
  PyLBA is labeled "Linear Ballistic Accumulators". Detail titles keep their icons.
- Opening a project sends its matching previews to the main detail figure's
  center over 700ms, with a delayed fade near arrival. All other previews retain
  the existing left-stack transition. Back restores the splash placements.
- The opt-in `projectArrival` behavior measures the figure in canvas coordinates;
  it uses the existing frozen previews, does not duplicate running animations,
  ignores pointer input on arriving previews, and respects reduced motion.
  D does not enable this behavior and remains unchanged.

Verification: `tests/prototype-e-arrival-browser.mjs` checks navigation labels,
title icons, every project's destinations/fade, back navigation, reduced motion
and D preservation without screenshots.
