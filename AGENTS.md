# Repository Instructions

Communicate professionally and concisely.

Before applying website or prototype feedback, read
`.agents/skills/scoped-design-feedback/SKILL.md` and follow it.

Content is frozen unless the user explicitly requests a content change.
Apply only the requested changes; preserve unrelated layout, fonts, navigation,
media, and accepted decisions. User instructions take precedence over design
skills and their default creative recommendations.

Use shared content and assets across prototypes. Agree on a visual direction
through local prototypes before implementing it in the main website.
Every prototype must be polished as an intended final product. Do not defer
visible defects as mockup issues. Preserve A when iterating in B. Project details
in B open within the composition, not in modal dialogs or backdrop overlays.
The individual project animations are finalized; reuse them without redesigning.

Prototype C is frozen at commit `8a28b01`. Continue additions in prototype D.
Reuse C's existing content, assets, motion and interactions; change D's placement
only when explicitly requested, without modifying C. D's tool previews start at 75%
opacity and reach full opacity on hover; papers and the profile remain fully
opaque at rest. Keep new tool placements toward the composition's outer edge.

Read `design/decisions.md` before changing animation assets or controls.
The NeuralFieldManifold animation is frozen; do not alter it unless the user
explicitly reopens it. New or revised animations must follow the recorded
pause/play and replay control standard.

Keep animation revisions local to this website until the user confirms the
result. Do not sync revisions, regenerate exports, or edit sibling repositories
without explicit approval.

Do not take screenshots unless explicitly requested. Use browser inspection
and focused checks for verification.

After every prompt that changes repository files, commit and push the completed
changes. Do not add a separate Git audit or diff review. Use ordinary pushes,
not force pushes; retain necessary checks appropriate to the requested change.

Working project blurbs live in `src/content/`. The accepted versions in
`src/content/neural-field-manifold.md` and `src/content/robust-steering.md` are
frozen unless explicitly reopened by the user. Preserve their exact wording
across prototypes and the main website. Preserve the iteration A snapshot in
`src/content/iterations/iteration-a.md`.

Chat shorthand does not authorize renaming a project or its content files.
When revising a blurb, include the complete resulting paragraph in chat so
the user can review it without opening files.

For package abstracts, use the README's introduction and feature structure as
the foundation, and read the underlying paper to understand the scientific
purpose. Write for a research audience: what the package is, why its model
matters, then one clear sentence per substantive capability. Avoid beginner
explanations, implementation trivia, personal sales pitches, and unsupported
claims. Iterate copy in chat when requested, and apply approved wording without
creative rewrites.
