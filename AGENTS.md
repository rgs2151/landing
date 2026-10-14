# Repository Instructions

Communicate professionally and concisely.

Before applying website or prototype feedback, read
`.agents/skills/scoped-design-feedback/SKILL.md` and follow it.

Content is frozen unless the user explicitly requests a content change.
Apply only the requested changes; preserve unrelated layout, fonts, navigation,
media, and accepted decisions. User instructions take precedence over design
skills and their default creative recommendations.

The accepted F design is now the main website at `/`, implemented in `src/site/`.
There are no active prototype routes. `src/prototypes/README.md` records the
iteration history; old implementations are recoverable from Git.
Preserve the accepted design and reuse shared content and animation assets.
Project details open within the composition, not in modal dialogs or overlays.
The individual project animations are finalized; do not redesign them implicitly.
Tool previews start at 75% opacity and reach full opacity on hover; papers and
the profile remain fully opaque at rest. Keep new tool placements toward the
composition's outer edge. Future design explorations must be explicitly scoped
and polished, not rough mockups, before promotion into the main website.

Read `design/decisions.md` before changing animation assets or controls.
The NeuralFieldManifold animation is frozen; do not alter it unless the user
explicitly reopens it. New or revised animations must follow the recorded
pause/play and replay control standard.

When an animation is approved, retain its shared high-quality fragments and
composite for project content, and export low-resolution transparent videos for
the landing page. Never mount live simulations as new landing previews. Keep
matching fragment/video/poster names grouped by project; see `src/assets/README.md`.
Preserve geometry, timing, transparency and controls when exporting. Do not run
performance benchmarks unless requested; use focused functional verification.

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
across the website. Historical copy iterations remain in Git history.

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
