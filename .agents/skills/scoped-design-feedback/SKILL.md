---
name: scoped-design-feedback
description: Apply website and prototype feedback in this repository while preserving content and all design decisions outside the user's requested change.
---

# Scoped Design Feedback

- Treat direct feedback as a bounded edit. Change only the named property, element, or behavior. A blue-to-red request changes that color; it does not authorize copy edits, navigation changes, or layout revisions.
- Preserve all text, names, author order, links, and media unless the user explicitly requests changing them. Design freedom does not authorize rewriting content.
- For broad feedback, work within the stated dimension. "Shorten the content" authorizes shortening content; it does not authorize changing fonts or navigation. An overall redesign authorizes visual redesign while preserving copy unless copy changes are also requested.
- Implement necessary dependent changes only when required for the requested result. Keep them minimal and explain any material dependency. Put unrelated improvement ideas in discussion instead of implementing them.
- Read the current state before editing. Preserve earlier accepted decisions and user edits. If feedback is ambiguous, clarify the uncertain part while continuing work that is clear.
- Keep prototype content and media in shared sources. Variants may change presentation, but must not carry independently rewritten copies of the same content. Scope variant-specific feedback to that variant; do not silently apply it to every direction.
- Prototype means a separate design direction, not reduced quality. Present finished layouts, interactions and states; never excuse visible defects as temporary mockup work. The accepted F direction now lives in `src/site/`; project expansion happens within the scene, without modals.
- For a new design exploration, use local prototypes first. Ordinary feedback now applies to the accepted main website; do not resurrect the removed A-F routes.
- Verify the diff matches the requested scope and that copy has not changed accidentally. Respect the user's instruction to avoid screenshots unless they explicitly request them.
- Report the concrete change concisely. Do not bundle unsolicited redesigns, refactors, or fixes into a feedback iteration.
