# AGENTS.md

## Project

This repository contains Rudramani Singha's personal academic landing page for `singha.io`.

The site is a Vite + React website with a monochrome ASCII terminal interface. GitHub Pages deployment is Actions-only. Do not preserve or reintroduce GitHub Pages "Deploy from a branch" support.

## Communication

- Be professional, concise, and direct.
- Prefer concrete status, exact commands, and file references over broad explanation.
- Do not over-explain simple changes.
- When correcting a mistake, state the correction plainly and move on.
- Avoid speculative recommendations unless the user asks for options.

## Source of Truth

- `src/main.jsx` is the React app, contains the locked homepage content, and renders ASCII media effects.
- `index.html` is the Vite HTML entrypoint and contains metadata.
- `stylesheet.css` is the current styling source imported by React.
- `images/` contains the profile and project imagery.
- `scripts/` contains research notebooks and artifacts linked from the site history.
- `tools/` contains small build helpers.
- `dist/` is generated output and should not be edited by hand or committed.

## Content Rules

- Treat the homepage content as locked. Do not alter wording, project descriptions, author lists, titles, links, metadata, images, or ordering unless the user explicitly asks for that exact content change.
- Do not rewrite biography, project descriptions, author lists, titles, metadata, or external links unless the user explicitly asks.
- Keep the Art of Neuron project link pointed at `https://art-of-neuron.github.io/`.
- Keep the custom domain as `singha.io`.
- Do not add placeholder links such as `example.com`.
- Preserve SEO/social metadata unless the requested change specifically concerns it.

## Frontend Rules

- Use Vite, React, npm, and Node 24, as pinned by `.node-version`.
- Keep dependencies minimal. The GitHub Actions path should stay well under 30 seconds whenever possible.
- Do not add Svelte, Astro, Tailwind, or another framework without an explicit user decision.
- Styling work should preserve the current content and improve presentation only.
- Keep the current visual direction ASCII-first unless the user asks for a different style.
- Prefer small, plain static-site changes over architectural churn.

## Commands

Use these commands from the repository root:

```bash
npm install
npm run dev
npm run build
npm run preview
```

In CI, use:

```bash
npm ci
npm run build
```

## Deployment

- GitHub Pages source must be set to GitHub Actions.
- `.github/workflows/pages.yml` builds with Vite.
- Pushes to `style` deploy the built `dist/` artifact.
- Pull requests into `style` should build for validation only.
- Ensure `CNAME`, `.nojekyll`, `404.html`, `favicon.ico`, and `stylesheet.css` are present in `dist/` after `npm run build`.

## Development Workflow

- Use the `style` branch as the main development and deployment branch.
- Keep changes scoped and review the diff before finishing.
- Run `npm run build` before reporting success.
- After any substantial completed step, commit the scoped changes and push `style`.
- If local Node is missing, use a temporary Node 24 runtime or state clearly that build verification could not be run.
