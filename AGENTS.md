# AGENTS.md

## Project

This repository contains Rudramani Singha's personal academic landing page for `singha.io`.

The site is a Vite-powered static website. GitHub Pages deployment is Actions-only. Do not preserve or reintroduce GitHub Pages "Deploy from a branch" support.

## Source of Truth

- `index.html` is the main page and contains the current homepage content.
- `stylesheet.css` is the current styling source.
- `images/` contains the profile and project imagery.
- `scripts/` contains research notebooks and artifacts linked from the site history.
- `tools/` contains small build helpers.
- `dist/` is generated output and should not be edited by hand or committed.

## Content Rules

- Do not rewrite biography, project descriptions, author lists, titles, metadata, or external links unless the user explicitly asks.
- Keep the Art of Neuron project link pointed at `https://art-of-neuron.github.io/`.
- Keep the custom domain as `singha.io`.
- Do not add placeholder links such as `example.com`.
- Preserve SEO/social metadata unless the requested change specifically concerns it.

## Frontend Rules

- Use Vite with npm and Node 24, as pinned by `.node-version`.
- Keep dependencies minimal. The GitHub Actions path should stay well under 30 seconds whenever possible.
- Do not add React, Svelte, Astro, Tailwind, or another framework without an explicit user decision.
- Styling work should preserve the current content and improve presentation only.
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
- If local Node is missing, use a temporary Node 24 runtime or state clearly that build verification could not be run.
