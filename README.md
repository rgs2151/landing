# Rudramani Singha Website

[![Deploy Static Site](https://github.com/rgs2151/landing/actions/workflows/pages.yml/badge.svg)](https://github.com/rgs2151/landing/actions/workflows/pages.yml)

This repository is a Vite-powered static website with no Jekyll dependency.

## Files

- `index.html`: Main homepage
- `stylesheet.css`: Main stylesheet
- `images/`: Hero and work images
- `scripts/`: Research notebooks and supporting artifacts
- `tools/`: Small build helpers
- `favicon.ico`: Site favicon at the repository root
- `CNAME`: Custom domain mapping (`singha.io`)

## Local Preview

Run from repository root:

```bash
npm install
npm run dev
```

For a production preview:

```bash
npm run build
npm run preview
```

## Deployment

GitHub Pages is deployed through GitHub Actions only. Configure Pages with "GitHub Actions" as the build and deployment source.

The workflow at `.github/workflows/pages.yml` builds the Vite site and publishes the generated `dist/` artifact on pushes to `style`. Pull requests into `style` run build validation without deploying.

## Domain

Custom domain is configured via `CNAME`:

```text
singha.io
```
