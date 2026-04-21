# Rudramani Singha Website

[![Deploy Static Site](https://github.com/rgs2151/landing/actions/workflows/pages.yml/badge.svg)](https://github.com/rgs2151/landing/actions/workflows/pages.yml)

This repository is now a plain static website with no Jekyll dependency.

## Files

- `index.html`: Main homepage
- `stylesheet.css`: Main stylesheet
- `images/`: Hero and work images
- `favicon.ico`: Site favicon at the repository root
- `CNAME`: Custom domain mapping (`singha.io`)

## Local Preview

Run from repository root:

```bash
python3 -m http.server 8000
```

Then open http://localhost:8000.

## Deployment

GitHub Pages deploys automatically from GitHub Actions using `.github/workflows/pages.yml`.
The workflow publishes a static artifact and includes `.nojekyll`.

## Domain

Custom domain is configured via `CNAME`:

```text
singha.io
```


