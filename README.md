# Rudramani Singha Website

This repository contains the React landing page for `singha.io` plus a single Docker image that builds the Vite site and serves it with nginx.

## Files

- `index.html`: Vite HTML entrypoint and metadata
- `src/main.jsx`: React homepage
- `public/stylesheet.css`: Main stylesheet
- `public/images/`: Hero and work images
- `public/scripts/`: Notebook artifacts referenced by the site
- `Dockerfile`: nginx container for the landing page
- `compose.yml`: local/server Compose entrypoint for the landing container
- `nginx.conf`: static nginx config with custom 404 support
- `public/favicon.ico`: Site favicon
- `public/CNAME`: Custom domain mapping (`singha.io`)

## Local Preview

Fast local preview:

```bash
npm install
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173`.

Preserved directions A, B and C are at `/prototype/a`, `/prototype/b` and
`/prototype/c`. C and D are frozen; D remains at `/prototype/d`.
E is frozen at `/prototype/e`. The current iteration F is at `/prototype/f`,
preserving E's layout and content while using prerecorded splash previews.
These local design routes are excluded from the production
build; the homepage remains unchanged. Shared copy and media paths are
in `src/content/site.js`; shared interactive visuals are in `src/shared/`.
The A findings and B decisions are in `design/prototype-b.md`; C's layout and
local-motion decisions are in `design/prototype-c.md`. D's additions are recorded
in `design/prototype-d.md`. E's baseline is recorded in `design/prototype-e.md`;
F's cache and fidelity checks are in `design/prototype-f.md`.

Tool additions use one shared structure:

- `src/content/<tool>.js` and `.md`: metadata and copy.
- `src/shared/<tool>/`: original assets, renderers, license and provenance.
- `src/shared/tools.js`: ordered figures paired with their text.
- `src/shared/ToolDetail.jsx`: shared title, author, links and wrapped figures.
- `src/prototypes/f/PrototypeF.jsx`: current splash placement; D and E remain frozen.

PyLBA and NeuralJAXlib use this structure. F's low-resolution transparent videos
are exported from the same high-quality sources used in project content;
the full gallery is unchanged. Asset organization and regeneration are documented
in [the asset guide](src/assets/README.md). Imported
source and cached data are documented in `src/shared/neuraljaxlib/README.md`.
Its current copy is taken from the source README/gallery, pending abstract review.

Build locally:

```bash
npm run build
```

## Paper Media

Original supplementary videos are in `public/assets/videos/`. Run
`bash scripts/build-paper-videos.sh` with FFmpeg installed to regenerate the
desktop strip, mobile grid, and posters. Shorter clips hold their final frame
until the longest clip finishes. The page loads only the matching layout near
the viewport, pauses offscreen, and respects reduced-motion preferences.

The Nature logo comes from the journal's website header; the GitHub mark comes
from [Primer Octicons](https://github.com/primer/octicons/blob/main/icons/mark-github-16.svg).

## Docker / Compose

Build and run the production container locally:

```bash
docker compose up --build
```

Open `http://localhost:8080`.

Detached mode:

```bash
docker compose up --build -d
docker compose logs -f
docker compose down
```

On every push to `expansion`, GitHub Actions publishes:

```text
ghcr.io/rgs2151/landing:expansion
ghcr.io/rgs2151/landing:latest
ghcr.io/rgs2151/landing:sha-<commit>
```

A server/infrastructure repository should pull the image instead of copying this source code:

```yaml
services:
  landing:
    image: ghcr.io/rgs2151/landing:expansion
    restart: unless-stopped
    expose:
      - "80"
```

Route `singha.io` to container port `80`. Docker Hub is optional; GHCR is the default because the code and Actions pipeline already live on GitHub.

## Domain

Custom domain is configured via `CNAME`:

```text
singha.io
```
