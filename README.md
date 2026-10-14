# Rudramani Singha Website

The accepted interactive website, built with React and Vite and served by nginx
in a single Docker container. The final F design is now the homepage at `/`.

## Local Development

```sh
npm install
npm run dev
```

Open the URL printed by Vite, normally `http://localhost:5173/`. Research,
packages and About use hash links, such as `/#neuraljaxlib`, within the scene.
There are no separate prototype routes.

```sh
npm run build
npm run preview
```

The preview command serves the actual production build. Browser and model checks
are documented in [tests/README.md](tests/README.md); no screenshots or performance
benchmarks are required for ordinary feedback.

## Repository Layout

```text
src/
  main.jsx                 React entry point
  site/                    Main scene, placements, portrait, motion and CSS
  content/                 Shared copy, authors, project metadata and links
  assets/                  Project fragment/composite entry points and video player
  shared/                  High-quality animation implementations and detail views
  prototypes/README.md     Completed iteration history only
public/
  assets/projects/         Project-grouped preview videos, posters and manifests
  assets/videos/           Original and composite paper videos
  assets/images/previews/  Prerecorded Nature preview frames
  assets/data/             Cached recording used by the neural-field animation
  images/                  Profile and original website images
design/
  decisions.md             Accepted design and animation rules
  references/              Scientific references, source fixtures and provenance
reports/                   Historical performance measurements
tests/                     Current model, browser and source-parity checks
scripts/                   Explicit asset import, cache and export commands
```

The old page iterations are recoverable from Git; see
[the prototype log](src/prototypes/README.md). Their code and routes are no longer
part of the app. Existing `pb-`/`pc-` CSS class names were retained to minimize
migration risk; they belong to the single production scene, not separate pages.

## Animation Assets

Content pages use the approved high-quality renderers and composites. The splash
uses transparent prerecorded videos, plus the existing Nature frame-sheet player.
See [the asset guide](src/assets/README.md) for grouped folders and export commands.
Do not regenerate approved media or edit sibling repositories without approval.

Original Nature supplementary videos are in `public/assets/videos/`.
`bash scripts/build-paper-videos.sh` regenerates their composite strips and posters;
`node scripts/build-splash-previews.mjs` regenerates the smaller Nature previews.
The Nature mark comes from the journal; the GitHub mark is from Primer Octicons.
Source licenses and attribution remain with the content/assets.

## Docker And Hosting

```sh
docker compose up --build -d
docker compose logs -f
docker compose down
```

The container serves the production site at `http://localhost:8080`.
The Docker build includes the two cached steering-data JSON files under
`design/references/robust-steering-animation/`, in addition to `src/` and `public/`.

Every push to `expansion` publishes these images through GitHub Actions:

```text
ghcr.io/rgs2151/landing:expansion
ghcr.io/rgs2151/landing:latest
ghcr.io/rgs2151/landing:sha-<commit>
```

The server repository should pull the image and route `singha.io` to its port 80:

```yaml
services:
  landing:
    image: ghcr.io/rgs2151/landing:latest
    restart: unless-stopped
    expose:
      - "80"
```

Publishing an image does not itself restart the server container. Its deployment
must pull and recreate the service. Domain metadata remains `singha.io`.
