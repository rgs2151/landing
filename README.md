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

Build locally:

```bash
npm run build
```

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
