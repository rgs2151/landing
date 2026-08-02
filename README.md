# Rudramani Singha Website

This repository contains the React landing page for `singha.io` plus a single Docker image that serves the built Vite output with nginx.

## Files

- `index.html`: Vite HTML entrypoint and metadata
- `src/main.jsx`: React homepage
- `public/stylesheet.css`: Main stylesheet
- `public/images/`: Hero and work images
- `public/scripts/`: Notebook artifacts referenced by the site
- `Dockerfile`: nginx container for the landing page
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

## Docker

Build the site, then build the container:

```bash
npm run build
docker build -t singha-landing .
```

Run locally on a machine where Docker can run containers:

```bash
docker run --rm -p 8080:80 singha-landing
```

Open `http://localhost:8080`.

For a server, the infrastructure repo can build this image or pull a published image and route `singha.io` to container port `80`.

## Domain

Custom domain is configured via `CNAME`:

```text
singha.io
```
