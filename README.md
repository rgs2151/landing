# Rudramani Singha Website

This repository contains the static landing page for `singha.io` plus a single Docker image that serves it with nginx.

## Files

- `index.html`: Main homepage
- `stylesheet.css`: Main stylesheet
- `images/`: Hero and work images
- `scripts/`: Notebook artifacts referenced by the site
- `Dockerfile`: nginx container for the landing page
- `nginx.conf`: static nginx config with custom 404 support
- `favicon.ico`: Site favicon at the repository root
- `CNAME`: Custom domain mapping (`singha.io`)

## Local Preview

Fast local preview does not need Docker:

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000`.

## Docker

Build:

```bash
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

