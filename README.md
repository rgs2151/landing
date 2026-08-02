# Singha Web Platform

This repository is the monorepo for `singha.io` and self-hosted subdomains.

## Layout

- `apps/landing/`: owned React/Vite app for `singha.io`
- `services/draw/`: ExcaliDash deployment wrapper for `draw.singha.io`
- `infra/`: Docker Compose, Traefik, and backup configuration
- `ops/`: deploy and backup scripts
- `docs/`: architecture and service map

## Landing Development

```bash
cd apps/landing
npm install
npm run dev
npm run build
```

## Server Deployment

```bash
cp infra/.env.example infra/.env
ops/deploy.sh
```

The server stack is Docker Compose based. Traefik receives ports `80` and `443`, then routes subdomains to individual containers by Docker labels.

## Current Services

See `docs/SERVICES.md`.

