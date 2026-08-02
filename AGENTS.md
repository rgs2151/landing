# AGENTS.md

## Project

This repository is Rudramani Singha's personal web platform for `singha.io` and related subdomains.

The repo is a monorepo:

- `apps/landing/` contains the owned React/Vite source for `singha.io`.
- `services/*/` contains deployment wrappers for pre-made self-hosted apps.
- `infra/` contains the shared Docker Compose, Traefik, and backup configuration.
- `ops/` contains operational scripts for deploy, backup, and restore.
- `docs/` contains architecture and service maps.

## Communication

- Be professional, concise, and direct.
- Prefer exact commands and file references.
- Do not over-explain simple changes.
- Correct mistakes plainly and move on.
- Avoid speculative recommendations unless the user asks for options.

## Content Rules

- Treat `apps/landing/` homepage content as locked.
- Do not alter biography, project descriptions, author lists, titles, links, metadata, images, or ordering unless the user explicitly asks for that exact content change.
- Keep the custom domain as `singha.io`.
- Do not add placeholder links.
- Preserve SEO/social metadata unless the requested change specifically concerns it.

## Architecture Rules

- The landing page is owned source code and lives in `apps/landing/`.
- Third-party self-hosted apps are not vendored by default. Put their Compose wrapper, env example, state paths, and backup notes under `services/<name>/`.
- Only vendor or fork a third-party app when the user explicitly decides to modify its source.
- Use pinned image tags for third-party services.
- Each public app must declare its route through Traefik labels.
- Runtime state belongs under `/srv/singha/state` on the server and must never be committed.
- Secrets belong in ignored env files on the server and must never be committed.

## Commands

From the repo root:

```bash
cd apps/landing
npm install
npm run dev
npm run build
```

Server deploy:

```bash
cp infra/.env.example infra/.env
ops/deploy.sh
```

Local preview:

```bash
ops/preview.sh
```

Backup:

```bash
cp infra/backup/restic.env.example infra/backup/restic.env
ops/backup.sh
```

## Workflow

- Use `expansion` for the platform refactor.
- Keep changes scoped and inspect the diff before finishing.
- Run `npm run build` in `apps/landing` when landing code changes.
- If Docker is available, run `docker compose --env-file infra/.env -f infra/compose.yaml config`.
- After a substantial completed step, commit and push the current branch.
