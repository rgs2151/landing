# Architecture

This repository is the control plane for a personal web platform.

## Principles

- One repository holds the full platform context.
- Owned code lives under `apps/`.
- Pre-made self-hosted services live under `services/` as deployment wrappers.
- Traefik is the only public entrypoint.
- Docker Compose starts the complete stack.
- Runtime state lives on the server under `/srv/singha/state`.
- Backups target runtime state, not the git repository.

## Request Flow

```text
Internet
  -> server ports 80/443
  -> Traefik
  -> container selected by Host(...) label
```

## Service Types

Owned apps:

```text
apps/<name>/src
apps/<name>/Dockerfile
apps/<name>/compose.yaml
```

Third-party services:

```text
services/<name>/compose.yaml
services/<name>/.env.example
services/<name>/backup.md
services/<name>/README.md
```

Only vendor third-party source when we explicitly decide to fork or modify it.
