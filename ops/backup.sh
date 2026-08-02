#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/infra/backup/restic.env}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE. Copy infra/backup/restic.env.example to infra/backup/restic.env and fill it."
  exit 1
fi

docker compose --env-file "$ENV_FILE" -f "$ROOT/infra/backup/compose.yaml" run --rm restic backup /data --tag singha-platform
