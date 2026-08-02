#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/infra/.env}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE. Copy infra/.env.example to infra/.env and fill secrets."
  exit 1
fi

cd "$ROOT"

git pull --ff-only

docker compose --env-file "$ENV_FILE" -f infra/compose.yaml pull --ignore-buildable
docker compose --env-file "$ENV_FILE" -f infra/compose.yaml build landing
docker compose --env-file "$ENV_FILE" -f infra/compose.yaml up -d --remove-orphans
