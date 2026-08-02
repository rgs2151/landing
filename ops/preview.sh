#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LANDING_PORT="${LANDING_PORT:-5173}"
DRAW_PORT="${DRAW_PORT:-6767}"
NODE_DIR="${NODE_DIR:-/tmp/node-v24.18.1-linux-x64}"
LANDING_PID=""
DRAW_STARTED=0
DOCKER_CMD=()

cleanup() {
  if [ -n "$LANDING_PID" ] && kill -0 "$LANDING_PID" 2>/dev/null; then
    kill "$LANDING_PID" 2>/dev/null || true
  fi

  if [ "$DRAW_STARTED" -eq 1 ]; then
    "${DOCKER_CMD[@]}" compose -f "$ROOT/services/draw/compose.local.yaml" down
  fi
}

trap cleanup EXIT INT TERM

if ! command -v node >/dev/null 2>&1; then
  if [ -x "$NODE_DIR/bin/node" ]; then
    export PATH="$NODE_DIR/bin:$PATH"
  else
    echo "Node 24 was not found. Install Node 24 or set NODE_DIR to a Node 24 runtime."
    exit 1
  fi
fi

if [ ! -d "$ROOT/apps/landing/node_modules" ]; then
  (cd "$ROOT/apps/landing" && npm ci)
fi

if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    DOCKER_CMD=(docker)
  elif command -v sudo >/dev/null 2>&1 && sudo -n docker info >/dev/null 2>&1; then
    DOCKER_CMD=(sudo docker)
  fi
fi

echo "Starting landing: http://127.0.0.1:${LANDING_PORT}"
(cd "$ROOT/apps/landing" && npm run dev -- --host 127.0.0.1 --port "$LANDING_PORT" --strictPort) &
LANDING_PID="$!"

if [ "${#DOCKER_CMD[@]}" -gt 0 ]; then
  echo "Starting draw: http://127.0.0.1:${DRAW_PORT}"
  if DRAW_PORT="$DRAW_PORT" "${DOCKER_CMD[@]}" compose -f "$ROOT/services/draw/compose.local.yaml" up -d; then
    DRAW_STARTED=1
  else
    echo "Draw preview failed to start. Landing will keep running."
    "${DOCKER_CMD[@]}" compose -f "$ROOT/services/draw/compose.local.yaml" down --remove-orphans || true
  fi
else
  echo "Docker is not available; skipping draw preview."
fi

echo
echo "Preview URLs:"
echo "  landing  http://127.0.0.1:${LANDING_PORT}"
if [ "$DRAW_STARTED" -eq 1 ]; then
  echo "  draw     http://127.0.0.1:${DRAW_PORT}"
else
  echo "  draw     unavailable until Docker can run containers"
fi
echo
echo "Press Ctrl-C to stop preview."

wait "$LANDING_PID"
