#!/usr/bin/env bash
# Starts the CARLA server with laptop-friendly settings.
#   ./scripts/start_carla.sh            windowed 1280x720, Low quality
#   ./scripts/start_carla.sh --headless no window (pair with --no-render for batch scoring)
set -euo pipefail
CARLA_VERSION="${CARLA_VERSION:-0.9.16}"
ROOT="${CARLA_ROOT:-$HOME/CARLA_$CARLA_VERSION}"
[ -x "$ROOT/CarlaUE4.sh" ] || { echo "CarlaUE4.sh not found in $ROOT (set CARLA_ROOT)"; exit 1; }
ARGS=(-quality-level="${QUALITY:-Low}" -carla-rpc-port="${PORT:-2000}" -nosound -prefernvidia)
if [ "${1:-}" = "--headless" ]; then ARGS+=(-RenderOffScreen); else ARGS+=(-windowed -ResX=1280 -ResY=720); fi
"$ROOT/CarlaUE4.sh" "${ARGS[@]}" &
echo "CARLA started (PID $!) on port ${PORT:-2000}; check with: python -m autocross check"
