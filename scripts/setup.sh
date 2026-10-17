#!/usr/bin/env bash
# One-time setup on Linux/macOS: creates .venv, installs the harness and (where available) the
# CARLA client matching the server, then runs a self-check. Without a CARLA client the bicycle
# backend still works; CARLA has no macOS build, so on a Mac that is the expected outcome.
#   CARLA_ROOT=~/CARLA_0.9.16 ./scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CARLA_VERSION="${CARLA_VERSION:-0.9.16}"

PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 &&
     "$c" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 12))' 2>/dev/null; then
    PY="$c"; break
  fi
done
[ -n "$PY" ] || { echo "Python 3.10-3.12 required (e.g. 'brew install python@3.12' or 'sudo apt install python3.12-venv')"; exit 1; }
echo "Using $PY"

[ -d .venv ] || "$PY" -m venv .venv
VPY=.venv/bin/python
"$VPY" -m pip install --upgrade pip -q
"$VPY" -m pip install -e ".[dev]" -q

if ! "$VPY" -m pip install "carla==$CARLA_VERSION" -q 2>/dev/null; then
  TAG="cp$("$VPY" -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')"
  WHL=$(ls "${CARLA_ROOT:-$HOME/CARLA_$CARLA_VERSION}"/PythonAPI/carla/dist/carla-*-"$TAG"-*.whl 2>/dev/null | head -n1 || true)
  if [ -n "$WHL" ]; then
    "$VPY" -m pip install "$WHL" -q
  else
    echo
    echo "NOTE: no carla==$CARLA_VERSION client for $TAG on this system (none exists for macOS)."
    echo "      The harness is installed; use --backend bicycle. To add CARLA later, set CARLA_ROOT and re-run."
  fi
fi

"$VPY" -m autocross check || true
echo
echo "Setup complete. Activate with:  source .venv/bin/activate"
echo "Dev loop without CARLA:  python -m autocross run -c controllers/baseline_pure_pursuit.py -t tracks/hackathon_test.yaml --backend bicycle --view"
