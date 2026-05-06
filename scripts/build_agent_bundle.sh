#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

APP_NAME="${APP_NAME:-synth-pipeline}"
BUILD_DIR="${BUILD_DIR:-.bundle-build}"
DIST_DIR="${DIST_DIR:-dist}"
BUNDLE_DIR="${BUNDLE_DIR:-$DIST_DIR/${APP_NAME}-bundle}"
PYINSTALLER_MODE="${PYINSTALLER_MODE:-onedir}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "Could not find python3.10+ on PATH. Set PYTHON_BIN=/path/to/python." >&2
  exit 2
fi

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Bundling requires Python 3.10+ because requirements.txt pins packages that require it." >&2
  echo "Selected interpreter: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))" >&2
  echo "Set PYTHON_BIN=/path/to/python3.10 or newer and rerun." >&2
  exit 2
fi

if [[ "$PYINSTALLER_MODE" != "onedir" && "$PYINSTALLER_MODE" != "onefile" ]]; then
  echo "PYINSTALLER_MODE must be 'onedir' or 'onefile'" >&2
  exit 2
fi

rm -rf "$BUILD_DIR" "$BUNDLE_DIR"
mkdir -p "$BUILD_DIR" "$DIST_DIR"

echo "Using Python: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"
"$PYTHON_BIN" -m venv "$BUILD_DIR/venv"
"$BUILD_DIR/venv/bin/python" -m pip install --upgrade pip wheel
"$BUILD_DIR/venv/bin/python" -m pip install -r requirements.txt pyinstaller

pyinstaller_args=(
  --clean
  --noconfirm
  --name "$APP_NAME"
  --distpath "$BUNDLE_DIR/bin"
  --workpath "$BUILD_DIR/pyinstaller"
  --specpath "$BUILD_DIR/spec"
  --collect-submodules langgraph
  main.py
)

if [[ "$PYINSTALLER_MODE" == "onefile" ]]; then
  pyinstaller_args=(--onefile "${pyinstaller_args[@]}")
  executable_path="$BUNDLE_DIR/bin/$APP_NAME"
else
  pyinstaller_args=(--onedir "${pyinstaller_args[@]}")
  executable_path="$BUNDLE_DIR/bin/$APP_NAME/$APP_NAME"
fi

"$BUILD_DIR/venv/bin/python" -m PyInstaller "${pyinstaller_args[@]}"

mkdir -p "$BUNDLE_DIR/share"
cp -R domains "$BUNDLE_DIR/share/domains"

cat > "$BUNDLE_DIR/README.txt" <<EOF
$APP_NAME bundle

Run locally:
  $executable_path --domain $BUNDLE_DIR/share/domains/benchmark_haiku.yaml --target-n 1 --run-id smoke

Mount into a task image:
  docker run --rm \\
    -v "\$(pwd)/$BUNDLE_DIR:/opt/$APP_NAME:ro" \\
    -v "\$(pwd)/runs/run-001:/work:rw" \\
    -w /work \\
    TASK_IMAGE \\
    /opt/$APP_NAME/${executable_path#"$BUNDLE_DIR"/} --domain /opt/$APP_NAME/share/domains/benchmark_haiku.yaml --run-id run-001

Notes:
  - The binary includes a Python runtime and pinned Python dependencies.
  - The task image still must be ABI-compatible with the build host target.
  - Provider credentials can be passed through the environment at runtime.
EOF

echo "Built bundle: $BUNDLE_DIR"
echo "Executable: $executable_path"
