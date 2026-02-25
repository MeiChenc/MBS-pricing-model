#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY_BIN=${PY_BIN:-/usr/bin/python3}
TARGET_ARCH=${TARGET_ARCH:-}

# Force consistent architecture on macOS to avoid numpy wheel mismatch.
if [[ -z "$TARGET_ARCH" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]] && [[ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" == "1" ]]; then
    # On Apple Silicon, default to native arm64 even if terminal runs under Rosetta.
    TARGET_ARCH="arm64"
  else
    TARGET_ARCH="$(uname -m)"
  fi
fi

if [[ "$TARGET_ARCH" != "arm64" && "$TARGET_ARCH" != "x86_64" ]]; then
  echo "Unsupported TARGET_ARCH=$TARGET_ARCH (expected arm64 or x86_64)"
  exit 1
fi

ARCH_PREFIX=(arch "-$TARGET_ARCH")

echo "Using architecture: ${ARCH_PREFIX[*]}"
echo "Using python: $PY_BIN"

if [[ -d .venv ]]; then
  rm -rf .venv
fi

"${ARCH_PREFIX[@]}" "$PY_BIN" -m venv .venv
source .venv/bin/activate

"${ARCH_PREFIX[@]}" python -m pip install --upgrade pip setuptools wheel
"${ARCH_PREFIX[@]}" python -m pip install --no-cache-dir -r requirements.txt

"${ARCH_PREFIX[@]}" python -c "import platform, numpy, pandas, lifelines, matplotlib; print('machine=', platform.machine()); print('numpy=', numpy.__version__)"

echo
echo "Done. Activate with: source .venv/bin/activate"
echo "Run model with: python mbs.py"
