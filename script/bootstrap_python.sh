#!/usr/bin/env bash
set -e

MARKER_FILE=".python_deps_ready"

check_talib() {
  python - <<'PY'
import sys
try:
    import talib
    print("[bootstrap] TA-Lib OK", getattr(talib, "__version__", "unknown"))
    sys.exit(0)
except Exception as exc:
    print("[bootstrap] TA-Lib failed:", exc)
    sys.exit(1)
PY
}

if [ ! -f "$MARKER_FILE" ]; then
  echo "[bootstrap] Installing Python deps..."
  python -m pip install --upgrade pip
  python -m pip install ccxt numpy pandas ta-lib || true

  if ! check_talib; then
    echo "[bootstrap] Installing pandas-ta fallback..."
    python -m pip install pandas-ta
  fi

  touch "$MARKER_FILE"
fi

check_talib || true
