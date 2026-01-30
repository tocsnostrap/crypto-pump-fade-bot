#!/usr/bin/env bash
set -e

MARKER_FILE=".python_deps_ready"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="python3"
PIP_BIN="python3 -m pip"

ensure_venv() {
  if [ ! -d "$VENV_DIR" ]; then
    echo "[bootstrap] Creating virtualenv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi

  PYTHON_BIN="$VENV_DIR/bin/python"
  PIP_BIN="$VENV_DIR/bin/pip"

  if [ ! -x "$PYTHON_BIN" ]; then
    echo "[bootstrap] ERROR: Python binary not found in $VENV_DIR"
    exit 1
  fi
}

check_core_deps() {
  "$PYTHON_BIN" - <<'PY'
import sys
try:
    import ccxt
    import pandas
    import numpy
    print("[bootstrap] Core deps OK (ccxt, pandas, numpy)")
    sys.exit(0)
except ImportError as exc:
    print("[bootstrap] Missing core dep:", exc)
    sys.exit(1)
PY
}

check_talib() {
  "$PYTHON_BIN" - <<'PY'
import sys
try:
    import talib
    print("[bootstrap] TA-Lib OK", getattr(talib, "__version__", "unknown"))
    sys.exit(0)
except Exception as exc:
    print("[bootstrap] TA-Lib not available, checking pandas-ta...")
    try:
        import pandas_ta
        print("[bootstrap] pandas-ta OK (fallback)")
        sys.exit(0)
    except ImportError:
        print("[bootstrap] Neither talib nor pandas-ta available")
        sys.exit(1)
PY
}

install_deps() {
  echo "[bootstrap] Installing Python dependencies..."
  "$PIP_BIN" install --upgrade pip --quiet
  "$PIP_BIN" install "numpy<2.3" "pandas>=2.0" ccxt --quiet || true

  # Try TA-Lib first, fall back to pandas-ta
  "$PIP_BIN" install ta-lib --quiet 2>/dev/null || {
    echo "[bootstrap] TA-Lib unavailable, installing pandas-ta fallback..."
    "$PIP_BIN" install pandas-ta --quiet
  }
}

# Ensure we have a writable virtualenv before any checks/install
ensure_venv

# Always verify core deps, reinstall if missing
if ! check_core_deps; then
  echo "[bootstrap] Core dependencies missing, installing..."
  install_deps
  touch "$MARKER_FILE"
fi

# Verify technical analysis library
if ! check_talib; then
  echo "[bootstrap] Installing pandas-ta..."
  "$PIP_BIN" install pandas-ta --quiet
fi

# Final verification
if check_core_deps && check_talib; then
  echo "[bootstrap] All dependencies ready!"
else
  echo "[bootstrap] ERROR: Failed to install required dependencies"
  exit 1
fi
