#!/usr/bin/env bash
set -e

MARKER_FILE=".python_deps_ready"
PYTHON_BIN="python3"

pip_install() {
  "$PYTHON_BIN" -m pip install --user "$@"
}

check_core_deps() {
  "$PYTHON_BIN" - <<'PY'
import os
import sys

def sanitize_sys_path():
    cwd = os.path.abspath(os.getcwd())
    cleaned = []
    for entry in sys.path:
        if not entry:
            continue
        abs_entry = os.path.abspath(entry)
        if abs_entry == cwd:
            continue
        if os.path.basename(abs_entry) == "numpy":
            continue
        cleaned.append(entry)
    sys.path = cleaned

sanitize_sys_path()

try:
    import ccxt
    import pandas
    import numpy
    import socketio
    import websocket
    print("[bootstrap] Core deps OK (ccxt, pandas, numpy, socketio, websocket-client)")
    sys.exit(0)
except Exception as exc:
    print("[bootstrap] Missing core dep or import error:", exc)
    sys.exit(1)
PY
}

check_talib() {
  "$PYTHON_BIN" - <<'PY'
import os
import sys

def sanitize_sys_path():
    cwd = os.path.abspath(os.getcwd())
    cleaned = []
    for entry in sys.path:
        if not entry:
            continue
        abs_entry = os.path.abspath(entry)
        if abs_entry == cwd:
            continue
        if os.path.basename(abs_entry) == "numpy":
            continue
        cleaned.append(entry)
    sys.path = cleaned

sanitize_sys_path()

try:
    import talib
    print("[bootstrap] TA-Lib OK", getattr(talib, "__version__", "unknown"))
    sys.exit(0)
except Exception:
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
  pip_install --upgrade pip --quiet
  pip_install "numpy<2.3" "pandas>=2.0" ccxt python-socketio websocket-client --quiet || true

  # Try installing 'ta' library (version 0.10.2 for Python 3.11 compatibility)
  echo "[bootstrap] Attempting to install 'ta' library..."
  pip_install "ta==0.10.2" --quiet || {
      echo "[bootstrap] Warning: Failed to install 'ta' library. Continuing with fallbacks."
  }

  # Try TA-Lib first, fall back to pandas-ta
  pip_install ta-lib --quiet 2>/dev/null || {
    echo "[bootstrap] TA-Lib unavailable, installing pandas-ta fallback..."
    pip_install pandas-ta --quiet
  }
}

# Always verify core deps, reinstall if missing
if ! check_core_deps; then
  echo "[bootstrap] Core dependencies missing, installing..."
  install_deps
  touch "$MARKER_FILE"
fi

# Verify technical analysis library
if ! check_talib; then
  echo "[bootstrap] Installing pandas-ta..."
  pip_install pandas-ta --quiet
fi

# Final verification
if check_core_deps && check_talib; then
  echo "[bootstrap] All dependencies ready!"
else
  echo "[bootstrap] ERROR: Failed to install required dependencies"
  exit 1
fi
