#!/usr/bin/env bash
# set -e removed to prevent exit on pandas-ta failure

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
    # Check for 'ta' library
    try:
        import ta
        print("[bootstrap] 'ta' library OK")
        sys.exit(0)
    except ImportError:
        pass

    print("[bootstrap] TA-Lib not available, checking pandas-ta...")
    try:
        import pandas_ta
        print("[bootstrap] pandas-ta OK (fallback)")
        sys.exit(0)
    except ImportError:
        print("[bootstrap] Neither talib, ta, nor pandas-ta available")
        sys.exit(1)
PY
}

install_deps() {
  echo "[bootstrap] Installing Python dependencies..."
  pip_install --upgrade pip --quiet
  pip_install "numpy<2.3" "pandas>=2.0" ccxt python-socketio websocket-client --quiet || true

  # Try installing 'ta' library (version 0.10.2 for Python 3.11 compatibility)
  echo "[bootstrap] Attempting to install 'ta' library..."
  pip_install "ta==0.10.2" || echo "[bootstrap] Warning: Failed to install 'ta' library. Continuing..."

  # Try TA-Lib first, fall back to pandas-ta
  # We use || true to ensure script doesn't exit even if this fails
  pip_install ta-lib --quiet 2>/dev/null || {
    echo "[bootstrap] TA-Lib unavailable, attempting pandas-ta fallback..."
    pip_install pandas-ta || {
        echo "[bootstrap] Warning: Failed to install pandas-ta. Using talib_compat fallback."
    }
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
  echo "[bootstrap] Technical analysis libraries missing. Attempting to install 'ta'..."
  
  # Try installing 'ta' first (more reliable on newer Python)
  pip_install "ta==0.10.2" || echo "[bootstrap] Warning: Failed to install 'ta'."
  
  # Check if we are good now
  if check_talib; then
      echo "[bootstrap] TA library installed successfully!"
      exit 0
  fi
  
  # If still missing, try pandas-ta one last time but don't fail hard
  echo "[bootstrap] Still missing TA libs. Trying to force install pandas-ta..."
  pip_install pandas-ta || echo "[bootstrap] Failed to install pandas-ta. Proceeding anyway..."
fi

# Final verification - Warning only
if check_core_deps && check_talib; then
  echo "[bootstrap] All dependencies ready!"
else
  echo "[bootstrap] WARNING: Some dependencies might be missing, but proceeding to start bot..."
  echo "[bootstrap] The bot will attempt to use available libraries via talib_compat."
fi
