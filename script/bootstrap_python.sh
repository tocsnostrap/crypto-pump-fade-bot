#!/usr/bin/env bash
set -e

MARKER_FILE=".python_deps_ready"

check_core_deps() {
  python3 - <<'PY'
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
  python3 - <<'PY'
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
        print("[bootstrap] pandas-ta not available, using compat fallback")
        try:
            import pandas
            print("[bootstrap] pandas OK (compat fallback)")
            sys.exit(0)
        except ImportError:
            print("[bootstrap] Neither talib nor pandas available")
            sys.exit(1)
PY
}

install_deps() {
  echo "[bootstrap] Installing Python dependencies..."
  python3 -m pip install --upgrade pip --quiet
  python3 -m pip install "numpy<2.3" "pandas>=2.0" ccxt --quiet || true

  # Try TA-Lib first, fall back to pandas-ta
  python3 -m pip install ta-lib --quiet 2>/dev/null || {
    echo "[bootstrap] TA-Lib unavailable, installing pandas-ta fallback..."
    python3 -m pip install pandas-ta --quiet || echo "[bootstrap] pandas-ta install failed; using compat fallback"
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
  echo "[bootstrap] pandas-ta missing, attempting install..."
  python3 -m pip install pandas-ta --quiet || echo "[bootstrap] pandas-ta install failed; using compat fallback"
fi

# Final verification
if check_core_deps && check_talib; then
  echo "[bootstrap] All dependencies ready!"
else
  echo "[bootstrap] ERROR: Failed to install required dependencies"
  exit 1
fi
