#!/usr/bin/env bash
set -e

MARKER_FILE=".python_deps_ready"

is_replit_env() {
  [ "$REPLIT" = "1" ] || [ -n "$REPL_ID" ] || [ -n "$REPLIT_DEPLOYMENT" ] || [ -n "$REPLIT_DB_URL" ] || [ -n "$REPLIT_DEV_DOMAIN" ]
}

clean_python_packages() {
  python3 - <<'PY'
import glob
import os
import shutil
import site

def site_paths():
    paths = []
    try:
        paths.extend(site.getsitepackages())
    except Exception:
        pass
    user_site = site.getusersitepackages()
    if user_site:
        paths.append(user_site)
    seen = set()
    result = []
    for path in paths:
        if path and os.path.isdir(path) and path not in seen:
            seen.add(path)
            result.append(path)
    return result

targets = []
for base in site_paths():
    for name in ("numpy", "pandas", "pandas_ta", "ccxt"):
        candidate = os.path.join(base, name)
        if os.path.isdir(candidate):
            targets.append(candidate)
    for pattern in ("numpy-*.dist-info", "pandas-*.dist-info", "pandas_ta-*.dist-info", "ccxt-*.dist-info"):
        targets.extend(glob.glob(os.path.join(base, pattern)))

for target in targets:
    try:
        shutil.rmtree(target)
        print(f"[bootstrap] Removed {target}")
    except Exception as exc:
        print(f"[bootstrap] Could not remove {target}: {exc}")
PY
}

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
  if is_replit_env; then
    echo "[bootstrap] Replit detected; skipping pip installs."
    echo "[bootstrap] Use Nix packages in .replit for numpy/pandas/ccxt."
    return 1
  fi

  echo "[bootstrap] Installing Python dependencies..."
  python3 -m pip install --upgrade pip --quiet
  clean_python_packages
  python3 -m pip install --upgrade --force-reinstall --no-cache-dir --only-binary=:all: \
    --index-url https://pypi.org/simple \
    "numpy<2.3" "pandas>=2.0" --quiet
  python3 -m pip install --upgrade --force-reinstall --no-cache-dir \
    --index-url https://pypi.org/simple \
    ccxt --quiet

  # Try TA-Lib first, fall back to pandas-ta
  python3 -m pip install ta-lib --quiet 2>/dev/null || {
    echo "[bootstrap] TA-Lib unavailable, installing pandas-ta fallback..."
    python3 -m pip install --index-url https://pypi.org/simple pandas-ta --quiet || echo "[bootstrap] pandas-ta install failed; using compat fallback"
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
  if is_replit_env; then
    echo "[bootstrap] pandas-ta missing; relying on compat fallback."
  else
    echo "[bootstrap] pandas-ta missing, attempting install..."
    python3 -m pip install --index-url https://pypi.org/simple pandas-ta --quiet || echo "[bootstrap] pandas-ta install failed; using compat fallback"
  fi
fi

# Final verification
if check_core_deps && check_talib; then
  echo "[bootstrap] All dependencies ready!"
else
  echo "[bootstrap] ERROR: Failed to install required dependencies"
  exit 1
fi
