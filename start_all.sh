#!/bin/bash

# Production startup script for Pump Fade Trading Bot
# Runs Express dashboard (foreground) + Python trading bot (background)

set -e

export BOT_MANAGED=1
export NODE_ENV=production
export PORT=${PORT:-5000}

echo "=== Pump Fade Trading Bot Production Startup ==="
echo "Starting at: $(date)"
echo "PORT: $PORT"

# Bootstrap Python deps if available
if [ -f "./script/bootstrap_python.sh" ]; then
    sh ./script/bootstrap_python.sh
fi

# Verify build exists
if [ ! -f "dist/index.cjs" ]; then
    echo "ERROR: dist/index.cjs not found!"
    echo "Run 'npm run build' first"
    exit 1
fi

# Cleanup function
cleanup() {
    echo "$(date): Shutting down..."
    if [ ! -z "$PYTHON_PID" ] && kill -0 $PYTHON_PID 2>/dev/null; then
        kill $PYTHON_PID 2>/dev/null
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Start Python trading bot in background with simple restart loop
(
    while true; do
        echo "$(date): [python-bot] Starting..."
        python3 main.py 2>&1 | while IFS= read -r line; do echo "[python-bot] $line"; done
        echo "$(date): [python-bot] Exited, restarting in 10s..."
        sleep 10
    done
) &
PYTHON_PID=$!
echo "Python bot started (PID: $PYTHON_PID)"

# Give Python a moment to initialize
sleep 1

# Run Express directly in foreground - this is what Replit monitors for port 5000
echo "$(date): Starting Express server on port $PORT..."
exec node dist/index.cjs
