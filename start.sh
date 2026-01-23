#!/bin/bash
echo "=== Pump Fade Trading Bot Startup ==="
echo "Starting at: $(date)"

export NODE_ENV=production
export PORT=${PORT:-5000}

# Start Python bot in background (if python3 available)
if command -v python3 &> /dev/null; then
    echo "Starting Python trading bot..."
    python3 main.py &
    echo "Python bot started"
else
    echo "Python not available - dashboard only mode"
fi

# Start Express server
echo "Starting Express server on port $PORT..."
exec node dist/index.cjs
