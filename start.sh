#!/bin/bash
echo "=== Pump Fade Trading Bot Production Startup ==="
echo "Starting at: $(date)"

export NODE_ENV=production
export PORT=${PORT:-5000}

# Install Python dependencies if needed
pip install ccxt numpy pandas ta-lib 2>/dev/null || true

# Build if needed
if [ ! -f "dist/index.cjs" ]; then
    echo "Building application..."
    npm run build
fi

# Start Python bot in background
echo "Starting Python trading bot..."
python3 main.py &
PYTHON_PID=$!
echo "Python bot started (PID: $PYTHON_PID)"

# Start Express server (foreground)
echo "Starting Express server on port $PORT..."
exec node dist/index.cjs
