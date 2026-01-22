#!/bin/bash

# Production startup script for Pump Fade Trading Bot
# Runs Express dashboard (foreground) + Python trading bot (background)

echo "=== Pump Fade Trading Bot Production Startup ==="
echo "Starting at: $(date)"
echo "Working directory: $(pwd)"

# Set environment variables
export BOT_MANAGED=1
export NODE_ENV=production
export PORT=${PORT:-5000}

echo "PORT: $PORT"
echo "NODE_ENV: $NODE_ENV"

# Function to log with timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1"
}

# Cleanup function
cleanup() {
    log "Shutting down..."
    if [ ! -z "$PYTHON_PID" ] && kill -0 $PYTHON_PID 2>/dev/null; then
        kill $PYTHON_PID 2>/dev/null
        log "Python bot stopped"
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Check if npm dependencies are installed
if [ ! -d "node_modules" ]; then
    log "Installing Node.js dependencies..."
    npm install
fi

# Build if dist doesn't exist or is older than source
if [ ! -f "dist/index.cjs" ]; then
    log "Building application..."
    npm run build
    if [ $? -ne 0 ]; then
        log "ERROR: Build failed!"
        # Try to start dev server as fallback
        log "Attempting dev server fallback..."
        exec npm run dev
    fi
fi

# Verify dist/index.cjs exists after build
if [ ! -f "dist/index.cjs" ]; then
    log "ERROR: dist/index.cjs not found after build!"
    log "Starting development server instead..."
    exec npm run dev
fi

log "Build verified: dist/index.cjs exists"

# Detect Python command (python3 preferred on Linux)
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
fi

if [ -z "$PYTHON_CMD" ]; then
    log "WARNING: Python not found, starting Express only"
else
    log "Using Python command: $PYTHON_CMD"
    
    # Verify Python can import required modules
    $PYTHON_CMD -c "import ccxt, talib, pandas, numpy" 2>/dev/null
    if [ $? -ne 0 ]; then
        log "WARNING: Python dependencies may be missing"
        log "Installing Python dependencies..."
        pip install ccxt numpy pandas ta-lib 2>/dev/null || pip3 install ccxt numpy pandas ta-lib 2>/dev/null
    fi
    
    # Start Python trading bot in background with restart loop
    (
        while true; do
            log "[python-bot] Starting trading bot..."
            $PYTHON_CMD main.py 2>&1 | while IFS= read -r line; do 
                echo "$(date '+%Y-%m-%d %H:%M:%S') [python-bot] $line"
            done
            EXIT_CODE=$?
            log "[python-bot] Exited with code $EXIT_CODE, restarting in 10s..."
            sleep 10
        done
    ) &
    PYTHON_PID=$!
    log "Python bot started in background (PID: $PYTHON_PID)"
fi

# Give Python a moment to initialize
sleep 2

# Run Express server in foreground - this is what Replit monitors for port 5000
log "Starting Express server on port $PORT..."
log "Server will be available at http://0.0.0.0:$PORT"

exec node dist/index.cjs
