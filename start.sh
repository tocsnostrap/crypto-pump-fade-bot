#!/bin/bash

# Production startup script for Pump Fade Trading Bot
# Runs Express dashboard; Python bot is spawned by Node server

set -e

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

# Run Express directly in foreground - this is what Replit monitors for port 5000
echo "$(date): Starting Express server on port $PORT..."
exec node dist/index.cjs
