#!/bin/bash

echo "===================================="
echo "TENSORBOARD - TETRIS AI METRICS"
echo "===================================="

# Check if runs directory exists
if [ ! -d "runs" ]; then
    echo "ERROR: No runs directory found!"
    echo "Run training first to generate logs."
    exit 1
fi

# Check if there are any log directories
if [ -z "$(ls -A runs 2>/dev/null)" ]; then
    echo "ERROR: No TensorBoard logs found in runs directory!"
    echo "Run training first to generate logs."
    exit 1
fi

echo "Available log directories:"
ls runs/
echo ""

echo "Starting TensorBoard..."
echo ""
echo "TensorBoard will be available at: http://localhost:6006"
echo "Press Ctrl+C to stop TensorBoard"
echo ""

# Start TensorBoard
tensorboard --logdir runs --port 6006 --reload_interval 1