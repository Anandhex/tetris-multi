#!/bin/bash

echo "===================================="
echo "TETRIS AI - FULL TRAINING SESSION"
echo "===================================="

# Create timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")

echo "Session: $timestamp"
echo ""




# Check Unity connection
echo "IMPORTANT: Make sure Unity is running with the Tetris scene!"
echo ""
read -p "Is Unity running and ready? (y/n): " unity_ready

if [ "$unity_ready" != "y" ] && [ "$unity_ready" != "Y" ]; then
    echo "Please start Unity first, then run this script again."
    exit 1
fi

# Start training
echo "Starting training..."
echo ""

episodes=2000
model_path="models/tetris_model_$timestamp.pth"
tensorboard_dir="runs/training_$timestamp"

python3 train_tetris.py --mode train --episodes $episodes --model_path "$model_path" --tensorboard_dir "$tensorboard_dir"

echo ""
echo "===================================="
echo "TRAINING SESSION COMPLETED"
echo "===================================="
echo ""
echo "Results saved to: $model_path"
echo "TensorBoard logs: $tensorboard_dir"
echo ""
echo "TensorBoard is still running at http://localhost:6006"
echo "Press Ctrl+C to stop TensorBoard when you're done."
echo ""

# Keep TensorBoard running
wait $TENSORBOARD_PID