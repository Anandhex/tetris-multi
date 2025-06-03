#!/bin/bash

echo "===================================="
echo "TETRIS AI - MODEL EVALUATION"
echo "===================================="

# Check for models
if [ ! -d "models" ]; then
    echo "ERROR: No models directory found!"
    exit 1
fi

echo "Available models:"
echo ""
if ls models/*.pth 2>/dev/null; then
    echo ""
    echo "Special models:"
    if ls models/*_best_score.pth 2>/dev/null; then
        echo "Best Score Models:"
        ls models/*_best_score.pth 2>/dev/null | xargs -n 1 basename
    fi
    if ls models/*_best_avg.pth 2>/dev/null; then
        echo "Best Average Models:"
        ls models/*_best_avg.pth 2>/dev/null | xargs -n 1 basename
    fi
else
    echo "No trained models found!"
    echo "Run training first to create models."
    exit 1
fi

echo ""
read -p "Enter model filename (without path): " model_name

if [ ! -f "models/$model_name" ]; then
    echo "ERROR: Model not found!"
    exit 1
fi

echo ""
read -p "Number of evaluation episodes (default 20): " episodes
episodes=${episodes:-20}

echo ""
echo "Evaluating model: $model_name"
echo "Episodes: $episodes"
echo ""
echo "Make sure Unity is running!"
read -p "Press Enter to continue..."

# Create timestamp for evaluation
timestamp=$(date +"%Y%m%d_%H%M%S")
tensorboard_dir="runs/eval_$timestamp"

python3 train_tetris.py --mode evaluate --eval_episodes $episodes --model_path "models/$model_name" --tensorboard_dir "$tensorboard_dir"

echo ""
echo "===================================="
echo "EVALUATION COMPLETED"
echo "===================================="