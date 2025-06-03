#!/bin/bash

echo "===================================="
echo "TETRIS AI TRAINING"
echo "===================================="

# Check if Python files exist
if [ ! -f "train_tetris.py" ]; then
    echo "ERROR: train_tetris.py not found!"
    echo "Make sure you have all Python files in this directory."
    exit 1
fi

# Create timestamp for this training session
timestamp=$(date +"%Y%m%d_%H%M%S")

echo "Training session: $timestamp"
echo ""

# Ask user for training mode
echo "Select training mode:"
echo "1. New training (fresh start)"
echo "2. Continue existing training"
echo "3. Evaluate existing model"
echo "4. Quick test (100 episodes)"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "Starting new training session..."
        episodes=2000
        model_path="models/tetris_model_$timestamp.pth"
        tensorboard_dir="runs/training_$timestamp"
        python3 train_tetris.py --mode train --episodes $episodes --model_path "$model_path" --tensorboard_dir "$tensorboard_dir"
        ;;
    2)
        echo "Available models:"
        if ls models/*.pth 2>/dev/null; then
            echo ""
            read -p "Enter model filename (without path): " model_name
            if [ -f "models/$model_name" ]; then
                episodes=1000
                tensorboard_dir="runs/continue_$timestamp"
                python3 train_tetris.py --mode continue --episodes $episodes --model_path "models/$model_name" --tensorboard_dir "$tensorboard_dir"
            else
                echo "Model not found! Starting new training..."
                episodes=2000
                model_path="models/tetris_model_$timestamp.pth"
                tensorboard_dir="runs/training_$timestamp"
                python3 train_tetris.py --mode train --episodes $episodes --model_path "$model_path" --tensorboard_dir "$tensorboard_dir"
            fi
        else
            echo "No existing models found. Starting new training..."
            episodes=2000
            model_path="models/tetris_model_$timestamp.pth"
            tensorboard_dir="runs/training_$timestamp"
            python3 train_tetris.py --mode train --episodes $episodes --model_path "$model_path" --tensorboard_dir "$tensorboard_dir"
        fi
        ;;
    3)
        echo "Available models:"
        if ls models/*.pth 2>/dev/null; then
            echo ""
            read -p "Enter model filename (without path): " model_name
            if [ -f "models/$model_name" ]; then
                eval_episodes=20
                tensorboard_dir="runs/eval_$timestamp"
                python3 train_tetris.py --mode evaluate --eval_episodes $eval_episodes --model_path "models/$model_name" --tensorboard_dir "$tensorboard_dir"
            else
                echo "Model not found!"
                exit 1
            fi
        else
            echo "No existing models found!"
            exit 1
        fi
        ;;
    4)
        echo "Starting quick test training..."
        episodes=100
        model_path="models/tetris_quick_test_$timestamp.pth"
        tensorboard_dir="runs/quick_test_$timestamp"
        python3 train_tetris.py --mode train --episodes $episodes --model_path "$model_path" --tensorboard_dir "$tensorboard_dir"
        ;;
    *)
        echo "Invalid choice. Starting new training..."
        episodes=2000
        model_path="models/tetris_model_$timestamp.pth"
        tensorboard_dir="runs/training_$timestamp"
        python3 train_tetris.py --mode train --episodes $episodes --model_path "$model_path" --tensorboard_dir "$tensorboard_dir"
        ;;
esac

echo ""
echo "===================================="
echo "TRAINING COMPLETED"
echo "===================================="
echo ""
echo "To view TensorBoard:"
echo "  ./tensorboard.sh"
echo ""
echo "To start another training session:"
echo "  ./train.sh"
echo ""