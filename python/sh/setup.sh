#!/bin/bash

echo "===================================="
echo "TETRIS AI TRAINING SETUP"
echo "===================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from your package manager"
    exit 1
fi

echo "Python found. Setting up environment..."

# Create directories
mkdir -p runs models logs data

echo "Created project directories."

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "ERROR: pip3 is not installed"
    echo "Please install pip3"
    exit 1
fi

# Install requirements
echo "Installing Python packages..."
pip3 install torch torchvision tensorboard numpy matplotlib pandas scikit-learn

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install packages"
    exit 1
fi

# Make scripts executable
chmod +x *.sh 2>/dev/null

echo ""
echo "===================================="
echo "SETUP COMPLETE!"
echo "===================================="
echo ""
echo "Next steps:"
echo "1. Start Unity with your Tetris scene"
echo "2. Run ./train.sh to start training"
echo "3. Run ./tensorboard.sh to view metrics"
echo ""