@echo off
echo ====================================
echo TETRIS AI TRAINING SETUP
echo ====================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found. Setting up environment...

REM Create directories
if not exist "runs" mkdir runs
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "data" mkdir data

echo Created project directories.

REM Install requirements
echo Installing Python packages...
pip install torch torchvision tensorboard numpy matplotlib pandas scikit-learn

if errorlevel 1 (
    echo ERROR: Failed to install packages
    pause
    exit /b 1
)

echo.
echo ====================================
echo SETUP COMPLETE!
echo ====================================
echo.
echo Next steps:
echo 1. Start Unity with your Tetris scene
echo 2. Run train.bat to start training
echo 3. Run tensorboard.bat to view metrics
echo.
pause