@echo off
title TensorBoard - Tetris AI

echo ====================================
echo TENSORBOARD - TETRIS AI METRICS
echo ====================================

REM Check if runs directory exists
if not exist "runs" (
    echo ERROR: No runs directory found!
    echo Run training first to generate logs.
    pause
    exit /b 1
)

REM Check if there are any log directories
dir runs /ad /b >nul 2>&1
if errorlevel 1 (
    echo ERROR: No TensorBoard logs found in runs directory!
    echo Run training first to generate logs.
    pause
    exit /b 1
)

echo Available log directories:
dir runs /ad /b
echo.

echo Starting TensorBoard...
echo.
echo TensorBoard will be available at: http://localhost:6006
echo Press Ctrl+C to stop TensorBoard
echo.

REM Start TensorBoard
tensorboard --logdir runs --port 6006 --reload_interval 1

pause