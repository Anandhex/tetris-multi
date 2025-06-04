@echo off
title Tetris AI - Model Evaluation

echo ====================================
echo TETRIS AI - MODEL EVALUATION
echo ====================================

REM Check for models
if not exist "models" (
    echo ERROR: No models directory found!
    pause
    exit /b 1
)

echo Available models:
echo.
dir models\*.pth /b 2>nul
if errorlevel 1 (
    echo No trained models found!
    echo Run training first to create models.
    pause
    exit /b 1
)

echo.
echo Special models:
if exist "models\*_best_score.pth" (
    echo Best Score Models:
    dir models\*_best_score.pth /b
)
if exist "models\*_best_avg.pth" (
    echo Best Average Models:
    dir models\*_best_avg.pth /b
)

echo.
set /p model_name="Enter model filename (without path): "

if not exist "models\%model_name%" (
    echo ERROR: Model not found!
    pause
    exit /b 1
)

echo.
set /p episodes="Number of evaluation episodes (default 20): "
if "%episodes%"=="" set episodes=20

echo.
echo Evaluating model: %model_name%
echo Episodes: %episodes%
echo.
echo Make sure Unity is running!
pause

REM Create timestamp for evaluation
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set timestamp=%datetime:~0,8%_%datetime:~8,6%

set tensorboard_dir=runs\eval_%timestamp%

python train_tetris.py --mode evaluate --eval_episodes %episodes% --model_path "models\%model_name%" --tensorboard_dir "%tensorboard_dir%"

echo.
echo ====================================
echo EVALUATION COMPLETED
echo ====================================
pause