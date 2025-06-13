@echo off
title Tetris AI - Full Training Session (Headless)

echo ====================================
echo TETRIS AI - FULL TRAINING SESSION
echo ====================================

REM Create timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set timestamp=%datetime:~0,8%_%datetime:~8,6%

echo Session: %timestamp%
echo.

REM Start TensorBoard in background
echo Starting TensorBoard...
start "TensorBoard" cmd /k "tensorboard --logdir runs --port 6006 --reload_interval 1"

REM Wait for TensorBoard to start
timeout /t 5 /nobreak >nul

echo TensorBoard started at http://localhost:6006
echo.

REM Start Unity Headless
echo Starting Unity (Headless Mode)...
start "" /B "D:\Edu\CMS\Sem_2\IMLD_Team_project\tetris-multi\build\tetris-multi.exe" -nographics -batchmode
timeout /t 10 /nobreak >nul
echo Unity started in headless mode
echo.

REM Start training
echo Starting Python training script...
set episodes=2000
set model_path=models\tetris_model_%timestamp%.pth
set tensorboard_dir=runs\training_%timestamp%

python train_tetris.py --mode train --episodes %episodes% --model_path "%model_path%" --tensorboard_dir "%tensorboard_dir%"

echo.
echo ====================================
echo TRAINING SESSION COMPLETED
echo ====================================
echo.
echo Results saved to: %model_path%
echo TensorBoard logs: %tensorboard_dir%
echo.
echo TensorBoard is still running at http://localhost:6006
echo Close the TensorBoard window when you're done reviewing metrics.
echo.
pause
