@echo off
echo ========================================================
echo Running Dataset Cleaner and Model Training Pipeline
echo ========================================================

set PYTHON_EXE=C:\Users\VARUN.M\AppData\Local\Python\pythoncore-3.14-64\python.exe

if exist "%PYTHON_EXE%" (
    echo [1/2] Cleaning and segmenting vibration dataset...
    "%PYTHON_EXE%" clean_dataset.py
    echo.
    echo [2/2] Training AI precision controller model...
    "%PYTHON_EXE%" train_model.py
) else (
    echo [1/2] Cleaning and segmenting vibration dataset...
    python clean_dataset.py
    echo.
    echo [2/2] Training AI precision controller model...
    python train_model.py
)

echo ========================================================
echo Pipeline complete.
echo ========================================================
pause

