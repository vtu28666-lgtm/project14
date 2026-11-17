@echo off
echo ========================================================
echo Starting Adaptive Approximate MAC Accelerator Dashboard
echo ========================================================

set PYTHON_EXE=C:\Users\VARUN.M\AppData\Local\Python\pythoncore-3.14-64\python.exe

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" server.py
) else (
    python server.py
)

pause

