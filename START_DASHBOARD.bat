@echo off
REM =====================================================================
REM  Explainable AI Hospital Admission Risk & Community-Care
REM  Decision Support System  -  one-click launcher
REM
REM  The trained models are already included, so nothing needs training.
REM  This script installs the required packages (first run only) and then
REM  opens the dashboard in your browser.
REM =====================================================================
title Admission Risk Decision Support - Dashboard
cd /d "%~dp0"

echo.
echo  ============================================================
echo   Explainable AI Admission Risk Decision Support System
echo  ============================================================
echo.

REM ---- check Python is available -------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python was not found on this system.
    echo.
    echo  Please install Python 3.11 or newer from https://www.python.org/downloads/
    echo  and make sure "Add Python to PATH" is ticked during installation.
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo  Python %%v detected.

REM ---- install dependencies on first run ------------------------------
if not exist ".setup_done" (
    echo.
    echo  First run detected - installing required packages.
    echo  This takes a few minutes and only happens once.
    echo.
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo  [ERROR] Package installation failed. See the messages above.
        pause
        exit /b 1
    )
    echo done > .setup_done
    echo.
    echo  Setup complete.
)

REM ---- launch ---------------------------------------------------------
echo.
echo  Starting the dashboard...
echo  It will open in your browser at http://localhost:8501
echo.
echo  Leave this window open while you use the dashboard.
echo  Close it (or press Ctrl+C) when you are finished.
echo.
python -m streamlit run app.py

pause
