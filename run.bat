@echo off
echo.
echo   Prompt Coroner v2 — LangGraph Edition
echo   ----------------------------------------

python --version >nul 2>&1
if errorlevel 1 (
    echo   Python not found. Download from https://python.org
    pause & exit /b
)

if "%GROQ_API_KEY%"=="" (
    echo.
    echo   Get your FREE key at: https://console.groq.com
    echo.
    set /p GROQ_API_KEY="  Paste your GROQ_API_KEY: "
)

if not exist "venv\" (
    echo   Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate

echo   Installing dependencies (first run takes ~2 min)...
pip install -q -r requirements.txt

echo.
echo   Ready! Open: http://localhost:5000
echo.
python app.py
pause
