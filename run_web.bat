@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Video2Docs Web UI starter (Windows)
REM - Activates .\venv if present
REM - Starts the Flask web app

cd /d "%~dp0"

if exist .env (
  echo Using .env in %CD%
) else (
  echo No .env found. You can copy .env.example to .env and adjust settings.
)

if exist "venv\Scripts\activate.bat" (
  echo Activating virtual environment...
  call "venv\Scripts\activate.bat"
) else (
  echo No virtual environment found at venv\Scripts\activate.bat. Continuing with system Python.
)

REM Start the web application
python -m src.webapp %*

endlocal
