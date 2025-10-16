@echo off
setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION
REM Install Video2Docs as a Windows service using NSSM (Non-Sucking Service Manager)
REM
REM Requirements:
REM   - NSSM installed and available in PATH (https://nssm.cc/download)
REM
REM Usage:
REM   install_service_windows.bat [ServiceName]
REM     ServiceName defaults to Video2Docs
REM
REM Notes:
REM   - The service will run in this repository directory
REM   - The app auto-loads environment from .env in this directory
REM   - Logs can be configured via NSSM (e.g., nssm set <name> AppStdout <file>)

set SERVICE_NAME=%~1
if "%SERVICE_NAME%"=="" set SERVICE_NAME=Video2Docs

REM Move to repo root
cd /d "%~dp0"
set REPO_DIR=%CD%

REM Detect NSSM
where nssm >nul 2>&1
if errorlevel 1 (
  echo ERROR: NSSM not found in PATH.
  echo Please install NSSM (https://nssm.cc/download) and ensure 'nssm.exe' is in PATH.
  echo After installing, run this script again.
  exit /b 1
)

REM Prefer venv Python if present
set PYTHON_BIN=
if exist "%REPO_DIR%\venv\Scripts\python.exe" (
  set "PYTHON_BIN=%REPO_DIR%\venv\Scripts\python.exe"
) else (
  for %%P in (python.exe py.exe) do (
    where %%P >nul 2>&1 && (
      if not defined PYTHON_BIN set PYTHON_BIN=%%~f$PATH:P
    )
  )
)

if "%PYTHON_BIN%"=="" (
  echo ERROR: Could not find Python. Please install Python 3 or create a venv.
  exit /b 1
)

echo Installing service %SERVICE_NAME%

echo - Working directory: %REPO_DIR%
echo - Python: %PYTHON_BIN%

REM Create or update the service
nssm stop "%SERVICE_NAME%" >nul 2>&1
nssm remove "%SERVICE_NAME%" confirm >nul 2>&1

REM Install service to run the launcher that activates venv and starts the app
nssm install "%SERVICE_NAME%" "%REPO_DIR%\run_web.bat"
if errorlevel 1 (
  echo ERROR: Failed to install service with NSSM.
  exit /b 1
)

nssm set "%SERVICE_NAME%" AppDirectory "%REPO_DIR%"
nssm set "%SERVICE_NAME%" DisplayName "Video2Docs Web Service"
nssm set "%SERVICE_NAME%" Start SERVICE_AUTO_START

REM Optional: set some restart policies
nssm set "%SERVICE_NAME%" AppThrottle 5000
nssm set "%SERVICE_NAME%" AppRestartDelay 5000

REM Start the service
nssm start "%SERVICE_NAME%"
if errorlevel 1 (
  echo Service installed but failed to start. Check Windows Event Viewer or nssm get %SERVICE_NAME% LastError.
  exit /b 2
)

echo Service "%SERVICE_NAME%" installed and started.
echo To view status: nssm status %SERVICE_NAME%
echo To stop:       nssm stop %SERVICE_NAME%
echo To remove:     nssm remove %SERVICE_NAME% confirm

endlocal
