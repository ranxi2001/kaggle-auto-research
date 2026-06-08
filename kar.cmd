@echo off
setlocal

set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo .venv not found. Run: py -3.12 -m venv .venv ^&^& .venv\Scripts\python.exe -m pip install -e .
  exit /b 1
)

"%PY%" -m cli.main %*
