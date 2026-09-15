@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0translate-all-en.ps1" %*
exit /b %ERRORLEVEL%
