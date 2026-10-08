@echo off
powershell.exe -NoProfile -File "%~dp0scripts\start_review_studio.ps1"
if errorlevel 1 pause
