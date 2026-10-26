@echo off
cd /d "%~dp0"
"%USERPROFILE%\miniconda3\envs\lerobot\python.exe" -B server.py --port 8443
pause
