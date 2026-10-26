@echo off
chcp 65001 >nul
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" lerobot
python "%~dp0analyze_alpha.py" "%~dp0results"
pause
