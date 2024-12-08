@echo off
REM Change directory to the location of your Python script
cd /d %~dp0

REM Activate the virtual environment
call .venv\Scripts\activate

REM Run the Python script
python main.py

REM Pause to keep the command prompt open
pause