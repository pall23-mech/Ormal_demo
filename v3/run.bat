@echo off
cd /d "%~dp0\.."
call .venv\Scripts\activate
python v3\app.py
