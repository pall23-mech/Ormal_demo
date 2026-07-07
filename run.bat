@echo off
cd /d "%~dp0"

if not exist ".venv\" (
    echo [error] Virtual environment not found.
    echo         Create it:
    echo           python -m venv .venv
    echo           .venv\Scripts\activate.bat
    echo           pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo           pip install -r requirements.txt
    echo           python setup.py
    exit /b 1
)

call .venv\Scripts\activate.bat
python app.py
