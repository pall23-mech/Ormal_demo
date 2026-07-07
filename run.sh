#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ ! -d ".venv" ]; then
    echo "[error] Virtual environment not found."
    echo "        Create it:"
    echo "          python -m venv .venv"
    echo "          source .venv/bin/activate"
    echo "          pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu"
    echo "          pip install -r requirements.txt"
    echo "          python setup.py"
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
exec python app.py
