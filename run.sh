#!/bin/bash

echo ""
echo "  🔬 Prompt Coroner v2 — LangGraph Edition"
echo "  ─────────────────────────────────────────"

# Check Python
if ! command -v python3 &> /dev/null; then
  echo "  ❌ Python3 not found. Install from https://python.org"
  exit 1
fi

# Ask for Groq key if not set
if [ -z "$GROQ_API_KEY" ]; then
  echo ""
  echo "  Get your FREE key at: https://console.groq.com"
  echo ""
  read -p "  Paste your GROQ_API_KEY: " GROQ_API_KEY
  export GROQ_API_KEY
fi

# Create venv
if [ ! -d "venv" ]; then
  echo "  📦 Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

# Install deps
echo "  📦 Installing dependencies (first run takes ~2 min for sentence-transformers)..."
pip install -q -r requirements.txt

echo ""
echo "  ✅ Ready! Open → http://localhost:5000"
echo "  (Ctrl+C to stop)"
echo ""

python3 app.py
