#!/bin/bash
# Unified setup script for Mac/Linux
# Sets up environment, installs dependencies, and prepares the project

set -e

echo "=========================================="
echo "Code Comment Generation - Setup"
echo "=========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Found Python $PYTHON_VERSION"
echo ""

# Step 1: Create virtual environment
if [ ! -d ".venv" ]; then
    echo "[1/6] Creating virtual environment..."
    python3 -m venv .venv
else
    echo "[1/6] Virtual environment already exists"
fi
echo ""

# Step 2: Activate and upgrade pip
echo "[2/6] Activating environment and upgrading pip..."
source .venv/bin/activate
pip install --upgrade pip setuptools wheel >/dev/null 2>&1
echo ""

# Step 3: Install root requirements
echo "[3/6] Installing root dependencies..."
pip install -r requirements.txt
echo ""

# Step 4: Install backend requirements
echo "[4/6] Installing backend dependencies..."
pip install -r backend/requirements.txt
echo ""

# Step 5: Setup algorithms
echo "[5/6] Setting up algorithms..."
cd backend
python setup_algos.py || echo "WARNING: Algorithm setup had issues, but continuing..."
cd ..
echo ""

# Step 6: Download NLTK data
echo "[6/6] Downloading NLTK data..."
python3 -c "import nltk; nltk.download('averaged_perceptron_tagger', quiet=True); nltk.download('wordnet', quiet=True)" 2>/dev/null || true
echo ""

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. Start backend: cd backend && python run_server.py"
echo "  3. Start frontend: cd frontend/client && npm install && npm run dev"
echo "  4. Run tests: python test.py"
echo ""

