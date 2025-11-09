@echo off
REM Unified setup script for Windows
REM Sets up environment, installs dependencies, and prepares the project

echo ==========================================
echo Code Comment Generation - Setup
echo ==========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [OK] Found Python %PYTHON_VERSION%
echo.

REM Step 1: Create virtual environment
if not exist ".venv" (
    echo [1/5] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        exit /b 1
    )
) else (
    echo [1/5] Virtual environment already exists
)
echo.

REM Step 2: Activate and upgrade pip
echo [2/5] Activating environment and upgrading pip...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel >nul 2>&1
echo.

REM Step 3: Install root requirements
echo [3/5] Installing root dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install root dependencies
    exit /b 1
)
echo.

REM Step 4: Install backend requirements
echo [4/5] Installing backend dependencies...
pip install -r backend\requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install backend dependencies
    exit /b 1
)
echo.

REM Step 5: Setup algorithms
echo [5/5] Setting up algorithms...
cd backend
python setup_algos.py
if errorlevel 1 (
    echo [WARNING] Algorithm setup had issues, but continuing...
)
cd ..
echo.

REM Step 6: Download NLTK data
echo [6/6] Downloading NLTK data...
python -c "import nltk; nltk.download('averaged_perceptron_tagger', quiet=True); nltk.download('wordnet', quiet=True)" 2>nul
echo.

echo ==========================================
echo [OK] Setup complete!
echo ==========================================
echo.
echo Next steps:
echo   1. Activate environment: .venv\Scripts\activate
echo   2. Start backend: cd backend ^&^& python run_server.py
echo   3. Start frontend: cd frontend\client ^&^& npm install ^&^& npm run dev
echo   4. Run tests: python test.py
echo.
pause

