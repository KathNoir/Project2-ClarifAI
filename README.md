# Code Comment Generation Project

Web application for generating code comments using Greedy and A* algorithms.

## Quick Start

### One-Command Setup

**Windows:**
```cmd
setup.bat
```

**Mac/Linux:**
```bash
bash setup.sh
```

This will:
- Create virtual environment
- Install all dependencies (backend Python + frontend Node.js)
- Copy algorithms to backend
- Download NLTK data
- Set up frontend dependencies

### Manual Setup

```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt

# 3. Setup algorithms
cd backend
python setup_algos.py
cd ..

# 4. Download NLTK data
python -c "import nltk; nltk.download('averaged_perceptron_tagger', quiet=True); nltk.download('wordnet', quiet=True)"
```

### Run Application

**Backend (Terminal 1):**
```bash
cd backend
python run_server.py  # Runs on http://localhost:8001
```

**Frontend (Terminal 2):**
```bash
cd frontend/client
npm install  # First time only
npm run dev  # Runs on http://localhost:3000
```

Visit http://localhost:3000 to use the application!

## Model Preparation

Both services expect trained models under `backend/models/` before you start the API.

- **Install dataset dependency**
  ```bash
  pip install datasets
  ```
- **Greedy G0_base (full dataset)**
  ```bash
  cd backend
  python tests/create_toy_greedy_model.py --max-samples 0 --language python
  ```
  - `--max-samples 0` streams the entire CodeSearchNet `python` split. Adjust the language if you want a different subset.
  - Use `--max-samples 200` (default) for a fast developmental build.
- **A* N-gram model (full dataset)**
  ```bash
  cd scripts
  python train_ngram_model.py --max-samples 0 --output ../backend/models/astar_A3_dependency
  ```
  - The script writes artifacts next to the backend so `AStarService` can load them.
  - Pass `--semantic-heuristic` if you have enough memory and want TF-IDF guidance.

Expect full-dataset training to take several hours and require ~10 GB of RAM plus 15–20 GB of disk space for the downloaded CodeSearchNet cache.

## Project Structure

For a detailed file structure guide with visual flowcharts, see **[STRUCTURE.md](STRUCTURE.md)**.

Quick overview:
- **`backend/`** - FastAPI backend (API, services, algorithms, models, tests, config)
- **`backend/config.py`** - Centralized configuration (ports, URLs, paths)
- **`backend/tests/`** - Application tests (includes `create_toy_greedy_model.py` for development)
- **`frontend/client/`** - Next.js frontend (React components, UI) - tracked in git
- **`testing/`** - Model training & experiments (research, most artifacts untracked)
- **`setup.sh` / `setup.bat`** - Unified setup script
- **`test.py`** - Unified test runner (run all tests from root)

## Testing

### Unified Test Runner

Run all tests:
```bash
python test.py
```

Run specific test suite:
```bash
python test.py --suite api      # Test API endpoints
python test.py --suite astar    # Test A* algorithm
python test.py --suite service  # Test service integration
python test.py --suite quick    # Quick smoke test
```

**Note**: Test files are in `backend/tests/`. The `testing/` directory is for model training/experiments, not application tests.

### Test Suites

- **`--suite api`**: Tests API endpoints (requires running server)
- **`--suite astar`**: Tests A* algorithm (trains model on sample data)
- **`--suite service`**: Tests full service integration (creates test model)
- **`--suite quick`**: Quick API smoke test (requires running server)
- **`--suite all`**: Runs all test suites (default)

## Algorithm Results

**Best Greedy (G0_base):**
- CodeBLEU: 0.223
- Runtime: 0.19ms

**Best A* (A3_dependency):**
- CodeBLEU: 0.363 (63% better!)
- Runtime: 13.5ms

## Team Tasks

### Katherine - Model Training (HIGH PRIORITY)
See `TEAM_TASKS.md`
- **Train models on FULL CodeSearchNet dataset** (not just samples)
- Place trained models in `backend/models/`:
  - `greedy_G0_base/`
  - `astar_A3_dependency/`

### Jarrod - Greedy Quality Improvement  
See `TEAM_TASKS.md`
- **Fix greedy comment quality** - currently generates trivial comments
- Debug algorithm/parameters after Katherine's full dataset training
- Verify service works correctly

### Jennifer - Completed
- A* algorithm service
- Frontend UI/UX with shadcn/ui
- Visualization component
- Accessibility features

## Development

### API Endpoints

- `GET /` - Health check and model status
- `POST /generate` - Generate comments for Python code
- `GET /models` - Check model loading status

### Request Example

```bash
# Default backend URL (configured in backend/config.py)
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"code": "def sum(a, b):\n    return a + b", "max_length": 20, "enable_visualization": false}'
```

### Response Example

```json
{
  "greedy_comment": "calculate sum of two numbers",
  "astar_comment": "return the sum of two input parameters",
  "code_tokens": ["sum", "a", "b"],
  "greedy_runtime_ms": 0.19,
  "astar_runtime_ms": 13.50,
  "greedy_loaded": true,
  "astar_loaded": true,
  "error": null
}
```

### API Documentation

When the server is running, visit (default port from `backend/config.py`):
- **Interactive API Docs**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

To change the port, edit `BACKEND_PORT` in `backend/config.py`.

## Documentation

- **`STRUCTURE.md`** - Detailed file structure guide with flowcharts (start here for codebase navigation)
- **`TEAM_TASKS.md`** - Team responsibilities and task breakdown

## Codebase Organization

### Configuration
All configuration is centralized in `backend/config.py`:
- Server ports and URLs
- Model paths and names
- Default parameters

### Testing Structure
- **`backend/tests/`** - Application tests (unit, integration, API)
  - Run via: `python test.py`
- **`testing/`** - Research/experiments (model training, ablation studies)
  - Artifacts (data, results) are gitignored
  - Scripts are tracked for reproducibility

## Troubleshooting

### "Python not found"
- Windows: Add Python to PATH or use `py` command
- Mac/Linux: Install via package manager

### "Models not loading"
- Check `backend/models/` directory exists
- Verify model files are present
- Run `python test.py --suite service` to create test model

### "Cannot connect to API"
- Make sure backend server is running: `cd backend && python run_server.py`
- Check port (default: 8001) is not used by another app
- Verify frontend is pointing to correct backend URL
- Check `backend/config.py` if you need to change ports

### "Import errors"
- Verify virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check you're in the correct directory

## License

See LICENSE file.
