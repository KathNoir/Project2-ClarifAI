# Project File Structure

Visual guide to the codebase organization and data flow.

## Directory Tree

```
Project2-ClarifAI/
│
├── README.md              # Main documentation (start here!)
├── STRUCTURE.md           # This file - file structure guide
├── LICENSE                 # License file
│
├── setup.sh / setup.bat   # Unified setup script
├── test.py                # Unified test runner
│
├── requirements.txt        # Root Python dependencies
├── environment.yml         # Conda environment (optional)
│
├── backend/                # FastAPI Backend (Python)
│   │
│   ├── requirements.txt    # Backend-specific dependencies
│   ├── config.py           # Configuration (ports, URLs, model paths)
│   ├── run_server.py       # Server startup script
│   │
│   ├── api/                # API Layer
│   │   ├── server.py          # FastAPI app, routes, middleware
│   │   └── models.py          # Request/response Pydantic models
│   │
│   ├── services/          # Business Logic Layer
│   │   ├── code_processor.py # Extracts tokens from Python code
│   │   ├── greedy_service.py # Wraps greedy algorithm
│   │   └── astar_service.py  # Wraps A* algorithm
│   │
│   ├── algos/              # Algorithm Implementations
│   │   ├── greedy_markov.py  # Greedy Markov chain generator
│   │   └── A_star_beam.py    # A* beam search generator
│   │
│   ├── models/             # Trained Models (gitignored)
│   │   ├── greedy_G0_base/
│   │   │   ├── config.json
│   │   │   ├── transitions.json
│   │   │   └── context_counts.json
│   │   └── astar_A3_dependency/
│   │       ├── config.json
│   │       ├── transitions.json
│   │       ├── vocab_probs.json
│   │       └── vectorizer.joblib
│   │
│   └── tests/               # Application Tests
│       ├── __init__.py
│       ├── create_toy_greedy_model.py # CLI to train the greedy model
│       ├── test_api.py        # API endpoint tests
│       ├── test_astar.py      # A* algorithm tests
│       ├── test_full_system.py # Service integration tests
│       └── test_quick.py       # Quick smoke tests
│
├── frontend/               # Next.js Frontend (TypeScript/React)
│   │
│   ├── client/                # React Application (tracked)
│   │   ├── package.json    # Node.js dependencies
│   │   ├── next.config.js  # Next.js configuration
│   │   ├── tailwind.config.js # Tailwind CSS config
│   │   │
│   │   ├── public/         # Static assets
│   │   │
│   │   └── src/
│   │       ├── app/           # Next.js app router pages
│   │       │   ├── layout.tsx # Root layout
│   │       │   └── page.tsx  # Home page
│   │       │
│   │       ├── components/    # React components
│   │       │   ├── CodeInput.tsx        # Code input textarea
│   │       │   ├── CommentDisplay.tsx   # Results display
│   │       │   ├── AlgorithmVisualization.tsx # Visualization
│   │       │   └── ui/        # shadcn/ui components
│   │       │       ├── button.tsx
│   │       │       ├── card.tsx
│   │       │       └── ...
│   │       │
│   │       └── lib/           # Utilities
│   │           └── utils.ts   # Helper functions
│   │
│   └── deploy/             # Deployment configuration
│       ├── package.json    # Node.js dependencies
│       ├── next.config.js  # Next.js configuration
│       ├── tailwind.config.js # Tailwind CSS config
│       │
│       ├── public/         # Static assets
│       │
│       └── src/
│           ├── app/           # Next.js app router pages
│           │   ├── layout.tsx # Root layout
│           │   └── page.tsx  # Home page
│           │
│           ├── components/    # React components
│           │   ├── CodeInput.tsx        # Code input textarea
│           │   ├── CommentDisplay.tsx   # Results display
│           │   ├── AlgorithmVisualization.tsx # Visualization
│           │   └── ui/        # shadcn/ui components
│           │       ├── button.tsx
│           │       ├── card.tsx
│           │       └── ...
│           │
│           └── lib/           # Utilities
│               └── utils.ts   # Helper functions
│
└── testing/                # Model Training & Experiments (Research Only)
    │                           # NOT application tests - those are in backend/tests/
    │                           # Most artifacts (data/, results/) are gitignored
    │
    ├── scripts/            # Training & experimentation scripts (tracked)
    │   ├── run_experiments.py # Main training script
    │   ├── preprocess.py      # Data preprocessing
    │   ├── eval.py            # Evaluation metrics
    │   ├── ablation_study.py  # Ablation studies
    │   ├── quick_test.py      # Quick validation tests
    │   │
    │   ├── comment_gen/    # Algorithm implementations (dev versions)
    │   │   ├── greedy_markov.py
    │   │   └── A_star_beam.py
    │   │
    │   ├── data/          # Processed datasets (gitignored)
    │   └── results/      # Experiment outputs (gitignored)
    │
    ├── notebooks/          # Jupyter notebooks
    │   ├── colab_training.ipynb # Google Colab training
    │   └── theoretical_analysis.ipynb
    │
    ├── data/              # Training datasets (gitignored)
    ├── codesearchnet_data/ # Raw dataset files (gitignored)
    └── .gitignore         # Local testing artifacts ignore rules
```

## Data Flow

### Request Flow

```
┌─────────────┐
│   Browser   │  http://localhost:3000
└──────┬──────┘
       │ User enters Python code
       │
┌──────▼─────────────────┐
│  Frontend (Next.js)    │
│  - CodeInput.tsx       │
│  - Form validation     │
└──────┬─────────────────┘
       │ POST /generate
       │ JSON: {code, max_length, ...}
       │
┌──────▼─────────────────┐
│  Backend API (FastAPI) │
│  - api/server.py       │
│  - POST /generate      │
└──────┬─────────────────┘
       │
       ├─→ Code Processor
       │   - extract_code_tokens()
       │   Returns: ["sum", "a", "b"]
       │
       ├─→ Greedy Service ────→ greedy_markov.py
       │                        ↓
       │                   models/greedy_G0_base/
       │                        ↓
       │                   "calculate sum of numbers"
       │
       └─→ A* Service ──────→ A_star_beam.py
                               ↓
                          models/astar_A3_dependency/
                               ↓
                          "return the sum of two parameters"
       │
┌──────▼─────────────────┐
│  Response JSON         │
│  {                     │
│    greedy_comment,     │
│    astar_comment,      │
│    code_tokens,        │
│    runtimes...         │
│  }                     │
└──────┬─────────────────┘
       │
┌──────▼─────────────────┐
│  Frontend Display      │
│  - CommentDisplay.tsx  │
│  - Side-by-side view   │
└────────────────────────┘
```

### Model Loading Flow

```
Backend Startup (run_server.py)
    ↓
Import Services
    ├─→ GreedyService.__init__()
    │       ↓
    │   Check models/greedy_G0_base/
    │       ↓
    │   Load config.json, transitions.json
    │       ↓
    │   Initialize greedy_markov.Generator
    │
    └─→ AStarService.__init__()
            ↓
        Check models/astar_A3_dependency/
            ↓
        Load config.json, vectorizer.joblib, etc.
            ↓
        Initialize A_star_beam.Generator
            ↓
        Models loaded, ready to serve requests
```

## Key Files

### Setup & Configuration
- **`setup.sh` / `setup.bat`**: One-command setup for entire project
- **`test.py`**: Unified test runner (all test suites)
- **`backend/config.py`**: Centralized configuration (ports, URLs, model names)
- **`requirements.txt`**: Root Python dependencies
- **`backend/requirements.txt`**: Backend-specific dependencies

### Core Backend
- **`backend/api/server.py`**: FastAPI application, routes, CORS
- **`backend/api/models.py`**: Request/response schemas
- **`backend/services/code_processor.py`**: Code token extraction
- **`backend/services/greedy_service.py`**: Greedy algorithm wrapper
- **`backend/services/astar_service.py`**: A* algorithm wrapper

### Algorithms
- **`backend/algos/greedy_markov.py`**: Greedy Markov implementation
- **`backend/algos/A_star_beam.py`**: A* beam search implementation

### Frontend
- **`frontend/client/src/app/page.tsx`**: Main page component
- **`frontend/client/src/components/CodeInput.tsx`**: Code input form
- **`frontend/client/src/components/CommentDisplay.tsx`**: Results display

### Testing
- **`test.py`**: Unified test runner (use this!)
- Individual test files in `backend/tests/` (used by test.py)
- **`backend/tests/create_toy_greedy_model.py`**: Create toy model for development/testing
- **Note**: 
  - `backend/tests/` = Application tests (tracked, run with `python test.py`)
  - `testing/` = Research/experiments (scripts tracked, data/results gitignored)

## Entry Points

| What | Command | Location |
|------|---------|----------|
| **Setup project** | `bash setup.sh` or `setup.bat` | Root |
| **Run backend** | `cd backend && python run_server.py` | backend/ |
| **Run frontend** | `cd frontend/client && npm run dev` | frontend/client/ |
| **Run tests** | `python test.py` | Root |
| **Train models** | `cd testing/scripts && python run_experiments.py` | testing/scripts/ |

## Finding Things

- **API endpoints**: `backend/api/server.py`
- **Request models**: `backend/api/models.py`
- **Algorithm logic**: `backend/algos/`
- **Service wrappers**: `backend/services/`
- **Frontend components**: `frontend/client/src/components/`
- **Training scripts**: `testing/scripts/`
- **Test files**: `backend/tests/` (use `test.py` to run)
- **Configuration**: `backend/config.py` (ports, URLs, model paths)

## Conventions

- **Python**: Backend uses FastAPI, services pattern
- **TypeScript**: Frontend uses Next.js 13+ app router, React hooks
- **Models**: Stored in `backend/models/` (gitignored, add manually)
- **Configuration**: All settings in `backend/config.py` (ports, URLs, model paths)
- **Tests**: Use `python test.py` from root for unified testing
- **Setup**: Use `setup.sh` / `setup.bat` for one-command setup
- **Testing**: Application tests in `backend/tests/`, research in `testing/`
