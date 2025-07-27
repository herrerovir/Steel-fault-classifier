from pathlib import Path

# === Project Root ===
ROOT_DIR = Path(__file__).resolve().parents[1]  # Assumes: /src/config.py

# === Data Directories ===
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_ENG_DIR = DATA_DIR / "feature-engineered"

# === Figures Directory ===
FIGURES_DIR = ROOT_DIR / "figures"

# === Model Directory ===
MODEL_DIR = ROOT_DIR / "models"

# === Notebooks Directory ===
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# === Results Directories ===
RESULTS_DIR = ROOT_DIR / "results"
METRICS_RESULTS_DIR = RESULTS_DIR / "metrics"


# === Ensure all required directories exist ===
for path in [
    # Data
    RAW_DIR,
    PROCESSED_DIR,
    FEATURE_ENG_DIR,

    # Figures
    FIGURES_DIR,
    
    # Models
    MODEL_DIR,
    
    # Notebooks
    NOTEBOOKS_DIR,

    # Results
    RESULTS_DIR,
    METRICS_RESULTS_DIR   
]:
    path.mkdir(parents = True, exist_ok = True)