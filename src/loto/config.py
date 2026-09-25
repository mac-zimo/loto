"""
Configuration centralisée du projet Loto Analyze.
"""

from pathlib import Path

# Runtime defaults are relative to the invocation directory.  This preserves the
# repository-root workflow without ever treating an installed package directory
# as writable storage.
PROJECT_ROOT = Path.cwd()
SRC_DIR = Path(__file__).parent
DATA_DIR = PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "output"

# CSV source files (relative to project root)
CSV_FILES = [
    "nouveau_loto.csv",
    "loto2017.csv",
    "loto_201902.csv",
    "loto_201911.csv",
]

# Database path
DB_PATH = PROJECT_ROOT / "data" / "loto_analyze.db"

# Loto game rules
NUM_BALLS = 5          # Number of main numbers drawn
MAX_BALL = 49          # Max number on a ball
NUM_CHANCE_MAX = 10    # Max chance number

# Statistics windows for analysis
FREQUENCY_WINDOWS = [30, 90, 182, 365]  # in draws

# Correlation thresholds
CORR_STRONG_THRESHOLD = 0.15  # absolute correlation considered "strong"

# Strategy simulation params
SIMULATION_ROUNDS = 100_000
SIMULATION_SEED = 42
