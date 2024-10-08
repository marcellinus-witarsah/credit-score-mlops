"""
A module for configuration.
"""

from pathlib import Path

from dotenv import load_dotenv

from credit_score_mlops.utils import logger, read_yaml

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Read parameters
GLOBAL_API = read_yaml(Path("api.yaml"))
CREDIT_SCORE_SCALING_PARAMS = GLOBAL_API.credit_score_scaling
GLOBAL_PARAMS = read_yaml(Path("params.yaml"))
DATA_PREPROCESSING_PARAMS = GLOBAL_PARAMS.data_preprocessing
TRAIN_PARAMS = GLOBAL_PARAMS.train
MLFLOW_PARAMS = GLOBAL_PARAMS.mlflow
