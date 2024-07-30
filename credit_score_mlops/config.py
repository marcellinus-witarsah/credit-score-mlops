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
CONFIG = read_yaml(PROJ_ROOT / "credit_score_mlops/config.yaml")
DATA_PREPROCESSING = CONFIG.data_preprocessing
TRAIN_CONFIG = CONFIG.train
EVALUATE_CONFIG = CONFIG.evaluate
INFERENCE_CONFIG = CONFIG.inference
