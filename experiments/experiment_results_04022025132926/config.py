import os
from loguru import logger

BASE_DIR = os.getenv("BASE_DIR")
logger.info(BASE_DIR)

SAMPLE_SIZE = 1000
N_FEATURES = 4
RESULTS_FILE = os.path.join(BASE_DIR, "overall_results.csv")
DATA_LIST = ["Credit Risk"] # "Credit Risk", "Bank Marketing", "Loan Status"