import os
from loguru import logger

BASE_DIR = os.getenv("BASE_DIR")
logger.info(BASE_DIR)

SAMPLE_SIZE = 400
N_FEATURES = 3
RESULTS_FILE = os.path.join(BASE_DIR, "overall_results.csv")
DATA_LIST = ["Credit Risk"]
KERNEL_USE = True

BOOTSTRAP = False
N_BOOTSTRAP = 500
N_SAMPELS = SAMPLE_SIZE

