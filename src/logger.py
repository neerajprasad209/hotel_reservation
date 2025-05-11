import logging
import os
import sys
from datetime import datetime

# Create logs directory
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log")

# Set up logging configuration
logging.basicConfig(
    filename=LOG_FILE,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    return logger