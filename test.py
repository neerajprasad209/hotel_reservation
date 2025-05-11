from src.logger import get_logger
from src.custom_exception import coustum_exception

import sys

logger = get_logger(__name__)

def divide(a, b):
    try:
        return a / b
        logger.info("Division operation completed successfully.")
        return result
    except Exception as e:
        logger.error("An error occurred during division operation.")
        raise coustum_exception("Custom exception zero division", sys)

if __name__ == "__main__":
    try:
        result = divide(10, 5)
        print(f"Result: {result}")
    except coustum_exception as e:
        logger.error(str(e))  # Log the exception message (str(e)
