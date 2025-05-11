import os
import pandas as pd
from src.logger import get_logger
from src.custom_exception import custom_exception
import yaml
import sys
import pandas as pd

logger = get_logger(__name__)



def read_yalm_file(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info("=========================================================")
            logger.info("YAML file read successfully.")
            return config
    except Exception as e:
        logger.error("Error occurred while reading the YAML file.")
        raise custom_exception("Custom exception in read_yaml_file", e)

def load_data(file_path):
    try:
        logger.info("Loading data from file: %s", file_path)
        return pd.read_csv(file_path)
    except FileNotFoundError as e:
        logger.error("File not found: %s", file_path)
        raise custom_exception("Custom exception in load_data", e)