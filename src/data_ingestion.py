import os
import pandas as pd
from google.cloud import storage
from src.logger import get_logger
from src.custom_exception import custom_exception
import yaml
from sklearn.model_selection import train_test_split
from config.path_config import *
from utils.common_function import read_yalm_file
import sys

logger = get_logger(__name__)


class Data_ingestion:
    def __init__(self, config):
        self.config = config['data_ingestion']
        self.bucket_name = self.config['bucket_name']
        self.file_name = self.config['bucket_file_name']
        self.train_ratio = self.config['train_ratio']
        self.test_ratio = self.config['test_ratio']
        
        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info("Data Ingestion started with {self.bucket_name} and file name {self.file_name}")
        
    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            
            blob.download_to_filename(RAW_FILE_PATH)
            logger.info("CSV file downloaded successfully from GCP to {RAW_FILE_PATH}.")
        except Exception as e:
            logger.error("Error occurred while downloading the CSV file from GCP.")
            raise custom_exception("Failed to download CSV file from GCP", e)
    
    def split_data(self):
        try:
            logger.info("Splitting data into train and test sets")
            data = pd.read_csv(RAW_FILE_PATH)
            
            train_data, test_data = train_test_split(data, test_size=1- self.test_ratio, random_state=42)
            
            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)
            
            logger.info("Data split successfully.")
            logger.info(f"Train data saved to {TRAIN_FILE_PATH}.")
            logger.info(f"Test data saved to {TEST_FILE_PATH}.")
        except Exception as e:
            logger.error("Error occurred while splitting the data.")
            raise custom_exception("Failed to split data", e)
        
    def run(self):
        try:
            logger.info("Starting data ingestion.")
            self.download_csv_from_gcp()
            self.split_data()
            logger.info("Data ingestion completed successfully.")
        except custom_exception as e:
            logger.error(f"Data ingestion failed: {str(e)}")
        
        finally:
            logger.info("Data ingestion process finished.")
            
if __name__ == "__main__":
    try:
        config = read_yalm_file(CONFIG_FILE_PATH)
        data_ingestion = Data_ingestion(config)
        data_ingestion.run()
    except custom_exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        sys.exit(1)