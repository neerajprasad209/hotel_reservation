from src.data_ingestion import Data_ingestion
from src.data_preprocessing import DataPreprocessing
from src.model_traning import ModelTraining
from config.path_config import *
from utils.common_function import read_yalm_file, load_data
import os




if __name__ == "__main__":
    
    ## 1. Data Ingestion
    config = read_yalm_file(CONFIG_FILE_PATH)
    data_ingestion = Data_ingestion(config)
    data_ingestion.run()
    
    ## 2. Data Preprocessing
    data_preprocessor = DataPreprocessing(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_FILE_PATH)
    data_preprocessor.process()
    
    ## 3. Model Training
    model_trainer = ModelTraining(PROCESSED_TRAIN_FILE_PATH, PROCESSED_TEST_FILE_PATH, MODEL_DIR)
    model_trainer.run()