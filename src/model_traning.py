import joblib
import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 
from config.path_config import *
from config.model_params import *
from src.logger import logging
from src.custom_exception import custom_exception
from utils.common_function import read_yalm_file, load_data
from scipy.stats import randint, uniform

import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)

class ModelTraining:
    def __init__(self, train_file_path, test_file_path, model_dir):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.model_dir = model_dir
        
        self.param_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SERCH_PARAMS
        
    def load_and_split_data(self):
        try:
            # Load the data
            logger.info(f"Loading train data from {self.train_file_path}")
            train_data = pd.read_csv(self.train_file_path)
            
            logger.info(f"Loading test data from {self.test_file_path}")
            test_data = pd.read_csv(self.test_file_path)
            
            logger.info("Train and test data loaded successfully.")
            
            # Split the data into features and target
            X_train = train_data.drop(columns=['booking_status'])
            y_train = train_data['booking_status']
            
            X_test = test_data.drop(columns=['booking_status'])
            y_test = test_data['booking_status']
            
            logger.info("Data split into features and target.")
            
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error in loading and splitting data: {str(e)}")
            raise custom_exception(f"Error in loading and splitting data: {str(e)}")
        
    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Starting LightGBM model training.")
            
            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params['random_state'])
            
            logger.info("Starting RandomizedSearchCV for hyperparameter tuning.")
            
            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.param_dist,
                n_iter=self.random_search_params['n_iter'],
                cv=self.random_search_params['cv'],
                n_jobs=self.random_search_params['n_jobs'],
                verbose=self.random_search_params['verbose'],
                random_state=self.random_search_params['random_state'],
                scoring=self.random_search_params['scoring']
            )
            
            logger.info("Starting our Hyperparameter Training.")
            
            random_search.fit(X_train, y_train)
            logger.info("Hyperparameter tuning completed.")
            
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            
            logger.info(f"Best parameters found: {best_params}")
            
            
            return best_lgbm_model
        
        except Exception as e:
            logger.error(f"Error in training LightGBM model: {str(e)}")
            raise custom_exception(f"Error in training LightGBM model: {str(e)}")
        
    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Starting model evaluation.")
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            logger.info(f"Model evaluation completed. Accuracy: {accuracy}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}")
            
            return {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        except Exception as e:
            logger.error(f"Failed to Evaluate model : {str(e)}")
            raise custom_exception(f"Error in model evaluation: {str(e)}")
        
    def save_model(self, model):
        try:
            logger.info(f"Saving model to {self.model_dir}")
            os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)
            
            joblib.dump(model, self.model_dir)
            logger.info("Model saved successfully.")
        except Exception as e:
            logger.error(f"Error in saving model: {str(e)}")
            raise custom_exception(f"Failed to saving model: {str(e)}")
        
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting model training process.")
                logger.info("Logging the Tranning and text dataset to MLflow")
                
                mlflow.log_artifact(self.train_file_path, artifact_path='datasets')
                mlflow.log_artifact(self.test_file_path, artifact_path='datasets')
                
                # Load and split data
                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
                self.save_model(best_lgbm_model)
                
                mlflow.log_artifact(self.model_dir)
                
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)
                
                
                logger.info("Model training process completed successfully.")
                
            return metrics
        except Exception as e:
            logger.error(f"Error in model training process: {str(e)}")
            raise custom_exception(f"Failed during model training process: {str(e)}")
        
if __name__ == "__main__":
    # Initialize the ModelTraining class
    model_trainer = ModelTraining(PROCESSED_TRAIN_FILE_PATH, PROCESSED_TEST_FILE_PATH, MODEL_DIR)
    
    # Run the model training process
    model_trainer.run()