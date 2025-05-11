from src.custom_exception import custom_exception
import os
import pandas as pd
import numpy as np
from config.path_config import *
from utils.common_function import read_yalm_file, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from src.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessing:
    def __init__(self, train_file_path, test_file_path, process_dir, config_file_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.process_dir = process_dir
        
        self.config_file_path = read_yalm_file(config_file_path)
        
        if not os.path.exists(self.process_dir):
            os.makedirs(self.process_dir)
        
    def process_data(self, df):
        try:
            logger.info("Start Processing data...")
            # Drop duplicates
            logger.info("Dropping the columns...")
            df.drop(columns=['Booking_ID'], inplace=True)
            
            logger.info("Dropping the duplicates...")
            df.drop_duplicates(inplace=True)
            
            # list of categorical columns
            categorical_columns = self.config_file_path['data_processing']['categorical_features']
            numeric_columns = self.config_file_path['data_processing']['numerical_features']
            
            logger.info("This is my categorical columns")
            logger.info(categorical_columns)
            
            logger.info("This is my numeric columns")
            logger.info(numeric_columns)
            
            
            # Doning Lable Encoding
            logger.info("Applying label encoding to categorical columns")
            label_encoder = LabelEncoder()
            mappinged_columns = {}
            for column in categorical_columns:
                df[column] = label_encoder.fit_transform(df[column])
                mappinged_columns[column] = {label:code for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
            
            logger.info("Label encoding applied successfully")
            for column, mapping in mappinged_columns.items():
                logger.info(f"Mapping for {column}: {mapping}")
            
            # Apply SMOTE for oversampling
            logger.info("Handling Skewness...")
            skewness_threshold = self.config_file_path['data_processing']['skewness_threshold']
            skewed_features = df[numeric_columns].apply(lambda x: x.skew())
            for column in skewed_features[skewed_features > skewness_threshold].index:
                df[column] = np.log1p(df[column])
                
            # Return DataFrame
            return df
        
        except Exception as e:
            logger.error("Error occurred during data processing.")
            raise custom_exception("Custom exception in process_data", e)
        
    def balance_data(self, df):
        try:
            logger.info("Handling Imbalance data...")
            # Separate features and target variable
            X = df.drop(columns=['booking_status'])
            y = df['booking_status']
            
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df['booking_status'] = y_resampled
            
            logger.info("Data balancing completed successfully.")
            return balanced_df
        
        except Exception as e:
            logger.error("Error occurred during data balancing.")
            raise custom_exception("Custom exception in balance_data", e)
        
    def select_features(self, df):
        try:
            logger.info("Feature Selection...")
            # Load the model
            model = RandomForestClassifier(random_state=42)
            
            # Separate features and target variable
            X = df.drop(columns=['booking_status'])
            y = df['booking_status']
            
            # Fit the model
            model.fit(X, y)
            
            feature_importances = model.feature_importances_
            
            feature_importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
            
            top_features_importance_df =feature_importances_df.sort_values(by='Importance', ascending=False)
            
            num_features = self.config_file_path['data_processing']['no_of_features_to_select']
            
            top_10_features = top_features_importance_df['Feature'].head(num_features).values
            top_10_df = df[top_10_features.tolist() + ['booking_status']]
            
            logger.info("Top 10 features selected:")
            logger.info(top_10_features)
            logger.info("Feature selection completed successfully.")
            
            return top_10_df
        
        except Exception as e:
            logger.error("Error occurred during feature selection.")
            raise custom_exception("Custom exception in select_features", e)
        
    def save_processed_data(self, df, file_path):
        try:
            logger.info("Saving processed data to %s", file_path)
            df.to_csv(file_path, index=False)
            logger.info("Processed data saved successfully.")
        except Exception as e:
            logger.error("Error occurred while saving processed data.")
            raise custom_exception("Custom exception in save_processed_data", e)
        
    def process(self):
        try:
            # Load the data
            logger.info("Loading data from RAW_FILE_PATH")
            train_df = load_data(self.train_file_path)
            test_df = load_data(self.test_file_path)
            
            train_df = self.process_data(train_df)
            test_df = self.process_data(test_df)
            
            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)
            
            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]
            
            
            # Save the processed data
            self.save_processed_data(train_df, PROCESSED_TRAIN_FILE_PATH)
            self.save_processed_data(test_df, PROCESSED_TEST_FILE_PATH)
            
            logger.info("Data processing completed successfully.")
        
        except Exception as e:
            logger.error("Error occurred during data processing.")
            raise custom_exception("Custom exception in process", e)
        
if __name__ == "__main__":
    data_preprocessor = DataPreprocessing(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_FILE_PATH)
    data_preprocessor.process()