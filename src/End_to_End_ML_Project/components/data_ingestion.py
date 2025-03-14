import os
import sys
from src.End_to_End_ML_Project.exception import CustomException
from src.End_to_End_ML_Project.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass 
class DataIngestionConfig:
    raw_data_path : str = os.path.join("raw_file", "data.csv")           

@dataclass 
class DataIngestionArtifact:                                             
    raw_data_path : str = os.path.join("raw_file", "data.csv")
    train_data_path : str = os.path.join("artifacts", "train.csv")    
    test_data_path : str = os.path.join("artifacts", "test.csv")   
    valid_data_path : str = os.path.join("artifacts", "valid.csv")    

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.artifact_config = DataIngestionArtifact()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            df = pd.read_csv("notebooks/data.csv")

            logging.info("Data has beed read")

            # creates the directory for the raw data file
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok = True)  

            # creates the directory for the training data file
            os.makedirs(os.path.dirname(self.artifact_config.train_data_path), exist_ok = True)

            #  writes the DataFrame to a CSV file at the path specified by raw_data_path
            df.to_csv(self.ingestion_config.raw_data_path, index = False)

            train_set, test_set = train_test_split(df, test_size= 0.2, random_state= 42)

            # writes the training set to a CSV file at the path specified by train_data_path
            train_set.to_csv(self.artifact_config.train_data_path)

            # writes the test set to a CSV file at the path specified by test_data_path
            test_set.to_csv(self.artifact_config.test_data_path)

            return(
                self.artifact_config.train_data_path,
                self.artifact_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)


