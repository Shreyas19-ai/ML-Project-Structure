from src.End_to_End_ML_Project import logger
from src.End_to_End_ML_Project.logger import logging     # either of both we can use

from src.End_to_End_ML_Project.exception import CustomException
import sys
import os
import numpy as np
import pandas as pd
import pickle
from Flask import Flask, render_template, request

from src.End_to_End_ML_Project.components.data_ingestion import DataIngestionConfig
from src.End_to_End_ML_Project.components.data_ingestion import DataIngestionArtifact
from src.End_to_End_ML_Project.components.data_ingestion import DataIngestion
from src.End_to_End_ML_Project.components.data_transformation import DataTransformationConfig
from src.End_to_End_ML_Project.components.data_transformation import DataTransformationArtifact
from src.End_to_End_ML_Project.components.data_transformation import DataTransformation
from src.End_to_End_ML_Project.components.model_trainer import ModelTrainerConfig
from src.End_to_End_ML_Project.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:

        data_ingestion = DataIngestion()
        train_data_path , test_data_path = data_ingestion.initiate_data_ingestion()
    
        data_transformation = DataTransformation()
        transformed_train_data_path, transformed_test_data_path,_ = data_transformation.initiate_data_transformation(train_data_path,
                                                                                                                      test_data_path)
    
        train_array = np.load(transformed_train_data_path)
        test_array = np.load(transformed_test_data_path)

        model_trainer = ModelTrainer()        
        print(model_trainer.initiate_train_model(train_array, test_array))


    except Exception as e:
        raise CustomException(e, sys)

    