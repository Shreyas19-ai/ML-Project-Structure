from src.End_to_End_ML_Project import logger
from src.End_to_End_ML_Project.logger import logging     # either of both we can use

from src.End_to_End_ML_Project.exception import CustomException
import sys
import os

from src.End_to_End_ML_Project.components.data_ingestion import DataIngestionConfig
from src.End_to_End_ML_Project.components.data_ingestion import DataIngestionArtifact
from src.End_to_End_ML_Project.components.data_ingestion import DataIngestion
from src.End_to_End_ML_Project.components.data_transformation import DataTransformationConfig
from src.End_to_End_ML_Project.components.data_transformation import DataTransformationArtifact
from src.End_to_End_ML_Project.components.data_transformation import DataTransformation


if __name__ == "__main__":
    try:

        data_ingestion = DataIngestion()
        train_data_path , test_data_path = data_ingestion.initiate_data_ingestion()
    
        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    except Exception as e:
        raise CustomException(e, sys)

    data_ingestion = DataIngestion()
    train_data , test_data = data_ingestion.initiate_data_ingestion()
    