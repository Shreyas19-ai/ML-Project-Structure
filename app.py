from src.End_to_End_ML_Project import logger
from src.End_to_End_ML_Project.logger import logging     # either of both we can use

from src.End_to_End_ML_Project.exception import CustomException
import sys
import os

from src.End_to_End_ML_Project.components.data_ingestion import DataIngestionConfig
from src.End_to_End_ML_Project.components.data_ingestion import DataIngestionArtifact
from src.End_to_End_ML_Project.components.data_ingestion import DataIngestion

if __name__ == "__main__":

    data_ingestion = DataIngestion()
    train_data , test_data = data_ingestion.initiate_data_ingestion()
    