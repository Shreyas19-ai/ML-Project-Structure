import os
import sys
from src.End_to_End_ML_Project.exception import CustomException
from src.End_to_End_ML_Project.logger import logging
import pandas as pd
import numpy as np  
import pickle

def save_object(file_path, obj):
    '''
    this function is responsible for saving object
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
            
        logging.info(f"object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)