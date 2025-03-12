import pandas as pd
import numpy as np
import sys
import os

from src.End_to_End_ML_Project.exception import CustomException
from src.End_to_End_ML_Project.logger import logging
from src.End_to_End_ML_Project.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        '''
        This method is responsible for prediction of new data points
        first the data will be transformed then it will be ready for prediction
        for that we have to load the preprocessor and model object
        processor object is the pipeline object, and we will transform the new data through this pipeline object
        '''
        try:
            model_path = 'artifacts/model.pkl'
            model = load_object(model_path)
            preprocessor_path = 'artifacts/preprocessor.pkl'
            preprocessor = load_object(preprocessor_path)
            scaled_features = preprocessor.transform(features)
            model_prediction = model.predict(scaled_features)
            return model_prediction
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, gender : str,
                       race_ethnicity : str,
                       parental_level_of_education ,
                       lunch : str,
                       test_preparation_course : str,
                       reading_score : int,
                       writing_score : int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_frame(self):
        '''
        This function is responsible for mapping the input into the DataFrame
        When the user will input the data, it will be mapped into the DataFrame by calling this method
        After conversion into dataframe, above prediction method will be called,
        which will do the prediction on the new data
        '''
        try:
            custom_data_input = {
                "gender" : [self.gender],
                "race_ethnicity" : [self.race_ethnicity],
                "parental_level_of_education" : [self.parental_level_of_education],
                "lunch" : [self.lunch],
                "test_preparation_course" : [self.test_preparation_course],
                "reading_score" : [self.reading_score],
                "writing_score" : [self.writing_score]
            }
            
            return pd.DataFrame(custom_data_input)
        
        except Exception as e:
            raise CustomException(e, sys)

