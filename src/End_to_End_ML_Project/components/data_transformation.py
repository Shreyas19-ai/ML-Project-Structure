import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.End_to_End_ML_Project.exception import CustomException
from src.End_to_End_ML_Project.logger import logging
from src.End_to_End_ML_Project.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

@dataclass
class DataTransformationArtifact:
    transformed_train_data_path: str = os.path.join('artifacts', 'transformed_train_data.npy')
    transformed_test_data_path: str = os.path.join('artifacts', 'transformed_test_data.npy')
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.artifact_config = DataTransformationArtifact()

    def get_data_transformation_object(self, df):
        '''
        This function is responsible for data transformation
        '''
        try:
            # Exclude the target column 'math_score' from the features
            num_features = df.select_dtypes(exclude='object').drop(columns=['math_score']).columns.tolist()
            cat_features = df.select_dtypes(include='object').columns.tolist()

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('std_scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("onehot", OneHotEncoder()),
                ("std_scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"categorical features: {cat_features}")
            logging.info(f"numerical features: {num_features}")

            preprocessor = ColumnTransformer([
                ('num', num_pipeline, num_features),
                ('cat', cat_pipeline, cat_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function is responsible for initiating data transformation
        '''
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"train_df shape: {train_df.shape}")
            logging.info(f"test_df shape: {test_df.shape}")

            # Drop the 'Unnamed: 0' column if it exists
            if 'Unnamed: 0' in train_df.columns:
                train_df = train_df.drop(columns=['Unnamed: 0'])
            if 'Unnamed: 0' in test_df.columns:
                test_df = test_df.drop(columns=['Unnamed: 0'])

            # Print column names to debug
            logging.info(f"train_df columns: {train_df.columns}")
            logging.info(f"test_df columns: {test_df.columns}")

            '''
            The returned preprocessing_obj is a ColumnTransformer 
            that will later be applied to both X_train and X_test.
            This only creates a preprocessing pipeline but does not apply it yet.
            '''
            preprocessing_obj = self.get_data_transformation_object(train_df)

            '''
            splitting the data into features and target
            train and test df are read in the start of the function
            '''
            X_train = train_df.drop('math_score', axis=1)
            y_train = train_df['math_score']

            X_test = test_df.drop('math_score', axis=1)
            y_test = test_df['math_score']

            logging.info(f"X_train shape: {X_train.shape}")
            logging.info(f"y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}")
            logging.info(f"y_test shape: {y_test.shape}")

            '''
            the returned pipeline object is used to transform the data
            '''
            X_train = preprocessing_obj.fit_transform(X_train)
            X_test = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            logging.info(f"train_arr shape: {train_arr.shape}")
            logging.info(f"test_arr shape: {test_arr.shape}")

            '''
            save the preprocessor object(pipeline) as pkl 
            for new data during predictions
            '''
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            '''
            save the transformed data as numpy array
            to the path specified in the config and artifact dataclassess
            '''
            np.save(self.artifact_config.transformed_train_data_path, train_arr)
            np.save(self.artifact_config.transformed_test_data_path, test_arr)

            return (
                self.artifact_config.transformed_train_data_path,
                self.artifact_config.transformed_test_data_path,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)