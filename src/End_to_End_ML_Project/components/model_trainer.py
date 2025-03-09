import os
import sys
import numpy as np
import pandas as pd

from src.End_to_End_ML_Project.exception import CustomException
from src.End_to_End_ML_Project.logger import logging
from src.End_to_End_ML_Project.utils import save_object
from src.End_to_End_ML_Project.utils import evaluate_models
from dataclasses import dataclass

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

@dataclass
class ModelTrainerConfig:
    train_model_file = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_train_model(self, train_array : np.ndarray, test_array : np.ndarray):
        try:
            logging.info("split the data into training and testing sets")
            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1]
            )

            models = {
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'XGBoost': XGBRegressor(),
                'CatBoost': CatBoostRegressor(verbose=False)
            }

            params = {
                'Lasso': {
                    'alpha': [0.1, 1.0, 10.0]
                },
                'Ridge': {
                    'alpha': [0.1, 1.0, 10.0]
                },
                'Decision Tree': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                'Random Forest': {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'Gradient Boosting': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'AdaBoost': {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'XGBoost': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'CatBoost': {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                }
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_name = max(model_report, key=model_report.get)
            logging.info(f"best model: {best_model_name}")

            best_model_score = model_report[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("Model is not performing well", sys)
            logging.info("No best model found")

            best_model = models[best_model_name]

            save_object(
                file_path = self.model_trainer_config.train_model_file,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_score_val = r2_score(y_test, predicted)
            return r2_score_val

        except Exception as e:
            raise CustomException(e, sys)