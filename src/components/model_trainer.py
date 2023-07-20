import sys
import os
import numpy as np
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
from src.utils import evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path =os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                
                
            )
            y_train_non_negative = np.maximum(0, y_train)
            model = {
             "Random Forest": RandomForestRegressor(),
             "Linear Regression": LinearRegression(),
             "Decision Tree": DecisionTreeRegressor(),
             "Gradient Boosting": GradientBoostingRegressor(),
             "XGBRegressor": XGBRegressor(),
             "CatBoosting Classifier": CatBoostRegressor(verbose=False, objective='Poisson'),
             "AdaBoost Classifier": AdaBoostRegressor(),
            }              

            
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Classifier":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Classifier":{
                    'learning_rate':[.1,.01,0.5,.001], 
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train_non_negative,X_test=X_test,y_test=y_test,models=model,params=params)
            
            
            ##To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            ##To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            
            best_model = model[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model Found")
            
            logging.info("Best model found on both training and testing dataset")
            
            
            
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
               
               
            )
            
            predicted = best_model.predict(X_test)
            
            R2_score = r2_score(y_test,predicted)
            return R2_score
        
        
        
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
     