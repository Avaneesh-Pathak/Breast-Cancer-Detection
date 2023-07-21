import sys
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from src.utils import save_obj
from src.exception import CustomException
from src.logger import logging
import os

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = load_breast_cancer().feature_names
            target_column_name = "target"

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self):
        try:
            # Load the breast cancer dataset from sklearn
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["target"] = data.target

            logging.info("Read data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "target"
            numerical_columns = data.feature_names

            input_feature_df = df.drop(columns=[target_column_name])
            target_feature_df = df[target_column_name]

            logging.info(
                f"Applying preprocessing object on the dataframe."
            )

            input_feature_arr = preprocessing_obj.fit_transform(input_feature_df)

            train_arr, test_arr, train_target, test_target = train_test_split(
                input_feature_arr, target_feature_df, test_size=0.2, random_state=42
            )

            logging.info(f"Saved preprocessing object.")

            # Save preprocessing_obj using the save_obj function if available
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                train_target,
                test_target,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
