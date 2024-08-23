import logging
import mlflow
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.base import RegressorMixin
from src.train_model import LinearRegressionModel
from src.train_model import RandomForestModel
from .config import ModelNameConfig
from src.utils import save_object

from urllib.parse import urlparse
@dataclass
class ModelTrainerConfig:
	trained_model_file_path=os.path.join("artifacts","model.pkl")


def train_model(X_train:pd.DataFrame,
				X_test:pd.DataFrame,
				y_train:pd.Series,
				y_test:pd.Series,
				)->RegressorMixin:

	try:			

		model=None
		

		with mlflow.start_run():
			if ModelNameConfig.model_name=="linearregression" :
					
				model=LinearRegressionModel()


			elif ModelNameConfig.model_name=="randomforest":
				
				model=RandomForestModel()


			else:
				raise ValueError(f"model {config.model_name} not supported")

			
		trained_model=model.train(X_train,y_train)
		logging.info("model was successfully trained")
		save_object(file_path=ModelTrainerConfig.trained_model_file_path,obj=trained_model)
		tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
		if tracking_url_type_store != "file":
			mlflow.sklearn.log_model(
                    trained_model,
                    "model_name",
                    registered_model_name=ModelNameConfig.model_name
                )
		else:
			print("in local file system")
			mlflow.sklearn.log_model(trained_model, "model_name")
		
		return trained_model




	except Exception as e:
		logging.error(e)
		raise e