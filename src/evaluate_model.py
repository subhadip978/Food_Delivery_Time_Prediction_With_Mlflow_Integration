import logging
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from abc import ABC, abstractmethod

class Evaluation(ABC):

	@abstractmethod
	def calculate_scores(self,y_true:np.ndarray, y_pred:np.ndarray):
		pass


class MSE(Evaluation):

	def calculate_scores(self,y_true,y_pred):
		try:
			logging.info("calculate mse")
			mse=mean_squared_error(y_true,y_pred)
			logging.info(f"mse: {mse}")
			return mse
		except Exception as e:
			logging.error("error in calculating mse")
			raise e


class R2(Evaluation):

	def calculate_scores(self,y_true,y_pred):
		try:
			logging.info("calculate R2")
			r2=r2_score(y_true,y_pred)
			logging.info(f"r2_scroe is : {r2}")
			
			print(f"r2_score absolute error :{r2}")
			return r2
		except Exception as e:
			logging.error("error in calculating r2_score")
			raise e




class RMSE(Evaluation):

	def calculate_scores(self,y_true,y_pred):
		try:
			logging.info("calculate RMSE")
			rmse=np.sqrt(mean_squared_error(y_true,y_pred))
			logging.info(f"rmse is :{rmse}")
			return rmse

		except  Exception as e:
			logging.error("error in calculating rmse")
			raise e