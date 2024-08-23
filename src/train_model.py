import logging
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor




class Model(ABC):
	@abstractmethod
	def train(self,X_train,y_train):
		pass


class LinearRegressionModel(Model):

	def train(self, X_train,y_train,**kwargs):
		try:
			reg=LinearRegression(**kwargs)
			reg.fit(X_train,y_train)
			logging.info("model train completed")
			return reg

		except Exception as e:
			logging.error("error in LinearRegression model :{}".format(e))
			raise e


class RandomForestModel(Model):
    """
    RandomForestModel that implements the Model interface.
    """

    def train(self, X_train, y_train, **kwargs):
        try:
           reg = RandomForestRegressor(**kwargs)
           reg.fit(X_train, y_train)
           logging.info("model training completed")
           return reg
        except Exception as e:
           logging.error("error in randomforest model:{}".format(e))
           raise e