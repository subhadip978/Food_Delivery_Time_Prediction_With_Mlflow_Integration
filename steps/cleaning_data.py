import logging
import pandas as pd
# from zenml import step
from src.data_cleaning import  DataPreProcessStrategy,  CategoricalEncodingStrategy , DataSplittingStrategy

from typing import Tuple
from typing_extensions import Annotated


def cleaning_df(data:pd.DataFrame)-> Tuple[
Annotated[pd.DataFrame,"X_train"],
Annotated[pd.DataFrame,"X_test"],
Annotated[pd.Series,"y_train"],
Annotated[pd.Series,"y_test"],

]:
	try:
		process_strategy=DataPreProcessStrategy()
		clean_data=process_strategy.handle_data(data)
		logging.info(clean_data.head())
		encode_strategy=CategoricalEncodingStrategy()
		encoded_data=encode_strategy.handle_data(clean_data)

		splitting_strategy=DataSplittingStrategy()
		X_train, X_test, y_train, y_test=splitting_strategy.handle_data(encoded_data)
		logging.info(y_train.shape)
		logging.info(X_train.shape)
		logging.info(y_test.shape)
		logging.info(X_test.shape)
		return X_train,X_test,y_train,y_test
		logging.info("data cleaning sucessfully completed")

	except Exception as e:
		logging.error("error in cleaning data:{}".format(e))
		raise e