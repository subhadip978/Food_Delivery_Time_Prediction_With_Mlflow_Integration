
import logging
import pandas as pd

# from zenml import step


class IngestData:

	"""
	ingesting data from the data path
	"""

	def __init__(self,data_path:str):
		self.data_path=data_path


	def get_data(self):
		return pd.read_csv(self.data_path)




def ingest_df(data_path):
	try:
		ingest_data=IngestData(data_path)
		df=ingest_data.get_data()
		logging.info("data ingestion successfully completed")
		return df

	except Exception as e:
		logging.error("Error in ingest_df: ()".format(e))
		raise e