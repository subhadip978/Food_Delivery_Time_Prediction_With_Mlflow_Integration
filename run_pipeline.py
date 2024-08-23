from pipelines.training_pipeline import train_pipeline
from urllib.parse import urlparse
import mlflow
if __name__=="__main__":
		
	print("Tracking URI:", urlparse(mlflow.get_tracking_uri()).scheme)
	train_pipeline(data_path="./data/train.csv")

# 