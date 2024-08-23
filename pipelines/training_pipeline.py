# from zenml import pipeline

from steps.ingest_data import ingest_df

from steps.cleaning_data  import cleaning_df

from steps.model_train import train_model

from steps.model_evaluation import evaluate_model


def train_pipeline(data_path):
	data=ingest_df(data_path)
	X_train,X_test,y_train,y_test=cleaning_df(data)
	model=train_model(X_train,X_test,y_train,y_test)
	r2,rmse=evaluate_model(model,X_test,y_test)