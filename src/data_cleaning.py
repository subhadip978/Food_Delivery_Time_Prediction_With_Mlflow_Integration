import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

import math
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



class DataStrategy(ABC):

	@abstractmethod
	def handle_data(self,data:pd.DataFrame)->pd.DataFrame:
		pass



class DataPreProcessStrategy(DataStrategy):
	def handle_data(self,data):
		try:
			
			data.replace('NaN', float(np.nan),regex=True, inplace=True)

			
			
			data.drop(['ID',
			'Delivery_person_ID',
			'Delivery_person_Age','Festival'],axis=1, inplace=True)


			data.replace('NaN', float(np.nan),regex=True, inplace=True)
		
			data['Delivery_person_Ratings'] = data['Delivery_person_Ratings'].astype("float64")
			data["multiple_deliveries"]=data["multiple_deliveries"].astype("float64")



			data['Delivery_person_Ratings'].fillna(data['Delivery_person_Ratings'].median(), inplace=True)
			data['multiple_deliveries'].fillna(data['multiple_deliveries'].mode()[0], inplace=True)
			data['City'].fillna(data['City'].mode()[0],inplace=True)
			# data['Festival'].fillna(data['Festival'].mode()[0],inplace=True)

# Earth's radius


			def distance(lat1,lon1,lat2,lon2):
				R=6371
				lat1,lon1,lat2,lon2=map(math.radians,[lat1,lon1,lat2,lon2])


				# Haversine formula
				dlat = lat2 - lat1
				dlon = lon2 - lon1
				a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
				c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
				return R * c


			data['total_distance']=data.apply(lambda row: distance(row['Restaurant_latitude'], row['Restaurant_longitude'],row['Delivery_location_latitude'], row['Delivery_location_longitude']),axis=1)
		
			data.drop(['Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude'],axis=1,inplace=True)
			data['Time_taken(min)'] = data['Time_taken(min)'].str.replace('(min)', '').str.strip()
			data['Time_taken(min)']= data['Time_taken(min)'].astype("int32")
		


			return data

		except Exception as e:
			logging.error("error in preprocessing data:{}".format(e))
			raise e



class CategoricalEncodingStrategy(DataStrategy):
	def handle_data(self,data):

		try:
			# data['Order_Date']=pd.to_datetime(data['Order_Date'])
			# data['year']=data['Order_Date'].dt.year
			# data['month']=data['Order_Date'].dt.month
			# data['Day']=data['Order_Date'].dt.day

			data.dropna(subset=["Time_Orderd"],inplace=True)
			# data['Time_Orderd'] = pd.to_datetime(data['Time_Orderd'],format='%H:%M:%S')
			# data['Hour'] = data['Time_Orderd'].dt.hour.astype("float64")
			# data['Minute'] = data['Time_Orderd'].dt.minute.astype("float64")    

			# data['Time_Order_picked']=pd.to_datetime(data["Time_Order_picked"])
			# data["Time_Order_picked_Hour"]=data["Time_Order_picked"].dt.hour.astype('float64')
			# data["Time_Order_picked_Minute"]=data["Time_Order_picked"].dt.minute.astype('float64')
	
			data.drop(["Time_Order_picked","Time_Orderd","Order_Date"], axis=1, inplace=True)

			categorical_cols=['Weatherconditions', 'Road_traffic_density', 'Type_of_order',
       		'Type_of_vehicle',  'City']

			labelencode=LabelEncoder()

			for col in categorical_cols:
			 	data[col]=labelencode.fit_transform(data[col])
			print(data.columns);
			return data





		except Exception as e:
			logging.info("Error in categorical encoding:{}".format(e))
			raise e




class DataSplittingStrategy(DataStrategy):
	def handle_data(self,data:pd.DataFrame):
		try:
			y=data['Time_taken(min)']
			X=data.drop(['Time_taken(min)'],axis=1)
			
			X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
			return X_train,X_test,y_train,y_test

		except Exception as e:
			logging.error("Error in  DataSplitting :{} ".format(e))
			raise e