import os
import streamlit as st
import pandas as pd
from src.data_cleaning import DataPreProcessStrategy, CategoricalEncodingStrategy
from src.utils import load_object

model_path=os.path.join("artifacts","model.pkl")
model=load_object(file_path=model_path)

def main():

    city={
        'City_A':1,
         'City_B':2, 
         'City_C':3

    }

    weather_conditions = {
    'Clear': 1,
    'Cloudy': 2,
    'Rainy': 3
    }

    road_traffic_density = {
    'Low': 1,
    'Medium': 2,
    'High': 3
    }

    type_of_order = {
    'Type_1': 1,
    'Type_2': 2,
    'Type_3': 3
    }

    type_of_vehicle = {
    'Vehicle_1': 1,
    'Vehicle_2': 2,
    'Vehicle_3': 3
    }
    st.title("Food Delivery Time Prediction")



    delivery_data = {
	
      
   
    
    'Delivery_person_Ratings': st.number_input('Delivery Person Ratings', format="%.1f"),
    
    
    

    # 'Hour': float(st.time_input('Time Ordered').hour),
    # 'Minute': float(st.time_input('Time Ordered Minute').minute),
    # 'Time_Order_picked_Hour': float(st.time_input('Time Order Picked hour').hour),
    # 'Time_Order_picked_Minute': float(st.time_input('Time Order Picked minute').minute),
    'Weatherconditions': weather_conditions[st.selectbox('Weather Conditions', list(weather_conditions.keys()))],
    'Road_traffic_density': road_traffic_density[st.selectbox('Road Traffic Density', list(road_traffic_density.keys()))],
    'Vehicle_condition':st.number_input('Vehicle_condition'),
    'Type_of_order': type_of_order[st.selectbox('Type of Order', list(type_of_order.keys()))],
    'Type_of_vehicle': type_of_vehicle[st.selectbox('Type of Vehicle', list(type_of_vehicle.keys()))] ,
    'multiple_deliveries': st.number_input('Multiple Deliveries', format="%.1f"),
    'City': city[st.selectbox('City',list(city.keys()))],
    'total_distance': st.number_input('total distance', format="%.6f"),
}


    df=pd.DataFrame([delivery_data])
    


    if st.button('Predict'):
        try:
        
            prediction = model.predict(df)
            st.write("Prediction Result:")
            st.write(prediction)
        except Exception as e:
            st.error(f"Error in prediction: {e}")

if __name__ == "__main__":
    main()