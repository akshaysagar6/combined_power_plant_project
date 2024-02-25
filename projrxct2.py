import streamlit as st
import pandas as pd
from pickle import load
import numpy as np
from xgboost import XGBRegressor

st.title('Combined Cycle Power Plant')
st.subheader('Regression Model To Predict The Energy Generated')


def page():
    
    temp=st.sidebar.number_input('Temperature')    
    exha=st.sidebar.number_input('Exhaust Vacuum')
    amb=st.sidebar.number_input('Ambient Pressure')
    hum=st.sidebar.number_input('Humidity')
    
    
    data_dict= {'temperature':temp,'exhaust_vacuum':exha,'amb_pressure':amb,'r_humidity':hum }
    df=pd.DataFrame(data_dict,index=[1])
    df_new=np.sqrt(df)
    return df_new

features=page()
if st.sidebar.button('Submit'):
    st.write(features)
    loaded_model= load(open('final_model.pkl','rb'))
    result=loaded_model.predict(features)
    result_1=result**2
    st.write('Energy Generated',result_1)
