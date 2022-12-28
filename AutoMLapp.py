#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = load_model('gb_model')

def predict(model, input_df):
    predictions_df = model.predict_model(input_df)

def main():
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.title("Predicting Customer Churn")
    if add_selectbox == 'Online':
        account_length=st.number_input('Enter account length :' , min_value=0.0, max_value=240.0, value=128.0)
    
        voice_mail_plan=st.selectbox('The customer has voice mail plan :' , ['','1', '0'])
    
        voice_mail_messages=st.slider('Number of voice-mail messages. :' , min_value=0.0, max_value=60.0, value=25.0)
    
        day_mins=st.slider('Total minutes of day calls :' , min_value=0.0, max_value=360.0, value=265.1)
    
        evening_mins=st.slider('Total minutes of evening calls :' , min_value=0.0, max_value=400.0, value=197.4)
    
        night_mins=st.slider('Total minutes of night calls :' , min_value=0.0, max_value=400.0, value=244.7)
    
        international_mins=st.slider('Total minutes of international calls :' , min_value=0.0, max_value=60.0, value=10.0)
    
        customer_service_calls=st.slider('Number of calls to customer service :' , min_value=0.0, max_value=10.0, value=1.0)
    
        international_plan=st.selectbox('The customer has international plan :' , ['','1', '0'])
    
        day_calls=st.slider('Total day calls :' , min_value=0.0, max_value=200.0, value=110.0)
    
        day_charge=st.slider('Total day charge :' , min_value=0.0, max_value=360.0, value=45.07)
    
        evening_calls=st.slider('Total number of evening calls :' , min_value=0.0, max_value=200.0, value=99.0)
    
        evening_charge=st.slider('Total  eve charge :' , min_value=0.0, max_value=360.0, value=16.78)
    
        night_calls=st.slider('Total number of night calls :' , min_value=0.0, max_value=200.0, value=91.0)
    
        night_charge = st.slider(' Total night charge : ' , min_value=0.0, max_value=400.0, value=11.01)   
    
        international_calls=st.slider('Total number of international calls :' , min_value=0, max_value=20, value=3)
    
        international_charge=st.slider('Total int charge :' , min_value=0.0, max_value=360.0, value=2.7)
    
        total_charge=st.slider('Total charge:' , min_value=0.0, max_value=360.0, value=75.56)
        
        output=""
        input_dict={'account_length':account_length, 'voice_mail_plan':voice_mail_plan, 'voice_mail_messages':voice_mail_messages,
                   'day_mins':day_mins, 'evening_mins':evening_mins, 'night_mins':night_mins, 'international_mins':international_mins,
                   'customer_service_calls':customer_service_calls, 'international_plan':international_plan, 'day_calls':day_calls,
                   'day_charge':day_charge, 'evening_calls':evening_calls, 'evening_charge':evening_charge, 'night_calls':night_calls,
                   'night_charge':night_charge, 'international_calls':international_calls, 'international_charge':international_charge,
                   'total_charge':total_charge}
        
        input_df = pd.DataFrame([input_dict])
        
        if st.button("Predict"):
            output = model.predict(input_df)
        st.success('Churn : {}'.format(output))
        
        if(output == 0):
            st.write('This Customer is NOT going to Churn')
            return('Not Churn')
        if(output == 1):
            st.write('This customer is going to Churn')
            return('Churn')
        
        
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = model.predict(data)
            st.write(predictions)
            
if __name__ == '__main__':
    main()

