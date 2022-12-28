
import streamlit as st
import pandas as pd
import numpy as np
import pickle
pickle_in = open('random_forest.pkl', 'rb')
model = pickle.load(pickle_in)

def predicts(input_df):
    predictions_df = np.asarray(input_df)
    pred_df_reshaped = predictions_df.reshape(1, -1)
    prediction = model.predict(pred_df_reshaped)
    prediction_proba = model.predict_proba(pred_df_reshaped)
    st.subheader('Prediction')
    st.write(prediction_proba)
    
    if(prediction == 0):
        st.write('This person is not going to Churn')
        return('Not Churn')
    else:
        st.write('This person is going to Churn')
        return('Churn')
    

add_selectbox = st.sidebar.selectbox("How would you like to predict?",("Online", "Batch"))
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
    
    
    output = ''
    if st.button('Predict'):
        output = predicts([account_length, voice_mail_plan, voice_mail_messages, day_mins, evening_mins, night_mins, international_mins, customer_service_calls, international_plan, day_calls, day_charge, evening_calls, evening_charge, night_calls, night_charge, international_calls, international_charge, total_charge])
        st.success(output)  


if add_selectbox == 'Batch':
    file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
    if file_upload is not None:
        data = pd.read_csv(file_upload)
        predictbatch = model.predict(data)
        st.write(predictbatch)
        