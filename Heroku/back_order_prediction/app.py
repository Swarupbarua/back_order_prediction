# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 02:20:31 2020

@author: Swarup Barua
"""


import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 
import tensorflow as tf

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("Col_list.pkl","rb")
init_col, req_col, col_objectType = pickle.load(pickle_in)

model = tf.keras.models.load_model('ann_model1',compile = False)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_backorder(input_data):
    
    # convert list to dataframe
    df = pd.DataFrame([input_data], columns = init_col)
    # convert all fields to numeric
    for i in col_objectType:
        df[i] = df[i].map({'No' : 0, 'Yes' : 1}).astype('int')
    # Move the lead time to new field
    df['lead_time_KNN_Imputed'] = df['lead_time']
    # Remove unwanted columns 
    df = df[req_col]
    # Predict backorder
    y_pred = model.predict([list(df.values[0])])
    if y_pred>0.5:
      return 'Backorder generated'
    else:
      return 'No Backorder generated'




def main():
    st.title("Backorder prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Backorder prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    sku = st.number_input("sku")
    national_inv = st.number_input("national_inv")
    lead_time = st.number_input("lead_time")
    in_transit_qty = st.number_input("in_transit_qty")
    forecast_3_month = st.number_input("forecast_3_month")
    forecast_6_month = st.number_input("forecast_6_month")
    forecast_9_month = st.number_input("forecast_9_month")
    sales_1_month = st.number_input("sales_1_month")
    sales_3_month = st.number_input("sales_3_month")
    sales_6_month = st.number_input("sales_6_month")
    sales_9_month = st.number_input("sales_9_month")
    min_bank = st.number_input("min_bank")
    potential_issue = st.text_input("potential_issue","Type Here")
    pieces_past_due = st.number_input("pieces_past_due")
    perf_6_month_avg = st.number_input("perf_6_month_avg")
    perf_12_month_avg = st.number_input("perf_12_month_avg")
    local_bo_qty = st.number_input("local_bo_qty")
    deck_risk = st.text_input("deck_risk","Type Here")
    oe_constraint = st.text_input("oe_constraint","Type Here")
    ppap_risk = st.text_input("ppap_risk","Type Here")
    stop_auto_buy = st.text_input("stop_auto_buy","Type Here")
    rev_stop = st.text_input("rev_stop","Type Here")
    
    data = [ sku,
             national_inv,
             lead_time,
             in_transit_qty,
             forecast_3_month,
             forecast_6_month,
             forecast_9_month,
             sales_1_month,
             sales_3_month,
             sales_6_month,
             sales_9_month,
             min_bank,
             potential_issue,
             pieces_past_due,
             perf_6_month_avg,
             perf_12_month_avg,
             local_bo_qty,
             deck_risk,
             oe_constraint,
             ppap_risk,
             stop_auto_buy,
             rev_stop]
    
    result=""
    if st.button("Predict"):
        result=predict_backorder(data)
        st.success(result)
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    
