#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:43:11 2024

@author: amitpharswan
"""

import numpy as np
import pickle
import streamlit as st


# Loading Model
loaded_model = pickle.load(open('/Users/amitpharswan/Desktop/PROJECT/trained_model.sav', 'rb'))




# Function For Prediction
def diabetes_prediction (input_data):
   
    #1. Here we will take value from user

    #2. Changing the input_data to numpy Array
    input_data_as_numpy_array =np.asarray(input_data)


    #3. Reshaping the array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


    #4. SVM Accuracy is more than other so we will use SVM_model."
    prediction =loaded_model.predict(input_data_reshaped)
    print(prediction)

    #5. Prediction
    if(prediction[0]==0):
      return("The person is Not Diabetic ")
    else:
      return("The person is Diabetic")
  
    
  
    
  
    
  
    
def main():
    
    #Title for Web App
    st.title('Diabetes Prediction Web App')
    
    
    # getting the input data from the user
    
    
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()