# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

loaded_model = pickle.load(open('/Users/amitpharswan/Desktop/PROJECT/trained_model.sav', 'rb'))


#1. Input Data
input_data=(197,	70	,45	,543	,30.5	,0.158,	53)


#2. Changing the input_data to numpy Array
input_data_as_numpy_array =np.asarray(input_data)


#3. Reshaping the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


#4. SVM Accuracy is more than other so we will use SVM_model."
prediction =loaded_model.predict(input_data_reshaped)


#5. Prediction
if(prediction[0]==0):
  print("The person is Not Diabetic ")
else:
  print("The person is Diabetic")