# -*- coding: utf-8 -*-
"""LSTM_RNN_Model.ipynb"""

"""Developed by Pradeep Kumar for receipt matching case study using Embedded layers in a LSTM - Model"""

# Import the required libraries for data mining and building the Model
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

# Reading the source file using pandas 
df = pd.read_excel("dummy_file.xlsx")
df

# Creating a 'MATCH_FLAG' based on matching values of feature_transaction_id and matched_transaction_id for all receipts
df['Match_Flag'] = np.where( (df['receipt_id']== df['receipt_id']) & (df['matched_transaction_id']== df['feature_transaction_id']) &  (df['company_id']== df['company_id']) , '1', '0')
df

# Declaring the PREDICTORS(Independent features) and RESPONSE(Dependent features) variables 
X = df.iloc[:,0:14]
X


Y = df.iloc[:,-1]
Y

# Splitting the dataset into train set and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) # 80% training

# Converting the data into numpy arrays 
x_train = np.array(x_train)
x_test = np.array(x_test)


y_train = np.array(y_train)
y_test = np.array(y_test)

# Verifying the shape of train data set
x_train.shape

# Declaring the paramerters of Bi-directional LSTM - RNN Model for learning the matching features
model = Sequential()
model.add(Embedding(80000, 32, input_length=None))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

# Compiling the LSTM MODEL using Binary_CrossEntropy
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

# Building the LSTM - RNN model 
print('Train...')

model.fit(x_train, y_train,
          batch_size=100,
          epochs=100,
          validation_data=[x_test, y_test])

# Verifying the Embedded layers and the Bi-directional parameteres of the model
model.summary()

# Predicting the output response on the test data
predicted_output = model.predict(x_test)

# Verifying the shape of response variables
predicted_output.shape

# To Save model and architecture to single file
model.save("model.h5")
print("Saved model to disk")

# Verifying the performance of the model on the complete dataset
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

