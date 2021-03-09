#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Dependencies
import math
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('fivethirtyeight')


# In[2]:


# get the stock quote
SBUX_df = web.DataReader('SBUX', data_source='yahoo', start='2000-01-01', end ='2021-12-31')
SBUX_df


# In[3]:


# get the total rows and columns in the AAPL Dataset
SBUX_df.shape


# In[5]:


# create a closing price dataset
Closing_Price_SBUX_df = SBUX_df.filter(['Close'])

# convert the the closing price dataset to a array
Closing_Price_array_SBUX_df = Closing_Price_SBUX_df.values

Closing_Price_array_SBUX_df


# In[6]:


# get the number of rows to train the model on (90 %)
train_closing_price_SBUX_len = math.ceil(len(Closing_Price_array_SBUX_df) * 0.9) # rounded up

train_closing_price_SBUX_len


# In[7]:


# Scale de closing price dataset between 0 and 1
scale_SBUX = MinMaxScaler(feature_range=(0,1))
scaled_closing_price_SBUX_df = scale_SBUX.fit_transform(Closing_Price_array_SBUX_df)
scaled_closing_price_SBUX_df


# In[8]:


# Training Data
train_closing_price_SBUX = scaled_closing_price_SBUX_df[0:train_closing_price_SBUX_len, :]
# Split Data
x_train =[]
y_train=[]

for i in range(80, len(train_closing_price_SBUX)):
  x_train.append(train_closing_price_SBUX[i-80:i,0])
  y_train.append(train_closing_price_SBUX[i,0])


# In[9]:


# convert to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)


# In[10]:


# Reshape the dataset
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


# In[11]:


x_train.shape


# In[12]:


# Building the LSTM model
#x_train = tf.convert_to_tensor(x_train)
#y_train = tf.convert_to_tensor(y_train)

SBUX_model = Sequential()
SBUX_model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1],1)))
SBUX_model.add(LSTM(50, return_sequences=False))
SBUX_model.add(Dense(25))
SBUX_model.add(Dense(1))


# In[13]:


# Compile  the LSTM model
SBUX_model.compile(optimizer='adam', loss='mean_squared_error')
#'categorical_crossentropy'


# In[14]:


# training the AAPL_model
SBUX_model.fit(x_train,y_train,epochs=1,batch_size=1)


# In[15]:


# Setting the testing Dataset
# scaled testing dataset set from index 1743 to 2003
SBUX_test_dataset= scaled_closing_price_SBUX_df[train_closing_price_SBUX_len - 80: , :]

#setting the x and y test data
x_test = []
y_test = Closing_Price_array_SBUX_df[train_closing_price_SBUX_len, :]
for i in range(80,len(SBUX_test_dataset)):
    x_test.append(SBUX_test_dataset[i-80:i, 0])


# In[16]:


# Convert the test dataset to array
x_test = np.array(x_test)
print (x_test)


# In[17]:


#reshape the test dataset

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))


# In[18]:


x_test.shape


# In[19]:


# getting the model predicted closing price value

predictions = SBUX_model.predict(x_test)
predictions = scale_SBUX.inverse_transform(predictions)


# In[20]:


# getting the root mean squared error 

rmse = np.sqrt(np.mean(predictions - y_test) **2)
rmse


# 

# In[21]:


# Plot the Dataset
train_data = Closing_Price_SBUX_df[:train_closing_price_SBUX_len]
validation_data = Closing_Price_SBUX_df[train_closing_price_SBUX_len:]

validation_data['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('SBUX Model', color='crimson')
plt.xlabel('date', fontsize=18, color='crimson')
plt.ylabel('closing price USD', color='crimson')
plt.plot(train_data['Close'], color='darkcyan')
plt.plot(validation_data['Close'], color='limegreen')
plt.plot(validation_data['Predictions'], color = 'lightcoral')
plt.legend(['Train', 'Validation', 'Predictions'])


# In[ ]:




