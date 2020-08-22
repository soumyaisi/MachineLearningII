"""
RNN for time series prediction, developed in tensorflow-2.
"""

#Import libraies............
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense,Flatten, SimpleRNN
from tensorflow.keras.models import Model
from  tensorflow.keras.optimizers import SGD,Adam
import pandas as pd


#make the original data.............
series = np.sin(0.1*np.arange(200))
#add noise
#series = np.sin(0.1*np.arange(200)) + np.random.randn(200)*0.1

plt.plot(series)
plt.show()


#Build the dataset...................
T = 10
X = []
D = 1
Y = []
for t in range(len(series) - T):
	x = series[t:t+T]
	X.append(x)
	y = series[t+T]
	Y.append(y)

X = np.array(X).reshape(-1,T, 1)#N*T*D
Y = np.array(Y)
N = len(X)
print(X.shape, Y.shape)


#Build the model...................
i = Input(shape = (T, 1))
x = SimpleRNN(5, activation = 'relu')(i)
x = Dense(1)(x)
model = Model(i, x)

model.compile(loss = 'mse', optimizer = Adam(lr=0.1))
r = model.fit(X[:-N//2], Y[:-N//2], epochs = 50, validation_data = (X[-N//2:], Y[-N//2:]),)

#Plot the loss...............
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()


#Wrong method for forecastng................
validation_target = Y[-N//2:]
validation_prediction = []
#index of first validation input
i = -N//2

while len(validation_prediction) < len(validation_target):
	p = model.predict(X[i].reshape(1,-1,1))[0,0]
	i += 1
	validation_prediction.append(p)

plt.plot(validation_target, label = 'forecast target')
plt.plot(validation_prediction, label = 'forecast predicition')
plt.legend()
plt.show()


#Correct way of forecasting......................
validation_target = Y[-N//2:]
validation_prediction = []
last_x = X[-N//2]#1-d array of length T

while len(validation_prediction) < len(validation_target):
	p = model.predict(last_x.reshape(1,-1,1))[0,0]
	validation_prediction.append(p)
	last_x = np.roll(last_x, -1)#make new input
	last_x[-1] = p


plt.plot(validation_target, label = 'forecast target')
plt.plot(validation_prediction, label = 'forecast predicition')
plt.legend()
plt.show()



