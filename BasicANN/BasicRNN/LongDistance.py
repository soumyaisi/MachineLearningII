"""
A classification model developed on custom data and is a classification method. As a classification 
algo we used simple RNN,LSTM,GRU and also linear model. And all of these developed in tensorflow-2.
"""

#Import libraies...............
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense,Flatten, SimpleRNN, LSTM, GRU, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from  tensorflow.keras.optimizers import SGD,Adam
import pandas as pd


#Create the custom data..............
T = 20
D = 1
X =[]
Y = []

def get_label(x, i1, i2, i3):
	if x[i1] < 0 and x[i2] < 0 and x[i3] < 0:
		return 1
	if x[i1] < 0 and x[i2] > 0 and x[i3] > 0:
		return 1
	if x[i1] > 0 and x[i2] < 0 and x[i3] > 0:
		return 1
	if x[i1] > 0 and x[i2] > 0 and x[i3] < 0:
		return 1
	return 0

for t in range(5000):
	x = np.random.randn(T)
	X.append(x)
	#y = get_label(x, -1,-2,-3)#short distance
	y = get_label(x, 0,1,2)#long distance
	Y.append(y)

X = np.array(X)
Y = np.array(Y)
N = len(X)


#Linear model.............
i = Input(shape = (T, ))
x = Dense(1, activation = 'sigmoid')(i)
model = Model(i, x)
model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr=0.01), metrics = ['accuracy'])

#train the model
r = model.fit(X, Y, epochs = 100, validation_split = 0.5,)

#plot
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()

#plot
plt.plot(r.history['accuracy'], label = 'accuracy')
plt.plot(r.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()


#Simple RNN............
inputs = np.expand_dims(X, -1)
i = Input(shape = (T, D))
x = SimpleRNN(5)(i)
x = Dense(1, activation = 'sigmoid')(x)

model = Model(i, x)
model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr=0.01), metrics = ['accuracy'])
r = model.fit(inputs, Y, epochs = 200, validation_split = 0.5,)

#plot
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()

#plot
plt.plot(r.history['accuracy'], label = 'accuracy')
plt.plot(r.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()


#Simple LSTM.............
inputs = np.expand_dims(X, -1)
i = Input(shape = (T, D))
x = LSTM(5)(i)
x = Dense(1, activation = 'sigmoid')(x)

model = Model(i, x)
model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr=0.01), metrics = ['accuracy'])
r = model.fit(inputs, Y, epochs = 200, validation_split = 0.5,)

#plot
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()

#plot
plt.plot(r.history['accuracy'], label = 'accuracy')
plt.plot(r.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()


#Simple GRU................
inputs = np.expand_dims(X, -1)
i = Input(shape = (T, D))
x = GRU(5)(i)
x = Dense(1, activation = 'sigmoid')(x)

model = Model(i, x)
model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr=0.01), metrics = ['accuracy'])
r = model.fit(inputs, Y, epochs = 400, validation_split = 0.5,)

#plot
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()

#plot
plt.plot(r.history['accuracy'], label = 'accuracy')
plt.plot(r.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()


#Get all the hidden states of LSTM...................
inputs = np.expand_dims(X, -1)
i = Input(shape = (T, D))
x = LSTM(5, return_sequences = True)(i)
x = GlobalMaxPooling1D()(x)
x = Dense(1, activation = 'sigmoid')(x)

model = Model(i, x)
model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr=0.01), metrics = ['accuracy'])
r = model.fit(inputs, Y, epochs = 100, validation_split = 0.5,)

#plot
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()

#plot
plt.plot(r.history['accuracy'], label = 'accuracy')
plt.plot(r.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()



