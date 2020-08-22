"""
Stock return forecasting developed in tensorflow-2.
"""

#Import libraies................
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense,Flatten, SimpleRNN, LSTM, GRU, MaxPooling1D
from tensorflow.keras.models import Model
from  tensorflow.keras.optimizers import SGD,Adam
import pandas as pd
from sklearn.preprocessing import  StandardScaler


#Read the data................
df = pd.read_csv('SBUX.csv')
df.head()
print(df.shape)


#wrong thing................
series = df['close'].values.reshape(-1, 1)

#Data processing....................
scaler = StandardScaler()
scaler.fit(series[:len(series) // 2])
series = scaler.transform(series).flatten()

T = 10
D = 1
X = []
Y = []
for t in range(len(series)-T):
	x = series[t: t+T]
	X.append(x)
	y = series[t+T]
	Y.append(y)

X = np.array(X).reshape(-1, T, 1)
Y = np.array(Y)
N = len(X)
print(X.shape, Y.shape)


#Build the model........................
i = Input(shape = (T,1))
x = LSTM(5)(i)
x = Dense(1)(x)
model = Model(i,x)
model.compile(loss='mse', optimizer = Adam(lr=0.1),)
r = model.fit(X[:-N//2], Y[:-N//2], epochs = 80, validation_data = (X[-N//2:], Y[-N//2:]),)


#One step forecast..................
outputs = model.predict(X)
print(outputs.shape)
prediction = outputs[:,0]

plt.plot(Y, label = 'targets')
plt.plot(prediction, label = 'prediciton')
plt.legend()
plt.show()


#multi step forecast.....................
validation_target = Y[-N//2:]
validation_prediction = []
last_x = X[-N//2]

while len(validation_prediction) < len(validation_target):
	p = model.predict(last_x.reshape(1,T,1))[0,0]
	validation_prediction.append(p)
	last_x = np.roll(last_x, -1)
	last_x[-1] = p

plt.plot(validation_target, label = 'targets')
plt.plot(validation_prediction, label = 'prediciton')
plt.legend()
plt.show()


#cauculate return .................
df['prevClose'] = df['close'].shift(1)
df['return'] = (df['close'] - df['prevClose']) / df['prevClose']
df['return'].hist()

series = df['return'].values[1:].reshape(-1,1)
scaler = StandardScaler()
scaler.fit(series[:len(series) // 2])
series = scaler.transform(series).flatten()

T = 10
D = 1
X = []
Y = []
for t in range(len(series)-T):
	x = series[t: t+T]
	X.append(x)
	y = series[t+T]
	Y.append(y)

X = np.array(X).reshape(-1, T, 1)
Y = np.array(Y)
N = len(X)
print(X.shape, Y.shape)


#Build the model..................
i = Input(shape = (T,1))
x = LSTM(5)(i)
x = Dense(1)(x)
model = Model(i,x)
model.compile(loss='mse', optimizer = Adam(lr=0.1),)
r = model.fit(X[:-N//2], Y[:-N//2], epochs = 80, validation_data = (X[-N//2:], Y[-N//2:]),)


#one step forecast.................
outputs = model.predict(X)
print(outputs.shape)
prediction = outputs[:,0]

plt.plot(Y, label = 'targets')
plt.plot(prediction, label = 'prediciton')
plt.legend()
plt.show()


#multi step forecast.........................
validation_target = Y[-N//2:]
validation_prediction = []
last_x = X[-N//2]

while len(validation_prediction) < len(validation_target):
	p = model.predict(last_x.reshape(1,T,1))[0,0]
	validation_prediction.append(p)
	last_x = np.roll(last_x, -1)
	last_x[-1] = p

plt.plot(validation_target, label = 'targets')
plt.plot(validation_prediction, label = 'prediciton')
plt.legend()
plt.show()


#Working on full data...............
input_data = df[['open', 'high', 'low', 'close', 'volume']].values
targets = df['return'].values

T = 10
D = input_data.shape[1]
N = len(input_data ) - T


#Train-Test split.....................
Ntrain = len(input_data)*2 // 3
scaler = StandardScaler()
scaler.fit(input_data[:Ntrain+T])
input_data = scaler.transform(input_data)

X_train = np.zeros((Ntrain, T, D))
Y_train = np.zeros(Ntrain)


for t in range(Ntrain):
	X_train[t, :, :] = input_data[t:t+T]
	Y_train[t] = (targets[t+T] > 0)

X_test = np.zeros((N-Ntrain, T, D))
Y_test = np.zeros(N-Ntrain)

for u in range(N-Ntrain):
	t = u + Ntrain
	X_test[u,:,:] = input_data[t:t+T]
	Y_test[u] = (targets[t+T] > 0)


#<odel developed and plots.............
i = Input(shape = (T,D))
x = LSTM(50)(i)
x = Dense(1, activation = 'sigmoid')(x)
model = Model(i,x)
model.compile(loss = 'bonary_crossentropy', optimizer = Adam(lr=0.001), metrics = ['accuracy'],)

r = model.fit(X_train, Y_train, batch_size = 32, epochs = 300, validation_data = (X_test, Y_test),)

plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'Val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label = 'accuracy')
plt.plot(r.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()







