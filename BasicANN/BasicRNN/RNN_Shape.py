"""
Basic of a RNN.............
"""

#Import  libraies..................
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense,Flatten, SimpleRNN
from tensorflow.keras.models import Model
from  tensorflow.keras.optimizers import SGD,Adam
import pandas as pd


#N - number of samples
#T - sequence length
#D = number of input features
#M - number of hidden units
#K - number of output units


#Make some data..............
N = 1
T = 20
D = 3
K = 2
X =np.random.randn(N, T, D)


#Make a RNN.................
M = 5
i = Input(shape = (T, D))
x = SimpleRNN(M)(i)
x = Dense(K)(x)
model = Model(i, x)

#Prediction......................
yhat = model.predict(X)
print(yhat)

print(model.summary())
print(model.layers[1].get_weights())
a,b,c = model.layers[1].get_weights()
print(a.shape, b.shape, c.shape)

Wx,Wh,bh = model.layers[1].get_weights()
w0,b0 = model.layers[2].get_weights()


h_last = np.zeros(M)#initial hidden state
x = X[0]#the one and only sample
yhat = []#store output

for t in range(T):
	h = np.tanh(x[t].dot(Wx) + h_last.dot(Wh) + bh)
	y = h.dot(w0) + b0
	yhat.append(y)
	h_last = h

print(yhat[-1])


