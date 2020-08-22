"""
Linear regression in tensorflow-2. 
"""

#Import libraies...................
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Read the csv file from local............
data = pd.read_csv('moore.csv', header = None, sep = '\t').values
X = data[:,0].reshape(-1, 1)
Y = data[:,1]

#Plot the data.........
plt.scatter(X, Y)

#Log transformation on data and plot it.............
Y = np.log(Y)
plt.scatter(X, Y)

#Calculate mean of the data...........
X = X - X.mean()


#Define the model................
model = tf.keras.models.Sequential([
	tf.keras.layers.Input(shape = (1,)),
	tf.keras.layers.Dense(1)
	])


#Compile the model................
model.compile(optimizer = tf.keras.optimizers.SGD(0.001, 0.9), loss = 'mse')


#Define custom scheduler.............
def schedule(epoch, lr):
	if epoch >= 50:
		return 0.0001
	return 0.001

scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)


#Train the model...................
r = model.fit(X, Y, epochs = 200, callbacks = [scheduler])


#Plot the loss..............
plt.plot(r.history['loss'], label= 'loss')
plt.show()


#Print layers and weights of the model...............
print("model layers.............",model.layers)
print("weights................",model.layers[0].get_weights())


#Slope of the line........
s = model.layers[0].get_weights()[0][0,0]
print("slope of the line...........", s)


#Analytical solution......
X = np.array(X).flatten()
Y = np.array(Y)
denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean()*X.sum()) / denominator
b = (Y.mean()*X.dot(X) - X.mean()) / denominator
print(a,b)


#Plot the predict data.............
yhat = model.predict(X).flatten()
plt.scatter(X, Y)
plt.plot(X, yhat)
plt.show()


#For checking purpose(ML model and analytic model)................
w,b = model.layers[0].get_weights()
X = X.reshape(-1, 1)
yhat2 = (X.dot(w) + b).flatten()
#don't use == for floating point
s = np.allclose(yhat, yhat2)
print(s)
