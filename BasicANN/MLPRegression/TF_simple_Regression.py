"""
A simple regression problem using tensorflow-2
"""

#Loading libraies................
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#Create the dataset...............
N = 1000
X = np.random.random((N, 2)) *6 -3 #(-3,3)
Y = np.cos(2*X[:,0]) + np.cos(3*X[:,1])


#Plot the data in 3d...............
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()


#ML model................
model = tf.keras.models.Sequential([
	tf.keras.layers.Dense(128, input_shape = (2,), activation = 'relu'),
	tf.keras.layers.Dense(1)
	])


#Compile and fit the model (as regression problem)...................
opt = tf.keras.optimizers.Adam(0.01)
model.compile(optimizer = opt, loss = 'mse')
r = model.fit(X, Y, epochs = 100)


#Plot the loss..............
plt.plot(r.history['loss'], label = 'loss')
plt.show()


#Surface and prediction plot..............
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:,0], X[:,1], Y)

line = np.linspace(-3,3,50)
xx,yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
Yhat = model.predict(Xgrid).flatten()
ax.plot_trisurf(Xgrid[:,0], Xgrid[:, 1], Yhat, linewidth = 0.2, antialiased = True)
plt.show()


#Can it Extrapolate....................
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:,0], X[:,1], Y)

line = np.linspace(-5,5,50)
xx,yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
Yhat = model.predict(Xgrid).flatten()
ax.plot_trisurf(Xgrid[:,0], Xgrid[:, 1], Yhat, linewidth = 0.2, antialiased = True)
plt.show()


