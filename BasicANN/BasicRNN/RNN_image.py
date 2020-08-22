"""
Image classification using LSTM. Developed in tensorflow-2.
"""

#Import libraies..................
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense,Flatten, SimpleRNN, LSTM
from tensorflow.keras.models import Model
from  tensorflow.keras.optimizers import SGD,Adam
import pandas as pd


#Read the dataset...............
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train.shape)


#Build the model.............
i = Input(shape = x_train[0].shape)
x = LSTM(128)(i)
x = Dense(10, activation = 'softmax')(x)
model = Model(i, x)

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
r = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs =20)

#plot the loss............
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()

#plot the accuracy............
plt.plot(r.history['accuracy'], label = 'accuracy')
plt.plot(r.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()


