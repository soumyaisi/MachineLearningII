"""
Example of tensorflow mirrored-strategy.
"""

#Import libraies..............
import tensorflow as tf
import numpy as np
import  matplotlib.pyplot as plt
import os
import subprocess
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization
from  tensorflow.keras.models import Model

#Loading the data.................
cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

K = len(set(y_train))

def create_model():
	i = Input(shape=x_train[0].shape)

	x = Conv2D(32, (3,3), activation='relu', padding='same')(i)
	x = BatchNormalization()(x)
	x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2,2))(x)
	x = Conv2D(64, (3,3), activation='relu', padding='same')(i)
	x = BatchNormalization()(x)
	x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2,2))(x)
	x = Conv2D(128, (3,3), activation='relu', padding='same')(i)
	x = BatchNormalization()(x)
	x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2,2))(x)

	x = Flatten()(x)
	x = Dropout(0.2)(x)
	x = Dense(1024, activation='relu')(x)
	x = Dropout(0.2)(x)
	x = Dense(K, activation='softmax')(x)

	model = Model(i,x)
	return model

strategy = tf.distribute.MirroredStrategy()
#strategy = tf.distribute.experimental.CentralStorageStrategy()
print("number of devices", strategy.num_replicas_in_sync)

with strategy.scope():
	model = create_model()

	model.compile(loss = 'sparse_categgorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


r = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=15)


