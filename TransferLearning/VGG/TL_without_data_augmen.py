"""
Classification using transfer learning and without data augmentation.Developed in tensorflow-2.
"""

#Import libraies............
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.applications.vgg16 import  VGG16 as PretrainedModel, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from  sklearn.utils import shuffle
from glob import glob
import sys,os


#Load the dataset..............
plt.imshow(image.load_image('training/0_808.jpg'))
plt.show()

plt.imshow(image.load_image('training/1_616.jpg'))
plt.show()

#!makid data/train
#mkdir data/test
#mkdir data/train/nonfood
#mkdir data/train/food
#mkdir data/test/nonfood
#mkdir data/test/food

#move the images...
#!mv training/0*.jpg data/train/nonfood
#!mv training/1*.jpg data/train/food
#!mv validation/0*.jpg data/train/nonfood
#!mv validation/1*.jpg data/train/food


train_path = 'data/train'
valid_path = 'data/test'

IMAGE_SIZE = [200, 200]

#getting number of files
image_files = glob(train_path + '/*/*.jpg')
valid_image_files = glob(valid_path + '/*/*.jpg')

#number of classes
folders = glob(train_path + '/*')
print(folders)

plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()


#Load the pre-trainded model.............
ptm = PretrainedModel(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)

x = Flatten()(ptm.output)

model = Model(inputs = ptm.input, outputs = x)

print(model.summary())


gen = ImageDataGenerator(preprocessing_function = preprocess_input)

batch_size = 128

#Create generator............
train_generator = gen.flow_from_directory(
	train_path,
	target_size = IMAGE_SIZE,
	batch_size = batch_size,
	class_mode = 'binary',
	)
valid_generator = gen.flow_from_directory(
	valid_path,
	target_size = IMAGE_SIZE,
	batch_size = batch_size,
	class_mode = 'binary',
	)


Ntrain = len(image_files)
Nvalid = len(valid_image_files)

feat = model.predict(np.random.random([1] + IMAGE_SIZE + [3]))
D = feat.shape[1]


X_train = np.zeros((Ntrain, D))
Y_train = np.zeros(Ntrain)
X_valid = np.zeros((Nvalid, D))
Y_valid = np.zeros(Nvalid)


#populate x_train, y_train
i = 0
for x,y in train_generator:
	#get features
	features = model.predict(x)
	#size of te batch
	sz = len(y)
	X_train[i:i+sz] = features
	Y_train[i:i+sz] = y

	i += sz
	print(i)

	if i >= Ntrain:
		print("breaking now")
		break
print(i)

#populate x_valid, y_valid
i = 0
for x,y in valid_generator:
	#get features
	features = model.predict(x)
	#size of te batch
	sz = len(y)
	X_valid[i:i+sz] = features
	Y_valid[i:i+sz] = y

	i += sz
	print(i)

	if i >= Nvalid:
		print("breaking now")
		break
print(i)

print(X_train.max(), X_train.min())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train2 = scaler.fit_transform(X_train)
x_valid2 = scaler.transform(X_valid)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr.fit(x_train2, Y_train)
print(logr.score(x_train2, Y_train))
print(logr.score(x_valid2, Y_valid))


#do logistix in tensorflow
i = Input(shape = (D, ))
x = Dense(1, activation = 'sigmoid')(i)
linearmodel = Model(i, x)

linearmodel.compile(loss = 'binary_crossentropy',
	optimizer = 'adam',
	metrics = ['accuracy'],
	)

r = linearmodel.fit(
	X_train, Y_train,
	batch_size = 128,
	epochs = 10,
	validation_data = (X_valid, Y_valid),
	)


plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()


plt.plot(r.history['accuracy'], label = 'accuracy')
plt.plot(r.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()

#data..............
#https://www.epfl.ch/labs/mmspg/downloads/food-image-datasets/