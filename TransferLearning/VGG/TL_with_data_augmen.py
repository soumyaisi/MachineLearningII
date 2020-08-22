"""
Classification using transfer learning and with data augmentation.Developed in tensorflow-2.
"""

#Import libraies................
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


#Load the data...........
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

#Load the pre-trained model................
ptm = PretrainedModel(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)

ptm.trainable = False

#Build the model..................
K = len(folders)
x = Flatten()(ptm.output)
x = Dense(K, activation = 'softmax')(x)


model = Model(inputs = ptm.input, outputs = x)

print(model.summary())


#Create a instance of generator..........
gen = ImageDataGenerator(
	rotation_range = 20,
	width_shift_range = 0.1,
	height_shift_range = 0.1,
	shear_range = 0.1,
	zoom_range = 0.2,
	horizontal_flip = True,
	preprocessing_function = preprocess_input
	)

batch_size = 128


#Create generator..............
train_generator = gen.flow_from_directory(
	train_path,
	shuffle = True,
	target_size = IMAGE_SIZE,
	batch_size = batch_size,
	)
valid_generator = gen.flow_from_directory(
	valid_path,
	target_size = IMAGE_SIZE,
	batch_size = batch_size,
	)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

r = model.fit_generator(
	train_generator,
	validation_data = valid_generator,
	epochs = 10,
	steps_per_epochs = int(np.ceil(len(image_files) / batch_size)),
	validation_steps = int(np.ceil(len(valid_image_files) / batch_size)),
	)

plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()


plt.plot(r.history['accuracy'], label = 'accuracy')
plt.plot(r.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()

#data............
#https://www.epfl.ch/labs/mmspg/downloads/food-image-datasets/

