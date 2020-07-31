"""
Example of tensorflow-serving.
"""

import tensorflow as tf
import requests
import numpy as np
import  matplotlib.pyplot as plt
import os
import subprocess
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout
from  tensorflow.keras.models import Model
import tempfile


#Return your IP address in a json..........
r = requests.get('https://api.ipify.org?format=json')
j = r.json()
print(j)

#Dataset loading.............
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#Preprocessing the data to fit in the model...............
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(x_train.shape)

K = len(set(y_train))

#Build the ML Model...............
i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3,3), strides=2, activation='relu')(i)
x = Conv2D(64, (3,3), strides=2, activation='relu')(x)
x = Conv2D(128, (3,3), strides=2, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i, x)
print(model.summary())

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = 3)

#Save the model to a directory...................
MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
print(export_path)
#if os.path.isdir(export_path):
	#!rm -r {export_path}

tf.saved_model.save(model, export_path)


#!ls -l {export_path}
#saved_model_cli show --dir {export_path} --all
#! apt-get install tensorflow-model-server

os.environ["MODEL_DIR"] = MODEL_DIR

#Start the server...............
"""
%%bash --bg
nohup tensorflow_model_server \
--rest_api_port = 8501 \
--model_name = gashion_model \
--model_base_path = "${MODEL_DIR}" > server.log 2>&1


#!tail server.log
"""

#Label Mapping.................
labels = '''T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot'''.split("\n")


def show(idx, title):
	plt.figure()
	plt.imshow(x_test[idx].reshape(28,28), cmap='gray')
	plt.axis('off')
	plt.title('\n\n{}'.format(title), fontdict={'size': 16})

i = np.random.randint(0, len(x_test))
show(i, labels[y_test[i]])

#Format some data to pass to the server..........
import json
data = json.dumps({"signature_name": "serving_default", "instances": x_test[0:3].tolist()})
print(data)

headers = {"content-type": "application/json"}
r = requests.post('http://localhost:8501/v1/models/fashion_model:predict', data = data, headers=headers)
j = r.json()
print(j.keys())
print(j)

pred = np.array(j['predictions'])
print(pred.shape)

pred = pred.argmax(axis=1)

pred = [labels[i] for i in pred]
print(pred)

actual = [labels[i] for i in y_test[:3]]
print(actual)

for i in range(0,3):
	show(i, "True: {actual[i]}, Predicted: {pred[i]}")

#Select a model by version..............
headers = {"content-type": "application/json"}
r = requests.post('http://localhost:8501/v1/models/fashion_model/versions/1:predict', data = data, headers=headers)
j = r.json()
pred = np.array(j['predictions'])
pred = pred.argmax(axis=1)
pred = [labels[i] for i in pred]
for i in range(0,3):
	show(i, "True: {actual[i]}, Predicted: {pred[i]}")


#Save version 2 of the model..................
version = 2
export_path = os.path.join(MODEL_DIR, str(version))
print(export_path)
#if os.path.isdir(export_path):
	#!rm -r {export_path}

tf.saved_model.save(model2, export_path)


headers = {"content-type": "application/json"}
r = requests.post('http://localhost:8501/v1/models/fashion_model/versions/2:predict', data = data, headers=headers)
j = r.json()
pred = np.array(j['predictions'])
pred = pred.argmax(axis=1)
pred = [labels[i] for i in pred]
for i in range(0,3):
	show(i, "True: {actual[i]}, Predicted: {pred[i]}")

