import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from  keras.layers import Dense, Embedding, Dropout, GRU, LSTM, CuDNNLSTM, CuDNNGRU 
from keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard
import os
import re
import string
import nltk
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.optimizers import Adam

num_words = 20000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)


max_len = 256
embedding_size = 10
batch_size = 128
n_epochs = 10

pad = 'pre'

x_train_pad = pad_sequences(x_train, maxlen=max_len, padding=pad, truncating=pad)
x_test_pad = pad_sequences(x_test, maxlen=max_len, padding=pad, truncating=pad)

model = Sequential()

model.add(Embedding(input_dim = num_words, output_dim=embedding_size, 
	input_length=max_len, name = 'layer_embedding'))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid', name = 'classification'))

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
callback_early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

print(model.summary())

model.fit(x_train_pad, y_train, 
	epochs=n_epochs, batch_size=batch_size, 
	validation_split=0.05, 
	callbacks=[callback_early_stopping])

eval_ = model.evaluate(x_test_pad, y_test)
print(eval_)

model.save('sentiment_lstm')

model_GRU = Sequential()
model_GRU.add(Embedding(input_dim = num_words, output_dim=embedding_size, 
	input_length=max_len, name = 'layer_embedding'))
model_GRU.add(CuDNNGRU(units=16, return_sequences=True))
model_GRU.add(CuDNNGRU(units=8, return_sequences=True))
model_GRU.add(CuDNNGRU(units=4, return_sequences=False))
model_GRU.add(Dense(1, activation='sigmoid'))
print(model_GRU.summary())

model_GRU.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
model_GRU.fit(x_train_pad, y_train, 
	epochs=n_epochs, batch_size=batch_size, 
	validation_split=0.05, 
	)

eval_GRU = model.evaluate(x_test_pad, y_test)
print(eval_GRU)

y_pred = model.predict(x_test_pad[:1000])
y_pred = y_pred.T[0]
labels_pred = np.array([1.0 if p > 0.5 else 0.0 for p in y_pred])
true_labels = np.array(y_test[:1000])

incorrect = np.where(labels_pred != true_labels)
incorrect = incorrect[0]
print(len(incorrect))

idx = incorrect[1]
text = x_test[idx]
print(text)

word_index = imdb.get_word_index()
print(word_index.items())

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
print(reverse_word_indexe)

def decode_index(text):
	return ' '.join([reverse_word_index.get(i) for i in text])

text_data = []
for i in range(x_train):
	text_data.append(decode_index(x_train[i]))

layer_embedding = model.get_layer('layer_embedding')
weights_embedding = layer_embedding.get_weights()[0]
print(weights_embedding[word_index.get('good')])

from scipy.spatial.distance import cdist

def print_similar_words(word, metric = 'cosine'):
	token = word_index.get(word)
	embedding = weights_embedding[token]
	distances = cdist(weights_embedding, [embedding], metric=metric).T[0]
	sorted_index = np.argsort(distances)
	sorted_distances = distances[sorted_index]
	sorted_words = [reverse_word_index[token] for token in sorted_index if token != 0]

	def print_words(words, distances):
		for word,distance in zip(words, distances):
			print(distance, word)

	N =10
	print(word)
	print_words(sorted_words[0:N], sorted_distances[0:N])
	print_words(sorted_words[-N:], sorted_distances[-N:])

print_similar_words('good', metric='cosine')


