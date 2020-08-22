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
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.layers.convolutional import MaxPooling1D, Conv1D
from  keras.layers import Dense, Embedding, Flatten, Dropout, GRU, LSTM, CuDNNLSTM, CuDNNGRU, Input 
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
embedding_size = 50
batch_size = 128

pad = 'post'

x_train_pad = pad_sequences(x_train, maxlen=max_len, padding=pad, truncating=pad)
x_test_pad = pad_sequences(x_test, maxlen=max_len, padding=pad, truncating=pad)


model0 = Sequential()

model0.add(Embedding(input_dim = num_words, output_dim=embedding_size, 
	input_length=max_len, name = 'layer_embedding'))
model0.add(Conv1D(filters = 128, kernel_size=3, padding='same', activation='relu'))
model0.add(MaxPooling1D(pool_size=2))
model0.add(Conv1D(filters = 128, kernel_size=3, padding='same', activation='relu'))
model0.add(MaxPooling1D(pool_size=2))
model0.add(Conv1D(filters = 128, kernel_size=3, padding='same', activation='relu'))
model0.add(MaxPooling1D(pool_size=2))
model0.add(Dropout(0.5))
model0.add(Flatten())
model0.add(Dense(250, activation='relu'))
model0.add(Dense(1, activation='sigmoid', name = 'classification'))

model0.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

print(model0.summary())

model0.fit(x_train_pad, y_train, 
	epochs=3, batch_size=batch_size, 
	validation_split=0.05)

eval_ = model0.evaluate(x_test_pad, y_test)
print(eval_)


model1 = Sequential()

model1.add(Embedding(input_dim = num_words, output_dim=embedding_size, 
	input_length=max_len, name = 'layer_embedding'))
model1.add(Conv1D(filters = 128, kernel_size=3, padding='same', activation='relu'))
model1.add(MaxPooling1D(pool_size=2))
model1.add(CuDNNLSTM(128))
model1.add(Dropout(0.5))
model1.add(Dense(1, activation='sigmoid', name = 'classification'))

model1.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

print(model1.summary())

model1.fit(x_train_pad, y_train, 
	epochs=3, batch_size=batch_size, 
	validation_split=0.05)

eval_ = model1.evaluate(x_test_pad, y_test)
print(eval_)


from keras.layers import  concatenate

conv = []
filter_sizes = [3,4,5]
embedding_layer = Embedding(input_dim = num_words, output_dim=embedding_size, 
	input_length=max_len, name = 'layer_embedding')
seq_input = Input(shape=(max_len,), dtype='int32')
embed_seq = embedding_layer(seq_input)

for f in filter_sizes:
	_conv = Conv1D(filters = 128, kernel_size=f, activation='relu')(embed_seq)
	_pool = MaxPooling1D(5)(_conv)
	conv.append(_pool)

_concat = concatenate(conv, axis=1)

_conv1 = Conv1D(128, 5, activation='relu')(_concat)
_pool1 = MaxPooling1D(5)(_conv1)
_pool1 = Dropout(0.5)(_pool1)

_conv2 = Conv1D(128, 5, activation='relu')(_concat)
_pool2 = MaxPooling1D(5)(_conv2)
_flat = Flatten()(_pool2)
_flat = Dropout(0.5)(_flat)
_dense = Dense(128, activation='relu')(_flat)

preds = Dense(1, activation='sigmoid')(_dense)

model = Model(seq_input, preds)
print(model.summary())


model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])

model.fit(x_train_pad, y_train, 
	epochs=3, batch_size=batch_size, 
	validation_split=0.05)

eval_ = model.evaluate(x_test_pad, y_test)
print(eval_)




