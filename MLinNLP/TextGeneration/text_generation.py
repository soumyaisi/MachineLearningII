import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

import numpy as np
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.models import Sequential, load_model
from keras.utils import  to_categorical
from random import randint

file = open('abc.txt', 'r')
text = file.read()
file.close()

tokens = text.lower()

n_chars = len(tokens)
unique_vocab = len(tokens)
print(n_chars)
print(unique_vocab)

characters = sorted(list(set(tokens)))
n_vocab = len(characters)

int_to_char = {n:char for n,char in enumerate(characters)}
char_to_int = {char:n for n,char in enumerate(characters)}

x = []
y = []
seq_length = 100
for i in range(0, n_chars - seq_length, 1):
	seq_in = tokens[i:i+seq_length]
	seq_out = tokens[i+seq_length]
	x.append([char_to_int[char] for char in seq_in])
	y.append([char_to_int[seq_out]])

print(x[0])
print(y[0])

x_new = np.reshape(x, (len(x), seq_length, 1))
x_new = x_new / float(n_vocab)
y_new = to_categorical(y)


model = Sequential()
model.add(LSTM(700, input_shape=(x_new.shape[1], x_new.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(700, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(700))
model.add(Dropout(0.2))
model.add(Dense(y_new.shape[1], activation='softmax'))
print(model.summary())

model.compile(loss = 'categorical_crossentropy', optimizer='adam')
model.fit(x_new, y_new, batch_size=64, epochs=5)


model.save('text_generation.h5')
model = load_model('text_generation.h5')


ini = np.random.randint(0, len(x)-1)
token_string = x[ini]

complete_string = [int_to_char[value] for value in token_string]


for i in range(500):
	x = np.reshape(token_string, (1, len(token_string), 1))
	x = x / float(n_vocab)

	prediction = model.predict(x, verbose=0)

	id_pred = np.argmax(prediction)
	seq_in = [int_to_char[value] for value in token_string]
	complete_string.append(int_to_char[id_pred])

	token_string.append(id_pred)
	token_string = token_string[1:len(token_string)]

text = ""
for char in complete_string:
	text = text + char
print(text)


