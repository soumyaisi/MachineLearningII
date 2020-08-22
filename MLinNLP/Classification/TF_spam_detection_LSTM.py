"""
Spam detection using LSTM in tensorflow-2.
"""

#Import libraies...............
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import LSTM, Embedding
from tensorflow.keras.models import Model


#Load the csv file.........
df = pd.read_csv('spam.csv', encoding = 'ISO-8859-1')
print(df.head())


#Drop the unnessasy columns............
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)
print(df.head())


#Rename the columns..............
df.columns = ['labels', 'data']
print(df.head())


#Create binary label.............
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
Y = df['b_labels'].values


#Split the data.............
df_train, df_test, Ytrain, Ytest = train_test_split(df['data'], Y, test_size = 0.33)


#Convert the sentences to sequences of words..............
MAX_VOCUB_SIZE = 20000
tokenizer = Tokenizer(num_words = MAX_VOCUB_SIZE)
tokenizer.fit_on_texts(df_train)
sequences_train = tokenizer.texts_to_sequences(df_train)
sequences_test = tokenizer.texts_to_sequences(df_test)


#word -> integer mapping..........
word2idx = tokenizer.word_index
V = len(word2idx)
print(V)


#Padding to get N*T matrix..........
data_train = pad_sequences(sequences_train)
print(data_train.shape)


#Get tthe sequence length.................
T = data_train.shape[1]


#Padding for test data............
data_test = pad_sequences(sequences_test, maxlen = T)
print(data_test.shape)


#Create the model...........
D = 20 #embedding dimension
M = 15 #hidden state vector size
i = Input(shape = (T,))
x = Embedding(V+1, D)(i)
x = LSTM(M, return_sequences = True)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1, activation = 'sigmoid')(x)
model = Model(i, x)


#Compile and fit the model..............
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
r = model.fit(data_train, Ytrain, epochs = 10, validation_data = (data_test, Ytest))


#Plot the losses.................
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()


#Plot the accuracy....................
plt.plot(r.history['accuracy'], label = 'accuracy')
plt.plot(r.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()







