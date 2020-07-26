"""
Binary classification of text data using CNN in tensorflow-2.
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
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Model


#Load data from csv............
df = pd.read_csv('spam.csv', encoding = 'ISO-8859-1')
print(df.head())


#Drop unnessasy columns from the data............
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)
print(df.head())


#Rename the columns.............
df.columns = ['labels', 'data']
print(df.head())


#Create binary label............
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
Y = df['b_labels'].values


#Split the data...........
df_train, df_test, Ytrain, Ytest = train_test_split(df['data'], Y, test_size = 0.33)


#Convert the sentences to sequences of words...............
MAX_VOCUB_SIZE = 20000
tokenizer = Tokenizer(num_words = MAX_VOCUB_SIZE)
tokenizer.fit_on_texts(df_train)
sequences_train = tokenizer.texts_to_sequences(df_train)
sequences_test = tokenizer.texts_to_sequences(df_test)


#word -> integer mapping............
word2idx = tokenizer.word_index
V = len(word2idx)
print(V)


#Padding to get N*T matrix.............
data_train = pad_sequences(sequences_train)
print(data_train.shape)


#Get the sequence length...........
T = data_train.shape[1]


#Padding for test data...............
data_test = pad_sequences(sequences_test, maxlen = T)
print(data_test.shape)


#Create the model..............
D = 20 #embedding dimension
M = 15 #hidden state vector size
i = Input(shape = (T,))
x = Embedding(V+1, D)(i)
x = Conv1D(32, 3, activation = 'relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(64, 3, activation = 'relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation = 'relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1, activation = 'sigmoid')(x)
model = Model(i, x)


#Compile and train the model................
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
r = model.fit(data_train, Ytrain, epochs = 5, validation_data = (data_test, Ytest))


#Plot the loss............
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()


#Plot the accuracy...................
plt.plot(r.history['accuracy'], label = 'accuracy')
plt.plot(r.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()


