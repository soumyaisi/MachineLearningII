"""
Initial text preprocessing in tensorflow-2.
"""

#Import libraies...............
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


sentences = ['I like egg and ham',
			'I love chocolate and bunnies',
			'I hate onions']

#Tokenizer and padding................
MAX_VOCUB_SIZE = 20000
tokenizer = Tokenizer(num_words = MAX_VOCUB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
print(tokenizer.word_index)

data = pad_sequences(sequences)
print(data)

MAX_SEQUENCE_LENGTH = 5
data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH)
print(data)

data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH, padding = 'post')
print(data)

data = pad_sequences(sequences, maxlen = 4)
print(data)

data = pad_sequences(sequences, maxlen = 4, truncating = 'post')
print(data)

