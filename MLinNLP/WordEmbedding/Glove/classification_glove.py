"""
Using glove classification in Keras.
"""

#Import Libraies................
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
from  keras.layers import Dense, Embedding, Flatten, Dropout
from keras.callbacks import EarlyStopping,ModelCheckpoint
import os
import re
import string
import nltk
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.manifold import TSNE


#Read Data.................
df = pd.read_csv('yelp.csv')
print(df.head())

df = df.dropna()
df = df[['text', 'stars']]
print(df.head())


#Preprocess the Data.................
labels = df['stars'].map(lambda x: 1 if int(x) > 3 else 0)
print(labels[10:20])


# "    text = re.sub(r\"n't\", \" not \", text)\n",
#     "    text = re.sub(r\"i'm\", \"i am \", text)\n",
#     "    text = re.sub(r\"\\'re\", \" are \", text)\n",
#     "    text = re.sub(r\"\\'d\", \" would \", text)\n",
#     "    text = re.sub(r\"\\'ll\", \" will \", text)\n",
#     "    text = re.sub(r\",\", \" \", text)\n",
#     "    text = re.sub(r\"\\.\", \" \", text)\n",
#     "    text = re.sub(r\"!\", \" ! \", text)\n",
#     "    text = re.sub(r\"\\/\", \" \", text)\n",
#     "    text = re.sub(r\"\\^\", \" ^ \", text)\n",
#     "    text = re.sub(r\"\\+\", \" + \", text)\n",
#     "    text = re.sub(r\"\\-\", \" - \", text)\n",
#     "    text = re.sub(r\"\\=\", \" = \", text)\n",
#     "    text = re.sub(r\"'\", \" \", text)\n",
#     "    text = re.sub(r\"(\\d+)(k)\", r\"\\g<1>000\", text)\n",
#     "    text = re.sub(r\":\", \" : \", text)\n",
#     "    text = re.sub(r\" e g \", \" eg \", text)\n",
#     "    text = re.sub(r\" b g \", \" bg \", text)\n",
#     "    text = re.sub(r\" u s \", \" american \", text)\n",
#     "    text = re.sub(r\"\\0s\", \"0\", text)\n",
#     "    text = re.sub(r\" 9 11 \", \"911\", text)\n",
#     "    text = re.sub(r\"e - mail\", \"email\", text)\n",


def clean_text(text):
	text = text.translate(string.punctuation)
	text = text.lower().split()
	stops = set(stopwords.words('english'))
	text = [w for w in text if not w in stops and len(w) >= 3]
	text = " ".join(text)
	#Clean the text
	text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
	text = re.sub(r"what's", "what is ", text)
	text = re.sub(r"\'s", " ", text)
	text = re.sub(r"\'ve", " have ", text)
	text = re.sub(r"j k", "jk", text)
	text = re.sub(r"\s{2,}", " ", text)

	text = text.split()
	stemmer = SnowballStemmer('english')
	stemmed_words = [stemmer.stem(word) for word in text]
	text = " ".join(stemmed_words)
	return text

df['text'] = df['text'].map(lambda x: clean_text(x))
print(df.head(10))

maxlen = 50
embed_dim = 100
max_words = 20000

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(df['text'])

sequences = tokenizer.texts_to_sequences(df['text'])

data = pad_sequences(sequences, maxlen = maxlen, padding='post')
print(data[0])

labels = np.asarray(labels)

print(data.shape)
print(labels.shape)

vocab_size = len(tokenizer.word_index)+1
print(vocab_size)

print(len(df))


#Split the data....................
validation_split = .2
indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]
labels = labels[indices]

val_samples = int(validation_split * data.shape[0])

x_train = data[:-val_samples]
y_train = labels[:-val_samples]
x_val = data[-val_samples:]
y_val = labels[-val_samples:]


#Load the glove model.......................
#https://nlp.stanford.edu/projects/glove/
dir = '/glove.6B'

embed_index = dict()
f = open(os.path.join(dir, 'glove.6B.100d.txt'), encoding='utf8')

for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embed_index[word] = coefs

f.close()

print(len(embed_index))

embed_matrix = np.zeros((max_words, embed_dim))
for word, i in tokenizer.word_index.items():
	if i < max_words:
		embed_vector = embed_index.get(word)
		if embed_vector is not None:
			embed_matrix[i]  = embed_vector

#Build the Model.................
model = Sequential()
model.add(Embedding(max_words, embed_dim, weights=[embed_matrix], input_length=maxlen))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())


model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

save_best = ModelCheckpoint('yelp.hdf', save_best_only=True, monitor='val_loss', mode='min')

model.fit(x_train, y_train, epochs=5, 
	validation_data=(x_val, y_val), 
	batch_size=128, 
	verbose=1, 
	callbacks=[early_stopping, save_best])

glove_embds = model.layers[0].get_weights()[0]
print(glove_embds.shape)

words = []
for word, i in tokenizer.word_index.items():
	words.append(word)

def plot_words(data, start, stop, step):
	trace = go.Scatter(x=data[start:stop:step,0], 
		y = data[start:stop:step:1], 
		mode = 'markers', 
		text = words[start:stop:step])
	layout = dcit(title='t-SNE', 
		yaxis = dict(title='factor-2'), 
		xaxis = dict(title='factor-1'), 
		hovermode = 'closest')

	fig = dict(data = [trace], layout = layout)
	py.iplot(fig)

#Using TSNE plot...............
glove_tsne_embds = TSNE(n_components=2).fit_transform(glove_embds)
plot_words(glove_tsne_embds, 0, 100, 1)

