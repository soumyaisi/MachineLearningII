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
from  keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from  keras.layers import Dense, Embedding, Lambda
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import keras.backend as K
import nltk

from  string import punctuation
from  nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


remove_terms = punctuation + '0123456789'


def preprocessing(text):
	words  = word_tokenize(text)
	#print(words)
	tokens = [w for w in words if w.lower() not in remove_terms]
	#print(tokens)
	#stop = stopwords.words('english')
	#tokens = [token for token in tokens if token not in stop]
	#tokens = [word for word in tokens if len(word) > 3]
	tokens = [word for word in tokens if word.isalpha()]
	#print(tokens)
	lemma = WordNetLemmatizer()
	tokens = [lemma.lemmatize(word) for word in tokens]
	#print(tokens)
	preprocessed_text = ' '.join(tokens)
	#print(preprocessed_text)
	return preprocessed_text


corpus = open('guttenberg_astronomy.txt', encoding = 'utf8').readlines()
#print(corpus)

corpus = [preprocessing(sentence) for sentence in corpus if sentence.strip() != '']
#print(corpus)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

X_train_tokens = tokenizer.texts_to_sequences(corpus)
#print(X_train_tokens)


word2id = tokenizer.word_index
id2word = dict([(value, key) for (key, value) in word2id.items()])

vocab_size = len(word2id)+1
#print(vocab_size)

embed_size = 300
window_size = 2

def generate_context_word_pairs(corpus, window_size, vocab_size):
	context_length = window_size*2
	for words in corpus:
		sentence_length = len(words)
		for index, word in enumerate(words):
			context_words = []
			label_word = []
			start = index - window_size
			end = index + window_size + 1
			context_words.append([words[i] for i in range(start, end) if 0 <= i < sentence_length and i != index])
			label_word.append(word)

			x = pad_sequences(context_words, maxlen = context_length)
			y = to_categorical(label_word, vocab_size)
			yield (x,y)

i = 0
for x,y in generate_context_word_pairs(corpus=X_train_tokens, window_size=window_size, vocab_size=vocab_size):
	if 0 not in x[0]:
		print("context:", [id2word[w] for w in x[0]])
		print("target:", id2word[np.argwhere(y[0])[0][0]])

		if i == 10:
			break
		i += 1


model = Sequential()

model.add(Embedding(input_dim = vocab_size, output_dim=embed_size, embeddings_initializer='glorot_uniform', input_length=window_size*2))
model.add(Lambda(lambda x: K.mean(x, axis = 1), output_shape=(embed_size,)))
model.add(Dense(vocab_size, kernel_initializer='glorot_uniform', activation='softmax'))
print(model.summary())


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

n_epochs = 5
for epoch in range(n_epochs):
	loss = 0
	for x,y in generate_context_word_pairs(corpus=X_train_tokens, window_size=window_size, vocab_size=vocab_size):
		loss += model.train_on_batch(x,y)
	print("epoch:", epoch, "\tloss:", loss)

weights = model.get_weights()[0]
weights = weights[1:]
pd.DataFrame(weights, index = list(id2word.values())).head(10)

distance_matrix = cosine_similarity(weights)
print(distance_matrix.shape)

simillar_words = {search_term: [id2word[idx] for idx in distance_matrix[word2id[search_term]-1].argsort()[1:6]+1]
for search_term in ['system', 'sun', 'halley', 'kepler', 'discovery', 'ancient']}
print(simillar_words)


words = sum([[k]+v for k,v in simillar_words.items()], [])
words_id = [word2id[w] for w  in words]
word_vector = np.array([weights[idx] for idx in words_id])

tsne = TSNE(n_components=2, random_state=2018, n_iter=1000,perplexity=3)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(word_vector)
labels = words

plt.figure(figsize=(14,8))
plt.scatter(T[:,0], T[:, 1], c = 'red')
for label, x, y in zip(labels, T[:,0], T[:,1]):
	plt.annotate(label, xy=(x+1, y+1), xytext=(0,0), textcoords='offset points')
plt.show()


