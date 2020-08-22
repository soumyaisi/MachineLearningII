import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

import numpy as np
from keras.preprocessing.text import Tokenizer
from  keras.preprocessing.sequence import skipgrams
from keras.models import Sequential, Model
from  keras.layers import Reshape, Embedding, Input, Activation
from keras.layers.merge import Dot
from keras.utils import np_utils
from keras.utils.data_utils import get_file
import gensim

from  string import punctuation
from  nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
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

vocab_size = len(tokenizer.word_index)+1
#print(vocab_size)

items = tokenizer.word_index.items()

dim_embedding = 300
#inputs
inputs = Input(shape=(1,), dtype = 'int32')
w = Embedding(vocab_size, dim_embedding)(inputs)

#context
c_inputs = Input(shape=(1,), dtype='int32')
c = Embedding(vocab_size, dim_embedding)(c_inputs)

d = Dot(axes=2)([w,c])

d = Reshape((1,), input_shape=(1,1))(d)
d = Activation('sigmoid')(d)

model = Model(inputs=[inputs, c_inputs], outputs = d)

print(model.summary())

model.compile(loss = 'binary_crossentropy', optimizer = 'adam')

n_epochs = 15
for epoch in range(n_epochs):
	loss = 0
	for i, doc in enumerate(X_train_tokens):
		#print(i,doc)
		data,labels = skipgrams(sequence = doc, vocabulary_size = vocab_size, window_size=4)
		#print(data, labels)
		x = [np.array(x) for x in zip(*data)]
		y = np.array(labels, dtype=np.int32)
		#print(x,y)

		if x:
			loss += model.train_on_batch(x,y)

	print("epoch:", epoch, '\tloss:',loss)

f = open('skipgram.txt', 'w', encoding='utf8')
f.write('{} {}\n'.format(vocab_size-1, dim_embedding))

weights = model.get_weights()[0]
for word, i in items:
	f.write('{} {}\n'.format(word, ' '.join(map(str, list(weights[i, :])))))
f.close()


w2v = gensim.models.KeyedVectors.load_word2vec_format('skipgram.txt', binary = False)

print(w2v.most_similar(positive=['solar']))
print(w2v.most_similar(positive=['kepler']))
