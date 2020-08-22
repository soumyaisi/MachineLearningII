"""
How to use Fasttext using Keras.
"""

#Import Libraies and configurations...............
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
from gensim.models.fasttext import FastText
import matplotlib.pyplot as plt
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import  word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk import WordPunctTokenizer
from nltk.corpus import gutenberg


#Load the Data.............................
#nltk.download('gutenberg')
#print(gutenberg.fileids())

bible = gutenberg.raw('bible-kjv.txt')

bible_sents = sent_tokenize(bible)
print(len(bible_sents))


#Preprocessing the Data..................
remove_terms = punctuation + '0123456789'
def preprocessing(text):
	words = word_tokenize(text)
	tokens = [w for w in words if w.lower() not in remove_terms]
	stops = stopwords.words('english')
	tokens = [token for token in tokens if token not in stops]
	tokens = [word for word in tokens if word.isalpha()]
	lemma = WordNetLemmatizer()
	tokens = [lemma.lemmatize(word) for word in tokens]
	preprocessed_text = ' '.join(tokens)
	return preprocessed_text


corpus = [preprocessing(sentence) for sentence in bible_sents if sentence.strip() != '']
#print(corpus)

wpt = nltk.WordPunctTokenizer()
tokenized_corpus = [wpt.tokenize(doc) for doc in corpus]
print(tokenized_corpus[1])


#Using Fasttext..............
feature_size = 50
window_context = 10
min_word_count = 5
sample = 1e-3

fasttext_model = FastText(tokenized_corpus, size=feature_size, 
	window=window_context, min_count=min_word_count, 
	sample=sample, sg=1, iter=20)#1-skipgram and 0-CBOW


print(fasttext_model.wv['god'])


#Finding similar words..............
similar_words = {search_term: [item[0] for item in fasttext_model.wv.most_similar([search_term], topn=5)]
for search_term in ['god', 'jesus', 'egypt', 'john']}
print(similar_words)

print(fasttext_model.wv.similarity(w1 = 'god', w2 = 'jesus'))

sentence_1 = "satan god jesus"
print(fasttext_model.wv.doesnt_match(sentence_1.split()))

fasttext_model.save('fasttext')
loaded_model = FastText.load('fasttext')
print(loaded_model)


#Using PCA plot similar words..................
from sklearn.decomposition import PCA

words = sum([[k] + v for k,v in similar_words.items()],[])
wvs = fasttext_model.wv[words]
pca = PCA(n_components=2)
p = pca.fit_transform(wvs)
labels = words

plt.figure(figsize=(16, 10))
plt.scatter(p[:,0], p[:,1], c='blue')
for label,x,y in zip(labels, p[:,0], p[:,1]):
	plt.annotate(label, xy=(x+0.6, y+0.03), xytext=(0,0), textcoords='offset points')
plt.show()






