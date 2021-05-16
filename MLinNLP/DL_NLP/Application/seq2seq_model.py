# Importing the required packages
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from gensim.models import KeyedVectors


import keras
print(keras.__version__)
import tensorflow
print(tensorflow.__version__)

#www.sciencedirect.com/science/article/pii/S1532046412000615
#http://evexdb.org/pmresources/vec-space-­models/wikipedia-pubmed-and-PMC-w2v.bin


EMBEDDING_FILE = 'wikipedia-pubmed-and-PMC-w2v.bin'
print('Indexing word vectors')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE,binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))


import copy
from keras.preprocessing.sequence import pad_sequences

#The ADE corpus used from the paper by Gurulingappa is distributed with three files: DRUG-AE.rel, DRUG-DOSE.rel, and ADE-NEG.txt. 
#We are making use of the DRUG-AE.rel file, which provides relationships between drugs and adverse effects.

"""
10082597 | METHODS: We report two cases of pseudoporphyria caused by naproxen and oxaprozin. | pseudoporphyria | 620 | 635| oxaprozin | 659 | 668
The format of the DRUG-AE.rel file is as follows, fields are separated by pipe delimiters:

Column-1: PubMed-ID
Column-2: Sentence
Column-3: Adverse-Effect
Column-4: Begin offset of Adverse-Effect at ‘document level’
Column-5: End offset of Adverse-Effect at ‘document level’
Column-6: Drug
Column-7: Begin offset of Drug at ‘document level’
Column-8: End offset of Drug at ‘document level’
"""

TEXT_FILE = 'DRUG-AE.rel'
f = open(TEXT_FILE, 'r')
for each_line in f.readlines():
	sent_list = np.zeros([0,200])
	labels = np.zeros([0,3])
	tokens = each_line.split("|")
	sent = tokens[1]
	if sent in sentences:
		continue
	sentences.append(sent)
	begin_offset = int(tokens[3])
	end_offset = int(tokens[4])
	mid_offset = range(begin_offset+1, end_offset)
	word_tokens = nltk.word_tokenize(sent)
	offset = 0
	for each_token in word_tokens:
		offset = sent.find(each_token, offset)
		offset1 = copy.deepcopy(offset)
		offset += len(each_token)
		if each_token in punctuation or re.search(r'\d', each_token):
			continue
		each_token = each_token.lower()
		each_token = re.sub("[^A-Za-z\-]+","", each_token)
		if each_token in word2vec.vocab:
			new_word = word2vec.word_vec(each_token)
		if offset1 == begin_offset:
			sent_list = np.append(sent_list, np.array([new_word]), axis=0)
			labels = np.append(labels, np.array([[0,0,1]]),axis=0)
	elif offset == end_offset or offset in mid_offset:
		sent_list = np.append(sent_list, np.array([new_word]), axis=0)
		labels = np.append(labels, np.array([[0,1,0]]),axis=0)
	else:
		sent_list = np.append(sent_list, np.array([new_word]), axis=0)
		labels = np.append(labels, np.array([[1,0,0]]),axis=0)
	input_data_ae.append(sent_list)
	op_labels_ae.append(labels)
input_data_ae = np.array(input_data_ae)
op_labels_ae  = np.array(op_labels_ae)


input_data_ae = pad_sequences(input_data_ae, maxlen=30,dtype='float64', padding='post')
op_labels_ae = pad_sequences(op_labels_ae, maxlen=30,dtype='float64', padding='post')

print(len(input_data_ae))
print(len(op_labels_ae))

from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,Bidirectional, TimeDistributed
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint


# Creating Train and Validation datasets, for 4271 entries,4000 in train dataset, and 271 in validation dataset
x_train= input_data_ae[:4000]
x_test = input_data_ae[4000:]
y_train = op_labels_ae[:4000]
y_test =op_labels_ae[4000:]


batch = 1      # Making the batch size as 1, as showing model each of the instances one-by-one
#Adding Bidirectional LSTM with Dropout, and Time Distributed layer with Dropout
#Finally using Adam optimizer for training purpose

xin = Input(batch_shape=(batch,30,200), dtype='float')
seq = Bidirectional(LSTM(300, return_sequences=True),merge_
mode='concat')(xin)
mlp1 = Dropout(0.2)(seq)
mlp2 = TimeDistributed(Dense(60, activation='softmax'))(mlp1)
mlp3 = Dropout(0.2)(mlp2)
mlp4 = TimeDistributed(Dense(3, activation='softmax'))(mlp3)
model = Model(inputs=xin, outputs=mlp4)
model.compile(optimizer='Adam', loss='categorical_crossentropy')
model.fit(x_train, y_train,batch_size=batch,epochs=50,validation_data=(x_test, y_test))


val_pred = model.predict(x_test,batch_size=batch)
labels = []
for i in range(len(val_pred)):
	b = np.zeros_like(val_pred[i])
	b[np.arange(len(val_pred[i])), val_pred[i].argmax(1)] = 1
	labels.append(b)
print(val_pred.shape)

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score  
score =[]
f1 = []
precision =[]
recall =[]
point = []

for i in range(len(y_test)):
	if(f1_score(labels[i],y_test[i],average='weighted')>.6):
		point.append(i)
	score.append(f1_score(labels[i],y_test[i],average='weighted'))
	precision.append(precision_score(labels[i],y_test[i],average='weighted'))
	recall.append(recall_score(labels[i],y_test[i],average='weighted'))

print(len(point)/len(labels)*100)
print(np.mean(score))
print(np.mean(precision))
print(np.mean(recall))

