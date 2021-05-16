import random
import collections
import math
import os
import zipfile
import time
import re
import numpy as np
import tensorflow as tf
from matplotlib import pylab
#%matplotlib inline
from six.moves import range
from six.moves.urllib.request import urlretrieve
"""Make sure the dataset link is copied correctly"""
dataset_link = 'http://mattmahoney.net/dc/'
zip_file = 'text8.zip'

"""Downloading the required file"""
def data_download(zip_file):
	if not os.path.exists(zip_file):
		zip_file, _ = urlretrieve(dataset_link + zip_file, zip_file)
		print('File downloaded successfully!')
	return None
data_download(zip_file)

#Extracting the dataset in separate folder
extracted_folder = 'dataset'
if not os.path.isdir(extracted_folder):
	with zipfile.ZipFile(zip_file) as zf:
		zf.extractall(extracted_folder)
with open('dataset/text8') as ft_ :
	full_text = ft_.read()


"""Replacing punctuation marks with tokens"""
def text_processing(ft8_text):
	ft8_text = ft8_text.lower()
	ft8_text = ft8_text.replace('.', ' <period> ')
	ft8_text = ft8_text.replace(',', ' <comma> ')
	ft8_text = ft8_text.replace('"', ' <quotation> ')
	ft8_text = ft8_text.replace(';', ' <semicolon> ')
	ft8_text = ft8_text.replace('!', ' <exclamation> ')
	ft8_text = ft8_text.replace('?', ' <question> ')
	ft8_text = ft8_text.replace('(', ' <paren_l> ')
	ft8_text = ft8_text.replace(')', ' <paren_r> ')
	ft8_text = ft8_text.replace('--', ' <hyphen> ')
	ft8_text = ft8_text.replace(':', ' <colon> ')
	ft8_text_tokens = ft8_text.split()
	return ft8_text_tokens
ft_tokens = text_processing(full_text)


"""Shortlisting words with frequency more than 7"""
word_cnt = collections.Counter(ft_tokens)
shortlisted_words = [w for w in ft_tokens if word_cnt[w] > 7 ]
print(shortlisted_words[:15])

#Check the stats of the total words present in the dataset.
print("Total number of shortlisted words : ",len(shortlisted_words))
print("Unique number of shortlisted words : ",len(set(shortlisted_words)))

"""The function creates a dictionary of the words present in dataset along with their frequency order"""
def dict_creation(shortlisted_words):
	counts = collections.Counter(shortlisted_words)
	vocabulary = sorted(counts, key=counts.get, reverse=True)
	rev_dictionary_ = {ii: word for ii, word in enumerate(vocabulary)}
	dictionary_ = {word: ii for ii, word in rev_dictionary_.items()}
	return dictionary_, rev_dictionary_
dictionary_, rev_dictionary_ = dict_creation(shortlisted_words)
words_cnt = [dictionary_[word] for word in shortlisted_words]


"""Creating the threshold and performing the subsampling"""
thresh = 0.00005
word_counts = collections.Counter(words_cnt)
total_count = len(words_cnt)
freqs = {word: count/total_count for word,count in word_counts.items()}
p_drop = {word:1 - np.sqrt(thresh/freqs[word]) for word in word_counts}
train_words = [word for word in words_cnt if p_drop[word] < random.random()]

"""The function combines the words of given word_window size next to the index, for the SkipGram model"""
def skipG_target_set_generation(batch_, batch_index, word_window):
	random_num = np.random.randint(1, word_window+1)
	words_start = batch_index - random_num if (batch_index - random_num) > 0 else 0
	words_stop = batch_index + random_num
	window_target = set(batch_[words_start:batch_index] + batch_[batch_index+1:words_stop+1])
	return list(window_target)

"""The function internally makes use of the skipG_target_set_generation() function and combines each of the label words in the shortlisted_words with the words of word_window size around"""
def skipG_batch_creation(short_words, batch_length, word_window):
	batch_cnt = len(short_words)//batch_length
	short_words = short_words[:batch_cnt*batch_length]  
	for word_index in range(0, len(short_words), batch_length):
		input_words, label_words = [], []
		word_batch = short_words[word_index:word_index + batch_length]
		for index_ in range(len(word_batch)):
			batch_input = word_batch[index_]
			batch_label = skipG_target_set_generation(word_batch, index_, word_window)
			# Appending the label and inputs to the initial list. Replicating input to the size of labels in the window
			label_words.extend(batch_label)
			input_words.extend([batch_input]*len(batch_label))
			yield input_words, label_words

tf_graph = tf.Graph()
with tf_graph.as_default():
	input_ = tf.placeholder(tf.int32, [None], name='input_')
	label_ = tf.placeholder(tf.int32, [None, None], name='label_')

with tf_graph.as_default():
	word_embed = tf.Variable(tf.random_uniform((len(rev_dictionary_), 300), -1, 1))
	embedding = tf.nn.embedding_lookup(word_embed, input_)

"""The code includes the following :
# Initializing weights and bias to be used in the softmax layer
 # Loss function calculation using the Negative Sampling
 # Usage of Adam Optimizer
 # Negative sampling on 100 words, to be included in the loss function
 # 300 is the word embedding vector size
"""
vocabulary_size = len(rev_dictionary_)
with tf_graph.as_default():
	sf_weights = tf.Variable(tf.truncated_normal((vocabulary_size, 300), stddev=0.1) )
	sf_bias = tf.Variable(tf.zeros(vocabulary_size) )
	loss_fn = tf.nn.sampled_softmax_loss(weights = sf_weights, biases = sf_bias, labels=label_, inputs=embedding,num_sampled=100, num_classes=vocabulary_size)
	cost_fn = tf.reduce_mean(loss_fn)
	optim = tf.train.AdamOptimizer().minimize(cost_fn)

"""The below code performs the following operations :
 # Performing validation here by making use of a random selection of 16 words from the dictionary of desired size
 # Selecting 8 words randomly from range of 1000    
 # Using the cosine distance to calculate the similarity between the words
"""
with tf_graph.as_default():
	validation_cnt = 16
	validation_dict = 100
	validation_words = np.array(random.sample(range(validation_dict), validation_cnt//2))
	validation_words = np.append(validation_words, random.sample(range(1000,1000+validation_dict), validation_cnt//2))
	validation_data = tf.constant(validation_words, dtype=tf.int32)
	normalization_embed = word_embed / (tf.sqrt(tf.reduce_sum(tf.square(word_embed), 1, keep_dims=True)))
	validation_embed = tf.nn.embedding_lookup(normalization_embed, validation_data)
	word_similarity = tf.matmul(validation_embed,tf.transpose(normalization_embed))

"""Creating the model checkpoint directory"""
#!mkdir model_checkpoint
epochs = 2            # Increase it as per computation resources. It has been kept low here for users to replicate the process, increase to 100 or more
batch_length = 1000
word_window = 10
with tf_graph.as_default():
	saver = tf.train.Saver()
with tf.Session(graph=tf_graph) as sess:
	iteration = 1
	loss = 0
	sess.run(tf.global_variables_initializer())
	for e in range(1, epochs+1):
		batches = skipG_batch_creation(train_words, batch_length, word_window)
		start = time.time()
		for x, y in batches:
			train_loss, _ = sess.run([cost_fn, optim],feed_dict={input_: x, label_: np.array(y)[:,None]})
			loss += train_loss
			if iteration % 100 == 0:
				end = time.time()
				print("Epoch {}/{}".format(e, epochs), "Iteraion: {}".format(iteration), "Avg Training loss: {:.4f}".format(loss/100))
				start_ = (end - start) / 100.0
				print("Processing: {:.4f} sec/batch".format(start_))
				loss = 0
				start = time.time()
			if iteration % 2000 == 0:
				similarity_ = word_similarity.eval()
				for i in range(validation_cnt):
					validated_words = rev_dictionary_[validation_words[i]]
					top_k = 8 # number of nearest neighbors
					nearest = (-similarity_[i, :]).argsort()[1:top_k+1]
					log = 'Nearest to %s:' % validated_words
					for k in range(top_k):
						close_word = rev_dictionary_[nearest[k]]
						log = '%s %s,' % (log, close_word)
					print(log)
			iteration += 1
		save_path = saver.save(sess, "model_checkpoint/skipGram_text8.ckpt")
		embed_mat = sess.run(normalization_embed)


"""The Saver class adds ops to save and restore variables to and from checkpoints."""
with tf_graph.as_default():
	saver = tf.train.Saver()
with tf.Session(graph=tf_graph) as sess:
	"""Restoring the trained network"""
	saver.restore(sess, tf.train.latest_checkpoint('model_checkpoint'))
	embed_mat = sess.run(word_embed)

from sklearn.manifold import TSNE
word_graph = 250
tsne = TSNE()
word_embedding_tsne = tsne.fit_transform(embed_mat[:word_graph, :])
print(word_embedding_tsne.shape)



import matplotlib.pyplot as plt
def skipgram_plot(embeddings, labels):
	#assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
	pylab.figure(figsize=(12,12))
	for i, label in enumerate(labels):
		x, y = embeddings[i,:]
		pylab.scatter(x, y)
		pylab.annotate(label, xy=(x, y), xytext=(5, 2),textcoords='offset points', ha='right', va='bottom')
	pylab.show()

words = [rev_dictionary_[i] for i in range(1, word_graph+1)]
#skipgram_plot(word_embedding_tsne, words)

labels = words
plt.figure(figsize=(12, 12))
plt.scatter(word_embedding_tsne[:, 0], word_embedding_tsne[:, 1], c='steelblue', edgecolors='k')
for label, x, y in zip(labels, word_embedding_tsne[:, 0], word_embedding_tsne[:, 1]):
	plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')
plt.show()