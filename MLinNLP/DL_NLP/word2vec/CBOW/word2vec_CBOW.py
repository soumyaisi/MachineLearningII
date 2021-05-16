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


data_index = 0
def cbow_batch_creation(batch_length, word_window):
	"""The function creates a batch with the list of the label words and list of their corresponding words in the context of the label word."""
	global data_index
	"""Pulling out the centered label word, and its next word_window count of surrounding words
	word_window : window of words on either side of the center word
	relevant_words : length of the total words to be picked in a single batch, including the center word and the word_ window words on both sides
	Format :  [ word_window ... target ... word_window ] """
	relevant_words = 2 * word_window + 1
	batch = np.ndarray(shape=(batch_length,relevant_words-1),dtype=np.int32)
	label_ = np.ndarray(shape=(batch_length, 1), dtype=np.int32)
	buffer = collections.deque(maxlen=relevant_words)   
	# Queue to add/pop
	#Selecting the words of length 'relevant_words' from the starting index
	for _ in range(relevant_words):
		buffer.append(words_cnt[data_index])
		data_index = (data_index + 1) % len(words_cnt)
	for i in range(batch_length):
		target = word_window  # Center word as the label
		target_to_avoid = [ word_window ] # Excluding the label, and selecting only the surrounding words
		# add selected target to avoid_list for next time
		col_idx = 0
		for j in range(relevant_words):
			if j==relevant_words//2:
				continue
			batch[i,col_idx] = buffer[j] # Iterating till the middle element for window_size length
			col_idx += 1
		label_[i, 0] = buffer[target]
		buffer.append(words_cnt[data_index])
		data_index = (data_index + 1) % len(words_cnt)
	assert batch.shape[0]==batch_length and batch.shape[1]==relevant_words-1
	return batch, label_


for num_skips, word_window in [(2, 1), (4, 2)]:
	data_index = 0
	batch, label_ = cbow_batch_creation(batch_length=8, word_window=word_window)
	print('\nwith num_skips = %d and word_window = %d:' % (num_skips, word_window))
	print('batch:', [[rev_dictionary_[bii] for bii in bi] for bi in batch])
	print('label_:', [rev_dictionary_[li] for li in label_.reshape(8)])


num_steps = 100001
#num_steps = 100

"""Initializing :
   # 128 is the length of the batch considered for CBOW
   # 128 is the word embedding vector size
   # Considering 1 word on both sides of the center label words
   # Consider the center label word 2 times to create the batches
"""
batch_length = 128
embedding_size = 128
skip_window = 1
num_skips = 2


"""The below code performs the following operations :
 # Performing validation here by making use of a random selection of 16 words from the dictionary of desired size
 # Selecting 8 words randomly from range of 1000    
 # Using the cosine distance to calculate the similarity between the words
"""
tf_cbow_graph = tf.Graph()
with tf_cbow_graph.as_default():
	validation_cnt = 16
	validation_dict = 100
	validation_words = np.array(random.sample(range(validation_dict), validation_cnt//2))
	validation_words = np.append(validation_words,random.sample(range(1000,1000+validation_dict), validation_cnt//2))
	train_dataset = tf.placeholder(tf.int32, shape=[batch_length,2*skip_window])
	train_labels = tf.placeholder(tf.int32, shape=[batch_length, 1])
	validation_data = tf.constant(validation_words, dtype=tf.int32)


"""
Embeddings for all the words present in the vocabulary
"""
with tf_cbow_graph.as_default() :
	vocabulary_size = len(rev_dictionary_)
	word_embed = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
	# Averaging embeddings accross the full context into a single embedding layer
	context_embeddings = []
	for i in range(2*skip_window):
		context_embeddings.append(tf.nn.embedding_lookup(word_embed, train_dataset[:,i]))
	embedding =  tf.reduce_mean(tf.stack(axis=0,values=context_embeddings),0,keep_dims=False)

"""The code includes the following :
 # Initializing weights and bias to be used in the softmax layer
 # Loss function calculation using the Negative Sampling
 # Usage of AdaGrad Optimizer
 # Negative sampling on 64 words, to be included in the loss function
"""
with tf_cbow_graph.as_default() :
	sf_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
	sf_bias = tf.Variable(tf.zeros([vocabulary_size]))
	loss_fn = tf.nn.sampled_softmax_loss(weights=sf_weights,biases=sf_bias, inputs=embedding,labels=train_labels, num_sampled=64,num_classes=vocabulary_size)
	cost_fn = tf.reduce_mean(loss_fn)
	"""Using AdaGrad as optimizer"""
	optim = tf.train.AdagradOptimizer(1.0).minimize(cost_fn)

"""
Using the cosine distance to calculate the similarity between
the batches and embeddings of other words
"""
with tf_cbow_graph.as_default() :
	normalization_embed = word_embed / tf.sqrt(tf.reduce_sum(tf.square(word_embed), 1, keep_dims=True))
	validation_embed = tf.nn.embedding_lookup(normalization_embed, validation_data)
	word_similarity = tf.matmul(validation_embed,tf.transpose(normalization_embed))


with tf.Session(graph=tf_cbow_graph) as sess:
	sess.run(tf.global_variables_initializer())
	avg_loss = 0
	for step in range(num_steps):
		batch_words, batch_label_ = cbow_batch_creation(batch_length, skip_window)
		_, l = sess.run([optim, loss_fn], feed_dict={train_dataset : batch_words, train_labels : batch_label_ })
		avg_loss += l
		if step % 2000 == 0 :
			if step > 0 :
				avg_loss = avg_loss / 2000
			print('Average loss at step %d: %f' % (step,np.mean(avg_loss) ))
			avg_loss = 0
		if step % 10000 == 0:
			sim = word_similarity.eval()
		for i in range(validation_cnt):
			valid_word = rev_dictionary_[validation_words[i]]
			top_k = 8 # number of nearest neighbors
			nearest = (-sim[i, :]).argsort()[1:top_k+1]
			log = 'Nearest to %s:' % valid_word
			for k in range(top_k):
				close_word = rev_dictionary_[nearest[k]]
				log = '%s %s,' % (log, close_word)
				print(log)
	final_embeddings = normalization_embed.eval(session=sess)

from sklearn.manifold import TSNE

num_points = 250
tsne = TSNE(perplexity=30, n_components=2, init='pca',n_iter=5000)
embeddings_2d = tsne.fit_transform(final_embeddings[1:num_points+1, :])

#The cbow_plot() function plots the dimensionally reduced vectors.
def cbow_plot(embeddings, labels):
	assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
	pylab.figure(figsize=(12,12))
	for i, label in enumerate(labels):
		x, y = embeddings[i,:]
		pylab.scatter(x, y)
		pylab.annotate(label, xy=(x, y), xytext=(5, 2),textcoords='offset points', ha='right', va='bottom')
	pylab.show()

words = [rev_dictionary_[i] for i in range(1, num_points+1)]
cbow_plot(embeddings_2d, words)










