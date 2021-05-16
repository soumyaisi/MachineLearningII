
#https://s3.amazonaws.com/text-datasets/imdb_full.pkl

# Importing TensorFlow and IMDb dataset from keras library
from keras.datasets import imdb
import tensorflow as tf
print(tf.__version__)
from __future__ import print_function
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Creating Train and Test datasets from labeled movie reviews
(X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb_full.pkl",num_words=None, skip_top=0,maxlen=None, seed=113, tart_char=1, oov_char=2, index_from=3)


t = [item for sublist in X_train for item in sublist]
vocabulary = len(set(t))+1  
a = [len(x) for x in X_train]
plt.plot(a)
plt.show()


max_length = 200 # specifying the max length of the sequence in
the sentence
x_filter = []
y_filter = []
# If the selected length is lesser than the specified max_length, 200, then appending padding (0), else only selecting desired length only from sentence
for i in range(len(X_train)):
	if len(X_train[i])<max_length:
		a = len(X_train[i])
		X_train[i] = X_train[i] + [0] * (max_length - a)
		x_filter.append(X_train[i])
		y_filter.append(y_train[i])
	elif len(X_train[i])>max_length:
		X_train[i] = X_train[i][0:max_length]

#declaring the hyper params
embedding_size = 100   # word vector size for initializing the word embeddings
n_hidden = 200
learning_rate = 0.06
training_iters = 100000
batch_size = 32
beta =0.0001

n_steps = max_length         # timestepswords
n_classes = 2                # 0 /1 : binary classification for negative and positive reviews
da = 350                     # hyper-parameter : Self-attention MLP has hidden layer with da units
r = 30                       #count of different parts to be extracted from sentence (= number of rows in matrix embedding)
display_step =10
hidden_units = 3000

#Transform the training dataset values and labels in the desired format of array post transformation and encoding, respectively.
y_train = np.asarray(pd.get_dummies(y_filter))
X_train = np.asarray([np.asarray(g) for g in x_filter])
#Create an internal folder to record logs.
logs_path = './recent_logs/'
#Create a DataIterator class, to yield random data in batches of given batch size.

class DataIterator:
	""" Collects data and yields bunch of batches of data Takes data sources and batch_size as arguments """
	def __init__(self, data1,data2, batch_size):
		self.data1 = data1
		self.data2 = data2
		self.batch_size = batch_size
		self.iter = self.make_random_iter()
	def next_batch(self):
		try:
			idxs = next(self.iter)
		except StopIteration:
			self.iter = self.make_random_iter()
			idxs = next(self.iter)
		X =[self.data1[i] for i in idxs]
		Y =[self.data2[i] for i in idxs]
		X = np.array(X)
		Y = np.array(Y)
		return X, Y

def make_random_iter(self):
	splits = np.arange(self.batch_size, len(self.data1),self.batch_size)
	it = np.split(np.random.permutation(range(len(self.data1))), splits)[:-1]
	return iter(it)


# TF Graph Input
with tf.name_scope("weights"):
	Win  = tf.Variable(tf.random_uniform([n_hidden*r, hidden_units],-1/np.sqrt(n_hidden),1/np.sqrt(n_hidden)), name='W-­input')
	Wout = tf.Variable(tf.random_uniform([hidden_units,n_classes],-1/np.sqrt(hidden_units),1/np.sqrt(hidden_units)), name='W-out')
	Ws1  = tf.Variable(tf.random_uniform([da,n_hidden],-1/np.sqrt(da),1/np.sqrt(da)), name='Ws1')
	Ws2  = tf.Variable(tf.random_uniform([r,da],-1/np.sqrt(r),1/np.sqrt(r)), name='Ws2')


with tf.name_scope("biases"):            
	biasesout = tf.Variable(tf.random_normal([n_classes]),name='biases-out')
	biasesin  = tf.Variable(tf.random_normal([hidden_units]),name='biases-in')

with tf.name_scope('input'):
	x = tf.placeholder("int32", [32,max_length], name='x-­input')
	y = tf.placeholder("int32", [32, 2], name='y-input')

with tf.name_scope('embedding'):
	embeddings = tf.Variable(tf.random_uniform([vocabulary,embedding_size],-1, 1), name='embeddings')
	embed = tf.nn.embedding_lookup(embeddings,x)


def length(sequence):
	#Computing maximum of elements across dimensions of a tensor
	used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))   
	length = tf.reduce_sum(used, reduction_indices=1)
	length = tf.cast(length, tf.int32)
	return length


#Reuse the weights and biases using the following:
with tf.variable_scope('forward',reuse=True):
	lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden)
with tf.name_scope('model'):
	outputs, states = rnn.dynamic_rnn(lstm_fw_cell,embed,sequence_length=length(embed),dtype=tf.float32,time_major=False)    
	#in the next step we multiply the hidden-vec matrix with the Ws1 by reshaping
	h = tf.nn.tanh(tf.transpose(tf.reshape(tf.matmul(Ws1,tf.reshape(outputs,[n_hidden,batch_size*n_steps])),  [da,batch_size,n_steps]),[1,0,2]))
	#in this step we multiply the generated matrix with Ws2
	a = tf.reshape(tf.matmul(Ws2,tf.reshape(h,[da,batch_size*n_steps])),[batch_size,r,n_steps])
	def fn3(a,x):
		return tf.nn.softmax(x)
	h3 = tf.scan(fn3,a)
with tf.name_scope('flattening'):
	# here we again multiply(batch) of the generated batch with the same hidden matrix
	h4 = tf.matmul(h3,outputs)
	# flattening the output embedded matrix
	last = tf.reshape(h4,[-1,r*n_hidden])

with tf.name_scope('MLP'):
	tf.nn.dropout(last,.5, noise_shape=None, seed=None,name=None)
	pred1 = tf.nn.sigmoid(tf.matmul(last,Win)+biasesin)
	pred  = tf.matmul(pred1, Wout) + biasesout
#Define loss and optimizer
with tf.name_scope('cross'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits =pred, labels = y) + beta*tf.nn.l2_loss(Ws2) )
with tf.name_scope('train'):
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	gvs = optimizer.compute_gradients(cost)
	capped_gvs = [(tf.clip_by_norm(grad,0.5), var) for grad,var in gvs]
	optimizer.apply_gradients(capped_gvs)
	optimized = optimizer.minimize(cost)

#Evaluate model
with tf.name_scope('Accuracy'):
	orrect_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	accuracy     = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
tf.summary.scalar("cost", cost)
tf.summary.scalar("accuracy", accuracy)


summary_op =tf.summary.merge_all()
# Initializing the variables
train_iter = DataIterator(X_train,y_train, batch_size)    
init = tf.global_variables_initializer()


with tf.Session() as sess:
	sess.run(init)
	# Creating log file writer object
	writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
	step = 1
	# Keep training until reach max iterations
	while step * batch_size < training_iters:
		batch_x, batch_y = train_iter.next_batch()
		sess.run(optimized, feed_dict={x: batch_x, y: batch_y})
		# Executing the summary operation in the session
		summary = sess.run(summary_op, feed_dict={x: batch_x,y: batch_y})
		# Writing the values in log file using the FileWriter object created above
		writer.add_summary(summary,  step*batch_size)
		if step % display_step == 2:
			# Calculate batch accuracy
			acc = sess.run(accuracy, feed_dict={x: batch_x, y:batch_y})
			# Calculate batch loss
			loss = sess.run(cost, feed_dict={x: batch_x, y:batch_y})
			print ("Iter " + str(step*batch_size) + ",Minibatch Loss= " + "{:.6f}".format(loss)+ ", Training Accuracy= " + "{:.2f}".format(acc*100) + "%")
		step += 1
	print ("Optimization Finished!")


#To start the TensorBoard, specify the port, per your choice:
#tensorboard --logdir=./ --port=6006.



















