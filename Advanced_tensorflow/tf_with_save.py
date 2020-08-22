from __future__ import print_function, division
from  builtins import range
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util import get_normalized_data, y2indicator

def error_rate(p, t):
	return np.mean(p != t)

class TFLogistic:
	def __init__(self, savefile, D = None, K = None):
		self.savefile = savefile
		if D and K:
			self.build(D, K)

	def build(self, D, K):
		W0 = np.random.randn(D, K) / np.sqrt(2.0 / D)
		b0 = np.zeros(K)

		self.inputs = tf.placeholder(tf.float32, shape = (None, D), name = 'inputs')
		self.targets = tf.placeholder(tf.int64, shape = (None, ), name = 'targets')
		self.W = tf.Variable(W0.astype(np.float32), name = 'W')
		self.b = tf.Variable(b0.astype(np.float32), name = 'b')

		self.saver = tf.train.Saver({'W': self.W, 'b': self.b})

		logits = tf.matmul(self.inputs, self.W) + self.b
		cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = self.targets))
		self.predict_op = tf.argmax(logits, 1)
		return cost

	def fit(self, X, Y, Xtest, Ytest):
		N,D = X.shape
		K = len(set(Y))
		max_iter = 10
		lr = 1e-3
		mu = 0.9
		reguralization = 1e-1
		batch_sz = 100
		n_batches = N // batch_sz

		cost = self.build(D, K)
		l2_penalty = reguralization*tf.reduce_mean(self.W**2) / 2
		cost += l2_penalty
		train_op = tf.train.MomentumOptimizer(lr, momentum = mu).minimize(cost)

		costs = []
		init = tf.global_variables_initializer()
		with tf.Session() as session:
			session.run(init)
			for i in range(max_iter):
				for j in range(n_batches):
					Xbatch = X[j*batch_sz: (j*batch_sz + batch_sz),]
					Ybatch = Y[j*batch_sz: (j*batch_sz + batch_sz),]
					session.run(train_op, feed_dict = {self.inputs: Xbatch, self.targets: Ybatch})
					if j%200 == 0:
						test_cost = session.run(cost, feed_dict = {self.inputs: Xtest, self.targets: Ytest})
						Ptest = session.run(self.predict_op, feed_dict = {self.inputs: Xtest})
						err = error_rate(Ptest, Ytest)
						print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
						costs.append(test_cost)
			self.saver.save(session, self.savefile)						
		self.D = D
		self.K = K

		plt.plot(costs)
		plt.show()


	def predict(self, X):
		with tf.Session() as session:
			self.saver.restore(session, self.savefile)
			p = session.run(self.predict_op, feed_dict = {self.inputs: X})
		return p



	def score(self, X, Y):
		return 1 - error_rate(self.predict(X), Y)

	def save(self, filename):
		j = {'D': self.D, 'K': self.K, 'model': self.savefile}
		with open(filename, 'w') as f:
			json.dump(j, f)

	@staticmethod
	def	load(filename):
		with open(filename) as f:
			j = json.load(f)
		return TFLogistic(j['model'], j['D'], j['K'])		

def main():
	Xtrain,Xtest, Ytrain, Ytest = get_normalized_data()
	model = TFLogistic("./tf.model")
	model.fit(Xtrain, Ytrain, Xtest, Ytest)
	print("final train accuracy:", model.score(Xtrain, Ytrain))
	print("final test accuracy:", model.score(Xtest, Ytest))
	model.save("my_trained.json")
	model = TFLogistic.load("my_trained.json")
	print("final train accuracy (after reload):", model.score(Xtrain, Ytrain))
	print("final test accuracy (after reload):", model.score(Xtest, Ytest))

if __name__ == '__main__':
	main()

