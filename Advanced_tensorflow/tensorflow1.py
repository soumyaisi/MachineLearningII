from __future__ import print_function, division
from  builtins import range

import  numpy as np
import tensorflow as tf

#specify the type
A = tf.placeholder(tf.float32, shape=(5, 5), name = 'A')
v = tf.placeholder(tf.float32)

w = tf.matmul(A, v)

with tf.Session() as session:
	output = session.run(w, feed_dict = {A: np.random.randn(5, 5), v: np.random.randn(5, 1)})
	print(output, type(output))


shape = (2, 2)
x = tf.Variable(tf.random_normal(shape))
t = tf.Variable(0)

#initialize the variable first
init = tf.global_variables_initializer()

with tf.Session() as session:
	out = session.run(init)
	print(out)
	print(x.eval())
	print(t.eval())

u = tf.Variable(20.0)
cost = u*u + u + 1.0

# One difference between Theano and TensorFlow is that you don't write the updates
# yourself in TensorFlow. You choose an optimizer that implements the algorithm you want.
# 0.3 is the learning rate. Documentation lists the params.
train_op = tf.train.GradientDescentOptimizer(0.3).minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as session:
	session.run(init)

	for i in range(12):
		session.run(train_op)
		print("i = %d, cost = %.3f, u = %.3f" % (i, cost.eval(), u.eval()))


