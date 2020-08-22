"""
Basic example of tensorflow variable and gradient descent.
"""


import tensorflow as tf

a = tf.Variable(5.)
b = tf.Variable(3.)
print(a*b)

a = a + 1
print(a)

c = tf.constant(4.)
print(a*b + c)

#gradient descent.........
w = tf.Variable(5.)
def get_loss(w):
	return w**2
def get_grad(w):
	with tf.GradientTape() as tape:
		L = get_loss(w)
	g = tape.gradient(L, w)
	return g
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1)

losses = []
for i in range(50):
	g = get_grad(w)
	optimizer.apply_gradients(zip([g], [w]))
	losses.append(get_loss(w))

import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()

w = tf.Variable(5.)
losses = []
for i in range(50):
	w = w - 0.1*2*w
	losses.append(w**2)

plt.plot(losses)
plt.show()

