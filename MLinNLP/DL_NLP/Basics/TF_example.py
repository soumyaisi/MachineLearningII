import tensorflow as tf
hello = tf.constant('Hello, Tensors!')
sess = tf.Session()
print(sess.run(hello))


# Mathematical computation
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a+b))


import tensorflow as tf
import numpy as np
mat_1 = 10*np.random.random_sample((3, 4))   # Creating NumPy arrays
mat_2 = 10*np.random.random_sample((4, 6))
# Creating a pair of constant ops, and including the above made matrices
tf_mat_1 = tf.constant(mat_1)
tf_mat_2 = tf.constant(mat_2)
#Multiplying TensorFlow matrices with matrix multiplication operation
tf_mat_prod = tf.matmul(tf_mat_1 , tf_mat_2)
sess = tf.Session()            # Launching a session
# run() executes required ops and performs the request to store output in 'mult_matrix' variable
mult_matrix = sess.run(tf_mat_prod)
print(mult_matrix)
# Performing constant operations with the addition and subtraction of two constants
a = tf.constant(10)
b = tf.constant(20)
print("Addition of constants 10 and 20 is %i " % sess.run(a+b))
print("Subtraction of constants 10 and 20 is %i " % sess.run(a-b))
sess.close() 
