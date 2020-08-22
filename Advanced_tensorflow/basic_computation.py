"""
Basic computation in tensorflow-2.
"""

import tensorflow as tf

a = tf.constant(3.0)
b = tf.constant(4.0)
c = tf.sqrt(a**2 + b**2)
print(c)
#print(f"c: {c}")

a = tf.constant([1,2,3])
b = tf.constant([4,5,6])
c = tf.tensordot(a,b,axes = [0,0])
print(c)


import  numpy as np
A0 = np.random.randn(3,3)
B0 = np.random.randn(3,1)
A = tf.constant(A0)
B = tf.constant(B0)
c = tf.matmul(A, B)
print(c)

A = tf.constant([[1,2],[3,4]])
b = tf.constant(1)
c = A + b
print(c)

A = tf.constant([[1,2], [3,4]])
B = tf.constant([[2,3], [4,5]])
c = A*B
print(c)

