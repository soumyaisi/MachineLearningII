from __future__ import print_function, division
from builtins import range

import theano.tensor as T

#variable
c = T.scalar('c')
v = T.vector('v')
A = T.matrix('A')

#define multiplication
w = A.dot(v)

#how it takes values
import theano

matrix_times_vector = theano.function(inputs = [A, v], outputs = w)

import numpy as np
A_val = np.array([[1,2], [3,4]])
v_val = np.array([5,6])

w_val = matrix_times_vector(A_val, v_val)
print(w_val)

x = theano.shared(20.0, 'x')
cost = x*x + x + 1

x_updates = x - 0.3*T.grad(cost, x)

train = theano.function(inputs = [], outputs = cost, updates = [(x, x_updates)])

for i in range(25):
	cost_val = train()
	print(cost_val)

#optimal value of x
print(x.get_value())