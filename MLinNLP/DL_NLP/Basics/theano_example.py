#sudo pip install Theano


import theano
import theano.tensor as T
import numpy
from theano import function
#Variables 'x' and 'y' are defined
x = T.dscalar('x')               # dscalar : Theano datatype
y = T.dscalar('y')
# 'x' and 'y' are instances of TensorVariable, and are of dscalar theano type
print(type(x))
print(x.type)
print(T.dscalar)
# 'z' represents the sum of 'x' and 'y' variables. Theano's pp function, pretty-print out, is used to display the computation of the variable 'z'
z = x + y
from theano import pp
print(pp(z))

