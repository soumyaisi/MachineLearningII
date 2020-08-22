"""
Implemented in keras.
"""


from __future__ import print_function, division
from builtins import range
from keras.models import Sequential
from keras.layers import Dense, Activation
from util import get_normalized_data, y2indicator
import matplotlib.pyplot as plt

Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

N, D = Xtrain.shape
K = len(set(Ytrain))

# by default Keras wants one-hot encoded labels
# there's another cost function we can use
# where we can just pass in the integer labels directly
# just like Tensorflow / Theano
Ytrain = y2indicator(Ytrain)
Ytest = y2indicator(Ytest)


model = Sequential()


# ANN with layers [784] -> [500] -> [300] -> [10]
model.add(Dense(units=500, input_dim=D))
model.add(Activation('relu'))
model.add(Dense(units=300)) 
model.add(Activation('relu'))
model.add(Dense(units=K))
model.add(Activation('softmax'))


# list of losses: https://keras.io/losses/
# list of optimizers: https://keras.io/optimizers/
# list of metrics: https://keras.io/metrics/
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# gives us back a <keras.callbacks.History object at 0x112e61a90>
r = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=15, batch_size=32)
print("Returned:", r)

# print the available keys
# should see: dict_keys(['val_loss', 'acc', 'loss', 'val_acc'])
print(r.history.keys())

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
