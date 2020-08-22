"""
Implemented in keras.
"""


from __future__ import print_function, division
from builtins import range
from keras.models import Model
from keras.layers import Dense, Input
from util import get_normalized_data, y2indicator
import matplotlib.pyplot as plt

# get the data, same as Theano + Tensorflow examples
# no need to split now, the fit() function will do it
Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

N, D = Xtrain.shape
K = len(set(Ytrain))

Ytrain = y2indicator(Ytrain)
Ytest = y2indicator(Ytest)


# ANN with layers [784] -> [500] -> [300] -> [10]
i = Input(shape=(D,))
x = Dense(500, activation='relu')(i)
x = Dense(300, activation='relu')(x)
x = Dense(K, activation='softmax')(x)
model = Model(inputs=i, outputs=x)


model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

r = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=10, batch_size=32)
print("Returned:", r)

print(r.history.keys())

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
