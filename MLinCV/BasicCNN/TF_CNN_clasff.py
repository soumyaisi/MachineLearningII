"""
A multiclass classification CNN model in tensorflow-2
"""


#Loading libraies......................
import tensorflow as tf
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model


#Load the data and scale the data...................
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("Train data shape....", x_train.shape)


#data has dim 2 but con net want dim 3(H*W*C)...............
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(x_train.shape)


#Number of classes...................
K = len(set(y_train))
print("Number of classes........",K)


#Build the CNN using functional API.........................
i = Input(shape = x_train[0].shape)
x = Conv2D(32, (3,3), strides = 2, activation = 'relu')(i)
x = Conv2D(64, (3,3), strides = 2, activation = 'relu')(x)
x = Conv2D(128, (3,3), strides = 2, activation = 'relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation = 'softmax')(x)

model = Model(i, x)


#Compile and fit the model(as multiclass classification)............
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
r = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 15)


#Plot the loss.....................
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()


#Plot the accuracy.................
plt.plot(r.history['accuracy'], label = 'accuracy')
plt.plot(r.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()


#Plot the confusion matrix....................
def plot_confusion_matrix(cm, classes, normalize = False, title = 'ConfusionMatrix',cmap = plt.cm.Blues):
	if normalize:
		cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
		print("with normalization........")
	else:
		print("without normalization")
	print(cm)

	plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation = 45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i,j], fmt),
			horizontalalignment = 'center', 
			color = 'white' if cm[i,j] > thresh else 'black')
	plt.tight_layout()
	plt.ylabel("True Label")
	plt.xlabel("Predicted Label")
	plt.show()

p_test = model.predict(x_test).argmax(axis = 1)
cm = confusion_matrix(y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))


#Show some misclassified examples.................
mis = np.where(p_test != y_test)[0]
i = np.random.choice(mis)
plt.imshow(x_test[i].reshape(28, 28), cmap = 'gray')
plt.title("True label: %s Predicted label: %s" %(y_test[i], p_test[i]))
plt.show()


