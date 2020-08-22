"""
A simple multiclass classification problem with tensorflow-2. 
"""

#Import libraries.......
import tensorflow as tf
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


#Load dataset and scalling the data...............
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("train data shape.........", x_train.shape)


#Build the model ML model.................
model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(input_shape = (28, 28)),
	tf.keras.layers.Dense(128, activation = 'relu'),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(10, activation = 'softmax')
	])


#Compile the model (as it is a multiclass classification).............
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
	metrics = ['accuracy'])


#Train the model with 10 epochs................
r = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 10)


#Plot the loss................
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_l0ss')
plt.legend()
plt.show()


#Plot the accuracy(sometime use 'acc' ans sometime use 'accuracy')...............
plt.plot(r.history['acc'], label = 'accuracy')
plt.plot(r.history['val_acc'], label = 'val_acc')
plt.legend()
plt.show()


#Evaluate the model on test data.............
print(model.evaluate(x_test, y_test))


#Plot the confusion matrix...............
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
plt.imshow(x_test[i], cmap = 'gray')
plt.title("True label: %s Predicted label: %s" %(y_test[i], p_test[i]))
plt.show()




