"""
A multiclass classification CNN model in tensorflow-2 using Dropout and BatchNormalization.
Also data augmentation is used here. 
"""

#Import libraies..............
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model


#Load the data and scale it.................
cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()
print("x_train shape....", x_train.shape)
print("y_train shape....", y_train.shape)


#Number of classes...................
K = len(set(y_train))
print("Number of classes........",K)


#Build the CNN using functional API...................
#comment layers are used in first time training (without data augmentation). 
i = Input(shape = x_train[0].shape)
x = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
#x = Dropout(0.2)(x)
x = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
#x = Dropout(0.2)(x)
x = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
#x = Dropout(0.2)(x)
#x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation = 'softmax')(x)

model = Model(i, x)
print(model.summary())


#Compile and fit the model.....................
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
r = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 10)


#Plot the loss...........
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()


#Plot the accuracy...............
plt.plot(r.history['accuracy'], label = 'accuracy')
plt.plot(r.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()


#Fit the model with data augmentation
#Remember:: If you run this after first fit, it will continue training where it left off.
batch_size = 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = True)
train_generator = data_generator.flow(x_train, y_train, batch_size)
steps_per_epochs = x_train.shape[0] // batch_size
r = model.fit_generator(train_generator, validation_data = (x_test, y_test), steps_per_epoch = steps_per_epochs, epochs = 7)


#Plot the loss...........
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()


#Plot the accuracy............
plt.plot(r.history['accuracy'], label = 'accuracy')
plt.plot(r.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()


#Plot the confusion matrix.............
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


#Show some misclassified examples...................
mis = np.where(p_test != y_test)[0]
i = np.random.choice(mis)
plt.imshow(x_test[i], cmap = 'gray')
plt.title("True label: %s Predicted label: %s" %(y_test[i], p_test[i]))
plt.show()


