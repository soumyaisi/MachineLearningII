"""
Binary classification using Linear classifier in tensorflow-2. 
"""

#Load libraies...................
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer


#Load the data..............
data = load_breast_cancer()
print(type(data))
print(data.keys())
print(data.data.shape)
print(data.target)
print(data.target_names)
print(data.target.shape)
print(data.feature_names)
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.33)
N,D = x_train.shape
print("N,D.............",N,D)


#Scale the data..................
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#Define the Linear model..................
model = tf.keras.models.Sequential([
	tf.keras.layers.Input(shape=(D,)),
	tf.keras.layers.Dense(1, activation = 'sigmoid')
	])

#model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Dense(1, input_shape = (D,),activation = 'sigmoid'))


#Compile and fit the model................
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
r = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 100)

print("Train score...", model.evaluate(x_train, y_train))
print("Test score....", model.evaluate(x_test, y_test))


#Plot the loss.................
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()


#Plot the accuracy.................
plt.plot(r.history['accuracy'], label = 'accuracy')
plt.plot(r.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()


#Predcition using test data.............
p = model.predict(x_test)
print(p)

P = np.round(p).flatten()
print(P)

print("accuracy..", np.mean(p == y_test))
print("evaluate......", model.evaluate(x_test, y_test))


#Saving and loading the model...........
model.save("linearclassification.h5")


#Works only if you not use Input() layer explicitly..............
model = tf.keras.models.load_model("linearclassification.h5")
print(model.layers)
model.evaluate(x_test, y_test)

