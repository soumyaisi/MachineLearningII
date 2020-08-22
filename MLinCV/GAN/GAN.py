"""
GAN implemented in Tensorflow-2.
"""

#Import libraies.....................
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np
import pandas as pd
import sys,os
import matplotlib.pyplot as plt


#Load the dataset.....................
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0 * 2 - 1, x_test / 255.0 * 2 -1
print("train data shape...............",x_train.shape)


#Flatten the data.............
N,H,W = x_train.shape
D = H*W
x_train = x_train.reshape(-1, D)
x_test = x_test.reshape(-1, D)


#latent dimension..............
latent_dim = 100


#Generator model...................
def build_generator(latent_dim):
	i = Input(shape = (latent_dim, ))
	x = Dense(256, activation = LeakyReLU(alpha = 0.2))(i)
	x = BatchNormalization(momentum = 0.8)(x)
	x = Dense(512, activation = LeakyReLU(alpha = 0.2))(i)
	x = BatchNormalization(momentum = 0.8)(x)
	x = Dense(1024, activation = LeakyReLU(alpha = 0.2))(i)
	x = BatchNormalization(momentum = 0.8)(x)
	x = Dense(D, activation = 'tanh')(x)

	model = Model(i, x)
	return model


#Discriminator model....................
def build_discriminator(img_size):
	i = Input(shape = (img_size, ))
	x = Dense(512, activation = LeakyReLU(alpha = 0.2))(i)
	x = Dense(256, activation = LeakyReLU(alpha = 0.2))(x)
	x = Dense(1, activation = 'sigmoid')(x)
	model = Model(i, x)
	return model


#Compile the Discriminator model...............
discriminator = build_discriminator(D)
discriminator.compile(loss = 'binary_crossentropy',
	optimizer = Adam(0.0002, 0.5), 
	metrics = ['accuracy']
	)


#Build and compile the Generator model...........
generator = build_generator(latent_dim)


#Create a noise sample from latent space.............
z = Input(shape = (latent_dim, ))


#Pass noise data through the generator...........
img = generator(z)


#Make sure only the generator is trained................
discriminator.trainable = False


#The true output is fake, but we label them real(interchangeing the class label).............
fake_pred = discriminator(img)


#Create the combied model................
combined_model = Model(z, fake_pred)


#Compile the combined model...................
combined_model.compile(loss = 'binary_crossentropy',
	optimizer = Adam(0.0002, 0.5)
	)


#Train the GAN model................
batch_size = 32
epochs = 30000
sample_period = 200


#Create batch labels to use calling train_on_batch............
ones = np.ones(batch_size)
zeros = np.zeros(batch_size)

d_losses = []
g_losses = []


if not os.path.exists('gan_images'):
	os.makedirs('gan_images')


#To generate a grid of random samples from the generator.............
def sample_images(epochs):
	rows, cols = 5,5
	noise = np.random.randn(rows*cols, latent_dim)
	imgs = generator.predict(noise)

	imgs = 0.5 * imgs + 0.5

	fig, ax = plt.subplots(rows, cols)
	idx = 0
	for i in range(rows):
		for j in range(cols):
			ax[i,j].imshow(imgs[idx].reshape(H,W), cmap = 'gray')
			ax[i,j].axis('off')
			idx += 1
	fig.savefig('gan_images/%d.png'%epochs)
	plt.close()		


#Main training loop..............
for epoch in range(epochs):
	idx = np.random.randint(0, x_train.shape[0], batch_size)
	real_imgs = x_train[idx]

	noise = np.random.randn(batch_size, latent_dim)
	fake_imgs = generator.predict(noise)

	d_loss_real, d_acc_real = discriminator.train_on_batch(real_imgs, ones)
	d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_imgs, zeros)
	d_loss = 0.5 * (d_loss_real + d_loss_fake)
	d_acc = 0.5 * (d_acc_real + d_acc_fake)

	#Train the generator
	noise = np.random.randn(batch_size, latent_dim)
	g_loss = combined_model.train_on_batch(noise, ones)

	d_losses.append(d_loss)
	g_losses.append(g_loss)

	if epoch % 100 == 0:
		print("epoch: {epoch +1}/{epochs}, d_loss: {d_loss:.2f}, d_acc: {d_acc: .2f}, g_loss: {g_loss: .2f}")
	if epoch % sample_period == 0:
		sample_images(epoch)


#Plo the losses..........................
plt.plot(g_losses, label = 'g_losses')
plt.plot(d_losses, label = 'd_losses')
plt.legend()
plt.show()


#Read one generated image................
from skimage.io import imread
a = imread('gan_images/ 0.png')
plt.imshow(a)


