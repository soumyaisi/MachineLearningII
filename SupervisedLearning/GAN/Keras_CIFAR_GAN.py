import matplotlib as mpl
# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')
import pandas as pd
import numpy as np
import os
from keras.layers import Dense, Reshape, Flatten, Dropout, LeakyReLU,
Activation, BatchNormalization, SpatialDropout2D
from keras.layers.convolutional import Convolution2D, UpSampling2D,
MaxPooling2D, AveragePooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.regularizers import l1l2
from keras_adversarial import AdversarialModel, ImageGridCallback,
simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous,
normal_latent_sampling, fix_names
import keras.backend as K
from cifar10_utils import cifar10_data
from image_utils import dim_ordering_fix, dim_ordering_unfix,
dim_ordering_shape



def model_generator():
	model = Sequential()
	nch = 256
	reg = lambda: l1l2(l1=1e-7, l2=1e-7)
	h = 5
	model.add(Dense(input_dim=100, output_dim=nch * 4 * 4, W_regularizer=reg()))
	model.add(BatchNormalization(mode=0))
	model.add(Reshape(dim_ordering_shape((nch, 4, 4))))
	model.add(Convolution2D(nch/2, h, h, border_mode='same', W_regularizer=reg()))
	model.add(BatchNormalization(mode=0, axis=1))
	model.add(LeakyReLU(0.2))
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Convolution2D(nch / 2, h, h, border_mode='same', W_regularizer=reg()))
	model.add(BatchNormalization(mode=0, axis=1))
	model.add(LeakyReLU(0.2))
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Convolution2D(nch / 4, h, h, border_mode='same', W_regularizer=reg()))
	model.add(BatchNormalization(mode=0, axis=1))
	model.add(LeakyReLU(0.2))
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Convolution2D(3, h, h, border_mode='same', W_regularizer=reg()))
	model.add(Activation('sigmoid'))
	return model



def model_discriminator():
	nch = 256
	h = 5
	reg = lambda: l1l2(l1=1e-7, l2=1e-7)
	c1 = Convolution2D(nch / 4, h, h, border_mode='same', W_regularizer=reg(),
	input_shape=dim_ordering_shape((3, 32, 32)))
	c2 = Convolution2D(nch / 2, h, h, border_mode='same', W_regularizer=reg())
	c3 = Convolution2D(nch, h, h, border_mode='same', W_regularizer=reg())
	c4 = Convolution2D(1, h, h, border_mode='same', W_regularizer=reg())
	def m(dropout):
		model = Sequential()
		model.add(c1)
		model.add(SpatialDropout2D(dropout))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(LeakyReLU(0.2))
		model.add(c2)
		model.add(SpatialDropout2D(dropout))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(LeakyReLU(0.2))
		model.add(c3)
		model.add(SpatialDropout2D(dropout))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(LeakyReLU(0.2))
		model.add(c4)
		model.add(AveragePooling2D(pool_size=(4, 4), border_mode='valid'))
		model.add(Flatten())
		model.add(Activation('sigmoid'))
		return model
	return m

def example_gan(adversarial_optimizer, path, opt_g, opt_d, nb_epoch, generator,discriminator, latent_dim, targets=gan_targets, loss='binary_crossentropy'):
	csvpath = os.path.join(path, "history.csv")
	if os.path.exists(csvpath):
		print("Already exists: {}".format(csvpath))
	return


print("Training: {}".format(csvpath))
# gan (x - > yfake, yreal), z is gaussian generated on GPU
# can also experiment with uniform_latent_sampling
d_g = discriminator(0)
d_d = discriminator(0.5)
generator.summary()
d_d.summary()
gan_g = simple_gan(generator, d_g, None)
gan_d = simple_gan(generator, d_d, None)
x = gan_g.inputs[1]
z = normal_latent_sampling((latent_dim,))(x)
# eliminate z from inputs
gan_g = Model([x], fix_names(gan_g([z, x]), gan_g.output_names))
gan_d = Model([x], fix_names(gan_d([z, x]), gan_d.output_names))




# build adversarial model
model = AdversarialModel(player_models=[gan_g, gan_d],
player_params=[generator.trainable_weights, d_d.trainable_weights],
player_names=["generator", "discriminator"])
model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
player_optimizers=[opt_g, opt_d], loss=loss)


# create callback to generate images
zsamples = np.random.normal(size=(10 * 10, latent_dim))
def generator_sampler():
	xpred = dim_ordering_unfix(generator.predict(zsamples)).transpose((0, 2, 3, 1))
	return xpred.reshape((10, 10) + xpred.shape[1:])
generator_cb = ImageGridCallback(os.path.join(path, "epoch-{:03d}.png"),
generator_sampler, cmap=None)

# train model
xtrain, xtest = cifar10_data()
y = targets(xtrain.shape[0])
ytest = targets(xtest.shape[0])
callbacks = [generator_cb]
if K.backend() == "tensorflow":
	callbacks.append(TensorBoard(log_dir=os.path.join(path, 'logs'),
	histogram_freq=0, write_graph=True, write_images=True))
history = model.fit(x=dim_ordering_fix(xtrain),y=y,
validation_data=(dim_ordering_fix(xtest), ytest),
callbacks=callbacks, nb_epoch=nb_epoch,
batch_size=32)
# save history to CSV
df = pd.DataFrame(history.history)
df.to_csv(csvpath)
# save models
generator.save(os.path.join(path, "generator.h5"))
d_d.save(os.path.join(path, "discriminator.h5"))


def main():
	# z in R^100
	latent_dim = 100
	# x in R^{28x28}
	# generator (z -> x)
	generator = model_generator()
	# discriminator (x -> y)
	discriminator = model_discriminator()
	example_gan(AdversarialOptimizerSimultaneous(), "output/gan-cifar10",
	opt_g=Adam(1e-4, decay=1e-5),
	opt_d=Adam(1e-3, decay=1e-5),
	nb_epoch=100, generator=generator, discriminator=discriminator,
	latent_dim=latent_dim)
if __name__ == "__main__":
	main()


#In order to have a complete view on the open source code, we need to include a few simple utility
#functions for storing the grid of images:

from matplotlib import pyplot as plt, gridspec
import os
def write_image_grid(filepath, imgs, figsize=None, cmap='gray'):
	directory = os.path.dirname(filepath)
	if not os.path.exists(directory):
	os.makedirs(directory)
	fig = create_image_grid(imgs, figsize, cmap=cmap)
	fig.savefig(filepath)
	plt.close(fig)
def create_image_grid(imgs, figsize=None, cmap='gray'):
	n = imgs.shape[0]
	m = imgs.shape[1]
	if figsize is None:
	figsize=(n,m)
	fig = plt.figure(figsize=figsize)
	gs1 = gridspec.GridSpec(n, m)
	gs1.update(wspace=0.025, hspace=0.025) # set the spacing between axes.
	for i in range(n):
	for j in range(m):
	ax = plt.subplot(gs1[i, j])
	img = imgs[i, j, :]
	ax.imshow(img, cmap=cmap)
	ax.axis('off')
	return fig


#In addition, we need some utility methods for dealing with different image ordering
import keras.backend as K
import numpy as np
from keras.layers import Input, Reshape
def dim_ordering_fix(x):
	if K.image_dim_ordering() == 'th':
		return x
	else:
		return np.transpose(x, (0, 2, 3, 1))
def dim_ordering_unfix(x):
	if K.image_dim_ordering() == 'th':
		return x
	else:
		return np.transpose(x, (0, 3, 1, 2))
def dim_ordering_shape(input_shape):
	if K.image_dim_ordering() == 'th':
		return input_shape
	else:
		return (input_shape[1], input_shape[2], input_shape[0])
def dim_ordering_input(input_shape, name):
	if K.image_dim_ordering() == 'th':
		return Input(input_shape, name=name)
	else:
		return Input((input_shape[1], input_shape[2], input_shape[0]), name=name)
def dim_ordering_reshape(k, w, **kwargs):
	if K.image_dim_ordering() == 'th':
		return Reshape((k, w, w), **kwargs)
	else:
		return Reshape((w, w, k), **kwargs)
# One more utility function is used to fix names
def fix_names(outputs, names):
	if not isinstance(outputs, list):
		outputs = [outputs]
	if not isinstance(names, list):
	names = [names]
	return [Activation('linear', name=name)(output)
	for output, name in zip(outputs, names)]


#https://github.com/bstriner/keras-adversarial/blob/master/examples/example_gan_cifar10.py
#https://github.com/bstriner/keras-adversarial/blob/master/keras_adversarial/legacy.py


#https://deepmind.com/blog/wavenet-generative-model-raw-audio/



