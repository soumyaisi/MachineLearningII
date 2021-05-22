from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)


# layer.name, layer.input_shape, layer.output_shape
('mixed10', [(None, 8, 8, 320), (None, 8, 8, 768), (None, 8, 8, 768), (None, 8, 8, 192)], (None, 8, 8, 2048))
('avg_pool', (None, 8, 8, 2048), (None, 1, 1, 2048))
('flatten', (None, 1, 1, 2048), (None, 2048))
('predictions', (None, 2048), (None, 1000))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)# let's add a fully-connected layer as first layer
x = Dense(1024, activation='relu')(x)# and a logistic layer with 200 classes as last layer
predictions = Dense(200, activation='softmax')(x)# model to train
model = Model(input=base_model.input, output=predictions)


# that is, freeze all convolutional InceptionV3 layers
for layer in base_model.layers: layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


# we chose to train the top 2 inception blocks, that is, we will freeze
# the first 172 layers and unfreeze the rest:
for layer in
model.layers[:172]: layer.trainable = False
for layer in
model.layers[172:]: layer.trainable = True


# we use SGD with a low learning rate
from keras.optimizers
import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
# we train our model again (this time fine-tuning the top 2 inception blocks)
# alongside the top Dense layers
model.fit_generator(...)