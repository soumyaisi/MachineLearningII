import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from  keras.layers import Dense, Embedding, Dropout, Flatten
from keras.datasets import imdb


num_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
print(len(x_train))
print(len(x_test))

max_len = 256
embedding_size = 32
batch_size = 128

print(len(x_train[0]))

pad = 'post'
x_train_pad = pad_sequences(x_train, maxlen = max_len, padding = pad, truncating = pad)
x_test_pad = pad_sequences(x_test, maxlen = max_len, padding = pad, truncating = pad)

model = Sequential()

model.add(Embedding(input_dim = num_words, output_dim=embedding_size, input_length=max_len, name = 'layer_embedding'))

model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())

model.fit(x_train_pad, y_train, epochs = 5, validation_data=(x_test_pad, y_test), batch_size=batch_size)

eval_ = model.evaluate(x_test_pad, y_test)
print(eval_)


print("loss:", eval_[0])
print("accuracy:", eval_[1])

