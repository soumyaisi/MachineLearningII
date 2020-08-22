import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

from keras.models import Sequential
from  keras.layers import CuDNNLSTM, LSTM
from  keras.layers import Dense
from  keras.layers import TimeDistributed
from  keras.layers import RepeatVector
from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal

def gen_seq(lenght, n):
	return [randint(0, n-1) for _ in range(lenght)]

sequences = gen_seq(6, 30)

def onehot_encoder(seq, n):
	encod = []
	for s in seq:
		v = [0 for _ in range(n)]
		v[s] = 1
		encod.append(v)
	return array(encod)

def onehot_decoder(encod_seq):
	return [argmax(idx) for idx in encod_seq]


onehot = onehot_encoder(sequences, 30)
#print(onehot)

decoded = onehot_decoder(onehot)
#print(decoded)

def generate_pair(n_in, n_out, n_total):
	seq_in = gen_seq(n_in, n_total)
	seq_out = seq_in[:n_out] + [0 for _ in range(n_in-n_out)]

	x = onehot_encoder(seq_in, n_total)
	y = onehot_encoder(seq_out, n_total)

	x = x.reshape((1, x.shape[0], x.shape[1]))
	y = y.reshape((1, y.shape[0], y.shape[1]))
	return x,y

x,y = generate_pair(6,3,30)
#print(onehot_decoder(x[0]))
#print(onehot_decoder(y[0]))
#print(x.shape)

n_feature = 50
n_timestep_in = 5
n_timestep_out = 3

model = Sequential()

model.add(LSTM(150, input_shape=(n_timestep_in, n_feature)))
model.add(RepeatVector(n_timestep_in))
model.add(LSTM(150, return_sequences=True))
model.add(TimeDistributed(Dense(n_feature, activation='softmax')))

print(model.summary())

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])


for epoch in range(500):
	x,y = generate_pair(n_timestep_in, n_timestep_out, n_feature)
	model.fit(x,y,epochs=1,verbose=1)


epochs = 100
correct = 0

for _ in range(epochs):
	x,y = generate_pair(n_timestep_in, n_timestep_out, n_feature)
	pred = model.predict(x)
	if array_equal(onehot_decoder(y[0]), onehot_decoder(pred[0])):
		correct += 1
print("accuracy:", float(correct)  / float(epochs)*100.0)


for _ in range(20):
	x,y = generate_pair(n_timestep_in, n_timestep_out, n_feature)
	pred = model.predict(x)
	print("expected:", onehot_decoder(y[0]), "predicted:", onehot_decoder(pred[0]))


#https://github.com/datalogue/keras-attention
from attention_decoder import AttentionDecoder

model_att = Sequential()

model_att.add(LSTM(150, input_shape=(n_timestep_in, n_feature), return_sequences=True))
model_att.add(LSTM(150, return_sequences=True))
model_att.add(LSTM(150, return_sequences=True))
model_att.add(AttentionDecoder(150, n_feature))

print(model_att.summary())
model_att.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
for epoch in range(500):
	x,y = generate_pair(n_timestep_in, n_timestep_out, n_feature)
	model_att.fit(x,y,epochs=1,verbose=1)


correct = 0

for _ in range(epochs):
	x,y = generate_pair(n_timestep_in, n_timestep_out, n_feature)
	pred = model_att.predict(x)
	if array_equal(onehot_decoder(y[0]), onehot_decoder(pred[0])):
		correct += 1
print("accuracy:", float(correct)  / float(epochs)*100.0)





