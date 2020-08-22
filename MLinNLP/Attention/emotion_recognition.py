import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)


from keras.models import Sequential
from keras import Model, Input
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from  keras.layers import CuDNNLSTM, LSTM
from  keras.layers import Dense, Embedding, Reshape
from  keras.layers import TimeDistributed, Activation
from  keras.layers import Dot, Dropout
from keras.layers.wrappers import Bidirectional
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import random 
import matplotlib.pyplot as plt
#from Ipython.core.display import display, HTML


df = pd.read_csv('emotion.data')
print(df.head(10))

df.emotions.value_counts().plot.bar()

text_tokens = [text.split(" ") for text in df['text'].values.tolist()]
text = df['text'].values.tolist()
labels = df['emotions'].values.tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

word2id = tokenizer.word_index
id2word = dict([(value, key) for (key, value) in word2id.items()])

vocab_size = len(word2id)+1

embedding_dim = 100
max_len = 150

x = [[word2id[word] for word in sent] for sent in text_tokens]
print(x[0])

pad = 'post'
x_pad = pad_sequences(x, maxlen = max_len, padding = pad, truncating = pad)
print(x_pad[0])

label2id = {l:i for i,l in enumerate(set(labels))}
id2label = {v:k for k,v in label2id.items()}
print(id2label)

y = [label2id[label] for label in labels]
y = to_categorical(y, num_classes=len(label2id), dtype='float32')

seq_input = Input(shape=(max_len, ), dtype='int32')
embedded = Embedding(vocab_size, embedding_dim, input_length=max_len)(seq_input)
embedded = Dropout(0.2)(embedded)
lstm = Bidirectional(LSTM(embedding_dim, return_sequences=True))(embedded)
lstm = Dropout(0.2)(lstm)
att_vector = TimeDistributed(Dense(1))(lstm)
att_vector = Reshape((max_len, ))(att_vector)
att_vector = Activation('softmax', name = 'attention_layer')(att_vector)
att_output = Dot(axes=1)([lstm, att_vector])
fc = Dense(embedding_dim, activation='relu')(att_output)
output = Dense(len(label2id), activation='softmax')(fc)

model = Model(inputs = [seq_input], outputs = output)
print(model.summary())

model.compile(loss='categorical_crossentropy', metrics = ['accuracy'], optimizer='adam')
model.fit(x_pad, y, epochs=2, batch_size=64, validation_split=0.2, shuffle=True, verbose=2)

model_att = Model(input=model.input, outputs=[model.output, model.get_layer('attention_layer').output])

sample_text = random.choice(df['text'].values.tolist())

tokenized_sample = sample_text.split(" ")
encoded_sample = [[word2id[word] for word in tokenized_sample]]
encoded_sample = pad_sequences(encoded_sample, maxlen=max_len)

label_probs, attention = model_att.predict(encoded_sample)
label_probs = {id2label[_id]: prob for (label, _id), prob in zip(label2id.items(), label_probs[0])}
print(label_probs)

token_attention_dict = {}
max_score = 0.0
min_score = 0.0
for token, attention_score in zip(tokenized_sample, attention[0][-len(tokenized_sample):]):
	token_attention_dict[token] = attention_score

def rgb_to_hex(rgb):
	return ''%rgb

def attention_color(attention_score):
	c = 255 - int(attention_score*255)
	color = rgb_to_hex((c,255,c))
	return str(color)

html_text = "<hr><p style='font-size: large">
for word, att in token_attention_dict.items():
	html_text += 
html_text ++ "<\p>"

display(HTML(html_text))

emotions = 
