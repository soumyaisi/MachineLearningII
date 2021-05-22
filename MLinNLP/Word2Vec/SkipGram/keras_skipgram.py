from keras.layers import Concatenate
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Sequential
vocab_size = 5000
embed_size = 300
word_model = Sequential()
word_model.add(Embedding(vocab_size, embed_size,embeddings_initializer="glorot_uniform",input_length=1))
word_model.add(Reshape((embed_size, )))
context_model = Sequential()
context_model.add(Embedding(vocab_size, embed_size,embeddings_initializer="glorot_uniform",input_length=1))
context_model.add(Reshape((embed_size,)))
model = Sequential()
model.add(Concatenate([word_model, context_model]))
#model.add(Merge([word_model, context_model], mode="dot", dot_axes=0))
model.add(Dense(1, init="glorot_uniform", activation="sigmoid"))
model.compile(loss="mean_squared_error", optimizer="adam")

from keras.preprocessing.text import *
from keras.preprocessing.sequence import skipgrams
text = "I love green eggs and ham ."
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
word2id = tokenizer.word_index
id2word = {v:k for k, v in word2id.items()}

wids = [word2id[w] for w in text_to_word_sequence(text)]
pairs, labels = skipgrams(wids, len(word2id))
print(len(pairs), len(labels))
for i in range(10):
	print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
	id2word[pairs[i][0]], pairs[i][0],
	id2word[pairs[i][1]], pairs[i][1],
	labels[i]))


"""
merge_layer = model.layers[0]
word_model = merge_layer.layers[0]
word_embed_layer = word_model.layers[0]
weights = word_embed_layer.get_weights()[0]
"""