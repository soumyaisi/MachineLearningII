from gensim.models import Word2Vec
from  sklearn.manifold import TSNE
from sklearn.decomposition import  PCA
import matplotlib.pyplot as plt
import numpy as np

sentences = [['this','is','a','very','good','NLP','course'],
['this','is','a','NLP','course'],
['Word2Vec','NLP','text'],
['word','embedding','NLP'],
['NLP','course','Word2Vec','embedding']]

model = Word2Vec(sentences, min_count=1)

print(model)

words = list(model.wv.vocab)
print(words)

print(model['course'])
print(len(model['course']))

model.save('word_embedding.bin')
new_model = Word2Vec.load('word_embedding.bin')
print(new_model)

X = model[model.wv.vocab]
print(X.shape)

pca = PCA(n_components=2)
result = pca.fit_transform(X)

plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	plt.annotate(word, xy=(result[i,0], result[i,1]))
plt.show()


from gensim.models import KeyedVectors
#download the google news word2vec model file
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary = True)

result = model.most_similar(positive=['woman','king'], negative=['man'], topn=1)
print(result)


