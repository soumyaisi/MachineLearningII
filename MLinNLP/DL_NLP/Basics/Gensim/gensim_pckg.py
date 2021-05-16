#Gensim is another important library. It is used primarily for topic modeling and document similarity. Gensim is most useful for tasks such as getting a word vector of a word.


from gensim.models import Word2Vec
min_count = 0
size = 50
window = 2
sentences= "bitcoin is an innovative payment network and a new kind of money."
sentences=sentences.split()
print(sentences)

model = Word2Vec(sentences, min_count=min_count, size=size, window=window)
print(model)

"""
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  
sentence = ["I", "hope", "it", "is", "going", "good", "for", "you"]
vectors = [model[w] for w in sentence]
print(vectors)
"""
