
#NLTK is the most common package you will encounter working with corpora, categorizing text, analyzing linguistic structure, and more.

#pip install nltk
#pip install -U nltk


import nltk
#nltk.download('all')
# Tokenization
sent_ = "I am almost dead this time"
tokens_ = nltk.word_tokenize(sent_)
print(tokens_)

from nltk.corpus import wordnet
word_ = wordnet.synsets("spectacular")
print(word_)

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()  
print(stemmer.stem("decreases"))


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()    
print(lemmatizer.lemmatize("decreases"))

