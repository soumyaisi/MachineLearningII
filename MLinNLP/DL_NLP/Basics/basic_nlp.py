#Accessing Text from the Web

import urllib3
from bs4 import BeautifulSoup
pool_object = urllib3.PoolManager()
target_url = 'https://www.gutenberg.org/files/43383/43383-h/43383-h.htm'
response_ = pool_object.request('GET', target_url)
final_html_txt = BeautifulSoup(response_.data)
print(final_html_txt)


#Removal of Stopwords

import nltk
from nltk import word_tokenize
sentence= "This book is about Deep Learning and Natural Language Processing!"
tokens = word_tokenize(sentence)
print(tokens)
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
new_tokens = [w for w in tokens if not w in stop_words]
print(new_tokens)


#Counter Vectorization

from sklearn.feature_extraction.text import CountVectorizer
texts=["Ramiess sings classic songs","he listens to old pop ","and rock music", ' and also listens to classical songs']
cv = CountVectorizer()
cv_fit=cv.fit_transform(texts)
print(cv.get_feature_names())
print(cv_fit.toarray())



#TF-IDF Score


from sklearn.feature_extraction.text import TfidfVectorizer
texts=["Ramiess sings classic songs","he listens to old pop",
"and rock music", ' and also listens to classical songs']
vect = TfidfVectorizer()
X = vect.fit_transform(texts)
print(X.todense())


#Text Classifier

from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
data = [
 ('I love my country.', 'pos'),
 ('This is an amazing place!', 'pos'),
 ('I do not like the smell of this place.', 'neg'),
 ('I do not like this restaurant', 'neg'),
 ('I am tired of hearing your nonsense.', 'neg'),
 ("I always aspire to be like him", 'pos'),
 ("It's a horrible performance.", "neg")
 ]
model = NaiveBayesClassifier(data)
print(model.classify("It's an awesome place!"))





