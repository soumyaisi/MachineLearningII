#TextBlob is a Python library for processing textual data. It provides a simple API for diving deep into common NLP tasks, such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, and much more.


from textblob import TextBlob

# Taking a statement as input
statement = TextBlob("My home is far away from my school.")
# Calculating the sentiment attached with the statement
print(statement.sentiment)


# Defining a sample text
text = '''How about you and I go together on a walk far away
from this place, discussing the things we have never discussed
on Deep Learning and Natural Language Processing.'''
blob_ = TextBlob(text)           # Making it as Textblob object
print(blob_)
print(blob_.tags)


sample_ = TextBlob("I thinkk the model needs to be trained more!")
print(sample_.correct())



# Language Translation
lang_ = TextBlob(u"Voulez-vous apprendre le français?")
t = lang_.translate(from_lang='fr', to='en')
print(t)