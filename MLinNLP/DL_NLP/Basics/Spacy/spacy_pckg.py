#SpaCy provides very fast and accurate syntactic analysis (the fastest of any library released) and also offers named entity recognition and ready access to word vectors


import spacy
# Run below command, if you are getting error
# python -m spacy download en
nlp = spacy.load("en")


william_wikidef = """William was the son of King William
II and Anna Pavlovna of Russia. On the abdication of his
grandfather  William I in 1840, he became the Prince of Orange.
On the death of his father in 1849, he succeeded as king of the
Netherlands. William married his cousin Sophie of Württemberg
in 1839 and they had three sons, William, Maurice, and
Alexander, all of whom predeceased him. """
nlp_william = nlp(william_wikidef)
print([ (i, i.label_, i.label) for i in nlp_william.ents])



# Noun Phrase extraction
senten_ = nlp('The book deals with NLP')
for noun_ in senten_.noun_chunks:
    print(noun_)
    print(noun_.text)
    print('---')
    print(noun_.root.dep_)
    print('---')
    print(noun_.root.head.text)