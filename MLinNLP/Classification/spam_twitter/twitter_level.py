# -*- coding: utf-8 -*-
"""
Created on Sun May 23 15:59:31 2021

@author: soumya
"""

import os
path  = "/home/soumya/spam_twitter"
input_filename = os.path.join(path, "twitter.json")
labels_filename = os.path.join(path, "python_classes.json")

import json
tweets = []

with open(input_filename) as inf:
    for line in inf:
        if len(line.strip()) == 0:
            continue
        tweets.append(json.loads(line))
print(tweets)

labels = []
if os.path.exists(labels_filename):
    with open(labels_filename) as inf:
        labels = json.load(inf)
        
        
def get_next_tweet():
    return tweet_sample[len(labels)]['text']
    
    
%%javascript
function set_label(label){
    var kernel = IPython.notebook.kernel;
    kernel.execute("labels.append(" + label + ")");
    load_next_tweet();
}    

function load_next_tweet(){
    var code_input = "get_next_tweet()";
    var kernel = IPython.notebook.kernel;
    var callbacks = { 'iopub' : {'output' : handle_output}};
    kernel.execute(code_input, callbacks, {silent:false});
}


function handle_output(out){
    var res = out.content.data["text/plain"];
    $("div#tweet_text").html(res);
}

%%html
<div name="tweetbox">
Instructions: Click in textbox. Enter a 1 if the tweet is
relevant, enter 0 otherwise.<br>
Tweet: <div id="tweet_text" value="text"></div><br>
<input type=text id="capture"></input><br>
</div>

<script>
    $("input#capture").keypress(function(e) {
    if(e.which == 48) {
        set_label(0);
        $("input#capture").val("");
    }else if (e.which == 49){
        set_label(1);
        $("input#capture").val("");
}
});
load_next_tweet();
</script>

with open(labels_filename, 'w') as outf:
    json.dump(labels, outf)


#Creating a replicable dataset from Twitter
import os
input_filename = os.path.join(os.path.expanduser("~"), "Data",
"twitter", "python_tweets.json")
labels_filename = os.path.join(os.path.expanduser("~"), "Data",
"twitter", "python_classes.json")
replicable_dataset = os.path.join(os.path.expanduser("~"),
"Data", "twitter", "replicable_dataset.json")

import json
tweets = []
with open(input_filename) as inf:
    for line in inf:
        if len(line.strip()) == 0:
            continue
        tweets.append(json.loads(line))
if os.path.exists(labels_filename):
    with open(classes_filename) as inf:
        labels = json.load(inf)

dataset = [(tweet['id'], label) for tweet, label in zip(tweets,
labels)]


with open(replicable_dataset, 'w') as outf:
    json.dump(dataset, outf)

import os
tweet_filename = os.path.join(os.path.expanduser("~"), "Data",
"twitter", "replicable_python_tweets.json")
labels_filename = os.path.join(os.path.expanduser("~"), "Data",
"twitter", "replicable_python_classes.json")
replicable_dataset = os.path.join(os.path.expanduser("~"),
"Data", "twitter", "replicable_dataset.json")


import json
with open(replicable_dataset) as inf:
    tweet_ids = json.load(inf)

actual_labels = []
label_mapping = dict(tweet_ids)

import twitter
consumer_key = "<Your Consumer Key Here>"
consumer_secret = "<Your Consumer Secret Here>"
access_token = "<Your Access Token Here>"
access_token_secret = "<Your Access Token Secret Here>"
authorization = twitter.OAuth(access_token, access_token_secret,
consumer_key, consumer_secret)
t = twitter.Twitter(auth=authorization)


all_ids = [tweet_id for tweet_id, label in tweet_ids]
with open(tweets_filename, 'a') as output_file:
    for start_index in range(0, len(tweet_ids), 100):
        id_string = ",".join(str(i) for i in all_ids[start_index:start_index+100])
        search_results = t.statuses.lookup(_id=id_string)
        for tweet in search_results:
            if 'text' in tweet:
                output_file.write(json.dumps(tweet))
                output_file.write("\n\n")
                actual_labels.append(label_mapping[tweet['id']])


with open(labels_filename, 'w') as outf:
    json.dump(actual_labels, outf)


from sklearn.base import TransformerMixin
class NLTKBOW(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [{word: True for word in word_tokenize(document)} for document in X]


from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import BernoulliNB

import os
input_filename = os.path.join(os.path.expanduser("~"), "Data",
"twitter", "python_tweets.json")
labels_filename = os.path.join(os.path.expanduser("~"), "Data",
"twitter", "python_classes.json")


tweets = []
with open(input_filename) as inf:
    for line in inf:
        if len(line.strip()) == 0:
            continue
        tweets.append(json.loads(line)['text'])

with open(classes_filename) as inf:
    labels = json.load(inf)

from sklearn.pipeline import Pipeline
pipeline = Pipeline([('bag-of-words', NLTKBOW()),
('vectorizer', DictVectorizer()),
('naive-bayes', BernoulliNB())
])


scores = cross_val_score(pipeline, tweets, labels, scoring='f1')
import numpy as np
print("Score: {:.3f}".format(np.mean(scores)))


#Getting useful features from models

model = pipeline.fit(tweets, labels)
nb = model.named_steps['naive-bayes']
top_features = np.argsort(-feature_probabilities[1])[:50]
dv = model.named_steps['vectorizer']

for i, feature_index in enumerate(top_features):
    print(i, dv.feature_names_[feature_index],
          np.exp(feature_probabilities[1][feature_index]))


















    