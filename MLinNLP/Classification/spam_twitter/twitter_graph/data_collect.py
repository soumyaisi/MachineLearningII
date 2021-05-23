# -*- coding: utf-8 -*-
"""
Created on Sun May 23 16:46:11 2021

@author: soumya
"""

import twitter
consumer_key = "<Your Consumer Key Here>"
consumer_secret = "<Your Consumer Secret Here>"
access_token = "<Your Access Token Here>"
access_token_secret = "<Your Access Token Secret Here>"
authorization = twitter.OAuth(access_token, access_token_secret,
consumer_key, consumer_secret)
t = twitter.Twitter(auth=authorization, retry=True)

import os
data_folder = os.path.join(os.path.expanduser("~"), "Data",
"twitter")
output_filename = os.path.join(data_folder, "python_tweets.json")

import json
original_users = []
tweets = []
user_ids = {}


search_results = t.search.tweets(q="python",
count=100)['statuses']
for tweet in search_results:
    if 'text' in tweet:
        original_users.append(tweet['user']['screen_name'])
        user_ids[tweet['user']['screen_name']] =
        tweet['user']['id']
        tweets.append(tweet['text'])
        
        
#Classifying with an existing model
 from sklearn.externals import joblib
output_filename = os.path.join(os.path.expanduser("~"), "Models",
"twitter", "python_context.pkl")

joblib.dump(model, output_filename)


model_filename = os.path.join(os.path.expanduser("~"), "Models",
"twitter", "python_context.pkl")



from sklearn.base import TransformerMixin
from nltk import word_tokenize

class NLTKBOW(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [{word: True for word in word_tokenize(document)} for document in X]       
        
        
from sklearn.externals import joblib
context_classifier = joblib.load(model_filename)

y_pred = context_classifier.predict(tweets)

relevant_tweets = [tweets[i] for i in range(len(tweets)) if y_pred[i]
== 1]
relevant_users = [original_users[i] for i in range(len(tweets)) if
y_pred[i] == 1]

#Getting follower information from Twitter

import time
def get_friends(t, user_id):
    friends = []
    cursor = -1
    while cursor != 0:
        try:
            results = t.friends.ids(user_id= user_id,cursor=cursor, count=5000)
            friends.extend([friend for friend in results['ids']])
            cursor = results['next_cursor']
            if len(friends) >= 10000:
                break
        except TypeError as e:
            if results is None:
                print("You probably reached your API limit,waiting for 5 minutes")
                sys.stdout.flush()
                time.sleep(5*60) # 5 minute wait
            else:
                raise e

        except twitter.TwitterHTTPError as e:
            break
            finally:
                time.sleep(60)
    return friends
    
    
import time
def get_friends(t, user_id):
    friends = []
    cursor = -1
    while cursor != 0:
        try:
            results = t.friends.ids(user_id= user_id,cursor=cursor, count=5000)
            friends.extend([friend for friend in results['ids']])
            cursor = results['next_cursor']
            if len(friends) >= 10000:
                break
        except TypeError as e:
            if results is None:
                print("You probably reached your API limit,waiting for 5 minutes")
                sys.stdout.flush()
                time.sleep(5*60) # 5 minute wait
            else:
                raise e
        except twitter.TwitterHTTPError as e:
            break
            finally:
                time.sleep(60)
    return friends
        
        
#Building the network
        
friends = {}
for screen_name in relevant_users:
    user_id = user_ids[screen_name]
    friends[user_id] = get_friends(t, user_id)
 
#not clear       
friends = {user_id:friends[user_id] for user_id in friends
if len(friends[user_id]) > 0}       
        
 
from collections import defaultdict
def count_friends(friends):
    friend_count = defaultdict(int)
    for friend_list in friends.values():
        for friend in friend_list:
            friend_count[friend] += 1
    return friend_count


friend_count
reverse=True) = count_friends(friends)
from operator import itemgetter
best_friends = sorted(friend_count.items(), key=itemgetter(1),

while len(friends) < 150:
    for user_id, count in best_friends:
        if user_id not in friends:
            break
        friends[user_id] = get_friends(t, user_id)
    for friend in friends[user_id]:
        friend_count[friend] += 1
    best_friends = sorted(friend_count.items(),key=itemgetter(1), reverse=True)       
        
import json
friends_filename = os.path.join(data_folder, "python_friends.json")
with open(friends_filename, 'w') as outf:
    json.dump(friends, outf)        
        
with open(friends_filename) as inf:
    friends = json.load(inf)        
        
        
#Creating a graph
import networkx as nx
G = nx.DiGraph()

main_users = friends.keys()
G.add_nodes_from(main_users)

for user_id in friends:
    for friend in friends[user_id]:
        if friend in main_users:
            G.add_edge(user_id, friend)
        
%matplotlib inline
nx.draw(G)

from matplotlib import pyplot as plt
plt.figure(3,figsize=(20,20))
nx.draw(G, alpha=0.1, edge_color='b')


#Creating a similarity graph

friends = {user: set(friends[user]) for user in friends}
def compute_similarity(friends1, friends2):
    return len(friends1 & friends2) / len(friends1 | friends2)

def create_graph(followers, threshold=0):
    G = nx.Graph()        
    for user1 in friends.keys():
        for user2 in friends.keys():
            if user1 == user2:
                continue        
            weight = compute_similarity(friends[user1],friends[user2])
            if weight >= threshold:
                G.add_node(user1)
                G.add_node(user2)
                G.add_edge(user1, user2, weight=weight)
    return G

G = create_graph(friends)


plt.figure(figsize=(10,10))
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos)
edgewidth = [ d['weight'] for (u,v,d) in G.edges(data=True)]
nx.draw_networkx_edges(G, pos, width=edgewidth)

#Finding Subgraph
G = create_graph(friends, 0.1)
sub_graphs = nx.connected_component_subgraphs(G)
for i, sub_graph in enumerate(sub_graphs):
    n_nodes = len(sub_graph.nodes())
    print("Subgraph {0} has {1} nodes".format(i, n_nodes))

G = create_graph(friends, 0.25)
sub_graphs = nx.connected_component_subgraphs(G)
for i, sub_graph in enumerate(sub_graphs):
    n_nodes = len(sub_graph.nodes())
    print("Subgraph {0} has {1} nodes".format(i, n_nodes))        
        
sub_graphs = nx.connected_component_subgraphs(G)
n_subgraphs = nx.number_connected_components(G)


fig = plt.figure(figsize=(20, (n_subgraphs * 3)))
for i, sub_graph in enumerate(sub_graphs):
    ax = fig.add_subplot(int(n_subgraphs / 3), 3, i)        
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, sub_graph.nodes(), ax=ax,node_size=500)
    nx.draw_networkx_edges(G, pos, sub_graph.edges(), ax=ax)        
  

#Optimizing
def compute_silhouette(threshold, friends):
    G = create_graph(friends, threshold=threshold)
    if len(G.nodes()) < 2:
        return -99        
    sub_graphs = nx.connected_component_subgraphs(G)
    if not (2 <= nx.number_connected_components() < len(G.nodes()) - 1):
        return -99

    label_dict = {}
    for i, sub_graph in enumerate(sub_graphs):
        for node in sub_graph.nodes():
            label_dict[node] = i
    labels = np.array([label_dict[node] for node in G.nodes()])
    X = nx.to_scipy_sparse_matrix(G).todense()
    X = 1 - X
    return silhouette_score(X, labels, metric='precomputed')


def inverted_silhouette(threshold, friends):
    return -compute_silhouette(threshold, friends)

result = minimize(inverted_silhouette, 0.1, args=(friends,))

#Learning Data Minning With Python Book



        