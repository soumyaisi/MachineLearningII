# -*- coding: utf-8 -*-
"""
Created on Sun May 23 10:35:27 2021

@author: soumya
"""

#http://grouplens.org/datasets/movielens/

import numpy as np
import os
import pandas as pd

data_folder = "/home/soumya/ml-latest-small"
ratings_filename = os.path.join(data_folder, "ratings.csv")
print(ratings_filename)

all_ratings = pd.read_csv(ratings_filename)
all_ratings.head(5)

all_ratings["timestamp"] = pd.to_datetime(all_ratings['timestamp'],unit='s')
print(all_ratings[:5])

#Apriori implementation

all_ratings["Favorable"] = all_ratings["rating"] > 3
print(all_ratings[10:15])

ratings = all_ratings[all_ratings['userId'].isin(range(200))]
favorable_ratings = ratings[ratings["Favorable"]]
favorable_reviews_by_users = dict((k, frozenset(v.values)) for k, v in favorable_ratings.groupby("userId")["movieId"])


num_favorable_by_movie = ratings[["movieId", "Favorable"]].groupby("movieId").sum()

num_favorable_by_movie.sort_values("Favorable", ascending=False)[:5]

frequent_itemsets = {}
min_support = 50
frequent_itemsets[1] = dict((frozenset((movie_id,)),row["Favorable"]) for movie_id, row in num_favorable_by_movie.iterrows()
if row["Favorable"] > min_support)
    
print(frequent_itemsets[1])


from collections import defaultdict
def find_frequent_itemsets(favorable_reviews_by_users, k_1_itemsets,min_support):
    counts = defaultdict(int)
    for user, reviews in favorable_reviews_by_users.items():
        for itemset in k_1_itemsets:
            if itemset.issubset(reviews):
                for other_reviewed_movie in reviews - itemset:
                    current_superset = itemset | frozenset((other_reviewed_movie,))
                    counts[current_superset] += 1
    return dict([(itemset, frequency) for itemset, frequency in counts.items() if frequency >= min_support])
    

import sys    
for k in range(2, 20):
    cur_frequent_itemsets = find_frequent_itemsets(favorable_reviews_by_users,frequent_itemsets[k-1],min_support)
    frequent_itemsets[k] = cur_frequent_itemsets
    if len(cur_frequent_itemsets) == 0:
        print("Did not find any frequent itemsets of length {}".format(k))
        sys.stdout.flush()
        break    
    else:
        print("I found {} frequent itemsets of length {}".format(len(cur_frequent_itemsets), k))
        sys.stdout.flush()    
    
del frequent_itemsets[1]


candidate_rules = []
for itemset_length, itemset_counts in frequent_itemsets.items():
    for itemset in itemset_counts.keys():
        for conclusion in itemset:
            premise = itemset - set((conclusion,))
            candidate_rules.append((premise, conclusion))
print(candidate_rules[:5])
#In the first case, if a reviewer recommends movie 1196, they are also likely to recommend movie 2858.            
            
correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
for user, reviews in favorable_reviews_by_users.items():
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule      
        if premise.issubset(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1
                    
rule_confidence = {candidate_rule: correct_counts[candidate_rule]
/ float(correct_counts[candidate_rule] +
incorrect_counts[candidate_rule])
for candidate_rule in candidate_rules}
    
from operator import itemgetter
sorted_confidence = sorted(rule_confidence.items(),key=itemgetter(1), reverse=True) 
for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]    
    print("Rule: If a person recommends {0} they will also recommend {1}".format(premise, conclusion))
    print(" - Confidence:{0:.3f}".format(rule_confidence[(premise, conclusion)]))
    print("")                    


movie_name_filename = os.path.join(data_folder, "movies.csv")
print(movie_name_filename)
movie_name_data = pd.read_csv(movie_name_filename, encoding = "mac-roman")
movie_name_data.head(5)


def get_movie_name(movie_id):
    title_object = movie_name_data[movie_name_data["movieId"] ==movie_id]["title"]
    title = title_object.values[0]
    return title
    
for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    premise_names = ", ".join(get_movie_name(idx) for idx in premise)
    conclusion_name = get_movie_name(conclusion)
    print("Rule: If a person recommends {0} they will also recommend {1}".format(premise_names, conclusion_name))
    print(" - Confidence: {0:.3f}".format(rule_confidence[(premise,conclusion)]))
    print("")


#Evaluation
test_dataset = all_ratings[~all_ratings['userId'].isin(range(200))]
test_favorable = test_dataset[test_dataset["Favorable"]]
test_favorable_by_users = dict((k, frozenset(v.values)) for k, v in test_favorable.groupby("userId")["movieId"])


correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
for user, reviews in test_favorable_by_users.items():
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        if premise.issubset(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1
                
                
test_confidence = {candidate_rule: correct_counts[candidate_rule]
/ float(correct_counts[candidate_rule] + incorrect_counts
[candidate_rule]) for candidate_rule in rule_confidence}


for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    premise_names = ", ".join(get_movie_name(idx) for idx in premise)
    conclusion_name = get_movie_name(conclusion)
    print("Rule: If a person recommends {0} they will also recommend {1}".format(premise_names, conclusion_name))
    print(" - Train Confidence:{0:.3f}".format(rule_confidence.get((premise, conclusion),-1)))
    print(" - Test Confidence:{0:.3f}".format(test_confidence.get((premise, conclusion),-1)))
    print("")
    
    
    
    
    
    