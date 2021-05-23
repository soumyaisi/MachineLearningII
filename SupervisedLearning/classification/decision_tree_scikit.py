# -*- coding: utf-8 -*-
"""
Created on Sat May 22 20:02:53 2021

@author: soumya
"""

import pandas as pd
dataset = pd.read_csv('/home/soumya/sports.csv', sep = '\t')

print(dataset.ix[:5])
dataset.head(4)

dataset["HomeWin"] = dataset["Visitor_PTS"] < dataset["Home_PTS"]
y_true = dataset["HomeWin"].values
print(y_true)

from collections import defaultdict
won_last = defaultdict(int)

HomeLastWin_ = []
VisitorLastWin_ = []
for index, row in dataset.iterrows():
    #print(index, row)
    home_team = row["Home"]
    visitor_team = row["Visitor"]
    #print(home_team, visitor_team)
    row["HomeLastWin"] = won_last[home_team]
    row["VisitorLastWin"] = won_last[visitor_team]
    #print(row)
    HomeLastWin_.append(row["HomeLastWin"])
    VisitorLastWin_.append(row["VisitorLastWin"])
    #dataset.ix[index] = row
    print(row)
    won_last[home_team] = row["HomeWin"]
    won_last[visitor_team] = not row["HomeWin"]
   
dataset['HomeLastWin'] = HomeLastWin_
dataset['VisitorLastWin'] = VisitorLastWin_
dataset.head(4)
 
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=14)
X_previouswins = dataset[["HomeLastWin", "VisitorLastWin"]].values

from sklearn.model_selection import cross_val_score
import numpy as np
scores = cross_val_score(clf, X_previouswins, y_true,
scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=14)
scores = cross_val_score(clf, X_previouswins, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


