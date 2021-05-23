# -*- coding: utf-8 -*-
"""
Created on Sun May 23 11:41:00 2021

@author: soumya
"""

import os
import pandas as pd

data_folder = "/home/soumya"
adult_filename = os.path.join(data_folder, "adult.data")
print(adult_filename)

adult = pd.read_csv(adult_filename, header=None,
names=["Age", "Work-Class", "fnlwgt",
"Education", "Education-Num","Marital-Status", "Occupation",
"Relationship", "Race", "Sex","Capital-gain", "Capital-loss",
"Hours-per-week", "Native-Country","Earnings-Raw"])

adult.head(5)

adult.dropna(how='all', inplace=True)
print(adult.columns)

print(adult["Hours-per-week"].describe())
print(adult["Education-Num"].median())
print(adult["Work-Class"].unique())

adult["LongHours"] = adult["Hours-per-week"] > 40



import numpy as np
X = np.arange(30).reshape((10, 3))
X[:,1] = 1
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold()
Xt = vt.fit_transform(X)
print(vt.variances_)

X = adult[["Age", "Education-Num", "Capital-gain", "Capital-loss",
"Hours-per-week"]].values
y = (adult["Earnings-Raw"] == ' >50K').values

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
transformer = SelectKBest(score_func=chi2, k=3)
Xt_chi2 = transformer.fit_transform(X, y)
print(transformer.scores_)


from scipy.stats import pearsonr
def multivariate_pearsonr(X, y):
    scores, pvalues = [], []
    for column in range(X.shape[1]):
        cur_score, cur_p = pearsonr(X[:,column], y)
        scores.append(abs(cur_score))
        pvalues.append(cur_p)
    return (np.array(scores), np.array(pvalues))
    
    
transformer = SelectKBest(score_func=multivariate_pearsonr, k=3)
Xt_pearson = transformer.fit_transform(X, y)
print(transformer.scores_)


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
clf = DecisionTreeClassifier(random_state=14)
scores_chi2 = cross_val_score(clf, Xt_chi2, y, scoring='accuracy')
scores_pearson = cross_val_score(clf, Xt_pearson, y, scoring='accuracy')
print(scores_chi2, scores_pearson)







