# -*- coding: utf-8 -*-
"""
Created on Sun May 23 11:58:13 2021

@author: soumya
"""

#t http://archive.ics.uci.edu/ml/datasets/Internet+Advertisements

import os
import numpy as np
import pandas as pd
data_folder = "/home/soumya"
data_filename = os.path.join(data_folder, "ad.csv")
print(data_filename)

def convert_number(x):
    try:
        return float(x)
    except ValueError:
        return np.nan
        
from collections import defaultdict
converters = defaultdict(convert_number)
#converters[1558] = lambda x: 1 if x.strip() == "ad." else 0
#print(converters[1558])
ads = pd.read_csv(data_filename, header=None, converters=converters, delimiter = '\t')
ads.head(5)


ads.dropna(how='all', inplace=True)
ads = ads.replace(np.nan,0)
#ads.replace(to_replace = r'^?',value = 0, regex=True)

ads = ads.replace('?',0)
#ads = ads[~ads.C.str.contains("?")]
X = ads.drop(1023, axis=1).values
y = ads[1023]


from sklearn.decomposition import PCA
pca = PCA(n_components=5)
Xd = pca.fit_transform(X)
np.set_printoptions(precision=3, suppress=True)
print(pca.explained_variance_ratio_)


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
clf = DecisionTreeClassifier(random_state=14)
scores_reduced = cross_val_score(clf, Xd, y, scoring='accuracy')

#%matplotlib inline
from matplotlib import pyplot as plt
classes = set(y)
colors = ['red', 'green']
for cur_class, color in zip(classes, colors):
    mask = (y == cur_class).values
    plt.scatter(Xd[mask,0], Xd[mask,1], marker='o', color=color,label=int(cur_class))
    plt.legend()
    plt.show()

#create own transformer
from sklearn.base import TransformerMixin
from sklearn.utils import as_float_array
class MeanDiscrete(TransformerMixin):
    def fit(self, X):
        X = as_float_array(X)
        self.mean = X.mean(axis=0)
        return self
    def transform(self, X):
        X = as_float_array(X)
        assert X.shape[1] == self.mean.shape[0]
        return X > self.mean
        
mean_discrete = MeanDiscrete()
X_mean = mean_discrete.fit_transform(X)


#unit testing
from numpy.testing import assert_array_equal
def test_meandiscrete():
    X_test = np.array([[ 0,2],
    [ 3,5],
    [ 6,8],
    [ 9,11],
    [12,14],
    [15,17],
    [18,20],
    [21,23],
    [24,26],
    [27,29]])
    mean_discrete = MeanDiscrete()
    mean_discrete.fit(X_test)
    assert_array_equal(mean_discrete.mean, np.array([13.5, 15.5]))
    X_transformed = mean_discrete.transform(X_test)
    X_expected = np.array([[ 0, 0],
    [ 0, 0],
    [ 0, 0],
    [ 0, 0],
    [ 0, 0],
    [ 1, 1],
    [ 1, 1],
    [ 1, 1],
    [ 1, 1],
    [ 1, 1]])
    assert_array_equal(X_transformed, X_expected)

test_meandiscrete()

mean_discrete = MeanDiscrete()
#X_mean = mean_discrete.fit_transform(X)
X_fit = mean_discrete.fit(X)
X_mean = X_fit.transform(X)
clf = DecisionTreeClassifier(random_state=14)
scores_reduced = cross_val_score(clf, X_mean, y, scoring='accuracy')
print("Mean Discrete performance:{0:.3f}".format(scores_reduced.mean()))

#Showing error......
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('mean_discrete', MeanDiscrete()),('classifier', DecisionTreeClassifier(random_state=14))])
scores_mean_discrete = cross_val_score(pipeline, X, y, scoring='accuracy')
print("Mean Discrete performance:{0:.3f}".format(scores_mean_discrete.mean()))



