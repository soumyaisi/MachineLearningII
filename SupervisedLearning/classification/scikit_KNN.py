# -*- coding: utf-8 -*-
"""
Created on Sat May 22 19:21:30 2021

@author: soumya
"""

import numpy as np
import csv
import os

data_folder = "/data_folder"
data_filename = os.path.join(data_folder, "/home/soumya","ionosphere.data")

X = np.zeros((351, 34), dtype='float')
y = np.zeros((351,), dtype='bool')

print(data_filename)
with open(data_filename, 'r') as input_file:
    reader = csv.reader(input_file)
    for i, row in enumerate(reader):
        data = [float(datum) for datum in row[:-1]]
        X[i] = data
        y[i] = row[-1] == 'g'
        
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)
from sklearn.neighbors import KNeighborsClassifier
estimator = KNeighborsClassifier()

estimator.fit(X_train, y_train)
y_predicted = estimator.predict(X_test)
accuracy = np.mean(y_test == y_predicted) * 100
print("The accuracy is {0:.1f}%".format(accuracy))


from sklearn.cross_validation import cross_val_score
scores = cross_val_score(estimator, X, y, scoring='accuracy')
average_accuracy = np.mean(scores) * 100
print("The average accuracy is {0:.1f}%".format(average_accuracy))

avg_scores = []
all_scores = []
parameter_values = list(range(1, 21)) # Include 20
for n_neighbors in parameter_values:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator, X, y, scoring='accuracy')
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)

from matplotlib import pyplot as plt 
plt.plot(parameter_values,avg_scores, '-o')
plt.show()

X_broken = np.array(X)
X_broken[:,::2] /= 10
estimator = KNeighborsClassifier()
original_scores = cross_val_score(estimator, X, y,scoring='accuracy')
print("The original average accuracy for is {0:.1f}%".format(np.mean(original_scores) * 100))
broken_scores = cross_val_score(estimator, X_broken, y,
scoring='accuracy')
print("The 'broken' average accuracy for is {0:.1f}%".format(np.mean(broken_scores) * 100))



from sklearn.preprocessing import MinMaxScaler
X_transformed = MinMaxScaler().fit_transform(X_broken)
estimator = KNeighborsClassifier()
transformed_scores = cross_val_score(estimator, X_transformed, y,scoring='accuracy')
print("The average accuracy for is{0:.1f}%".format(np.mean(transformed_scores) * 100))



from sklearn.pipeline import Pipeline
scaling_pipeline = Pipeline([('scale', MinMaxScaler()),('predict', KNeighborsClassifier())])
scores = cross_val_score(scaling_pipeline, X_broken, y,
scoring='accuracy')
print("The pipeline scored an average accuracy for is {0:.1f}%".format(np.mean(transformed_scores) * 100))









