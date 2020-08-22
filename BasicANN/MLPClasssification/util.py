from __future__ import print_function, division
from builtins import range
import numpy as np
import pandas as pd
from sklearn.utils import  shuffle

def init_weight_and_bias(M1, M2):
	W = np.random.randn(M1, M2) / np.sqrt(M1)
	b = np.zeros(M2)
	return W.astype(np.float32), b.astype(np.float32)

def init_filter(shape, poolsz):
	w = np.random.randn(*shape) * np.sqrt(2) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
	return w.astype(np.float32)

def relu(x):
	return x*(x > 0)

def sigmoid(A):
	return 1 / (1 + np.exp(-A))

def softmax(A):
	expA = np.exp(A)
	return expA / expA.sum(axis = 1, keepdims = True)

def sigmoid_cost(T, Y):
	return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()

def cost(T, Y):
	return -(T*np.log(Y)).sum()

def cost2(T, Y):
	N = len(T)
	return -np.log(Y[np.arange(N), T]).mean()

def error_rate(targets, predictions):
	return np.mean(targets != predictions)

def y2indicator(y):
	N = len(y)
	K = len(set(y))
	ind = np.zeros((N, K))
	for i in range(N):
		ind[i, y[i]] = 1
	return ind

def getData(balance_ones = True, Ntest = 1000):
	Y = []
	X = []
	first = True
	for line in open('fer2013.csv'):
		if first:
			first = False
		else:
			row = line.split(',')
			Y.append(int(row[0]))
			X.append([int(p) for p in row[1].split()])
	X, Y = np.array(X) / 255.0, np.array(Y)

	X, Y = shuffle(X, Y)
	Xtrain, Ytrain = X[:-Ntest], Y[:-Ntest]
	Xvalid, Yvalid = X[-Ntest:], Y[-Ntest:]

	if balance_ones:
		X0, Y0 = Xtrain[Ytrain != 1], Ytrain[Ytrain != 1]
		X1 = Xtrain[Ytrain == 1, :]
		X1 = np.repeat(X1, 9, axis = 0)
		Xtrain = np.vstack([X0, X1])
		Ytrain = np.concatenate((Y0, [1]*len(X1)))

	return Xtrain, Ytrain, Xvalid, Yvalid

def getImageData():
	Xtrain, Ytrain, Xvalid, Yvalid = getData()
	N, D = Xtrain.shape
	d = int(np.sqrt(D))
	Xtrain = Xtrain.reshape(-1, 1, d, d)
	Xvalid = Xvalid.reshape(-1, 1, d, d)
	return Xtrain, Ytrain, Xvalid, Yvalid

def getBinaryData():
	Y = []
	X = []
	first = True
	for line in open('fer2013.csv'):
		if first:
			first = False
		else:
			row = line.split(',')
			y = int(row[0])
			if y == 0 or y == 1:
				Y.append(y)
				X.append([int(p) for p in row[1].split()])
	return np.array(X) / 255.0, np.array(Y)

def crossValidation(model, X, Y, k=5):
	X, Y = shuffle(X, Y)
	sz = len(Y) // k
	errors = []
	for i in range(k):
		xtr = np.concatenate([X[:i*sz, :], X[(i*sz + sz):, :]])
		ytr = np.concatenate(Y[:i*sz], Y[(i*sz + sz):])
		xte = X[i*sz: (i*sz + sz), :]
		yte = Y[i*sz: (i*sz + sz)]

		model.fit(xtr, ytr)
		err = model.score(xte, yte)
		errors.append(err)

	print("errors:", errors)
	return np.mean(errors)

