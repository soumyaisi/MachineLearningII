"""
Recommendation system by rating prediction in tensorflow-2.
"""


#Import libraies..............
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from  sklearn.utils import shuffle


#Load the csv file................
df = pd.read_csv('ml-25m/ml-25m/ratings.csv')
print(df.head())


#Data preprocessing.................
df.userId = pd.Categorical(df.userId)
df['new_user_id'] = df.userId.cat.codes

df.movieId = pd.Categorical(df.movieId)
df['new_movie_id'] = df.movieId.cat.codes

user_ids = df['new_user_id'].values
movie_ids = df['new_movie_id'].values
ratings = df['rating'].values

N = len(set(user_ids))
M = len(set(movie_ids))

K = 10


#Create the model using LU decomposition...............
u = Input(shape=(1,))
m = Input(shape=(1,))

u_emb = Embedding(N, K)(u)#(no of sample, 1, K)
m_emb = Embedding(M,K)(m)

u_emb = Flatten()(u_emb)#(no of sample, K)
m_emb = Flatten()(m_emb)

x = Concatenate()([u_emb, m_emb])
x = Dense(1024, activation = 'relu')(x)
x = Dense(1)(x)


#Compile the model..............
model = Model(inputs = [u,m], outputs = x)
model.compile(loss = 'mse', optimizer = SGD(lr = 0.08, momentum = 0.9))


#Split the data........................
user_ids, movie_ids, ratings = shuffle(user_ids, movie_ids, ratings)
Ntrain = int(0.8*len(ratings))
train_user = user_ids[:Ntrain]
train_movie = movie_ids[:Ntrain]
train_ratings = ratings[:Ntrain]

test_user = user_ids[Ntrain:]
test_movie = movie_ids[Ntrain:]
test_ratings = ratings[Ntrain:]


#Scale the data...............
avg_ratings = train_ratings.mean()
train_ratings = train_ratings - avg_ratings
test_ratings = test_ratings - avg_ratings


#Fit the model..................
r = model.fit(x = [train_user, train_movie], y = train_ratings, epochs = 25, batch_size = 1024, verbose = 2, validation_data = ([test_user, test_movie], test_ratings),)


#Plot the loss.................
plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()



