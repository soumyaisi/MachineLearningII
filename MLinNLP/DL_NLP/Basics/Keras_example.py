import keras
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
# Dataset Link : # https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
# Save the dataset as a .csv file :
tran_ = np.genfromtxt('transfusion.data', delimiter=',')
#tran_  = 
X=tran_[:,0:4]           # The dataset offers 4 input variables
Y=tran_[:,4]             # Target variable with '1' and '0'
print(X)

# Creating our first MLP model with Keras
mlp_keras = Sequential()
mlp_keras.add(Dense(8, input_dim=4, init='uniform', activation='relu'))
mlp_keras.add(Dense(6, init='uniform', activation='relu'))
mlp_keras.add(Dense(1, init='uniform', activation='sigmoid'))
mlp_keras.compile(loss = 'binary_crossentropy', optimizer='adam',metrics=['accuracy'])
mlp_keras.fit(X,Y, epochs=200, batch_size=8, verbose=0)
accuracy = mlp_keras.evaluate(X,Y)
print("Accuracy : %.2f%% " %  (accuracy[1]*100 ))

# Using a different set of optimizer
from keras.optimizers import SGD
opt = SGD(lr=0.01)
mlp_optim = Sequential()
mlp_optim.add(Dense(8, input_dim=4, init='uniform', activation='relu'))
mlp_optim.add(Dense(6, init='uniform', activation='relu'))
mlp_optim.add(Dense(1, init='uniform', activation='sigmoid'))
# Compiling the model with SGD
mlp_optim.compile(loss = 'binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# Fitting the model and checking accuracy
mlp_optim.fit(X,Y, validation_split=0.3, epochs=150, batch_size=10, verbose=0)
results_optim = mlp_optim.evaluate(X,Y) 
print("Accuracy : %.2f%%" % (results_optim[1]*100 ))



