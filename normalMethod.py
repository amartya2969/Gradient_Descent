import pandas as pd
import numpy as np

file_name = 'data.csv'
df=pd.read_csv(file_name,header=None)
df = df.values
X = df[:,:2]
Y = df[:,2]
X = X.astype(float)
Y = Y.astype(float)
Y = np.expand_dims(Y,axis=0)
Y =Y.T
Y = Y.reshape(len(Y),1)
to_test=int(X.shape[0])
size_to_train=int(X.shape[0]*0.7)
X_train,X_test = X[:size_to_train,:],X[size_to_train:,:]
Y_train,Y_test = Y[:size_to_train,:],Y[size_to_train:,:]



num_features = X_train.shape[1]
num_train = X_train.shape[0]
num_test = X_test.shape[0]



phi_train = np.concatenate((np.ones((1,num_train)),X_train.T)).T
phi_inv = np.linalg.pinv(phi_train)
theta = np.matmul(phi_inv,Y_train)
X_test = np.concatenate((np.ones((1,num_test)),X_test.T))
prediction = np.matmul(theta.T,X_test)
loss = (0.5) * np.sum((prediction-Y_test.T)**2)


print ('0.5 * Sum of Squared Errors =', loss)
print('0.5 * Mean of Sum of Squared Errors =', loss/num_test)
