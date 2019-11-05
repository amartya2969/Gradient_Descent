import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read():
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
    to_test=int(X.shape[0]*0.6)
    to_cross_validate=int(X.shape[0]*0.8)
    X_train,X_cv,X_test = X[:to_test,:],X[to_test:to_cross_validate,:],X[to_cross_validate:,:]
    Y_train,Y_cv,Y_test = Y[:to_test,:],Y[to_test:to_cross_validate,:],Y[to_cross_validate:,:]
    return X_train,X_cv,X_test,Y_train,Y_cv,Y_test

def normalize(X_train,X_cv,X_test):
	mn = np.mean(X_train,axis=0)
	std = np.std(X_train,axis=0);
	X_train = X_train-mn
	X_train = X_train/std

	X_test = X_test-mn
	X_test = X_test/std

	X_cv=X_cv-mn
	X_cv=X_cv/std

	return X_train,X_cv,X_test

def l1_regularization(theta_init,lambd,X,Y):
	m = X.shape[0]
	num_features = X.shape[1]

	theta0 = 0
	theta  = theta_init
	learning_rate = 0.5
	epochs = 200
	for epoch in range(epochs):
		prediction = theta0 + np.dot(X,theta.T)
		error = prediction - Y
		derivative = np.dot(X.T,error) + lambd*np.sign(theta.T)
		cost = (0.5  / m) * (np.sum(np.square(error))) + lambd * np.sum(np.abs(theta.T))
		theta0 = theta0 - (learning_rate*np.sum(error) / m)
		theta = theta - ((learning_rate) * derivative.T / m)
	return theta0, theta

def l2_regularization(theta_init,lambd,X,Y):
	m = X.shape[0]
	num_features = X.shape[1]
	theta0 = 0
	theta  = theta_init
	learning_rate = 0.5
	epochs = 200
	for epoch in range(epochs):
		prediction = theta0 + np.dot(X,theta.T)
		error = prediction - Y
		derivative = np.dot(X.T,error) + lambd*theta.T
		theta0 = theta0 - learning_rate * np.sum(error) / m
		theta = theta - learning_rate * derivative.T / m
	return theta0, theta


def predict(theta0,theta,X):
	return theta0 + np.dot(X,theta.T)


X_train,X_cv,X_test,Y_train,Y_cv,Y_test = read()
num_features = X_train.shape[1]
mcv = X_cv.shape[0]
mtest = X_test.shape[0]

X_train,X_cv,X_test = normalize(X_train,X_cv,X_test)

l1_params = np.logspace(0, 79, num=60, base=1.1)
theta_intial = np.ones([1,num_features])
l1_loss_array = []
for lamb in l1_params:
	theta0, theta =l1_regularization(theta_intial,lamb,X_train,Y_train)
	cvpred = predict(theta0 , theta, X_cv)
	cvloss = (0.5 / mcv) * np.sum((cvpred - Y_cv)**2)
	l1_loss_array.append(cvloss)

l1_best = l1_params[np.argmin(l1_loss_array)]


l2_params = np.logspace(0, 48, num=60, base=1.1)
l2_loss_array = []
for lamb in l2_params:
	theta0, theta = l2_regularization(theta_intial,lamb,X_train,Y_train)
	cvpred = predict(theta0, theta, X_cv)
	cvloss = (0.5 / mcv) * np.sum((cvpred - Y_cv)**2)
	l2_loss_array.append(cvloss)
l2_best = l2_params[np.argmin(l2_loss_array)]


f1,a1 = plt.subplots()
a1.plot(l1_params,l1_loss_array)
a1.set(xlabel='Lambda', ylabel='Cross-Validation Loss', title='L1 Regularization')
a1.grid()
plt.xscale('log')
plt.show()

f2,a2 =plt.subplots()
a2.plot(l2_params,l2_loss_array)
a2.set(xlabel='Lambda', ylabel='Cross-Validation Loss', title='L2 Regularization')
a2.grid()
plt.xscale('log')
plt.show()



theta0, theta = l1_regularization(theta_intial,l1_best,X_train,Y_train)
l1_prediction = predict(theta0, theta,X_test)
l1_loss = (0.5  ) * np.sum((l1_prediction - Y_test)**2)
print('Best L1 Regularization Lambda =', l1_best)
print('0.5 * Sum of Squared Errors (L1 Regularization) =', l1_loss)
print('0.5 * Mean of Sum of Squared Errors(L1 Regularization) =', l1_loss/mtest)

theta0, theta = l2_regularization(theta_intial,l2_best,X_train,Y_train)
l2_prediction = predict(theta0, theta,X_test)
l2_loss = (0.5  ) * np.sum((l2_prediction - Y_test)**2)
print('Best L2 Regularization Lambda =', l2_best)
print('0.5 *  Sum of Squared Errors (L2 Regularization) =', l2_loss)
print('0.5 *  Mean of Sum of Squared Errors (L2 Regularization) =', l2_loss/mtest)
""""
Best L1 Regularization Lambda = 93.05097044136404
Best L2 Regularization Coefficient = 1.6446318218438827
0.5 * Mean Squared Test Loss (L1 Regularization) = 10.935661352378938
0.5 * Mean Squared Test Loss (L2 Regularization) = 10.939519556544177
"""
