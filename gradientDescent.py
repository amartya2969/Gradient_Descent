import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

global X_train,Y_train,X_test,Y_test,size_to_train,size_to_test
file_name='data.csv'
df=pd.read_csv(file_name,header=None)
df=df.values
X=df[:,:2]
Y=df[:,2]
X=X.astype(float)
Y=Y.astype(float)
size_to_train=math.ceil(X.shape[0]*0.7)
size_to_test=X.shape[0]-size_to_train
X_train=X[:size_to_train,:]
X_test=X[size_to_test:,:]
Y_train=Y[:size_to_train:]
Y_train=np.asarray(Y_train).reshape(len(Y_train),1)
Y_test=Y[size_to_train:]
Y_test=np.asarray(Y_test).reshape(len(Y_test),1)
def normalize(X_train,X_test):
	mn = np.mean(X_train,axis=0)
	std = np.std(X_train,axis=0)
	X_train = X_train-mn
	X_train = X_train/std
	X_test = X_test-mn
	X_test = X_test/std
	return X_train,X_test
def gradient_descent(X_train,Y_train):

	m = X_train.shape[0]

	learning_rate = 0.0005

	loss_array = []
	theta_array = []
	theta0_array = []
	alpha = []
	for j in range(10):
		alpha.append(learning_rate)

		loss = 100
		prev_loss = 200
		epoch = 0
		theta = np.zeros([1,X.shape[1]])

		theta0  = 0
		while abs(loss-prev_loss) > 0.00001:
			prediction = theta0 + np.dot(X_train,theta.T)
			error = prediction -Y_train



			prev_loss = loss
			loss = (0.5/m) * (np.sum(np.square(error)))

			derivative = np.dot(X_train.T,error)

			theta0 = theta0 - learning_rate * np.sum(error) / m
			theta = theta - learning_rate * (derivative.T) / m
			epoch = epoch +1
			print(loss,epoch)
		print('New Loop')
		loss_array.append(loss)
		theta0_array.append(theta0)
		theta_array.append(theta)
		learning_rate = learning_rate *2

	loss_array = np.asarray(loss_array)
	theta0_array = np.asarray(theta0_array)
	theta_array = np.asarray(theta_array)

	pos=np.argmin(loss_array)
	best_alpha = alpha[pos]
	print("Best learning rate = ",best_alpha)
	f1,a1 = plt.subplots()
	a1.plot(alpha,loss_array)
	a1.set(xlabel='learning_rate', ylabel='Mean Loss', title='Gradient_Descent')
	plt.xscale('log')
	a1.grid()
	plt.show()
	return theta0_array[pos],theta_array[pos]
X_train,X_test = normalize(X_train,X_test)
gradient_descent(X_train,Y_train)
