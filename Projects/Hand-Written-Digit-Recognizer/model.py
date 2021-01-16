from NeuralNetworkClass import nn2 as network
import numpy as np
import pickle
import matplotlib.pyplot as plt

features = [i for i in range(10)]

x_train = np.load('Data/x_train.npy')
x_test  = np.load('Data/x_test.npy')
y_train = np.load('Data/y_train.npy')
y_test  = np.load('Data/y_test.npy')

# Can change network structure or hyperparameter
model = network.NeuralNetwork(784, [16,16], 10, 0.03)
# nn = pickle.load(open('my_model.pickle', 'rb'))

def normalize(data):
	return data / 255

def testAll(x_test, y_test):
	count = (x_test.shape)[0]
	val = 0

	for i in range(len(x_test)):
		prediction = model.predict(x_test[i])
		x = np.where(prediction == max(prediction))[0][0]
		y = y_test[i]
		if x==y:
			val += 1
	return (val/count)*100

def shuffle(x,y):
	data = np.concatenate((x, y), axis=1)

	np.random.shuffle(data)

	x = data[:, :-1]
	y = data[:, -1:]

	return x,y

def train(x_train, y_train, epoch=1):
	
	for x in range(epoch):
		x_train, y_train = shuffle(x_train, y_train)

		for i in range(len(x_train)):
			inputs = x_train[i]
			y = int(y_train[i,0])
			targets = [1 if y == i else 0 for i in range(10)]
			model.train(inputs, targets)

		print('Epoch {0} accuracy {1}%'.format(x+1,testAll(x_test, y_test)))

x_train = normalize(x_train)
y_train = y_train.reshape(-1,1)

train(x_train, y_train, 1)

x_test = normalize(x_test)
print("Model accuracy: {0}%".format(testAll(x_test, y_test)))

# Saving trained model
# with open('my_model.pickle', 'wb') as f:
# 	pickle.dump(model, f)
# pickle_in = open('my_model.pickle', 'rb')
