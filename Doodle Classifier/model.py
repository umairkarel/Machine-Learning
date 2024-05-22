from NeuralNetworkClass import nn2 as network
import numpy as np
import pickle

features = ['cat', 'bat', 'rainbow']

x_train = np.load('Data/x_train.npy')
x_test = np.load('Data/x_test.npy')
y_train = np.load('Data/y_train.npy')
y_test = np.load('Data/y_test.npy')

# Can change network structure or hyperparameter
# model = network.NeuralNetwork(784, [10,10], 3, 0.01)
nn = pickle.load(open('doodle_classifier.pickle', 'rb'))

def normalize(data):
	return data / 255

def testAll(x_test, y_test):
	count = (x_test.shape)[0]
	val = 0

	for i in range(len(x_test)):
		prediction = model.predict(x_test[i])
		x = np.where(prediction == max(prediction))[0][0]
		y = np.where(y_test[i] == 1)[0][0]
		if x==y:
			val += 1
	return (val/count)*100

def shuffle(x,y):
	data = np.concatenate((x, y), axis=1)

	np.random.shuffle(data)

	x = data[:, :-3]
	y = data[:, -3:]

	return x,y

def train(x_train, y_train, epoch=1):
	
	for x in range(epoch):
		x_train, y_train = shuffle(x_train, y_train)

		for i in range(len(x_train)):
			inputs = x_train[i]
			targets = y_train[i]
			model.train(inputs, targets)

		print('Epoch {0} correct {1}'.format(x+1,testAll(x_test, y_test)))


# Saving trained model
# with open('doodle_classifier1.pickle', 'wb') as f:
# 	pickle.dump(model, f)
# pickle_in = open('doodle_classifier1.pickle', 'rb')
# print(testAll(x_test, y_test))