import numpy as np
import math
import random
import pickle

# Classes
class NeuralNetwork:
	def __init__(self, numI, numH, numO, alpha=0.1):
		self.input_nodes = numI
		self.hidden_nodes = len(numH)
		self.output_nodes = numO
		self.weights = []
		self.biases = []
		self.alpha = alpha
		self.dimnesions = [self.input_nodes] + numH + [self.output_nodes]
		self.layers = len(numH) + 2

		for layer in range(self.layers - 1):
			weight = np.random.uniform(-1, 1, (self.dimnesions[layer+1], self.dimnesions[layer]))
			self.weights.append(weight)

		for layer in range(1, self.layers):
			bias = np.random.uniform(-1, 1, (self.dimnesions[layer], 1))
			self.biases.append(bias)

		# self.weights_ih = np.random.uniform(-1, 1, (self.hidden_nodes, self.input_nodes))
		# self.weights_ho = np.random.uniform(-1, 1, (self.output_nodes, self.hidden_nodes))
		# self.bias_h = np.random.uniform(-1, 1, (self.hidden_nodes, 1))
		# self.bias_o = np.random.uniform(-1, 1, (self.output_nodes, 1))

	def sigmoid(self, x):
		return 1/(1+np.exp(-x))

	def drev_sigmoid(self, y):
		return np.multiply(y,(1-y))

	def predict(self, inputs):
		weights = [inputs] + self.weights

		for layer in range(self.layers-2):
			if layer == 0:
				hidden = (weights[layer+1] @ weights[layer]).reshape(-1,1)
				hidden = np.add(hidden, self.biases[layer])
				hidden = self.sigmoid(hidden)
			else:
				hidden = (weights[layer+1] @ hidden).reshape(-1,1)
				hidden = np.add(hidden, self.biases[layer])
				hidden = self.sigmoid(hidden)

		output = (self.weights[-1] @ hidden).reshape(-1,1)
		output = np.add(output, self.biases[-1])
		output = self.sigmoid(output)

		return output.flatten()

	def feedforward(self, inputs):
		weights = [inputs] + self.weights
		hidden_layers = []

		# FeedForward
		for layer in range(self.layers-2):
			if layer == 0:
				hidden = (weights[layer+1] @ weights[layer]).reshape(-1,1)
				hidden = np.add(hidden, self.biases[layer])
				hidden = self.sigmoid(hidden)
			else:
				hidden = (weights[layer+1] @ hidden).reshape(-1,1)
				hidden = np.add(hidden, self.biases[layer])
				hidden = self.sigmoid(hidden)
			hidden_layers.append(hidden)

		return hidden_layers

	def backpropagate(self, inputs, layers, error):

		for layer in reversed((range(len(layers)))):
			gradient = self.drev_sigmoid(layers[layer])
			gradient = np.multiply(gradient, error)
			gradient *= self.alpha
			delta = gradient @ layers[layer-1].T

			if layer != 0:
				self.weights[layer] = np.add(self.weights[layer], delta)
				self.biases[layer] = np.add(self.biases[layer], gradient)
				error = self.weights[layer].T @ error

		hidden_grad = self.drev_sigmoid(layers[0])
		hidden_grad = np.multiply(hidden_grad, error)
		hidden_grad *= self.alpha
		delta = hidden_grad @ inputs.reshape(1,-1)

		self.weights[0] = np.add(self.weights[0], delta)
		self.biases[0] = np.add(self.biases[0], hidden_grad)

	def train(self, inputs, targets):
		inputs = np.array(inputs)
		targets = np.array(targets)

		# FeedForward
		hidden_layers = self.feedforward(inputs)

		output = (self.weights[-1] @ hidden_layers[-1]).reshape(-1,1)
		output = np.add(output, self.biases[-1])
		output = self.sigmoid(output)

		backprop_layers = hidden_layers + [output]

		# Calculating Errors
		targets = targets.reshape(-1,1)
		error = np.subtract(targets, output)

		# BackPropagate
		self.backpropagate(inputs, backprop_layers, error)