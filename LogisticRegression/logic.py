import numpy as np

def sigmoid(z):
    return (1/(1+ np.exp(-z)))

def cost(h, y):
    m = len(y)
    J = (-1/m) * (y.T @ np.log(h) + (1-y).T @ np.log(1-h))

    return J[0][0]

def fit(X, y, theta, iters=1000):
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    m = len(y)
    alpha = 0.01
    cost_hist = []

    for i in range(iters):
        h = sigmoid(X@theta)
        grad = (1/m) * (X.T @ (h-y))
        theta = theta - grad*alpha
        cost_hist.append(cost(h,y))

    return theta