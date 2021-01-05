import numpy as np

def linearRegression_OLS(data, m, b):
    xsum = sum([i for i,j in data])
    ysum = sum([j for i,j in data])

    xmean = xsum/len(data)
    ymean = ysum/len(data)

    num = 0
    den = 0
    for x,y in data:
        num += (x-xmean)*(y-ymean)
        den += (x-xmean)**2

    m = num/den
    b = ymean - m*xmean

    return m,b

def linearRegression_SGD(data, m, b, iters=1000, alpha=0.05):
    for _ in range(iters):
        for x,y in data:
            guess = m*x + b
            error = y-guess
            m += (error*x*alpha)
            b += (error*alpha)

    return m,b

def linearRegression_GD(data, m, b, iters=500, alpha=0.05):
    theta = np.array([b,m]).reshape(-1,1)
    n = len(data)
    X = np.array([[1,x] for x,y in data])
    y = np.array([[y] for x,y in data])

    for i in range(iters):
    	theta = theta - alpha*(1/n) * (X.T @ (X@theta- y))

    b,m = theta[:, 0]

    return m,b