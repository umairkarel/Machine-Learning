import tkinter as ttk
from tkinter import *
import time
import numpy as np
import matplotlib.pyplot as plt

# Optimal Branch

root = Tk()

data = np.array([[0,0]])
target = np.array([[-1]])
line = None
Height = 400
Width = 500
point = None
color = 'Red'
theta = np.zeros((3,1))
predicting = False

def mousePress(event):
    global data
    global target
    global color

    x = event.x
    y = event.y

    if predicting:
        color = Predict(x/Width,y/Height)
    else:
        val = color=='Red'
        data = np.concatenate((data,np.array([[x/Width, y/Height]])), axis=0)
        target = np.concatenate((target,np.array([[val]])), axis=0)

    point = canvas.create_oval(x, y, x+4, y+4, fill=color)

def drawLine():
    global line

    if line:
        canvas.delete(line)
        
    m = -theta[1][0]/theta[2][0]
    b = -theta[0][0]/theta[2][0]

    x1 = 0
    y1 = (m*x1 + b)
    x2 = 1
    y2 = (m*x2 + b)

    x1 = Width*x1+2
    x2 = Width*x2+2
    y1 = y1*Height+2
    y2 = y2*Height+2
    line = canvas.create_line(x1, y1, x2, y2,fill='red')

def changeColor(clr):
    global color
    global predicting
    global theta

    if not clr:
        predicting = True
        theta = fit(data[1:, :], target[1:, :], theta, 10000)
    else:
        color = clr
        predicting = False

    if predicting == True:
        predict_button.config(fg='green')
        Rbutton.config(fg='black')
        Bbutton.config(fg='black')
    elif color == 'Blue':
        Bbutton.config(fg='Blue')
        Rbutton.config(fg='black')
        predict_button.config(fg='black')
    else:
        Rbutton.config(fg='Red')
        Bbutton.config(fg='black')
        predict_button.config(fg='black')

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

    # plt.plot([i for i in range(10000)], cost_hist)
    # plt.show()
    return theta

def Predict(x,y):
    global data
    global target
    
    drawLine()
    prediction = sigmoid(theta[0] + theta[1]*x + theta[2]*y) > 0.5

    return 'Red' if prediction >= 0.5 else 'Blue'

canvas = Canvas(root, width=Width, height=Height, background='white')
canvas.bind('<Button-1>', mousePress)
canvas.pack()

Rbutton = Button(root, text='Red', font=('Courier',20), fg='Red', command=lambda: changeColor('Red'))
Rbutton.pack(side=LEFT, expand=True)

Bbutton = Button(root, text='Blue', font=('Courier',20), command=lambda: changeColor('Blue'))
Bbutton.pack(side=LEFT, expand=True)

predict_button = Button(root, text='Predict', font=('Courier',20), command=lambda: changeColor(''))
predict_button.pack(side=LEFT, expand=True)

root.mainloop()