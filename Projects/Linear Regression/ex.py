import tkinter as ttk
from tkinter import *
import time
import numpy as np

root = Tk()

class LinearRegression:

	def __init__(self, X, y, theta_starting, num_iteration=1000, learning_rate=0.01):
		self.X = X
		self.y = y
		self.theta = theta
		self.iteration = num_iteration
		self.alpha = learning_rate

data = []
m = 1
b = 0
line = None
Height = 600
Width = 1000
point = None

def mousePress(event):
	x = event.x
	y = event.y
	data.append([x/Width, y/Height])
	point = canvas.create_oval(x, y, x+4, y+4, fill='blue')

	if len(data)>1:
		linearRegression_GD()
		print(data)
		draw_line()


def linearRegression_OLS():
	global data
	global m
	global b

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

def linearRegression_SGD():
	global data
	global m
	global b

	alpha = 0.05

	for _ in range(1000):
		for x,y in data:
			guess = m*x + b
			error = y-guess
			m += (error*x*alpha)
			b += (error*alpha)

def linearRegression_GD():
	global data
	global m
	global b

	theta = np.array([b,m]).reshape(-1,1)
	n = len(data)
	X = np.array([[1,x] for x,y in data])
	y = np.array([[y] for x,y in data])

	alpha = 0.05

	for i in range(500):
		theta = theta - alpha*(1/n) * (X.T @ (X@theta- y))

	b,m = theta[:, 0]
	# print(m,b)

def draw_line():
	global m
	global b
	global line

	if line:
		canvas.delete(line)

	x1 = 0
	y1 = (m*x1 + b)
	x2 = 1
	y2 = (m*x2 + b)

	x1 = Width*x1+2
	x2 = Width*x2+2
	y1 = y1*Height+2
	y2 = y2*Height+2

	line = canvas.create_line(x1, y1, x2, y2,fill='red')

canvas = Canvas(root, width=Width, height=Height, background='white')
canvas.bind('<Button-1>', mousePress)
canvas.pack()


root.mainloop()