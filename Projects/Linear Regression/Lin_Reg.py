import tkinter as ttk
from tkinter import *
import time
import numpy as np

root = Tk()

data = []
m = 1
b = 0
line = None


def mousePress(event):
	x = event.x
	y = event.y
	data.append([x/Width, y/Height])
	canvas.create_oval(x, y, x+4, y+4, fill='blue')

def draw():
	if len(data) > 1:
		linearRegression_GD()
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

def linearRegression_GD():
	global data
	global m
	global b
	m = 0
	b = 0
	theta = np.array([b,m]).reshape(-1,1)
	n = len(data)
	X = np.array([[1,x] for x,y in data])
	y = np.array([[y] for x,y in data])

	alpha = 0.5

	for i in range(1000):
		theta = theta - alpha*(1/n) * (X.T @ (X@theta- y))
		b,m = theta[:, 0]
		label.config(text='Epoch : '+str(i))
		lable_y.config(text='y = {:.2f}x + {:.2f}'.format(m,b))
		draw_line()
		time.sleep(0.03)


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
	root.update()

canvas = Canvas(root, width=Width, height=Height, background='white')
canvas.bind('<Button-1>', mousePress)
canvas.pack()

button = Button(root, width=10, text='Start', command=draw)
button.pack()

label = Label(root, width=15,text='Epoch : ')
label.pack()

lable_y = Label(root, width=20, text='y = '+str(m)+'x + '+str(b))
lable_y.pack()

root.mainloop()