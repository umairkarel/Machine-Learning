from tkinter import *
from lin_reg_algos import *

root = Tk()

data = []
m = 1 # Slope
b = 0 # y-intercept
line = None
Height =400
Width = 400
point = None

def mousePress(event):
    global data
    global m
    global b

    x = event.x
    y = event.y
    data.append([x/Width, y/Height])
    point = canvas.create_oval(x, y, x+4, y+4, fill='blue')

    if len(data)>1:
    	m,b = linearRegression_GD(data, m, b)
    	draw_line()

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