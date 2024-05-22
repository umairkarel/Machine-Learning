import tkinter as ttk
from tkinter import *
from logic import *

root = Tk()
Height = 400
Width = 500

# Global Variables
data = np.array([[0,0]])
target = np.array([[-1]])
theta = np.zeros((3,1))
line = None
point = None
predicting = False
trained = False
currColor = 'Red'

def mousePress(event):
    global data
    global target
    global currColor

    x = event.x
    y = event.y

    if predicting:
        currColor = Predict(x/Width,y/Height)
    else:
        val = currColor=='Red'
        data = np.concatenate((data,np.array([[x/Width, y/Height]])), axis=0)
        target = np.concatenate((target,np.array([[val]])), axis=0)

    point = canvas.create_oval(x, y, x+4, y+4, fill=currColor)

def drawLine():
    global line

    if not trained:
        displayMsg("Please Train the Model First")
    else:
        # Deleting Previous Line
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

def displayMsg(msg):
    warn_display = Label(root, text=msg, font=("arial", "16"))
    warn_display.place(relx=0.3, rely=0.78)
    root.after(2000, warn_display.destroy)

def changeState(clr):
    global currColor
    global predicting

    currColor = clr

    if currColor == 'Green':
        predicting = True
        predict_button.config(fg='green')
        Rbutton.config(fg='black')
        Bbutton.config(fg='black')
    elif currColor == 'Blue':
        predicting = False
        Bbutton.config(fg='Blue')
        Rbutton.config(fg='black')
        predict_button.config(fg='black')
    else:
        predicting = False
        Rbutton.config(fg='Red')
        Bbutton.config(fg='black')
        predict_button.config(fg='black')

def train():
    global theta
    global trained

    if len(data) == 1:
        displayMsg("Please Add Some Data Points")
    else:
        theta = fit(data[1:,:], target[1:,:], theta, 10000)
        displayMsg("Your Model is Trained for 5000 iters!! Start Predicting")
        trained = True
        changeState('Green')

def startPrediction():
    global predicting

    if not trained:
        displayMsg("Please Train the Model First")
    else:
        predicting = True
        changeState('Green')

def Predict(x,y):
    if not trained:
        warn_display = Label(root, text="Please Train the Model First", font=("arial", "16"))
        warn_display.place(relx=0.2, rely=0.78)
        root.after(2000, warn_display.destroy)
    else:
        prediction = sigmoid(theta[0] + theta[1]*x + theta[2]*y) > 0.5

    return 'Red' if prediction >= 0.5 else 'Blue'
    

canvas = Canvas(root, width=Width, height=Height, background='white')
canvas.bind('<Button-1>', mousePress)
canvas.pack()

Data_Label = Label(root, text="Choose Data Point Color", font=("arial", "11"))
Data_Label.place(x=0, rely=0.83)

Rbutton = Button(root, text='Red', font=('Courier',20), fg='Red', command=lambda: changeState('Red'))
Rbutton.pack(side=LEFT, expand=True)

Bbutton = Button(root, text='Blue', font=('Courier',20), command=lambda: changeState('Blue'))
Bbutton.pack(side=LEFT, expand=True)

train_button = Button(root, text='Train Model', font=('Courier',20), command=lambda: train())
train_button.pack(side=LEFT, expand=True)

predict_button = Button(root, text='Predict', font=('Courier',20), command=startPrediction)
predict_button.pack(side=LEFT, expand=True)

boundary_button = Button(root, text='Decision Boundary', font=('Courier',20), command=lambda: drawLine())
boundary_button.pack(side=LEFT, expand=True)

root.mainloop()