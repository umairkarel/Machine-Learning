from NeuralNetwork import model
import pygame
import random
import math
import numpy as np
import pickle

color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
white = (225,225,225)
black = (0,0,0)

width = 500
height = 400
screen = pygame.display.set_mode((width,height))
clock = pygame.time.Clock()
FPS = 60

# model = pickle.load(open('color_predictor.pickle', 'rb'))

def map(n, start1, stop1, start2, stop2):
	return ((n-start1)/(stop1-start1))*(stop2-start2)+start2

def is_in_circle(x,y,pos):
	sqy = (y - 190)**2
	if pos:
		sqx = (x - 125*3)**2
	else:
		sqx = (x - 125)**2
	if math.sqrt(sqx+sqy) < 50:
		return True
	return False

def predict(r,g,b):
	return round(model.feedforward([r,g,b])[0])

def train(inputs, target):
	model.train(inputs, target)

def draw(guess):
	pygame.draw.line(screen, black, (250,0), (250,400), 3)
	pygame.draw.circle(screen, white, (125, 190), 50)
	pygame.draw.circle(screen, black, (125*3, 190), 50)

	if guess == 0:
		pygame.draw.circle(screen, white, (125, 100), 15)
	elif guess == 1:
		pygame.draw.circle(screen, black, (125*3, 100), 15)

def reset_window():
	pygame.display.flip()
	screen.fill(color)
	clock.tick(FPS)

def change_color():
	global r,g,b
	global guess

	r,g,b = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
	# print((r,g,b))
	# print('------')
	guess = predict(r,g,b)
	# print(guess)
	# print('----')
	

r,g,b = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
guess = predict(r, g, b)
running = True

while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
		if event.type == pygame.MOUSEBUTTONDOWN:
			x,y = pygame.mouse.get_pos()
			pos = 1 if x > 250 else 0

			if is_in_circle(x,y,pos):
				color = (r,g,b)
				target = pos

				# print('###')
				# print((r,g,b))
				# print('###')
				# print(guess)
				train([r,g,b], target)

				change_color()
			
	color = (r,g,b)
	draw(guess)
	reset_window()
	
with open('color_predictor.pickle', 'wb') as f:
	pickle.dump(model, f)
pickle_in = open('color_predictor.pickle', 'rb')

pygame.quit()