import pygame
from model import nn, features
import numpy as np
from PIL import Image

width = 400
height = 400
FPS = 60
white = (255,255,255)
black = (0,0,0)
radius = 10

pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((width,height))
clock = pygame.time.Clock()
screen.fill(white)
# fnt = pygame.font.SysFont("comicsans", 40)

def draw(scr, start, end, radius=1):
	dx = end[0] - start[0]
	dy = end[1] - start[1]
	dist = max(abs(dx), abs(dy))
	for i in range(dist):
		x = int(start[0] + float(i)/dist*dx)
		y = int(start[1] + float(i)/dist*dy)
		pygame.draw.circle(screen, black, (x,y), radius)

def predict():
	global msg
	
	img = Image.open('image.jpg')
	img = img.resize((28,28))
	img = img.convert('L')
	img = 255 - np.array(img)
	img = img.reshape(784,1) / 255
	p = nn.predict([img])
	result = str(np.argmax(p))
	print(result)

	# i = 0 if all(x < 0.5 for x in result) else 1

	# if i == 1:
	# text = fnt.render(str("It's a " + result), 1, (0,0,0))
	# else:
	# 	text = fnt.render(str("Not Sure!"), 1, (0,0,0))
	# screen.blit(text, (width/2 - 100, height-40))

last_pos = (0,0)
drawing = False
running = True
while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
		if event.type == pygame.MOUSEBUTTONDOWN:
			pygame.draw.circle(screen, black, event.pos, radius)
			drawing = True
		if event.type == pygame.MOUSEBUTTONUP:
			drawing = False
		if event.type == pygame.MOUSEMOTION:
			if drawing:
				pygame.draw.circle(screen, black, event.pos, radius)
				draw(screen, event.pos, last_pos, radius)
			last_pos = event.pos

		if event.type==pygame.KEYDOWN:
		    if event.key==pygame.K_RETURN:
		    	pygame.image.save(screen, "image.jpg")
		    	screen.fill(white)
		    	predict()

	pygame.display.flip()
	clock.tick(FPS)
pygame.quit()