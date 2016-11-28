from config import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

''' This code will make a movie of a series of 2d plots. '''

dataFolder = homeDir + 'surface/'
rawData = np.load(dataFolder + 'COField.npy')

fig, ax = plt.subplots()

def animate(i):
	ax.imshow(rawData[i, :, :], interpolation='None', origin='lower', cmap='hot')
	plt.title('Slide ' + str(i))

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

anim = animation.FuncAnimation(fig, animate, np.arange(0, len(rawData)), interval=30)
anim.save('coMovie.mp4', writer=writer)

plt.show()
print('done.')