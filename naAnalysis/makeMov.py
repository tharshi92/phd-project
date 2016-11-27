from config import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs

''' This code will make a movie of a series of 2d plots. '''

dataFolder = homeDir + 'surface/'
rawData = np.load(dataFolder + 'COField.npy')

fig, ax = plt.subplots()

def animate(i):
	ax.imshow(rawData[i, :, :], interpolation='None', origin='lower', cmap='hot')
	plt.title('Slide ' + str(i))

anim = animation.FuncAnimation(fig, animate, np.arange(0, len(rawData)), interval=120)
#anim.save('animation.mp4', fps=20, extra_args=['-vcodec', 'libx264'])

plt.show()
print('done.')

# Show map