from config import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.basemap import Basemap, cm

''' This code will make a movie of a series of 2d plots. '''

dataFolder = homeDir + 'surface/'
rawData = np.load(dataFolder + 'COField.npy')

fig = plt.figure()

m = Basemap(projection='mill', \
			resolution='c', \
			llcrnrlat=lats[lat_i], \
			urcrnrlat=lats[lat_f], \
            llcrnrlon=lons[lon_i], \
            urcrnrlon=lons[lon_f])

m.drawcoastlines()
m.drawcountries()

data = rawData[0, :, :]

x = np.linspace(0, m.urcrnrx, data.shape[1])
y = np.linspace(0, m.urcrnry, data.shape[0])

xx, yy = np.meshgrid(x, y)

p = m.imshow(data, cmap='Blues', interpolation='None', vmin=0, vmax=400)
cb = m.colorbar(p, "right", size="5%", pad='2%')

def animate(i):
	data = rawData[i, :, :]
	p = m.imshow(data, cmap='Blues', interpolation='None', vmin=0, vmax=400)

anim = animation.FuncAnimation(fig, animate, np.arange(0, len(rawData)), interval=60)

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=60)
# anim.save(dataFolder + 'COFieldMovie.mp4', writer=writer)

plt.show()
print('done.')