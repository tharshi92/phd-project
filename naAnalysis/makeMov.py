from config import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.basemap import Basemap, cm

''' This code will make a movie of a series of 2d plots. '''

dataFolder = homeDir + 'surfaceData/'
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

p = m.imshow(data, cmap='Blues', interpolation='None', vmin=0, vmax=400)
cb = m.colorbar(p, "right", size="5%", pad='2%')

def animate(i):
	data = rawData[i, :, :]
	m.imshow(data, cmap='Blues', interpolation='None', vmin=0, vmax=400)

anim = animation.FuncAnimation(fig, animate, np.arange(0, len(rawData)), interval=200)

FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title='Surface CO Field Data', \
                artist='tsrikann@physics.utoronto.ca')
writer = FFMpegWriter(fps=15, metadata=metadata)

#anim.save(dataFolder + 'COFieldMovie.mp4', writer=writer)

plt.show()
print('done.')