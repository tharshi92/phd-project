from configMaps import *
import numpy as np
from time import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.basemap import Basemap, cm

# Import data
start = 24 * (365 + 250 - 1)
end = 24 * (365 + 300 - 1)
dataFolder = homeDir + 'surfaceData/'
d = np.load(dataFolder + 'COField.npy')#[start:end, :, :]

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
m = Basemap(llcrnrlon=lng1, \
			llcrnrlat=lat1, \
			urcrnrlon=lng2, \
			urcrnrlat=lat2,)
m.drawcoastlines()
a = np.random.random((d.shape[1], d.shape[2]))
im = m.imshow(d[0, :, :], \
			interpolation='None', \
			animated=True, \
			origin='lower', \
			cmap='jet')

# # Initialization Function: plot the background of each frame
# def init():
#     im.set_data(np.random.random((d.shape[1], d.shape[2])))
#     return [im]

# # Animation function (updates each frame)
# def animate(i):
#     a = im.get_array()
#     a = d[i, :, :]
#     im.set_array(a)
#     plt.title('{0} days from the 250th day of 2007'.format(i/24))
#     return [im]

# # Calculate the interval
# f = 30. # frames per second
# dt = 1.0 / f
# t0 = time()
# animate(0)
# t1 = time()
# interval = 1000 * dt - (t1 - t0)

# # Call the animator
# anim = animation.FuncAnimation(fig, animate, init_func=init, \
# 								frames=len(d), \
# 								interval=interval, \
# 								blit=False)
# print('saving animation...')
# # anim.save(dataFolder + 'mapMovie.mp4', fps=f, \
# # 		extra_args=['-vcodec', 'libx264'])

# print('done.')
plt.show()