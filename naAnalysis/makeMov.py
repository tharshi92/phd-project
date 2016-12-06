from config import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.basemap import Basemap, cm

# Import data
dataFolder = homeDir + 'surfaceData/'
d = np.load(dataFolder + 'COField.npy')

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
a = np.random.random((d.shape[1], d.shape[2]))
im = plt.imshow(d[0, :, :], \
			interpolation='None', \
			animated=True, \
			origin='lower', \
			cmap='inferno')

# Initialization Function: plot the background of each frame
def init():
    im.set_data(np.random.random((d.shape[1], d.shape[2])))
    return [im]

# Animation function (updates each frame)
def animate(i):
    a = im.get_array()
    a = d[i, :, :]
    im.set_array(a)
    return [im]

# Call the animator
anim = animation.FuncAnimation(fig, animate, init_func=init, \
								frames=len(d), \
								interval=1, \
								blit=False)

plt.show()