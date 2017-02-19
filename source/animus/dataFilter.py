from config import *
import sys

# Name the data folder
folderName = sys.argv[1]

# Path to data folder
dumpFolder = homeDir + folderName + '/'
if not os.path.exists(dumpFolder):
    os.makedirs(dumpFolder)

# Number of levels to average oveer
levels = int(sys.argv[2])

# Describe the geometry of the area to analyze
lngIndex1 = int(sys.argv[3])
lngIndex2 = int(sys.argv[4])
latIndex1 = int(sys.argv[5])
latIndex2 = int(sys.argv[6])

# Main loop
for metadata in metadatum:
    
    var_name = metadata[0]
    title = metadata[1]
    units = metadata[2]
    
    print('Preparing ' + title + ' Data..')

    # Navigate to bulk data folder
    os.chdir(homeDir + 'geosChem4x5/')

    dataFile = np.mean(np.load('{0}.npy'.format(title))\
        [:, :levels, latIndex1:latIndex2, lngIndex1:lngIndex2], axis=1)

    # Navigate to data folder
    os.chdir(dumpFolder)

    np.save(title, dataFile)
    del dataFile
    print('done.')