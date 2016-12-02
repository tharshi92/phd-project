from config import *
import sys

folderName = sys.argv[1]
dumpFolder = homeDir + folderName + '/'
if not os.path.exists(dumpFolder):
    os.makedirs(dumpFolder)

os.chdir(dumpFolder)

metadata = pbl_metadata
var_name = metadata[0]
title = metadata[1]
units = metadata[2]

ghost_file = dumpFolder + title + '.npy'
if os.path.isfile(ghost_file):
    print(ghost_file + ' already exists!! Goodbye.')
    sys.exit()

print('Preparing ' + title + ' Data..')

for i in range(d):

    trFile = fnames[i]
    data = Dataset(trFile)

    for j in range(24):
        idx = 24 * i + j
        dataMap = data.variables[var_name][j, lat_i:lat_f, lon_i:lon_f]
        mapMean = np.mean(dataMap, dtype=np.float64)
        rawData[idx, :] = mapMean

    data.close()

np.save(dumpFolder + title, rawData)
print('done.')

