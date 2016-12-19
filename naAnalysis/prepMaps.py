from configMaps import *
from netCDF4 import Dataset

metadata = field_metadata

var_name = metadata[0]
title = metadata[1]
units = metadata[2]

print('Preparing ' + title + ' Data..')

for i in range(d):
    trFile = fnames[i]
    data = Dataset(trFile)
    
    for j in range(24):
        
        idx = 24 * i + j

        dataMap = np.mean(data.variables[var_name]\
            [j, :18, latIndex1:latIndex2, lngIndex1:lngIndex2], axis=0)
        
        rawData[idx, :, :] = dataMap
    
    data.close()

np.save(saveDir + title, rawData)
print('done.')




