from configMaps import *
from netCDF4 import Dataset

for metadata in metadatum:
    
    var_name = metadata[0]
    title = metadata[1]
    units = metadata[2]
    
    ghost_file = saveDir + title + '.npy'
    if os.path.isfile(ghost_file):
        print(ghost_file + ' already exists!! Skipping to next data stream.')
        continue
    
    print('Preparing ' + title + ' Data..')
    
    for i in range(d):
        print(fnames[i])
        trFile = fnames[i]
        data = Dataset(trFile)
        
        for j in range(24):
            
            idx = 24 * i + j
            
            if var_name == 'PBLDEPTH__PBL_M':
                dataMap = data.variables[var_name][j, lat_i:lat_f, lon_i:lon_f]
            else:
                dataMap = data.variables[var_name][j, 0, lat_i:lat_f, lon_i:lon_f]
            
            rawData[idx, :, :] = dataMap
        
        data.close()

    np.save(saveDir + title, rawData)
    print('done.')




