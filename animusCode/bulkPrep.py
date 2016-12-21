from config import *
import sys

# Name the data folder
folderName = 'geosChem4x5'

dumpFolder = homeDir + folderName + '/'
if not os.path.exists(dumpFolder):
    os.makedirs(dumpFolder)

os.chdir(dumpFolder)

for metadata in metadatum:

    M = np.zeros((n, 46, 72, 29))
    
    var_name = metadata[0]
    title = metadata[1]
    units = metadata[2]
    
    ghost_file = dumpFolder + title + '.npy'
    if os.path.isfile(ghost_file):
        print(ghost_file + ' already exists!! Skipping to next data stream.')
        continue
    
    print('Preparing ' + title + ' Data..')
    
    for i in range(d):
        
        trFile = fnames[i]
        data = Dataset(trFile)
        
        for j in range(24):
            
            idx = 24 * i + j
            
            if var_name == 'PBLDEPTH__PBL_M':
                dataMap = data.variables[var_name][j]
            else:
                dataMap = data.variables[var_name][j]
            
            M[idx, :, :, :] = dataMap
        
        data.close()

    np.save(dumpFolder + title, M)
    del M
    print('done.')

