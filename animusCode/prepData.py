from config import *
import sys

# Name the data folder
folderName = sys.argv[1]
# How many levels to average over
levels = int(sys.argv[2])
# Describe the geometry of the area to analyze
lngIndex1 = sys.argv[3]
lngIndex2 = sys.argv[4]
latIndex1 = sys.argv[5]
latIndex2 = sys.argv[6]

dumpFolder = homeDir + folderName + '/'
if not os.path.exists(dumpFolder):
    os.makedirs(dumpFolder)

os.chdir(dumpFolder)

for metadata in metadatum:
    
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
                dataMap = data.variables[var_name]\
                    [j, latIndex1:latIndex2, lngIndex1:lngIndex2]
            else:
                dataMap = np.mean(data.variables[var_name]\
                    [j, :levels, latIndex1:latIndex2, lngIndex1:lngIndex2], axis=0) 
                
            mapMean = np.mean(dataMap, dtype=np.float64)
            
            rawData[idx, :] = mapMean
        
        data.close()

    np.save(dumpFolder + title, rawData)
    print('done.')

