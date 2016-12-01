from config2yr import *
import sys

folderName = sys.argv[1]
dumpFolder = homeDir + folderName + '/'
if not os.path.exists(dumpFolder):
    os.makedirs(dumpFolder)

os.chdir(dumpFolder)

levels = int(sys.argv[2])

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
                    [j, lat_i:lat_f, lon_i:lon_f]
            else:
                dataMap = np.mean(data.variables[var_name]\
                    [j, :levels, lat_i:lat_f, lon_i:lon_f], axis=0) 
                
            mapMean = np.mean(dataMap, dtype=np.float64)
            
            rawData[idx, :] = mapMean
        
        data.close()

    np.save(dumpFolder + title, rawData)
    print('done.')

