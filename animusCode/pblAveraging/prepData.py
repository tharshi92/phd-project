from config import *
import sys

folderName = sys.argv[1]
dumpFolder = homeDir + folderName + '/'
if not os.path.exists(dumpFolder):
    os.makedirs(dumpFolder)

os.chdir(dumpFolder)

pbls = np.load('PBL.npy')

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
            level = int(round(pbls[idx]))
            if level < 5:
                level = 5

            dataMap = np.mean(data.variables[var_name]\
                [j, :level, lat_i:lat_f, lon_i:lon_f], axis=0) 
                
            mapMean = np.mean(dataMap, dtype=np.float64)
            
            rawData[idx, :] = mapMean
        
        data.close()

    np.save(dumpFolder + title, rawData)
    print('done.')

