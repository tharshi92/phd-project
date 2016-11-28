from configFinal import *

dumpFolder = homeDir + 'finalRun/'

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
                    [j, :18, lat_i:lat_f, lon_i:lon_f], axis=0) 
                
            mapMean = np.mean(dataMap, dtype=np.float64)
            
            rawData[idx, :] = mapMean
        
        data.close()

    np.save(dumpFolder + title, rawData)
    print('done.')

#%%

metadata = source_metadata

var_name = metadata[0]
title = metadata[1]
units = metadata[2]

print('Preparing ' + title + ' Data..')

calendar_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
leap_calendar_days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
count = [0]
temp = 0
for year in yrs:
    y = int(float(year))
    if y % 4 != 0:
        for days in calendar_days:
            temp += days
            count.append(temp * 24)
    else:
        for days in leap_calendar_days:
            temp += days
            count.append(temp * 24)

for i in range(numYears):

    ghost_file = dumpFolder + title + '.npy'
    if os.path.isfile(ghost_file):
        print(ghost_file + ' already exists!!')
        break

    data = Dataset(emDirs[i])
    emData = data.variables[var_name]\
        [:, latInitial:latFinal, lonInitial:lonFinal]
    for k in range(12):
	   start = count[k + 12 * i]
	   end = count[k + 1 + 12 * i] 
	   rawData[start:end, :] = np.ones(((end - start), 1)) * float(np.mean(emData[k]))

    data.close()
        
np.save(dumpFolder + title, rawData)
print('done.')


