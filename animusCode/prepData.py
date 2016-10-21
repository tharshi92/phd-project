from config import *

dumpFolder = homeDir + '6yrdata/'

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
        print(fnames[i])
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
    
    if plot:    
        print('Plotting data..')
        fig = plt.figure(figsize=(14, 8))
        width = 0.4
        plt.plot(temp1, training_data, color=[50/255, 100/255, 75/255], \
            alpha=0.7, label='Training', linewidth=width)
        plt.plot(temp2, testing_data, 'g', \
            alpha=0.7, label='Testing', linewidth=width)
        plt.legend(fontsize=10)
        plt.grid('on')
        plt.title(title)
        plt.xlabel('Hours since 010106 0000')
        y_label = units
        plt.ylabel('Mean ' + units)
        if saveplot:
            plt.savefig(title, extension='pdf')

    np.save(dumpFolder + title, rawData)
    print('done.')




