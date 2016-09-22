from config import *

for metadata in metadatum[:2]:
    
    var_name = metadata[0]
    title = metadata[1]
    units = metadata[2]
    
    print('Preparing ' + title + ' Data..')
    
    for i in range(d):
        
        trFile = fnames[i]
        teFile = fnames[i + 365]
        tr_f = Dataset(trFile)
        te_f = Dataset(teFile)
        
        for j in range(24):
            
            idx = 24*i + j
            
            tr_map = tr_f.variables[var_name][j, 0, :, :]
            te_map = te_f.variables[var_name][j, 0, :, :]
                
            tr_map2 = tr_map[latInitial:latFinal, lonInitial:lonFinal]
            
            te_map2 = te_map[latInitial:latFinal, lonInitial:lonFinal]
            
            training_data[idx, :] = tr_map2
            testing_data[idx, :] = te_map2
        
        tr_f.close()
        te_f.close()
    
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
        plt.xlabel('Days since 010106')
        y_label = units
        plt.ylabel('Mean ' + units)
        if saveplot:
            plt.savefig(title, extension='png')

    np.save('surface_' + title, training_data)
    np.save('surface_' + title, testing_data)
    print('done.')


