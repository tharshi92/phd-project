from config import *

for metadata in metadatum:
    
    var_name = metadata[0]
    title = metadata[1]
    units = metadata[2]
    
    ghost_file = 'tr' + title + '.npy'
    if os.path.isfile(ghost_file):
        print(ghost_file + ' already exists!! Skipping to next data stream.')
        continue
    
    print('Preparing ' + title + ' Data..')
    
    for i in range(d):
        
        trFile = fnames[i]
        teFile = fnames[i + 365]
        tr_f = Dataset(trFile)
        te_f = Dataset(teFile)
        
        for j in range(24):
            
            idx = 24*i + j
            
            if var_name == 'PBLDEPTH__PBL_M':
                tr_map = tr_f.variables[var_name][j, :, :]
                te_map = te_f.variables[var_name][j, :, :]
            else:
                tr_map = np.mean(tr_f.variables[var_name][j, :18, :, :], axis=0)
                te_map = np.mean(te_f.variables[var_name][j, :18, :, :], axis=0)
                
            tr_map = tr_map[latInitial:latFinal, lonInitial:lonFinal]
            tr_mean = np.mean(tr_map, dtype=np.float64)
            
            te_map = te_map[latInitial:latFinal, lonInitial:lonFinal]
            te_mean = np.mean(te_map, dtype=np.float64)
            
            training_data[idx, :] = tr_mean
            testing_data[idx, :] = te_mean
        
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
            plt.savefig(title, extension='pdf')

    np.save('tr' + title, training_data)
    np.save('te' + title, testing_data)
    print('done.')

#%%

metadata = source_metadata

var_name = metadata[0]
title = metadata[1]
units = metadata[2]

ghost_file = 'tr' + title + '.npy'
if os.path.isfile(ghost_file):
    print(ghost_file + ' already exists!!')

print('Preparing ' + title + ' Data..')

tr_f = Dataset(emDirs[0])
te_f = Dataset(emDirs[1])

tr_em = tr_f.variables[var_name][:, latInitial:latFinal, lonInitial:lonFinal]
te_em = te_f.variables[var_name][:, latInitial:latFinal, lonInitial:lonFinal]
calendar_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
count = [0]
temp = 0
for days in calendar_days:
    temp += days
    count.append(temp*24)

for i in range(n):
    for j in range(12):
        start = count[j]
        end = count[j + 1]
        if i in range(start, end):
            training_data[i, :] = np.mean(tr_em[j])
            testing_data[i, :] = np.mean(te_em[j])

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
        plt.savefig(title, extension='png', dpi=300)
    
np.save('tr' + title, training_data)
np.save('te' + title, testing_data) 
print('done.')



