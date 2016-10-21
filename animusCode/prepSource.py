from config import *
dumpFolder = homeDir + '6yrdata/'

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

    ghost_file = dumpFolder  + title + '.npy'
    if os.path.isfile(ghost_file):
        print(ghost_file + ' already exists!!')
        break

    data = Dataset(emNames[i])
    emData = data.variables[var_name]\
        [:, lat_i:lat_f, lon_i:lon_f]
    for k in range(12):
	   start = count[k + 12 * i]
	   end = count[k + 1 + 12 * i] 
	   rawData[start:end, :] = np.ones(((end - start), 1)) * float(np.mean(emData[k]))

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
        plt.xlabel('Days since 010106')
        y_label = units
        plt.ylabel('Mean ' + units)
        if saveplot:
            plt.savefig(title, extension='png', dpi=300)
        
np.save(dumpFolder + title, rawData)
print('done.')