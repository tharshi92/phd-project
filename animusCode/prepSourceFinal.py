from config import *
dumpFolder = homeDir + 'finalRun/'

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
        
np.save(dumpFolder + title, rawData)
print('done.')