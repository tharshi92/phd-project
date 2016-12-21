from config import *
import sys

# Name the data folder
folderName = sys.argv[1]
# Describe the geometry of the area to analyze
lngIndex1 = int(sys.argv[2])
lngIndex2 = int(sys.argv[3])
latIndex1 = int(sys.argv[4])
latIndex2 = int(sys.argv[5])

dumpFolder = homeDir + folderName + '/'
if not os.path.exists(dumpFolder):
    os.makedirs(dumpFolder)

os.chdir(dumpFolder)

metadata= source_metadata
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
        [:, latIndex1:latIndex2, lngIndex1:lngIndex2]
    for k in range(12):
	   start = count[k + 12 * i]
	   end = count[k + 1 + 12 * i] 
	   rawData[start:end, :] = np.ones(((end - start), 1)) * float(np.mean(emData[k]))

    data.close()
        
np.save(dumpFolder + title, rawData)
print('done.')
