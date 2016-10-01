# This is a config file for the neural network project!
import numpy as np
from netCDF4 import Dataset
import os.path
from time import gmtime, strftime

animus = 1
mac = 0
linux = 0
runName = strftime("RUN_%Y%m%d", gmtime())
plot = 0
saveplot = 0

# path information
if animus:
    homeDir = '/home/tsri/'
if linux:
    homeDir = '/home/tharshisri/'
if mac:
    homeDir = '/Users/tsri/'

# check if folder exists and put files inside there
saveDir = homeDir + 'phd-project/'  + runName + '/'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

# navigate to working directory
os.chdir(homeDir + 'phd-project/')

dataDir = homeDir + 'netcdf_data/'
yrs= ['2006', '2007', '2008', '2009']
mnths = ['0' + str(i) for i in range(1, 10)] + [str(i) for i in range(10, 13)]
days = ['0' + str(i) for i in range(1, 10)] + [str(i) for i in range(10, 32)]
emDirs = []
fnames = []
tmp = []

for y in yrs:

    emDirs.append(dataDir + 'emissions' + y + '.nc')

    for m in mnths:
        for d in days:
            fnames.append(dataDir + '{0}{1}{2}.nc'.
            format(y, m, d))
    
    for d in range(29, 32):
	if y != '2008':
            tmp.append(dataDir + '{0}02{1}.nc'
            .format(y, d))

    for d in range(30, 32):
	if y == '2008':
	    tmp.append(dataDir + '200802{0}.nc'.
	    format(d))

    for m in ['04', '06', '09', '11']:
        tmp.append(dataDir + '{0}{1}{2}.nc'
        .format(y, m, '31'))

for t in tmp:
    fnames.remove(t)

# control what data is let into the network

field = 1
pressure = 1
winds = 1
temp = 1
humidity = 1
pbl = 1

uwind_metadata = ['DAO_3D_S__UWND', 'Uwind', 'm/s']
vwind_metadata = ['DAO_3D_S__VWND', 'Vwind', 'm/s']
pressure_metadata = ['PEDGE_S__PEDGE', 'Pressure', 'hPa']
temp_metadata = ['DAO_3D_S__TMPU', 'Temperature', 'K']
humidity_metadata = ['TIME_SER__RH', 'Humidity', '%']
pbl_metadata = ['PBLDEPTH__PBL_M', 'PBL', 'm']
source_metadata = ['CO__SRCE__COanth', 'COSource', 'kg']
field_metadata = ['IJ_AVG_S__CO', 'COField', 'ppbv']

metadatum = []

if winds:
    metadatum.append(uwind_metadata)
    metadatum.append(vwind_metadata)

if pressure:
    metadatum.append(pressure_metadata)

if temp:
    metadatum.append(temp_metadata)
    
if humidity:
    metadatum.append(humidity_metadata)
    
if pbl:
    metadatum.append(pbl_metadata)

if field:
    metadatum.append(field_metadata)

# describe the geometry of the area to analyze

lonInitial = 20
lonFinal = 21
latInitial = 32
latFinal = 33

# total days, training days, and testing days
numYears = len(yrs)
d = 0
n = 0
for yr in yrs:
    y = int(float(yr))
	if y % 4 != 0:
		n += 365 * 24
		d += 365
	else:
		n += 366 * 24
		d += 366

rawData = np.zeros((n, 1))
