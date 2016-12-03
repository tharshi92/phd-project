# This is a config file for the neural network project!

import numpy as np
from netCDF4 import Dataset
import os.path

runName = 'surfaceData'
saveplot = 0

# path information
homeDir = os.path.expanduser("~") + '/'

# check if folder exists and put files inside there
saveDir = homeDir + 'phd-project/'  + 'naAnalysis/' + runName + '/'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

dataDir = '/users/jk/13/dbj/NN_CO/run.v8-02-01.G5_tagco_new_3Dchem/timeseries2/'
emDir = '/users/jk/13/dbj/NN_CO/'
yrs= ['2007']
mnths = ['0' + str(i) for i in range(1, 10)] + [str(i) for i in range(10, 13)]
days = ['0' + str(i) for i in range(1, 10)] + [str(i) for i in range(10, 32)]
emNames = []
fnames = []
tmp = []
prefix = 'v8.G5_4x5_tagCO_ts.'
suffix = '_new.nc'

for y in yrs:

    emNames.append(emDir + 'GEOS-Chem_CO_emiss_mass_NN_' + y + '.nc')

    for m in mnths:
        for d in days:
            fnames.append(dataDir + prefix + '{0}{1}{2}'.format(y, m, d) + suffix)
    
    for d in range(29, 32):
        if y != '2008':
            tmp.append(dataDir + prefix + '{0}02{1}'.format(y, d) + suffix)

    for d in range(30, 32):
        if y == '2008':
            tmp.append(dataDir + prefix + '200802{0}'.format(d) + suffix)

    for m in ['04', '06', '09', '11']:
        tmp.append(dataDir + prefix + '{0}{1}{2}'.format(y, m, '31') + suffix)

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
lon_i = 0
lon_f = 36
lat_i = 22
lat_f = 45

lons = np.arange(-180, 180, 5)
lats1 = np.arange(-86, 87, 4)
lats2 = np.append(lats1, 89)
lats = np.insert(lats2, 0, -89)

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

rawData = np.zeros((n, lon_f - lon_i, lat_f - lat_i))
