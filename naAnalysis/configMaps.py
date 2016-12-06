# This is a config file for the neural network project!

import numpy as np
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
yrs= ['2006', '2007', '2008', '2009', '2010', '2011']
mnths = ['0' + str(i) for i in range(1, 10)] + [str(i) for i in range(10, 13)]
days = ['0' + str(i) for i in range(1, 10)] + [str(i) for i in range(10, 32)]
dates = []
emNames = []
fnames = []
tmp = []
tmp2 = []
prefix = 'v8.G5_4x5_tagCO_ts.'
suffix = '_new.nc'

for y in yrs:

    emNames.append(emDir + 'GEOS-Chem_CO_emiss_mass_NN_' + y + '.nc')

    for m in mnths:
        for d in days:
            fnames.append(dataDir + prefix + '{0}{1}{2}'.format(y, m, d) + suffix)
            dates.append('{0}{1}{2}'.format(m, d, y))
    
    for d in range(29, 32):
        if y != '2008':
            tmp.append(dataDir + prefix + '{0}02{1}'.format(y, d) + suffix)
            tmp2.append('02{0}{1}'.format(d, y))


    for d in range(30, 32):
        if y == '2008':
            tmp.append(dataDir + prefix + '200802{0}'.format(d) + suffix)
            tmp2.append('02{0}2008'.format(d))

    for m in ['04', '06', '09', '11']:
        tmp.append(dataDir + prefix + '{0}{1}{2}'.format(y, m, '31') + suffix)
        tmp2.append('{0}31{1}'.format(m, y))

for t in tmp:
    fnames.remove(t)

for t2 in tmp2:
    dates.remove(t2)

del tmp, tmp2

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
lngIndex1 = 0
lngIndex2 = 36
latIndex1 = 22
latIndex2 = 45

lons = np.arange(-180, 180, 5)
tmp = np.arange(-86, 87, 4)
tmp2 = np.append(tmp, 89)
lats = np.insert(tmp2, 0, -89)

del tmp, tmp2

lng1 = lons[lngIndex1]
lng2 = lons[lngIndex2]
lat1 = lats[latIndex1]
lat2 = lats[latIndex2]

rawData = np.zeros((n, latIndex2 - latIndex1, lngIndex2 - lngIndex1))

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
