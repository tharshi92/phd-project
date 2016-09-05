# This is a config file for the neural network project!
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os.path

# directories
dataDir = '/home/tharshi/GEOSdata/'
yrs= ['2006', '2007']
mnths = ['0' + str(i) for i in range(1, 10)] + [str(i) for i in range(10, 13)]
days = ['0' + str(i) for i in range(1, 10)] + [str(i) for i in range(10, 32)]
fnames = []
tmp = []

for y in yrs:
    for m in mnths:
        for d in days:
            fnames.append(dataDir + 'v8.G5_4x5_tagCO_ts.{0}{1}{2}_new.nc'.
            format(y, m, d))
    
    for d in range(29, 32):
        tmp.append(dataDir + 'v8.G5_4x5_tagCO_ts.{0}{1}{2}_new.nc'
        .format(y, '02', d))

    for m in ['04', '06', '09', '11']:
        tmp.append(dataDir + 'v8.G5_4x5_tagCO_ts.{0}{1}{2}_new.nc'
        .format(y, m, '31'))

for t in tmp:
    fnames.remove(t)
    
stringy = 'GEOS-Chem_CO_emiss_mass_NN_'
emDirs = [dataDir + stringy + str(2006) + '.nc', dataDir + stringy + str(2007) + '.nc']

plot = 1
saveplot = 1

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

# Below we describe the geometry of the box to analyze

lonInitial = 20
lonFinal = 21
latInitial = 32
latFinal = 33

# total days, training days, and testing days
d = 365
n = d*24

training_data = np.zeros((n, 1), dtype=np.float64)
testing_data = np.zeros((n, 1), dtype=np.float64)

temp1 = np.arange(0, n)/24
temp2 = np.arange(n, 2*n)/24

reg = 1e-1
nn = 5