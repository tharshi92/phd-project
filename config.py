# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 21:35:51 2016

@author: tharshi

tsrikann@physics.utoronto.ca, University of Toronto: Jones Group

This program will take simulated GEOS CO FIELD DATA and GEOS CO SURFACE FLUX data
over new york and compile it into a target matrix and an input matrix

Author: Tharshi Sri, tsrikann@physics.utoronto.ca, University of Toronto: Jones Group

Version 1: 062616
           - CO Flux data was chosen to be a 3x3 box over the NewYork Area
           - Field Data is also 3x3 taken at a "level" of 20 
Version 2: 062916
            - CO Flux data was averaged over the day to capture whatever
            little variation there was
            - Extraction of UWind Data Added to code
            - Added a global config file to hold all parameters
Version 3: 070116 (Happy Canada Day!)
            - Brought in Vwinds
            - Added Network config file
            - Brought in Pressures
            - Brought in U/V angles
            - Wrote main.py to lump all files together
Version 4: 071416
            - Brought in the entire year of data
            - I plan on removing the angular information from the inputs
            - Merged all data prep files into prepData.py (Except CO Sources)
Version 4: 071516
            - Slimmed prepMap.py down
Version 5: 072016
            - Added support for datasets from 2006 and 2007

"""

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os.path

# This is a config file for the neural network project!

print('-------------------------------------------------')
print('Preparing Physical Configuration..')
print('-------------------------------------------------\n')

plot = 1

# Define folders that hold the data for the training/testing (on USB)
years = ['2006', '2007']
dir0 = 'C:/Users/tharshi/iCloudDrive/research2016/nnco/GEOS_data/'
coDirs = [dir0 + year + '/' for year in years]
emDirs = [coDir + 'GEOS-Chem_CO_emiss_mass_NN_' + year + '.nc' \
    for coDir, year in zip(coDirs, years)]

fnames = [year + 'GEOS4x5input' for year in years]

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

lonInitial = 21
lonFinal = 22
latInitial = 32
latFinal = 34

dLon = lonFinal - lonInitial
dLat = latFinal - latInitial

# total days, training days, and testing days
n_y = len(years)
d = 365
n = d*24

# temporary arrays for plotting data

training_data = np.zeros((n, 1), dtype=np.float64)
testing_data = np.zeros((n, 1), dtype=np.float64)

temp1 = np.arange(0, n)/24
temp2 = np.arange(n, 2*n)/24
