# Imports
from analysis import disk_analysis as da
import numpy as np
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import gizmo_analysis as gizmo
import utilities as ut
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import filters
from scipy.signal import gaussian


# First open the directory where simualtion data is
simulation_directory = '/directory/of/snapshots/'

# Get the star particle data at redshift 0
part = gizmo.io.Read.read_snapshots('star', 'redshift', 0, simulation_directory, assign_hosts_rotation=True)

# Initialize universe class and create a galaxy object
data = da.Universe()
data.create_galaxy(name='m12i', choosecolor='mediumpurple', simulation_directory=simulation_directory, part=part,
                   index=None, galaxy_data=None)

# Running save_data will create dictionary-arrays for offset angle, time, mass, and axis ratio
data.galaxy_dictionary['m12i'].save_data(part, simulation_directory)

# Running delta will create one for d(theta)
data.galaxy_dictionary['m12i'].delta()

# Put values in a dictionary, you can now save them to a file if you want
m12i = dict()
m12i['offset'] = data.galaxy_dictionary['m12i'].galaxy_data['offset']
m12i['times'] = data.galaxy_dictionary['m12i'].galaxy_data['times']
m12i['delta'] = data.galaxy_dictionary['m12i'].galaxy_data['delta']
m12i['axis ratio'] = data.galaxy_dictionary['m12i'].galaxy_data['axis ratio']
m12i['mass'] = data.galaxy_dictionary['m12i'].galaxy_data['mass']



# Now let's plot, start by creating an alter galaxy object
altdata_m12i = da.AlterGalaxy(galaxy_data_m12i, 50, 600, 50, 'mediumpurple')

# You can choose if you want to put a marker at a suspected settling time
altdata_m12i.multiplot(21, 50, markerx = altdata_m12i.galaxy_data['times'][CHOOSEX], markery = altdata_m12i.galaxy_data['times'][CHOOSEY])
