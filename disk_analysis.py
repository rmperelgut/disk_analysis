# importing all the necessary modules
import numpy as np
import gizmo_analysis as gizmo
import utilities as ut
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import filters
from scipy.signal import gaussian


class Universe:
    def __init__(self):
        # Store all of the galaxies in a dictionary where the name is the key and the value is galaxy class
        # This is to initialize
        self.galaxy_dictionary = dict()
        self.grand_dict = dict()

    def create_galaxy(self, name, choosecolor, simulation_directory, part, index, galaxy_data):
        # To fill up dictionary with galaxies by calling the galaxy class
        self.galaxy_dictionary[name] = Galaxy(name, choosecolor, simulation_directory, part, index, galaxy_data)

    def add_to_grand_dict(self, name, filename, data_dict, data_key, data_value):
        # smaller dictionary that has the data you want to save
        data_dict[data_key] = data_value

        # dictionary that contains all the galaxies, so we have a dictionary of dictionaries
        self.grand_dict[name] = data_dict[data_key]
        ut.io.file_hdf5(file_name_base='/Users/rpere/Desktop/Wetzel/{}'.format(filename),
                        dict_or_array_to_write=self.grand_dict)

    def get_from_grand_dict(self, name, filename):
        # getting desired dictionary from file
        des_dict = ut.io.file_hdf5('{}'.format(filename))['{}'.format(name)]
        return des_dict

class Galaxy:
    def __init__(self, name=None, choosecolor=None, simulation_directory=None, part=None, index=None, galaxy_data=None):
        self.name = name
        self.galaxy_data = dict()
        # raw galaxy information separated by 'times','offset','ratio', 'delta', "LProduct"
        self.choosecolor = choosecolor
        self.index = index

    def save_data(self, part, simulation_directory):
        if self.index is not None:
            # for the case that there is more than one galaxy
            rotation_array = part.hostz['rotation'][:, self.index, 2]
            z0 = np.asarray(rotation_array[-1])
            info_from_snapshots = ut.simulation.read_snapshot_times(directory=simulation_directory)
            time_array = info_from_snapshots['time']
            
            
            # axis ratios
            ratio_array = part.hostz['axis.ratios'][:, self.index, 0]
            
      
            angle_array = []
            # we will store the angles in here

            for i in range(len(rotation_array)):
                curr_rotation = (rotation_array[i])
                raw_angle = np.arccos(np.dot(z0, curr_rotation))
                final_angle = np.degrees(raw_angle)
                angle_array.append(final_angle)
           

            # Let's get the mass
            prop = ut.particle.get_galaxy_properties(part, host_index=self.index, axis_kind='both')
            mass_z0 = prop['mass']


        else:
            # getting rotation data
            rotation_array = part.hostz['rotation']
            z0 = np.asarray(rotation_array[-1][:, 2][0])
            # we use hostz because we want all the particle data, if we dont have the z, it will only give us redshift zero
            # we want to find the angle between the third row of z = 0, and all the other third rows

            info_from_snapshots = ut.simulation.read_snapshot_times(directory=simulation_directory)
            time_array = info_from_snapshots['time']
            # Note that snapshot times correlate with time passed after Big Bang

            ratio_array = part.hostz['axis.ratios'][:, 0, 0]

            angle_array = []
            # We will store the angles in here
            for i in range(len(rotation_array)):
                curr_rotation = (rotation_array[i][:, 2][0])
                raw_angle = np.arccos(np.dot(z0, curr_rotation))
                final_angle = np.degrees(raw_angle)
                angle_array.append(final_angle)

            # getting mass data
            properties = ut.particle.get_galaxy_properties(part, axis_kind='both')
            
            # For the mass at redshift zero
            mass_z0 = properties['mass']

        # for the dictionary, we want the name as the key, to get that, we will take it as an argument
        self.galaxy_data['offset'] = np.asarray(angle_array)
        self.galaxy_data['times'] = time_array
        self.galaxy_data['axis ratio'] = ratio_array
        self.galaxy_data['mass'] = mass_z0
        
    def tot_mass(self, part_0, part_st):
        # Let's get the total baryonic mass
        if self.index == 1: # For the second host
            # Redshift zero
            properties_0 = ut.particle.get_galaxy_properties(part_0, host_index=self.index, axis_kind='both')
            minor_radius_0 = properties_0['radius.minor']
            major_radius_0 = properties_0['radius.major']
            
            # For the stars: index 0 to get radial component, index 2 to get z component 
            major_mask_0 = part_0['star'].prop('host2.distance.principle.cylindrical')[:, 0] < major_radius_0
            
            minor_mask_0 = (part_0['star'].prop('host2.distance.principle.cylindrical')[:, 2] > -minor_radius_0) * \
                           (part_0['star'].prop('host2.distance.principle.cylindrical')[:, 2] < minor_radius_0)

            final_mask = major_mask_0 * minor_mask_0
            
            star_mass = part_0['star']['mass'][final_mask]
            
            # For the gas
            major_mask_0 = part_0['gas'].prop('host2.distance.principle.cylindrical')[:, 0] < major_radius_0

            minor_mask_0 = (part_0['gas'].prop('host2.distance.principle.cylindrical')[:, 2] > -minor_radius_0) * \
                           (part_0['gas'].prop('host2.distance.principle.cylindrical')[:, 2] < minor_radius_0)

            final_mask = major_mask_0 * minor_mask_0

            gas_mass = part_0['gas']['mass'][final_mask]

            # Add them together
            tot_mass_0 = np.sum(star_mass) + np.sum(gas_mass)
            
            
            # Now for settling time
            properties_st = ut.particle.get_galaxy_properties(part_st, host_index=self.index, axis_kind='both')
            minor_radius_st = properties_st['radius.minor']
            major_radius_st = properties_st['radius.major']

            # For the stars: index 0 to get radial component, index 2 to get z component 
            major_mask_st = part_st['star'].prop('host2.distance.principle.cylindrical')[:, 0] < major_radius_st

            minor_mask_st = (part_st['star'].prop('host2.distance.principle.cylindrical')[:, 2] > -minor_radius_st) * \
                            (part_st['star'].prop('host2.distance.principle.cylindrical')[:, 2] < minor_radius_st)

            final_mask = major_mask_st * minor_mask_st

            star_mass = part_st['star']['mass'][final_mask]

            # For the gas
            major_mask_st = part_st['gas'].prop('host2.distance.principle.cylindrical')[:, 0] < major_radius_st

            minor_mask_st = (part_st['gas'].prop('host2.distance.principle.cylindrical')[:, 2] > -minor_radius_st) * \
                            (part_st['gas'].prop('host2.distance.principle.cylindrical')[:, 2] < minor_radius_st)

            final_mask = major_mask_st * minor_mask_st

            gas_mass = part_st['gas']['mass'][final_mask]


            # Add them together
            tot_mass_st = np.sum(star_mass) + np.sum(gas_mass)
            
        else:
            # Let's start with redshift 0
            properties_0 = ut.particle.get_galaxy_properties(part_0, axis_kind='both')
            minor_radius_0 = properties_0['radius.minor']
            major_radius_0 = properties_0['radius.major']

            # For the stars: index 0 to get radial component, index 2 to get z component 
            major_mask_0 = part_0['star'].prop('host.distance.principle.cylindrical')[:, 0] < major_radius_0

            minor_mask_0 = (part_0['star'].prop('host.distance.principle.cylindrical')[:, 2] > -minor_radius_0) * \
                           (part_0['star'].prop('host.distance.principle.cylindrical')[:, 2] < minor_radius_0)

            final_mask = major_mask_0 * minor_mask_0

            star_mass = part_0['star']['mass'][final_mask]

            # For the gas
            major_mask_0 = part_0['gas'].prop('host.distance.principle.cylindrical')[:, 0] < major_radius_0

            minor_mask_0 = (part_0['gas'].prop('host.distance.principle.cylindrical')[:, 2] > -minor_radius_0) * \
                           (part_0['gas'].prop('host.distance.principle.cylindrical')[:, 2] < minor_radius_0)

            final_mask = major_mask_0 * minor_mask_0

            gas_mass = part_0['gas']['mass'][final_mask]


            # Add them together
            tot_mass_0 = np.sum(star_mass) + np.sum(gas_mass)


            # Now for settling time
            properties_st = ut.particle.get_galaxy_properties(part_st, axis_kind='both')
            minor_radius_st = properties_st['radius.minor']
            major_radius_st = properties_st['radius.major']

            # For the stars: index 0 to get radial component, index 2 to get z component 
            major_mask_st = part_st['star'].prop('host.distance.principle.cylindrical')[:, 0] < major_radius_st

            minor_mask_st = (part_st['star'].prop('host.distance.principle.cylindrical')[:, 2] > -minor_radius_st) * \
                            (part_st['star'].prop('host.distance.principle.cylindrical')[:, 2] < minor_radius_st)

            final_mask = major_mask_st * minor_mask_st

            star_mass = part_st['star']['mass'][final_mask]

            # For the gas
            major_mask_st = part_st['gas'].prop('host.distance.principle.cylindrical')[:, 0] < major_radius_st

            minor_mask_st = (part_st['gas'].prop('host.distance.principle.cylindrical')[:, 2] > -minor_radius_st) * \
                            (part_st['gas'].prop('host.distance.principle.cylindrical')[:, 2] < minor_radius_st)

            final_mask = major_mask_st * minor_mask_st


            gas_mass = part_st['gas']['mass'][final_mask]

            # Add them together
            tot_mass_st = np.sum(star_mass) + np.sum(gas_mass)
            
            
        # Save to dictionary    
        self.galaxy_data['totmass_0'] = tot_mass_0
        self.galaxy_data['totmass_st'] = tot_mass_st

    def momdotL(self, part):
        if self.index is not None:
            # for the case that there is more than one galaxy
            rotation_array = part.hostz['rotation'][:, self.index, 2, :]
   
        else:
            # getting rotation data
            # array order is [time, row, column], we want all the times and rows, but just the z column 
            rotation_array = part.hostz['rotation'][:, 0, 2, :]
            
            
        raw_L = self.galaxy_data['angmom']


        # We will store the angles in here
        angle_array = np.zeros(len(rotation_array))
        
        for i in range(len(rotation_array)):
            # normalizing the raw L data to make it dimensionless so we can use the dot product
            norm = np.sqrt(raw_L[i][0] ** 2 + raw_L[i][1] ** 2 + raw_L[i][2] ** 2)
            raw_L[i] = raw_L[i] / norm

            raw_angle = np.arccos(np.dot(raw_L[i], rotation_array[i]))
            final_angle = np.degrees(raw_angle)
            angle_array[i] = final_angle

        self.galaxy_data['momdotL'] = angle_array
        
    def delta(self):
        time_array = self.galaxy_data['times']
        angle_array = self.galaxy_data['offset']

        dtheta = []
        for i in range(len(time_array) - 1):
            dtheta.append((angle_array[i + 1] - angle_array[i]) / (time_array[i + 1] - time_array[i]))

        self.galaxy_data['delta'] = np.asarray(dtheta)

    def cumulative_L(self, D, T, part):
        # let's first get the momentum of inertia matrix
        new_moi = ut.particle.get_principal_axes(part, distance_max=D, mass_percent=90, age_limits=[0, T])

        # next part is like what we did in save_data to get the MOI at redshift 0
        rotation_array = new_moi['rotation']
        z0 = np.asarray(rotation_array[-1])

        # (1) mask that encloses the stars within 10 kpc of the galactic
        d_mask = part['star'].prop('host.distance.total') < D

        # apply it to distance and mass so we can find the values within R_90
        # we need it to use the ut.math.percentile_weighted() function
        particle_distances = part['star'].prop('host.distance.total')[d_mask]
        particle_masses = part['star']['mass'][d_mask]

        # (2) R_90 mask that we will apply to get distance that encloses 90% of the mass
        r_90 = ut.math.percentile_weighted(particle_distances, 90, particle_masses)

        # applying both the initial distance mask and the R_90 mask
        mask_d_r90 = part['star'].prop('host.distance.total')[d_mask] < r_90
        # mask that represents stars within 10 kpc and stars enclosed by radius that has 90% of the mass

        # getting particle ages
        ages = part['star'].prop('age')[d_mask][mask_d_r90]
        # once again getting the masses to use the ut.math.percentile_weighted() function
        particle_masses = part['star']['mass'][d_mask][mask_d_r90]

        # (3) getting first part of the last mask, we just want the T youngest stars
        # creating mask to get the final ages
        age_new = part['star'].prop('age')[d_mask][mask_d_r90] < T

        # applying the final mask
        final_ages = part['star'].prop('age')[d_mask][mask_d_r90][age_new]

        # obtaining the position vectors, note that each row is for a single star (x, y ,z)
        pos = part['star'].prop('host.distance')[d_mask][mask_d_r90][age_new]

        # obtaining the velocity vectors
        vel = part['star'].prop('host.velocity')[d_mask][mask_d_r90][age_new]

        # recall formula for momentum = v * m, we need mass
        mass = part['star']['mass'][d_mask][mask_d_r90][age_new]

        # calculate momnetum for each point; allocatinf memory for the array beforehand
        momentum = np.zeros((len(mass), 3))

        for i in range(len(momentum) - 1):
            momentum[i] = mass[i] * vel[i]

        # to get angular momentum, we need to cross the normal momentum with position
        L = np.cross(pos, momentum)

        # we need to get the total mass in order to ....
        total_mass = np.sum(mass)

        # divide each part of the vector by the total mass
        total_x = np.sum(L[:, 0]) / total_mass
        total_y = np.sum(L[:, 1]) / total_mass
        total_z = np.sum(L[:, 2]) / total_mass

        # normalizing
        norm = np.sqrt(total_x ** 2 + total_y ** 2 + total_z ** 2)
        final_L = np.array((total_x, total_y, total_z)) / norm

        # return the result of the dot product between the final L and the MOI
        dotted = np.dot(final_L, z0)
        final = np.arccos(dotted) * 180 / np.pi

        return final

    def differential_L(self, lower_T, upper_T, part, D=None):
        # Let's start with calculating the necessary information to find angular momentum
        # We have to do MOI at the end so we can apply the masks

        if D is None:
            D = 10

        # (1) mask that encloses the stars within 10 kpc of the galactic
        d_mask = part['star'].prop('host.distance.total') < D

        # apply it to distance and mass so we can find the values within R_90
        # we need it to use the ut.math.percentile_weighted() function
        particle_distances = part['star'].prop('host.distance.total')[d_mask]
        particle_masses = part['star']['mass'][d_mask]

        # (2) R_90 mask that we will apply to get distance that encloses 90% of the mass
        r_90 = ut.math.percentile_weighted(particle_distances, 90, particle_masses)

        # applying both the initial distance mask and the R_90 mask
        mask_d_r90 = part['star'].prop('host.distance.total')[d_mask] < r_90
        # mask that represents stars within 10 kpc and stars enclosed by radius that has 90% of the mass

        # getting particle ages
        ages = part['star'].prop('age')[d_mask][mask_d_r90]
        # once again getting the masses to use the ut.math.percentile_weighted() function
        particle_masses = part['star']['mass'][d_mask][mask_d_r90]

        # (3) getting first part of the last mask, we just want the youngest stars in a certain range
        youngest_lower = lower_T
        youngest_upper = upper_T

        # creating mask to get the final ages
        age_new = ((youngest_lower <= part['star'].prop('age')[d_mask][mask_d_r90]) *
                   (part['star'].prop('age')[d_mask][mask_d_r90] <= youngest_upper))

        # applying the final mask
        final_ages = part['star'].prop('age')[d_mask][mask_d_r90][age_new]

        # obtaining the position vectors, note that each row is for a single star (x, y ,z)
        pos = part['star'].prop('host.distance')[d_mask][mask_d_r90][age_new]

        # obtaining the velocity vectors
        vel = part['star'].prop('host.velocity')[d_mask][mask_d_r90][age_new]

        # recall formula for momentum = v * m, we need mass
        mass = part['star']['mass'][d_mask][mask_d_r90][age_new]

        # calculate momentum for each point; allocatinf memory for the array beforehand
        momentum = np.zeros((len(mass), 3))

        for i in range(len(momentum) - 1):
            momentum[i] = mass[i] * vel[i]

        # to get angular momentum, we need to cross the normal momentum with position
        L = np.cross(pos, momentum)

        # we need to get the total mass in order to ....
        total_mass = np.sum(mass)

        # divide each part of the vector by the total mass
        total_x = np.sum(L[:, 0]) / total_mass
        total_y = np.sum(L[:, 1]) / total_mass
        total_z = np.sum(L[:, 2]) / total_mass

        # normalizing
        norm = np.sqrt(total_x ** 2 + total_y ** 2 + total_z ** 2)
        final_L = np.array((total_x, total_y, total_z)) / norm

        # Now let's find the MOI, we have all the masks we need
        # this gives us z0

        # sus ignore
        moi_slide = ut.particle.get_principal_axes(part, distance_max=D, mass_percent=90,
                                                   age_limits=[youngest_lower, youngest_upper])['rotation'][-1]

        # return the result of the dot product between the final L and the MOI
        dotted = np.dot(final_L, moi_slide)
        final = np.arccos(dotted) * 180 / np.pi

        return final
    
    def addtodic(self, name, array):
        self.galaxy_data['{}'.format(name)] = array
    
    def vsigma(self, part, ifyoungest, index):
        # if at z=0, put in snapshot 600, if at settling time, but in respective index
        
        if ifyoungest: 
            if index == 1:
                # for host2 in local groups
                ages = part['star'].prop('age')
                age_mask = (ages < 0.25)
                d_mask = part['star'].prop('host2.distance.principal.total')[age_mask] < 10
                particle_distances = part['star'].prop('host2.distance.principal.total')[age_mask][d_mask]
                particle_masses = part['star']['mass'][age_mask][d_mask]
                r_90 = ut.math.percentile_weighted(particle_distances, 90, particle_masses)
                mask_d_r90 = part['star'].prop('host2.distance.principal.total')[age_mask][d_mask] < r_90
                ages = part['star'].prop('age')[age_mask][d_mask][mask_d_r90]
                particle_masses = part['star']['mass'][age_mask][d_mask][mask_d_r90]
                youngest = ut.math.percentile_weighted(ages, 25, particle_masses)
                age_new = part['star'].prop('age')[age_mask][d_mask][mask_d_r90] < youngest

                pos = part['star'].prop('host2.distance.principal.cylindrical')[age_mask][d_mask][mask_d_r90][age_new]
                vel = part['star'].prop('host2.velocity.principal.cylindrical')[age_mask][d_mask][mask_d_r90][age_new]

            else:
                # mask for getting ages for the youngest stars (we want stars made from past 250 megayears)
                ages = part['star'].prop('age')
                age_mask = (ages < 0.25)
                d_mask = part['star'].prop('host.distance.principal.total')[age_mask] < 10
                particle_distances = part['star'].prop('host.distance.principal.total')[age_mask][d_mask]
                particle_masses = part['star']['mass'][age_mask][d_mask]
                r_90 = ut.math.percentile_weighted(particle_distances, 90, particle_masses)
                mask_d_r90 = part['star'].prop('host.distance.principal.total')[age_mask][d_mask] < r_90
                ages = part['star'].prop('age')[age_mask][d_mask][mask_d_r90]
                particle_masses = part['star']['mass'][age_mask][d_mask][mask_d_r90]
                youngest = ut.math.percentile_weighted(ages, 25, particle_masses)
                age_new = part['star'].prop('age')[age_mask][d_mask][mask_d_r90] < youngest

                pos = part['star'].prop('host.distance.principal.cylindrical')[age_mask][d_mask][mask_d_r90][age_new]
                vel = part['star'].prop('host.velocity.principal.cylindrical')[age_mask][d_mask][mask_d_r90][age_new]

        else:
        # masks are now similar to when we calculated L
            if (index == 1):
                # First find the particles within 10 kpc
                d_mask = part['star'].prop('host2.distance.principal.total') < 10
                particle_distances = part['star'].prop('host2.distance.principal.total')[d_mask]

                # Get the masses of the particles within 10 kpc
                particle_masses = part['star']['mass'][d_mask]

                # Calculate what R90 is for the stars within 10 kpc, and weight them by their mass
                r_90 = ut.math.percentile_weighted(particle_distances, 90, particle_masses)

                # Create a mask for the stars within 10 kpc, and within R90
                mask_r90 = part['star'].prop('host2.distance.principal.total')[d_mask] < r_90

                # Select the ages and masses of stars within 10 kpc and within R90
                ages = part['star'].prop('age')[d_mask][mask_r90]
                particle_masses = part['star']['mass'][d_mask][mask_r90]

                # Calculate what the age is that will include the 25% youngest stars 
                # (within 10 kpc and within R90) and create a mask for it.
                youngest = ut.math.percentile_weighted(ages, 25, particle_masses)
                age_new = part['star'].prop('age')[d_mask][mask_r90] < youngest

                # Get the 3D positions and velocities of stars within 10 kpc, within R90, and the 25% youngest.
                pos = part['star'].prop('host2.distance.principal.cylindrical')[d_mask][mask_r90][age_new]
                vel = part['star'].prop('host2.velocity.principal.cylindrical')[d_mask][mask_r90][age_new]

            else: 
                d_mask = part['star'].prop('host.distance.principal.total') < 10
                particle_distances = part['star'].prop('host.distance.principal.total')[d_mask]
                particle_masses = part['star']['mass'][d_mask]
                r_90 = ut.math.percentile_weighted(particle_distances, 90, particle_masses)
                mask_d_r90 = part['star'].prop('host.distance.principal.total')[d_mask] < r_90
                ages = part['star'].prop('age')[d_mask][mask_d_r90]
                particle_masses = part['star']['mass'][d_mask][mask_d_r90]
                youngest = ut.math.percentile_weighted(ages, 25, particle_masses)
                age_new = part['star'].prop('age')[d_mask][mask_d_r90] < youngest

                pos = part['star'].prop('host.distance.principal.cylindrical')[d_mask][mask_d_r90][age_new]
                vel = part['star'].prop('host.velocity.principal.cylindrical')[d_mask][mask_d_r90][age_new]
                # we do not need the ages here as that is taken care of with the masks above

        '''
        Calculations are the same for all three cases from this point and below
        
        '''
        v_phi = np.median(vel[:,1])
        
        # first R
        upper_R = np.percentile(vel[:,0], 84)
        lower_R = np.percentile(vel[:,0], 16)
        scatter_R = (upper_R - lower_R)/2
        
        # next phi
        upper_phi = np.percentile(vel[:,1], 84)
        lower_phi = np.percentile(vel[:,1], 16)
        scatter_phi = (upper_phi - lower_phi)/2
        
        # finally Z
        upper_Z = np.percentile(vel[:,2], 84)
        lower_Z = np.percentile(vel[:,2], 16)
        scatter_Z = (upper_Z - lower_Z)/2
        
        sigma = np.sqrt(scatter_R**2 + scatter_phi**2 + scatter_Z**2)

        # Now we have v_phi and sigma, so let's calculate and return
        
        return v_phi/sigma
    
        
    def thetaL(self):
        raw_L = self.galaxy_data['angmom']
        thetaL = np.zeros(len(raw_L))
        
        # normalize L0
        norm = np.sqrt(raw_L[-1][0]** 2 + raw_L[-1][1] ** 2 + raw_L[-1][2] ** 2)
        L0 = self.galaxy_data['angmom'][-1] / norm
        
        for i in range(len(raw_L)):
            if (raw_L[i] == np.array([-1, -1, -1])).all():
                # for the case of the snapshot not having proper data
                thetaL[i] = -500
                
            else:
                # every other case
                # we need to normalize by dividing raw_L by the norm of raw_L 
                # we are turning L into a vector with components between 0 and 1 (unit vectors)
                
                norm = np.sqrt(raw_L[i][0] ** 2 + raw_L[i][1] ** 2 + raw_L[i][2] ** 2)
                raw_L[i] = raw_L[i] / norm
                
                dotted = np.dot(raw_L[i], L0)
                thetaL[i] = np.arccos(dotted) * 180 / np.pi

        # when plotting we will have to get rid of the -500 cases by masking the time and L array
        self.galaxy_data['thetaL'] = thetaL
        
    def deltaL(self):
        time_array = self.galaxy_data['times']
        thetaL = self.galaxy_data['thetaL']
        
        deltaL = np.zeros(len(time_array) - 1)
        for i in range(len(deltaL)):
            if (thetaL[i] == -500) or (thetaL[i+1] == -500):
                # for the case of the snapshot not having proper data
                deltaL[i] = np.nan
                
            else:
                deltaL[i] = (thetaL[i + 1] - thetaL[i]) / (time_array[i + 1] - time_array[i])
                
        self.galaxy_data['deltaL'] = deltaL
          
    def L(self, part, index):

        if (index == 1):
            # for the second galaxy in the local group simulations
            # (1) mask that encloses the stars within 10 kpc of the galactic disk
            d_mask = part['star'].prop('host2.distance.total') < 10

            # apply it to distance and mass so we can find the values within R_90
            # we need it to use the ut.math.percentile_weighted() function
            particle_distances = part['star'].prop('host2.distance.total')[d_mask]
            particle_masses = part['star']['mass'][d_mask]

            # (2) R_90 mask that we will apply to get distance that encloses 90% of the mass
            r_90 = ut.math.percentile_weighted(particle_distances, 90, particle_masses)

            # applying both the initial distance mask and the R_90 mask
            # mask that represents stars within 10 kpc and stars enclosed by radius that has 90% of the mass
            mask_d_r90 = part['star'].prop('host2.distance.total')[d_mask] < r_90

            # once again getting the masses to use the ut.math.percentile_weighted() function
            particle_masses = part['star']['mass'][d_mask][mask_d_r90]

            # now let's get the ages for the particles
            ages = part['star'].prop('age')[d_mask][mask_d_r90]

            youngest = ut.math.percentile_weighted(ages, 25, particle_masses)
            age_new = part['star'].prop('age')[d_mask][mask_d_r90] < youngest

            # obtaining the position vectors, note that each row is for a single star (x, y ,z)
            pos = part['star'].prop('host2.distance')[d_mask][mask_d_r90][age_new]

            # obtaining the velocity vectors
            vel = part['star'].prop('host2.velocity')[d_mask][mask_d_r90][age_new]

        else:
            # (1) mask that encloses the stars within 10 kpc of the galactic
            d_mask = part['star'].prop('host.distance.total') < 10

            # apply it to distance and mass so we can find the values within R_90
            # we need it to use the ut.math.percentile_weighted() function
            particle_distances = part['star'].prop('host.distance.total')[d_mask]
            particle_masses = part['star']['mass'][d_mask]

            # (2) R_90 mask that we will apply to get distance that encloses 90% of the mass
            r_90 = ut.math.percentile_weighted(particle_distances, 90, particle_masses)

            # applying both the initial distance mask and the R_90 mask
            # mask that represents stars within 10 kpc and stars enclosed by radius that has 90% of the mass
            mask_d_r90 = part['star'].prop('host.distance.total')[d_mask] < r_90

            # once again getting the masses to use the ut.math.percentile_weighted() function
            particle_masses = part['star']['mass'][d_mask][mask_d_r90]

            # now let's get the ages for the particles
            ages = part['star'].prop('age')[d_mask][mask_d_r90]

            youngest = ut.math.percentile_weighted(ages, 25, particle_masses)
            age_new = part['star'].prop('age')[d_mask][mask_d_r90] < youngest

            # obtaining the position vectors, note that each row is for a single star (x, y ,z)
            pos = part['star'].prop('host.distance')[d_mask][mask_d_r90][age_new]

            # obtaining the velocity vectors
            vel = part['star'].prop('host.velocity')[d_mask][mask_d_r90][age_new]

        '''
        Code is the same from this point on
        '''

        # recall formula for momentum = v * m, we need mass
        mass = part['star']['mass'][d_mask][mask_d_r90][age_new]

        # calculate momentum for each point; allocatinf memory for the array beforehand
        momentum = np.zeros((len(mass), 3))

        for i in range(len(momentum) - 1):
            momentum[i] = mass[i] * vel[i]

        # to get angular momentum, we need to cross the normal momentum with position
        L = np.cross(pos, momentum)

        # we need to get the total mass in order to ....
        total_mass = np.sum(mass)

        # divide each part of the vector by the total mass
        total_x = np.sum(L[:, 0]) / total_mass
        total_y = np.sum(L[:, 1]) / total_mass
        total_z = np.sum(L[:, 2]) / total_mass

        return np.array([total_x, total_y, total_z])

        
class AlterGalaxy:
    def __init__(self, galaxy_data, cutoff_back, cutoff_front, cutoff_graph, thiscolor):
        self.galaxy_data = galaxy_data
        self.cutoff_front = cutoff_front
        self.cutoff_back = cutoff_back
        self.thiscolor = thiscolor
        self.cutoff_graph = cutoff_graph

    def find_thresh(self, threshold, windows, chooseifval, name=None):
        # threshold and windows will be an array of different potential values
        # this function will return a dictionary where the key is the threshold and the value is the where it is broken
        degree = 3
        window_dict = dict()
        time_array = self.galaxy_data['times']
        angle_array = self.galaxy_data['delta']

        angle_array = angle_array[self.cutoff_back:self.cutoff_front]

        for window_index, window_value in enumerate(windows):
            thresh_dict = dict()  # thresh_dict will reset when the window changes value
            altered_angle = savgol_filter(angle_array, window_value, degree)
            # we are reading the graph from right to left

            for thresh_index, thresh_value in enumerate(threshold):
                for angle_index, angle_value in reversed(list(enumerate(altered_angle))):
                    # allows us to iterate backwards
                    if abs(angle_value) >= thresh_value:
                        if chooseifval:
                            thresh_dict['threshold: {}'.format(thresh_value)] = time_array[-1] - time_array[
                                angle_index + self.cutoff_back]
                            break
                            # this one reads differently than the index, it gives it in look back time rather than raw time
                        else:
                            thresh_dict['threshold: {}'.format(thresh_value)] = angle_index + self.cutoff_back
                            break
            # the index being returned here is under the context that the array has been flipped and subtracted from

            window_dict['window: {}'.format(window_value)] = thresh_dict

        table = pd.DataFrame(window_dict)
        print(table)

    def gauss_method(self, m_factor, stdev, thresh_value, chooseifval, theta_array):
        time_array = self.galaxy_data['times']
        dtheta = self.galaxy_data[theta_array][self.cutoff_back:self.cutoff_front]

        for i in range(0, len(stdev)):
            y_filter = gaussian(M=m_factor * stdev[i], std=stdev[i])
            # M is the window size and stdev is the standard deviation, the window size should be at least double the stdev

            y_smooth = filters.convolve1d(dtheta, y_filter / y_filter.sum())
            # "convolve1d" is the smoothing function, just like the savgol filter
            # We feed it the noisy data, and the gaussian filter
            # We want to normalize the filter such that it integrates to 1

            for angle_index, angle_value in reversed(list(enumerate(y_smooth))):
                # allows us to iterate backwards
                if abs(angle_value) >= thresh_value:
                    if chooseifval:
                        print("Gaussian settling time (lookback):", time_array[-1] - time_array[angle_index])
                        break
                        # this one reads differently than the index, it gives it in look back time rather than raw time
                    else:
                        print("Gaussian time/angle index:", angle_index + self.cutoff_back)
                        break

        return y_smooth

    def multiplot(self, window, threshold, markerx=None, markery=None):
        # Main four initial variables
        time_array = self.galaxy_data['times']
        angle_array = self.galaxy_data['offset']
        delta_array = self.galaxy_data['delta']
        axis_ratio = self.galaxy_data['axis ratio']
        degree = 3

        alt_angle = savgol_filter((delta_array[self.cutoff_back:]), window, degree)

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
        fig.tight_layout(pad=0.5)
        plt.figure(figsize=[10, 10])

        # First plot is the standard offset angle v time
        axs[0].plot(time_array[-1] - time_array, angle_array, color=self.thiscolor)
        axs[0].set_ylabel("Offset Angle (Degrees)", fontsize=20)
        axs[0].invert_xaxis()
        axs[0].tick_params(axis='both', which='both', labelsize=17, bottom=True, top=True, left=True, right=True, direction='in')
        axs[0].minorticks_on()
        
        # Second plot will be the Gaussian Filter
        thresh_gauss = 50
        stdev = [11]
        m_factor = 2
        y_smooth = self.gauss_method(m_factor, stdev, thresh_gauss, False, 'delta')
        
        
        # Applying the Filter and Plotting
        axs[1].plot(time_array[-1] - time_array[1:], 1000 * delta_array, color='lightgray', label='Unsmoothed 'r'$\dfrac{dθ}{dt}$')
        axs[1].plot(time_array[-1] - time_array[self.cutoff_graph:self.cutoff_front], y_smooth, color=self.thiscolor,
                    label="Gaussian Smooth")
        
        # Adding horizontal threshold lines and excluded values
        axs[1].axhline(y=0, color='black', linestyle='--')
        axs[1].axhline(y=thresh_gauss, color='black', linestyle='--')
        axs[1].axhline(y=-thresh_gauss, color='black', linestyle='--')
        axs[1].axvspan(0, time_array[-self.cutoff_front], color='mistyrose', label='Excluded values', alpha=0.5)
        axs[1].axvspan(time_array[-self.cutoff_back], time_array[-1], color='mistyrose', alpha=0.5)

        axs[1].set_ylabel(r'$\dfrac{d\theta}{dt}$ (Degree / Gyr)', fontsize=20)
        axs[1].set_ylim(-200, 200)
        axs[1].tick_params(axis='both', which='both', labelsize=17, bottom=True, top=True, left=True, right=True, direction='in')
        axs[1].minorticks_on()
        axs[1].invert_xaxis()

        
        
        # Third plot is simple, it is just the axis ratio
        axs[2].plot(13.7 - time_array, axis_ratio, color=self.thiscolor)
        axs[2].set_ylabel("Major-to-Minor \n Axis Ratio", fontsize=20)
        axs[2].set_xlabel("Lookback Time [Gyr]", fontsize=25)
        axs[2].set_xlim(0,time_array[-1])
        axs[2].tick_params(axis='both', which='both', labelsize=17, bottom=True, top=True, left=True, right=True, direction='in')
        axs[2].minorticks_on()
        axs[2].invert_xaxis()


        # This is for plotting the vertical lines for settling times
        if markerx is not None and markery is not None:
            axs[0].axvline(time_array[-1] - markery, color='green', label='Settling Time [Gyr]', alpha=0.7)
            axs[0].legend(fontsize=15)

            axs[1].axvline(time_array[-1] - markery, color='green', alpha=0.7)
            axs[1].legend(loc='upper right',fontsize=13)
                
            axs[2].axvline(time_array[-1] - markery, color='green', alpha=0.7)
           
               
    def plot_L(self, markerx=None, markery=None):
        time_array = self.galaxy_data['times']

#         raw_L = self.galaxy_data['angmom'] 
        thetaL = self.galaxy_data['thetaL']
        deltaL = self.galaxy_data['deltaL']
        momdotL = self.galaxy_data['momdotL']
        
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
        fig.tight_layout(pad=0.5)
        plt.figure(figsize=[10, 10])
        
               
        # We will have to mask some of the data to account for the -500s and the NaNs
        mask_thetaL = (thetaL != -500.0) * (np.isfinite(thetaL))
        thetaL = thetaL[mask_thetaL]
        
        # First plot is thetaL v lookback time
        axs[0].plot(time_array[-1] - time_array[mask_thetaL], thetaL, color=self.thiscolor)
        axs[0].set_ylabel("Offset Angle (Degrees)", fontsize=20)
        axs[0].tick_params(axis='both', which='both', labelsize=17, bottom=True, top=True, left=True, right=True, direction='in')
        axs[0].minorticks_on()
        axs[0].invert_xaxis()
        
        # Second plot is delta L and delta L smoothed v lookback time, using the Gaussian Filter
        thresh_gauss = 50
        stdev = [11]
        m_factor = 2
        y_smooth = self.gauss_method(m_factor, stdev, thresh_gauss, False, 'deltaL')
        
        # Adding horizontal threshold lines and excluded values
        axs[1].plot(time_array[-1] - time_array[1:], 1000 * deltaL, color='lightgray', label='Unsmoothed 'r'$\dfrac{dθ}{dt}$')
        axs[1].plot(time_array[-1] - time_array[self.cutoff_graph:self.cutoff_front], y_smooth, color=self.thiscolor, 
                        label="Gaussian Smooth")

        axs[1].axhline(y=0, color='black', linestyle='--')
        axs[1].axhline(y=thresh_gauss, color='black', linestyle='--')
        axs[1].axhline(y=-thresh_gauss, color='black', linestyle='--')
        axs[1].axvspan(0, time_array[-self.cutoff_front], color='mistyrose', label='Excluded Values', alpha=0.5)
        axs[1].axvspan(time_array[-self.cutoff_back], time_array[-1], color='mistyrose', alpha=0.5)

        axs[1].set_ylabel(r'$\dfrac{d\theta}{dt}$ (Degrees / Gyrs)', fontsize=20)
        axs[1].set_ylim(-200, 200)
        axs[1].tick_params(axis='both', which='both', labelsize=17, bottom=True, top=True, left=True, right=True, direction='in')
        axs[1].minorticks_on()
        axs[1].invert_xaxis()
        
        # Third plot is momdotL v time
        axs[2].plot(time_array[-1] - time_array, momdotL, color=self.thiscolor)
        axs[2].set_ylabel('MOI and L Dot Product', fontsize=20)
        axs[2].set_xlabel('Lookback Time [Gyr]', fontsize=25)
        axs[2].tick_params(axis='both', which='both', labelsize=17, bottom=True, top=True, left=True, right=True, direction='in')
        axs[2].set_xlim(0,time_array[-1])
        axs[2].minorticks_on()
        axs[2].invert_xaxis()
        
        if markerx is not None and markery is not None:
            axs[0].axvline(time_array[-1] - markery, color='green', label='Settling Time [Gyr]', alpha=0.7)
            axs[0].legend(fontsize=15)
            
            axs[1].axvline(time_array[-1] - markery, color='green', alpha=0.7)
            axs[1].legend(loc='upper right', fontsize=13)            
            
            axs[2].axvline(time_array[-1] - markery, color='green', alpha=0.7)
            
