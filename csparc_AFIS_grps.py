#!/usr/bin/env python

"""
Sort EPU AFIS data into Exposure Groups for cryoSPARC v4.1+

Written By Dustin Reed Morado for RELION v3.1
Modified By Tom Laughlin for cryoSPARC

"""

import os 
import shutil
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import cryosparc.tools as cs
import click

from xml.etree import ElementTree 
from sklearn.cluster import AgglomerativeClustering

def main(xml_dir, particle_fn):
	"""
	Modifies a input cryoSPARC particle.cs file to possess Exposure Groups based on EPU beam shifts
	Interactively assigns groups based on user input after plotting shifts

	NOTE: Overwrites the input file
		  Does write a back-up of the input file
	"""

	metadata_fns = get_xml_list(xml_dir)
	shift_array = get_shift_array(metadata_fns)
	n_groups = plot_shifts(shift_array)

	exposure_groups, group_centers = cluster_groups(n_groups, shift_array)
	exposure_group_ids, sorted_radii = sort_groups(shift_array, exposure_groups, group_centers)

	plot_groups(shift_array,exposure_group_ids, group_centers, sorted_radii)
	
	particle_data = cs.Dataset.load(particle_fn)

	particle_data_grouped = apply_groups(metadata_fns,exposure_group_ids,group_centers,sorted_radii)

	# Back-up input file
	particle_fn_bak = particle_fn + ".bak"

	click.echo(
		f"Writing a back-up of the input cryoSPARC particle file as: {particle_fn_bak} \n")

	shutil.copyfile(particle_fn, particle_fn_bak)

	# Write out new file to input file name
	particle_data_grouped.save(particle_fn)
	
	click.echo(
		f"Modified cryoSPARC particle file was written out as: {particle_fn} \n")
	

	return



def get_xml_list(xml_dir):
	"""
	Walk through the EPU data directory and load all movie metadata xml files into a list

	Input: Path
	Return: list
	"""
	metadata_fns = []

    for dirpath, dirnames, filenames in os.walk(xml_dir, followlinks=True):
        for filename in filenames:
            if (fnmatch(filename, 'FoilHole_*_Data_*.xml'))
         		metadata_fns.append(os.path.join(dirpath, filename))

    # Error check
    if len(metadata_fns) == 0:
    	raise ValueError('No EPU XML metadata files found')

    return metadata_fns


def get_shift_array(metadata_fns):
	"""
	Parse the movie metadata xmls for applied beamshifts for each movie

	Input: list
	Return: np.array
	"""
	# TFS namespaces
	name_space = {'fei' : 'http://schemas.datacontract.org/2004/07/Fei.SharedObjects'}

	beam_shifts = []

	for metadata_fn in metadata_fns:
    	metadata = ElementTree.parse(metadata_fn)
    	beam_shift = metadata.findall('fei:microscopeData/fei:optics/fei:BeamShift', name_space)

    	# Error check
    	if len(beam_shift) == 0:
            raise ValueError('No BeamShift found in {}'.format(metadata_fn))
        elif len(beam_shift[0]) != 2:
            raise ValueError('Improper BeamShift found in {}'.format(metadata_fn))

		beam_shifts.append([float(x.text) for x in beam_shift[0]])

	return np.array(beam_shifts)


def plot_shifts(shift_array):
	"""
	Plots beam shifts for user to indicate number of groups
	
	Input: np.array
	Return: int
	"""

	n_groups = 1

	if n_groups == 1:
    	figure = plt.figure()
    	axes = figure.add_subplot(111)
    	axes.set_title('EPU AFIS Beam Shift Values')
    	axes.set_xlabel('Beam Shift X (a.u.)')
    	axes.set_ylabel('Beam Shift Y (a.u.)')
    	axes.scatter(shift_array[:, 0], shift_array[:, 1])
    	plt.show(block = False)

        while n_groups <= 1:
            n_user = input('How many Exposure Groups (or Ctrl-C to abort): ')

            try:
                n_groups = int(n_user)
            except ValueError:
                print('Please enter a positive integer greater than 1!')
            else:
                if n_groups <=1:
                    print('Please enter a positive integer greater than 1!')

        plt.close('all')

    if len(beam_shifts) < n_groups:
        raise ValueError('Number of groups greater than number of points')

    return n_groups


def cluster_groups(n_groups, shift_array):
	"""
	Clusters beam shifts into specified number of groups by hierarchical ascending classification (hac)
	Input: int, np.array
	Returns: np.array,list
	"""

	hac = AgglomerativeClustering(n_clusters = n_groups, 
									linkage = 'complete'.fit(shift_array))

	exposure_groups = hac.labels_

	group_centers = [[np.average(shift_array[exposure_groups==x][:,0]),
					  np.average(shift_array[exposure_groups==x[:,1]])]
					  for x in set(sorted(exposure_groups))]

	return exposure_groups, group_centers

def sort_groups(shift_array, exposure_groups, group_centers):
	"""
	Plot exposure groups and assign exposure group IDs
	Input: np.array, np.array, list
	Returns: list,list
	"""

	# arrange points using polar coordinates
	radii = [x[0]**2 + x[1]**2 for x in group_centers]
	sorted_radii = sorted(enumerate(radii), key=lambda x:x[1])
	
	# set origin as the smallest radius
	origin = group_centers[sorted_radii[0][0]]

	# initialize dummy variables
	x_vec = (1,0)
	x_unit = (1,0)
	x_angle = np.pi
	x_length = 1

	y_vec = (1,0)
	y_unit = (1,0)
	y_angle = np.pi
	y_length = 1

	# base grid axes on four smallest points after the origin
	for idx in range(1,5):
    	point = group_centers[sorted_radii[idx][0]]
    	x_coord = point[0] - origin[0]
    	y_coord = point[1] - origin[1]
    
   	 	# set the x-axis
    	if x_coord > 0 and x_coord > abs(y_coord):
        	x_vec = (x_coord, y_coord)
        	x_angle = np.arctan2(y_coord,x_coord)
       		x_length = np.sqrt(x_vec[0]**2 + x_vec[1]**2)
        	x_unit = (x_vec[0] / x_length, x_vec[1] / x_length)
        
    	# set the y-axis
    	if y_coord > 0 and y_coord > abs(x_coord):
        	y_vec = (x_coord, y_coord)
        	y_angle = np.arctan2(y_coord, x_coord)
        	y_length = np.sqrt(y_vec[0]**2 + y_vec[1]**2)
        	y_unit = (y_vec[0] / y_length, y_vec[1] / y_length)


    # initialize list to store distances on polar grid
	grid_dists = []

	for idx, center in enumerate(group_centers):
    	# normalize the point w.r.t. the origin
    	point = center[0] - origin[0], center[1] - origin[1]
    
    	# project points along unit vectors
    	grid_x = np.round((point[0] * x_unit[0] + point[1] * x_unit[1])/x_length)
    	grid_y = np.round((point[0] * y_unit[0] + point[1] * y_unit[1])/y_length)
    
    	dist = max(abs(grid_x),abs(grid_y))
    
    	# compute angle and sort CCW
    	angle = (np.degrees(np.arctan2(grid_y, grid_x)) + 360) % 360
    	grid_dists.append((idx, grid_x, grid_y, dist, angle))

		# sort by dist and then by angle
		sorted_rays = [x[0] for x in sorted(grid_dists, key=lambda x:(x[3], x[4]))]

		# apply sorting back to original groups
		exposure_group_ids = [sorted_rays.index(x) + 1 for x in group_centers]

	return exposure_group_ids,sorted_radii

def plot_groups(shift_array,exposure_group_ids, group_centers, sorted_radii):
	"""
	Plot exposure groups

	Input: np.array, list,list,list
	Output: None
	"""
	figure = plt.figure()
	axes = figure.add_subplot(111)
	plt.scatter(shift_array[:,0], shift_array[:,1], c=exposure_group_ids, cmap='tab20b')

	for exposure_group in range(len(exposure_group_ids)):
    	idx = sorted_radii[exposure_group][0]
    	axes.annotate('{0:d}'.format(exposure_group + 1), xy=group_centers[idx], textcoords='offset pixels', xytext=(5, 5))

    plt.show(block = False)
    _ = input('If happy with grouping press any key (Ctrl-C to abort)')
    plt.close('all')
    
    return None


def apply_groups(particle_data, metadata_fns, exposure_group_ids):
	"""
	Modify the input particle file to contain the exposure groups

	Input: np.array, list,list,list
	Output: dict (like)
	"""
	mics_with_groups = sorted(zip(metadata_fns,exposure_group_ids), key=lambda x:(x[0],x[1]))

	particle_names = particle_data['blob/path']

	field_no = len(os.path.basename(mics_with_groups[0][0]).split('_'))

	particle_mic_names = []

	for particle_name in particle_names:
    	base = os.path.basename(particle_name)
    	root,ext = os.path.splitext(base)
    	mic_base = '_'.join(root.split('_')[0:field_no])
    	particle_mic_names.append(mic_base)

    exposure_dic = {}

	for metadata_fn, exposure_group in mics_with_groups:
    	base = os.path.basename(metadata_fn)
    	mic_base,ext = os.path.splitext(base)
    	exposure_dic[mic_base] = expsoure_group


	exposure_groups_assigned = []

	for particle_mic_name in particle_mic_names:
    	expoure_groups_assigned.append(exposure_dic[particle_mic_name])


    particle_data['ctf/exp_group_id'] = exposure_groups_assigned
	
	return particle_data

# Argument parsing with click
@click.command()
@click.option('--input_xml_dir', '-i',
              prompt='Input EPU XML directory path',
              type=click.Path(),
              required=True)
@click.option('--cryosparc_particle_file', '-cs',
              prompt='cryoSPARC particle file path',
              type=click.Path(),
              required=True)

if __name__ == '__main__':
	main(input_xml_dir,cryosparc_particle_file)

