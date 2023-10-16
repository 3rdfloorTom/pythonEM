#!/usr/bin/env python

"""
Threshold ("mask") the tilt/elevation of particles from a cryoSPARC .cs file
Potentially useful after local refinement of helical specimen

Written by Tom Laughlin (2023)
"""
import sys
import math
import numpy as np 
import matplotlib.pyplot as plt 
import cryosparc.tools as cs
import time
import argparse 

from eulerangles import matrix2euler
from mpl_toolkits.axes_grid1 import make_axes_locatable

def main(particle_fn, upper_bound, lower_bound):
    """
    Modifies the input cryoSPARC .cs file remove images with excessive out-of-plane tilt ("elevation")
    NOTE: writes a back-up of the original .cs file
    """

    # load particle metadata into dataset format
    particle_data = cs.Dataset.load(particle_fn)
    print('Extracting angles...this can take a few moments...\n')
    original_azimuth,original_elevation = eulers_from_rodrigues(particle_data)

    # get bound on tilt/elevation from user
    if upper_bound == None or lower_bound == None:
        original_plot = plot_histogram(original_azimuth,original_elevation,"Original Viewing Angle Distribution")
        
    while upper_bound == None:
        user_upper_bound = input('Please specify an upper bound for the tilt/elevation: ')

        try:
            upper_bound = float(user_upper_bound)
        except ValueError:
            print('Please enter a numerical value for the upper bound!')
    
    while lower_bound == None:
        user_lower_bound = input('Please specify an lower bound for the tilt/elevation: ')

        try:
            lower_bound = float(user_lower_bound)
        except ValueError:
            print('Please enter a numerical value for the lower bound!')
        else:
            if lower_bound >= upper_bound:
                print('Please specify a lower bound less than the upper bound!')

    plt.close('all')

    print(f'Masking elevation data between {lower_bound} and {upper_bound} ...\n')

    # Apply threshold mask on tilt/elevation
    mask= make_mask(original_elevation,upper_bound,lower_bound)
    masked_azimuth = list(np.array(original_azimuth)[mask])
    masked_elevation = list(np.array(original_elevation)[mask])
    masked_particle_data = particle_data.mask(mask)

    # Prompt user for acceptance of writing files
    masked_plot = plot_histogram(masked_azimuth,masked_elevation,"Masked Viewing Angle Distribution")

    _ = input('If happy with masking, then press any key (Ctrl-C to abort)\n')
    plt.close('all')
    
    print('Writing new files to disk.\n')

    # writing files to disk
    backup_fn, original_plot_fn, masked_plot_fn = write_files(particle_fn,particle_data,masked_particle_data,original_plot,masked_plot)

    # Notify user of new files
    print('New files written to disk:')
    print(f'Original particle metadata has been moved to {backup_fn} with {len(particle_data)} particles.')
    print(f'Masked particle metadata has been written to {particle_fn} with {len(masked_particle_data)} particles.')
    print(f'Plots for the original and masked data have been written as {original_plot_fn} and {masked_plot_fn}, respectively\n')

def rod2matrix(v):
    """
    Convert a Rodrigues vector into a rotaion matrix
    """
    theta = np.linalg.norm(v)
    
    if theta < sys.float_info.epsilon:              
        rotation_mat = np.eye(3, dtype=float)
    else:
        r = v / theta
        I = np.eye(3, dtype=float)
        r_rT = np.array([
            [r[0]*r[0], r[0]*r[1], r[0]*r[2]],
            [r[1]*r[0], r[1]*r[1], r[1]*r[2]],
            [r[2]*r[0], r[2]*r[1], r[2]*r[2]]
        ])
        r_cross = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])
        rotation_matrix = math.cos(theta) * I + (1 - math.cos(theta)) * r_rT + math.sin(theta) * r_cross
    return rotation_matrix


def eulers_from_rodrigues(particle_data):
    """"
    Returns lists of Euler angles in radians from input cryoSPARC poses
    """
    # extract the particle poses (represented as a Rodrigues vectors)
    rodrigues_vectors = particle_data['alignments3D/pose']

    # convert the Rodrigues vectors to rotation matrices
    rotation_matrix_list = []
    for vector in rodrigues_vectors:
        rotation_matrix = rod2matrix(vector)
        rotation_matrix_list.append(rotation_matrix)

    # convert rotation matrices into Euler angles (Relion Convention) in radians
    euler_angle_list = []
    for matrix in rotation_matrix_list:
        euler_angles = np.deg2rad(matrix2euler(matrix, axes='zyz',intrinsic=True,right_handed_rotation=True))
        euler_angle_list.append(euler_angles)

    # make lists of Euler angles
    azimuth = [angles[0] for angles in euler_angle_list]
    elevation = [np.pi/2-angles[1] for angles in euler_angle_list] #offset to match the plotting in the cryoSPARC output log

    return azimuth, elevation


def plot_histogram(azimuth,elevation,title):
    """
    Plot a 2D histogram of number of particles per Euler angle pair.
    """
 
    fig, ax = plt.subplots()
    hb = ax.hexbin(azimuth, elevation, bins='log', cmap='jet', gridsize=50)
    
    ax.set(xlim=(-np.pi, np.pi), ylim=(-np.pi/2, np.pi/2))
    ax.set_xlabel('Azimuth')
   # ax.set_xticks(np.arange(-np.pi, np.pi + np.pi/2, np.pi/2),[r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
                 
    ax.set_ylabel('Elevation')
   # ax.set_yticks(np.arange(-np.pi/2, np.pi/2 + np.pi/4, np.pi/4),[r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$', r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$'])
        
    ax.set_title(title)
    fig.gca().set_aspect('equal', adjustable='box')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cb = fig.colorbar(hb, ax=ax, cax=cax)
    cb.set_label('# of images')
    fig.tight_layout()
    plt.show(block=False)

    return fig

def make_mask(elevation, upper_bound, lower_bound):
    """
    Make a logical array of indices to retain in the particle data
    """
    mask_lower = np.array(elevation) > lower_bound
    mask_upper = np.array(elevation) < upper_bound
    mask = np.logical_and(mask_lower,mask_upper)

    return mask

def write_files(particle_fn,particle_data, masked_particle_data,original_plot,masked_plot):
    """
    Write out original and masked particle metadata and plots
    """
    # format strings for file names
    basename = particle_fn.split('.')[0]
    time_stamp = time.strftime('%Y%m%d-%H%M%S')    
    backup_fn = basename + "_" + time_stamp + ".cs"

    # write backup of original data and overwrite original name with the masked data
    particle_data.save(backup_fn)
    masked_particle_data.save(particle_fn)

    # write out histograms for reference
    original_plot_fn = basename + "_" + time_stamp + "_original.png"
    original_plot.figsize = (11.80,8.85)
    #original_plot.dpi = 300
    original_plot.savefig(original_plot_fn, dpi=300)  

    masked_plot_fn = basename + "_" + time_stamp + "_masked.png"
    masked_plot.figsize = (11.80,8.85)
    #masked_plot.dpi = 300
    masked_plot.savefig(masked_plot_fn, dpi=300)

    return backup_fn, original_plot_fn, masked_plot_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = "Threshold ('mask') the elevation angles of a cryoSPARC particle file")
    
    parser.add_argument('--input', '-i', type=str, required = True,
                        help = ('Filename of the cryoSPARC particle metadata file to be modified.'
                                '[REQUIRED]'))
    
    parser.add_argument('--upperbound', '-u', type=float, required = False,
                        help = ('Upper limit/threshold on the elevation (in radians).'))
    
    parser.add_argument('--lowerbound', '-l', type=float, required = False,
                        help = ('Lower limit/thresold on the elevation (in radians).'))
    
    args = parser.parse_args()

    main(particle_fn =args.input,
         upper_bound = args.upperbound,
         lower_bound = args.lowerbound)
