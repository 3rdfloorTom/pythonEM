#!/usr/bin/env python

"""

Split a cryoSPARC .cs particle metadata file by per-particle scale factore ('alpha')
Writes out upper (exclusive) and lower alpha .cs metadata files.

Written by Tom Laughlin (2023)

"""

import numpy as np 
import matplotlib.pyplot as plt 
import cryosparc.tools as cs
import argparse


def main(particle_fn, cutoff, bin_width):
    """
    Split input .cs file by the user specified alpha value.
    """

    # load particle metadata into dataset format
    particle_data = cs.Dataset.load(particle_fn)
    particle_alpha = particle_data['alignment3D/alpha']

    # get cutoff for alpha from user
    if cutoff == None:
        original_plot = plot_histograms([particle_alpha],bin_width,"Original Per-particle scale distribution")
        
    while cutoff == None:
        user_cutoff = input('Please specify a cutoff to split the alpha/scale: ')

        try:
            cutoff = float(user_cutoff)
        except ValueError:
            print('Please enter a numerical value for the cutoff!')
    
    plt.close('all')

    # separate the metadata by user-specified cutoff
    print(f'Splitting data at {cutoff} ...\n')
    data_above, data_below = split_data(particle_data,cutoff,'alignment3D/alpha')

    # check with user for acceptance of split
    split_plot = plot_histograms([data_above['alignment3D/alpha'],data_below['alignment3D/alpha']],bin_width,'Split Per-particle scale distributions') 
    
    _ = input('If happy with split, then press any key (Ctrl-C to abort)\n')
    plt.close('all')
    
    print('Writing new files to disk.\n')

    # writing files to disk
    data_above_fn, data_below_fn, original_plot_fn, split_plot_fn = write_files(particle_fn,data_above,data_below,cutoff,original_plot,split_plot)

    # Notify user of new files
    print('New files written to disk:')
    print(f'Particle metadata for those with an alpha greater than {cutoff} was written to {data_above_fn} with {len(data_above)} particles.')
    print(f'Particle metadata for those with an alpha less than or equal to {cutoff} was written to {data_below_fn} with {len(data_below)} particles.')
    print(f'Plots for the original and masked data have been written as {original_plot_fn} and {split_plot_fn}, respectively\n')
    
    return None

def plot_histograms(particle_alpha_lists, bin_width, title):
    """
    Takes a list of items to plot as a 1D-histogram with a specified bin width and title
    Returns a figure object
    """
    fig = plt.figure()
    counts_list = []
    bins_list = []
    
    for particle_alpha_list in particle_alpha_lists:
        counts, bins = np.histogram(particle_alpha_list, np.arange(min(particle_alpha_list), max(particle_alpha_list) + bin_width, bin_width))   
        counts_list.append(counts)
        bins_list.append(bins)

    for idx in range(len(particle_alpha_lists)):
        plt.stairs(counts_list[idx],bins_list[idx],fill=True, figure=fig)
    
    plt.title(title, figure=fig)
    plt.ylabel('# of images', figure=fig)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.xlabel('refined scale factor', figure=fig)
    plt.xlim([0,2])
    plt.show()
    
    return fig

def split_data(particle_data, cutoff, field):
    alpha_mask = particle_data[field] > cutoff
    data_above = particle_data.mask(alpha_mask)
    data_below = particle_data.mask(~alpha_mask)
    return data_above,data_below

def write_files(particle_fn,data_above,data_below,cutoff,original_plot,split_plot):
    """
    Write out original and masked particle metadata and plots
    """
    # format strings for file names
    basename = particle_fn.split('.')[0]
    data_above_fn = basename + "_above_" + cutoff + ".cs"
    data_below_fn = basename + "_below_" + cutoff + ".cs"
    
    # write split files with formatted names
    data_above.save(data_above_fn)
    data_below.save(data_below_fn)

    # write out histograms for reference
    original_plot_fn = basename + "_original.png"
    original_plot.figsize = (11.80,8.85)
    original_plot.dpi = 300
    original_plot.savefig(original_plot_fn)  

    split_plot_fn = basename + "_split.png"
    split_plot.figsize = (11.80,8.85)
    split_plot.dpi = 300
    split_plot.savefig(split_plot_fn)

    return data_above_fn, data_below_fn, original_plot_fn, split_plot_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = "Split cryoSPARC particle metadata file by alpha/per-particle scale factor")
    
    parser.add_argument('--input', '-i', type=str, required = True,
                        help = ('Filename of the cryoSPARC particle metadata file to be modified.'
                                '[REQUIRED]'))
    
    parser.add_argument('--cutoff', '-c', type=float, required = False,
                        help = ('cutoff for splitting per-particle alpha/scale factors'))
    
    parser.add_argument('--bin_width', '-b', type=float, required = False, default=0.01,
                        help = ('width of bins for histogram plotting'))
    
    args = parser.parse_args()

    main(particle_fn =args.input,
         cutoff = args.cutoff,
         bin_width = args.bin_width)