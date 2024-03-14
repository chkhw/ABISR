import os
import gc
import glob
import warnings
import argparse
from dataclasses import dataclass, field, asdict

import h5py
import xarray as xr
import numpy as np
import pandas as pd

import tensorflow as tf

from joblib import Parallel, delayed

import abi, domains

def decode_rad(array, encoding):
    '''
    Inputs
    ------
    array : numpy.ndarray
        The array containing scaled radiance values to be decoded. The array is modified in-place but also returned for convenience.
    encoding : dict
        A dictionary containing encoding parameters, specifically 'add_offset', 'scale_factor', and '_FillValue'. These parameters
        are used to decode the array from its scaled integer representation back to floating point radiance values.

    Outputs
    -------
    numpy.ndarray
        The decoded radiance data as a `float32` numpy array, with scale factors and offsets applied. Fill values are replaced with NaN.
    '''

    array = array.astype('float32')
    array[array==encoding['_FillValue']] = np.nan
    ao = encoding['add_offset']
    sf = encoding['scale_factor']
    array = array*sf + ao
    return array

def create_rad_file(band, filenames, rad, output_directory, domain):

    """
    Create a L1b file with super-resolved data, using domain-specific cut outs and attributes.

    Inputs
    ------
    band : str
        Band identifier for which the radiance file is created.
    filenames : dict
        Dictionary containing file paths. Keys are band names, and values are the L1b file paths.
    rad : numpy.ndarray
        Array of radiance values to be saved.
    output_directory : str
        Path where the output file will be saved. Defaults to current directory.
    domain : str
        Domain identifier used to specify a subset of data for processing.

    Outputs
    -------
    None
    """
    
    # Get the coordinates of the domain we ran the model for
    domain_coords = domains.domain_inds[config.domain]
    
    # Need to double check issues with these warnings, for now, supress them
    warnings.filterwarnings("ignore", category=xr.SerializationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Engine 'cfgrib' loading failed:")
        
    #rad = processed_image[:,:,band_index]
    rad = np.around(rad, decimals=0) # Is this necessary to do? I'm not sure.
    
    # Open the band-2 file and the LR band file
    ds_band2 = xr.open_dataset(filenames['02'], mode='r', engine='netcdf4')
    ds = xr.open_dataset(filenames[band], mode='r', engine='netcdf4')
    scale_ratio = ds_band2['Rad'].shape[0]//ds['Rad'].shape[0]

    # Remove LR coordinates from xarray object
    ds = ds.drop('y')
    ds = ds.drop('x')
    
    # Hold LR rad in memory but remove it from the xarray object
    old_rad = ds['Rad']
    old_rad_encoding_dict = ds['Rad'].encoding # also hold on to encoding information
    ds = ds.drop('Rad')
    
    # Hold LR dqf in memory but remove it from the xarray object, not currently used?
    old_dqf = ds['DQF']
    old_dqf_encoding_dict = ds['DQF'].encoding # also hold on to encoding information
    ds = ds.drop('DQF')
    
    # Put new coords into the LR band xarray object (ds)
    ds = ds.assign_coords({'y': ds_band2['y'][domain_coords[0]:domain_coords[1]],
                           'x': ds_band2['x'][domain_coords[2]:domain_coords[3]]})
    
    # Get the fill value mask from the old data, upsample, and apply it to new data
    fv_mask = np.isnan(old_rad[:])
    fv_mask = np.repeat(fv_mask, scale_ratio, axis=0)
    fv_mask = np.repeat(fv_mask, scale_ratio, axis=1)
    fv_mask = fv_mask[domain_coords[0]:domain_coords[1], domain_coords[2]:domain_coords[3]]
    rad[fv_mask] = old_rad_encoding_dict['_FillValue']

    # Decode from scaled radiance to radiance
    rad = decode_rad(rad, old_rad_encoding_dict)
    
    # Turn rad from numpy to DataArray with specified dimensions + encoding paramters
    new_rad = xr.DataArray(name='Rad', data=rad, dims=('y', 'x'))
    old_rad_encoding_dict['complevel'] = 1 # Increase compression level to save space, but take longer to read
    new_rad.encoding = old_rad_encoding_dict
    
    # Add new super-resolved rad to old band ds
    ds['Rad'] = new_rad
    
    # Add attributes, and fix some
    ds['Rad'].attrs = old_rad.attrs
    ds.attrs['spatial_resolution'] = "0.5km at nadir (super-resolved)"
    ds.attrs['SR_ROW_MIN'] = domain_coords[0]
    ds.attrs['SR_ROW_MAX'] = domain_coords[1]
    ds.attrs['SR_COL_MIN'] = domain_coords[2]
    ds.attrs['SR_COL_MAX'] = domain_coords[3]
    
    # Write out the copied ds with new variables to the output file
    output_fname = os.path.basename(filenames[band])[0:-3] + '_05km.nc'
    ds.to_netcdf(f'{output_directory}{output_fname}')
    return

def read_all_abi_bands(filenames, domain):
    '''
    Inputs
    ------
    filenames: dict
        Dictionary of filenames where keys are the band names. Values
        are the names of each L1b file.
    domain: str
        String with name of domain. Must be specified in domains.py
        
    Outputs
    -------
    data: dict
        Dictionary of numpy arrays where keys are the band name
    
    '''
    data = {}
    coords = domains.domain_inds[domain]

    for band in abi.all_bands:
        
        # Create h5py file object
        f = h5py.File(filenames[band], 'r')

        # Get scale so we know how to change domain coords for each channel
        if band in abi.abi_2km_bands: scale = 4
        elif band in abi.abi_1km_bands: scale = 2
        else: scale = 1

        # Read section of data corresponding to domain
        data[band] = f['Rad'][coords[0]//scale:coords[1]//scale,
                              coords[2]//scale:coords[3]//scale].astype('float32')

        # Set pixels with fill value to zero (needed for running the CNN)
        fv = f['Rad'].attrs['_FillValue']
        data[band][data[band]==fv] = 0.0

        # Close file
        f.close()
        
        print(band, data[band].shape)

    return data


def fetch_filenames(input_path, timestring):
    """
    Inputs
    ------
    inputs_path : str
        path to l1b files
    timestring : str
        timestamp starting with 's' the denotes when the file is from (should be the
        same for all l1b files)

    Outputs
    -------
    filenames : dict
        A dictionary where keys are band identifiers and values are the first matching
        filename for each band.
    """
    filenames = {}
    for band in abi.all_bands:
        matching_files = glob.glob(f'{input_path}*C{band}_G16*{timestring}*.nc')
        if matching_files: filenames[band] = matching_files[0]
    
    # Ensure all expected channels are found
    assert len(filenames) == 16, "Some channels not found."
    
    return filenames

def main(config):    
    # Set some configs on how the model will be run on the gpu
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[config.gpu_index], 'GPU')
    tf.keras.mixed_precision.set_global_policy(tf.keras.mixed_precision.Policy('mixed_float16'))
    
    # Make output directory using timestring if it doesn't already exist
    output_directory = f'{config.output_path}ABISR_{config.timestring}/'
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    
    # Load the model
    model = tf.keras.models.load_model(config.model_path, compile=False)
    
    # Fetch the filenames from the specified input directory
    filenames = fetch_filenames(config.input_path, config.timestring)
    
    # Read in the ABI data from the detected files
    data = read_all_abi_bands(filenames, config.domain)
    
    # Compute the number of chunks along each dimension    
    stride = config.chunk_size - config.overlap
    image_x_size = data['02'].shape[0]
    image_y_size = data['02'].shape[1]
    print(image_x_size, image_y_size)
    num_chunks_x = int(np.ceil((image_x_size - config.overlap) / stride))
    num_chunks_y = int(np.ceil((image_y_size - config.overlap) / stride))
    
    # Initialize an empty array to store the processed image
    num_channels = len(abi.lr_bands) # band-2 not super-resolved so doesn't count
    processed_image = np.zeros([image_x_size, image_y_size, num_channels], dtype='float32')
    
    # Initialize an array to keep track of how many times each pixel is processed (accounts for overlap later)
    count_map = np.zeros_like(data['02'], dtype='float32')

    # Loop over the chunks
    print('Running predictions on subtiles of full image...')
    
    for i in range(num_chunks_x)[:]:
        for j in range(num_chunks_y)[:]:
            
            # Calculate the start indices for the chunk
            start_x = i * stride
            start_y = j * stride
            start_x = min(start_x, image_x_size - config.chunk_size)
            start_y = min(start_y, image_y_size - config.chunk_size)
    
            # Calculate the end indices for the chunk
            end_x = start_x + config.chunk_size
            end_y = start_y + config.chunk_size
    
            # Get chunk indicies for LR channels
            start_x_1km, start_y_1km, end_x_1km, end_y_1km = start_x//2, start_y//2, end_x//2, end_y//2
            start_x_2km, start_y_2km, end_x_2km, end_y_2km = start_x//4, start_y//4, end_x//4, end_y//4

            # Extract the chunks
            chunk_05km = np.stack([data[band][start_x:end_x,start_y:end_y] for band in abi.abi_05km_bands],axis=-1)
            chunk_1km  = np.stack([data[band][start_x_1km:end_x_1km,start_y_1km:end_y_1km] for band in abi.abi_1km_bands],axis=-1)
            chunk_2km = np.stack([data[band][start_x_2km:end_x_2km,start_y_2km:end_y_2km] for band in abi.abi_2km_bands],axis=-1)

            # Check if this chunk has nonzero values. If not, then it is outside the FD and skip.
            if (np.nansum(chunk_2km[:,:,-1])!=0)==0:
                continue

            print(f'Running chunk {start_x} {end_x} {start_y} {end_y}')

            # Standardize the chunks
            chunk_05km = (chunk_05km - abi.means_05km) / abi.stds_05km
            chunk_1km = (chunk_1km - abi.means_1km) / abi.stds_1km
            chunk_2km = (chunk_2km - abi.means_2km) / abi.stds_2km
            
            # Process the chunk with the model
            processed_chunk = model.predict_on_batch((chunk_2km[None,...], chunk_1km[None,...], chunk_05km[None,...]))
    
            # Unstandardize
            processed_chunk = processed_chunk * abi.stds_lr + abi.means_lr
    
            # Add the processed chunk to the processed_image array
            processed_image[start_x:end_x, start_y:end_y] += processed_chunk[:end_x - start_x, :end_y - start_y].squeeze()
            
            # Update the count_map array
            count_map[start_x:end_x, start_y:end_y] += 1.0

    # Divide by number of counts to make predicted image
    processed_image /= count_map[:,:,None]

    # Remove data dict from memory since we don't need it anymore
    del data, count_map, processed_chunk, chunk_05km, chunk_1km, chunk_2km, model
    gc.collect()

    # Write out each of the channels in parallel
    print('Saving super-resolved output to disk...')
    r = Parallel(n_jobs=config.n_parallel_out, verbose=50)(delayed(create_rad_file)(abi.lr_bands[k],
                                                                                    filenames,
                                                                                    processed_image[:,:,k],
                                                                                    output_directory=output_directory,
                                                                                    domain=config.domain) for k in range(num_channels))

@dataclass
class DefaultConfig:
    model_path: str = 'models/PansharpeningCNN_12_layers_256_filters/' # path to the model used
    input_path: str = '' # path to the L1b files 
    output_path: str = '' # path to where you want the 0.5km output saved
    timestring: str = 's20221251520205' # timestamp in the ABI L1b filenames.
    gpu_index: int = 0 # Integer index of the GPU you want to use.
    chunk_size: int = 1024 # Bigger means more GPU VRAM needed, but potentially faster. Needs to be smaller than size of domain at 0.5km.
    overlap: int = 32 # Number of pixels to average between adjacent tiles. Cuts down on edge artifacts. 32 is probably most you need.
    domain: str = 'FD_Example4' # Use named examples present in domains.py
    n_parallel_out: int = 8 # number of parallel jobs to write out to disk. Not much benefit after 4ish.
    
def parse_arguments(config: DefaultConfig):
    parser = argparse.ArgumentParser(description="This script runs predictions using the ABI super-res model. ")
    parser.add_argument('--model_path', type=str, default=config.model_path)
    parser.add_argument('--input_path', type=str, default=config.input_path)
    parser.add_argument('--output_path', type=str, default=config.output_path)
    parser.add_argument('--timestring', type=str, default=config.timestring)
    parser.add_argument('--gpu_index', type=int, default=config.gpu_index)
    parser.add_argument('--chunk_size', type=int, default=config.chunk_size)
    parser.add_argument('--overlap', type=int, default=config.overlap)
    parser.add_argument('--domain', type=str, default=config.overlap)
    parser.add_argument('--n_parallel_out', type=int, default=config.n_parallel_out)
    args = parser.parse_args()

    # Update config based on provided arguments
    for field in asdict(config):
        setattr(config, field, getattr(args, field))

    return config

if __name__=='__main__':
    
    # Set up default config
    config = DefaultConfig()
    
    # Update config based on arguments
    config = parse_arguments(config)

    print(config)

    # Train models
    main(config)