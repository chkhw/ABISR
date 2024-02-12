import os
import gc
import glob
import warnings
import argparse

import h5py
import xarray as xr
import numpy as np
import pandas as pd

import tensorflow as tf

from joblib import Parallel, delayed

import abi

def decode_rad(array, encoding):
    array = array.astype('float32')
    array[array==encoding['_FillValue']] = np.nan
    ao = encoding['add_offset']
    sf = encoding['scale_factor']
    array = array*sf + ao
    return array

def create_rad_file(band, filenames, rad, output_directory=''):
    
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
    ds = ds.assign_coords({'y': ds_band2['y'], 'x': ds_band2['x']})
    
    # Get the fill value mask from the old data, upsample, and apply it to new data
    fv_mask = np.isnan(old_rad[:])
    fv_mask = np.repeat(fv_mask, scale_ratio, axis=0)
    fv_mask = np.repeat(fv_mask, scale_ratio, axis=1)
    rad[fv_mask] = old_rad_encoding_dict['_FillValue']

    # Decode from scaled radiance to radiance
    rad = decode_rad(rad, old_rad_encoding_dict)
    
    # Turn rad from numpy to DataArray with specified dimensions + encoding paramters
    new_rad = xr.DataArray(name='Rad', data=rad, dims=('x', 'y'))
    old_rad_encoding_dict['complevel'] = 1 # Increase compreesion level to save space, but take longer to read
    new_rad.encoding = old_rad_encoding_dict
    
    # Add new super-resolved rad to old band ds
    ds['Rad'] = new_rad
    
    # Add attributes, and fix some
    ds['Rad'].attrs = old_rad.attrs
    ds.attrs['spatial_resolution'] = "0.5km at nadir (super-resolved)"
    
    # Write out the copied ds with new variables to the output file
    output_fname = os.path.basename(filenames[band])[0:-3] + '_05km' + '.nc'
    ds.to_netcdf(output_directory + output_fname)
    
    return

def main(path_to_model, path_to_input, path_to_output, timestring):
    
    # Set some configs on how the model will be run on the gpu
    print('Available GPUs: ')
    print(tf.config.list_physical_devices('GPU'))
    gpus = tf.config.list_physical_devices('GPU')
    print('Picking the first one')
    gpu = gpus[0]
    tf.config.set_visible_devices(gpu, 'GPU')
    tf.keras.mixed_precision.set_global_policy(tf.keras.mixed_precision.Policy('mixed_float16'))
    
    # Make output directory using timestring if it doesn't already exist
    output_directory = path_to_output + 'ABISR_' + timestring + '/'
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    
    # Load the model
    model = tf.keras.models.load_model(path_to_model, compile=False)
    
    # Fetch the filenames from the specified input directory
    filenames = {}
    print('Filenames found: ')
    for band in abi.all_bands :
        filenames[band] = glob.glob(path_to_input + f'*C{band}_G16*{timestring}*.nc')[0]
        print(os.path.basename(filenames[band]))
    if len(filenames)!=16:
        print('Some channels not found. Ensure that all 15 LR channels and the single Band-2 file are present')
    
    # Read in the ABI data from the detected files
    data = {}
    for band in abi.all_bands:
        f = h5py.File(filenames[band], 'r')
        fv = f['Rad'].attrs['_FillValue']
        data[band] = f['Rad'][:].astype('float32')
        data[band][data[band]==fv] = 0.0
        f.close()
    
    # Define the chunk size and overlap, these were tuned for speed and to minimize issues along tile edges
    chunk_size = 1024
    overlap = 16
    
    image_x_size = data['02'].shape[0]
    image_y_size = data['02'].shape[1]
    
    # Compute the number of chunks along each dimension
    stride = chunk_size - overlap
    num_chunks_x = int(np.ceil((image_x_size - overlap) / stride))
    num_chunks_y = int(np.ceil((image_y_size - overlap) / stride))
    
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
            
            # Adjust the start indices for the last chunk in each row and column
            start_x = min(start_x, image_x_size - chunk_size)
            start_y = min(start_y, image_y_size - chunk_size)
    
            # Calculate the end indices for the chunk
            end_x = start_x + chunk_size
            end_y = start_y + chunk_size
    
            # Get chunk indicies for LR channels
            start_x_1km, start_y_1km, end_x_1km, end_y_1km = start_x//2, start_y//2, end_x//2, end_y//2
            start_x_2km, start_y_2km, end_x_2km, end_y_2km = start_x//4, start_y//4, end_x//4, end_y//4
            
            # Extract the chunks
            chunk_05km = np.stack([data[band][start_x:end_x,start_y:end_y] for band in abi.abi_05km_bands],axis=-1)
            chunk_1km  = np.stack([data[band][start_x_1km:end_x_1km,start_y_1km:end_y_1km] for band in abi.abi_1km_bands],axis=-1)
            chunk_2km = np.stack([data[band][start_x_2km:end_x_2km,start_y_2km:end_y_2km] for band in abi.abi_2km_bands],axis=-1)
    
            # Standardize the chunks
            chunk_05km = (chunk_05km - abi.means_05km) / abi.stds_05km
            chunk_1km = (chunk_1km - abi.means_1km) / abi.stds_1km
            chunk_2km = (chunk_2km - abi.means_2km) / abi.stds_2km
            
            # Process the chunk with the model
            processed_chunk = model.predict_on_batch((chunk_2km[None,...], chunk_1km[None,...], chunk_05km[None,...]))
    
            # Unstandardize
            processed_chunk = processed_chunk*abi.stds_lr + abi.means_lr
    
            # Add the processed chunk to the processed_image array
            processed_image[start_x:end_x, start_y:end_y] += processed_chunk[:end_x - start_x, :end_y - start_y].squeeze()
            
            # Update the count_map array
            count_map[start_x:end_x, start_y:end_y] += 1.0
    
    
    # Divide by number of counts to make predicted image
    processed_image /= count_map[:,:,None]
    
    # Remove data dict from memory since we don't need it anymore
    del data, count_map, processed_chunk, chunk_05km, chunk_1km, chunk_2km
    gc.collect()

    # Write out each of the channels in parallel
    print('Saving super-resolved output to disk...')
    r = Parallel(n_jobs=8, verbose=50)(delayed(create_rad_file)(abi.lr_bands[k], filenames, processed_image[:,:,k], output_directory=output_directory) for k in range(num_channels))

def parse_arguments():
    parser = argparse.ArgumentParser(description="This script runs the super-res ABI model on a full ABI image and saves the ouput to disk")
    parser.add_argument("--model", type=str, default='models/PansharpeningCNN_12_layers_256_filters/', help="Path to the model directory")
    parser.add_argument("--input", type=str, required=True, help="Path to the input data directory")
    parser.add_argument("--output", type=str, default='', help="Path to the output directory")
    parser.add_argument("--timestring", type=str, required=True, help="Timestring value")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    print(f"Path to Model: {args.model}")
    print(f"Path to Input: {args.input}")
    print(f"Path to Output: {args.output}")
    print(f"Timestring: {args.timestring}")
    main(args.model, args.input, args.output, args.timestring)