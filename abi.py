import numpy as np
import pandas as pd

def convert_raw_to_rad(f, raw):
    '''
    
    '''
    
    # Convert from 'scaled radiance' to true radiance
    sf = f['Rad'].attrs['scale_factor']
    ao = f['Rad'].attrs['add_offset']
    raw = raw.astype('float32') # raw is 'Scaled Radiance'
    rad = raw*sf + ao # now converted to true radiance
    return rad

def convert_raw_to_bt(f, raw):
    # apply scale factor and add offset to convert to radiance
    rad = convert_raw_to_rad(f, raw)
    # Convert radiance to BT
    fk1 = f['planck_fk1'][()]
    fk2 = f['planck_fk2'][()]
    bc1 = f['planck_bc1'][()]
    bc2 = f['planck_bc2'][()]
    T = (fk2 / (np.log((fk1/rad)+1)) - bc1) / bc2
    return T

def convert_raw_to_refl_or_bt(f, raw, band_number=1):
    # If IR band, then use BT func
    if band_number>6:
        return convert_raw_to_bt(f,raw)
    # Otherwise assume VIS, and use refl func
    return convert_raw_to_refl(f, raw)

def convert_raw_to_refl(f, raw):
    # First convert to true radiance
    rad = convert_raw_to_rad(f, raw)
    # Convert radiance to refl (multiply by constant)
    k0 = np.array(f['kappa0'],dtype='float32') 
    refl = rad*k0
    return refl

stats = pd.read_csv('stats.csv', index_col='metric')

idx_2km = [3,5,6,7,8,9,10,11,12,13,14,15]
idx_1km = [0,2,4]
idx_05km = [1]

abi_2km_bands = ['04', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16']
abi_1km_bands = ['01', '03', '05']
abi_05km_bands = ['02']
all_bands = abi_2km_bands + abi_1km_bands + abi_05km_bands
lr_bands = abi_2km_bands + abi_1km_bands
bands = [f'{k:02d}' for k in np.arange(1,17)]
band_sorting = np.argsort(all_bands)

means_2km = np.array([stats[band]['mean'] for band in abi_2km_bands],dtype='float32')
means_1km = np.array([stats[band]['mean'] for band in abi_1km_bands],dtype='float32')
means_05km = np.array([stats[band]['mean'] for band in abi_05km_bands],dtype='float32')
means_lr = np.concatenate([means_2km, means_1km])
means = np.concatenate([means_2km, means_1km, means_05km])

stds_2km = np.array([stats[band]['std'] for band in abi_2km_bands],dtype='float32')
stds_1km = np.array([stats[band]['std'] for band in abi_1km_bands],dtype='float32')
stds_05km = np.array([stats[band]['std'] for band in abi_05km_bands],dtype='float32')
stds_lr = np.concatenate([stds_2km, stds_1km])
stds = np.concatenate([stds_2km, stds_1km, stds_05km])