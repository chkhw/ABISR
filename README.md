# Super-resolution ABI Model

This repository contains code for super-resolving GOES-16 ABI channels to 0.5 km resolution. These scripts are primarily for running inference using already trained models. 


## Installation

Assuming you have anaconda installed, you can build the environment (abisr_test) using the included environment.yml file. Currently, these scripts require a GPU. Depending on your GPU, you may need to install a different version of the CUDA libraries. This was tested on an NVIDIA RTX A6000 and uses Tensorflow 2.15.0

Run this once:

```bash
conda env create -f environment.yml 
```

This will create an environment named 'abisr_test'.

Now, activate the environment. Run this in each new terminal session before running the script:

```bash
conda activate abisr_test
```

## Model weights
TF/Keras model weights used in the paper can be obtained at this link:
https://drive.google.com/drive/folders/1F0MfzEg64oIuCls5knAPwh6Vky9l2pSi?usp=sharing

Make a new directory (ideally named 'models') and extract this file there. 


## Usage
The predict.py script will run the model by tiliing across the entire full-disk image. It will produce 0.5-km resolution files for all 16 ABI channels. The predict.py script can be executed from the command line with the following arguments:

--model: Path to the model directory.

--input: Path to the input data directory. This directory should contain all 16 ABI channels. It can contain multiple times because we specify exactly which time to use with the 'timestring' argument

--output: Path to the output directory (optional, default is current directory). A directory will be created here named 'ABISR_{timestring}' and 15 super-resolved files will be saved here with a suffix: '_05km.nc'

--timestring: This is the file start time that you can find in the name of the ABI L1b files beginning with 'sYYYYDDD....'
This is unique for each timestamp, but should be same for each set of 16 channels. Example: 's20221251520205'

Example command:
```bash
python predict.py --model=models/PansharpeningCNN_12_layers_256_filters/ --input=/path/to/dir/containing/ABI/L1b/ --timestring=s20221251520205
```

The script requires that all 16 L1b files (one for channel) are present in the --input directory. 

## Functionality
Currently only full-disk images from GOES-16 ABI have been tested. The script performs the following steps:

- Loads the specified model.
- Reads ABI data from the input directory.
- Applies the super-resolution model to make all channels at 0.5-km resolution.
- Saves the super-resolved images to the output directory while attempting to mimic the format of the original ABI L1b files.

Runtime is currently ~5 minutes on a modern GPU for a full-disk image, but there is lots of room for improvement on that front. 


## To-do
- Various planned speed optimizations
- Add support for CONUS/Meso sectors
- Train models for a few channel subsets
- Add ability to save only selected channels to disk
- Add ability to specify cut-out of FD/CONUS/Meso sectors to run SR model selectively on that region.

