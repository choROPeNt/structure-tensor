## import packages
import os
import argparse
import yaml

## calculation packages
import numpy as np
from structure_tensor import eig_special_3d, structure_tensor_3d

## export packages
import vtk as vtk; from vtk.util import numpy_support
import h5py as h5

def load_config():
    parser = argparse.ArgumentParser(description='structure tensor')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    # Get a device to train on
    return config

def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))

def hdf5_loader(file_path,config):
    import h5py as h5
    raw_internal_path = config["raw_internal_path"]

    with h5.File(file_path, 'r') as hdf5_file:
            # Assuming the dataset is named 'data'
            raw_data = hdf5_file[raw_internal_path][:]

    return raw_data

def raw_loader(file_path,config):

    dtype = np.uint16  
    shape = config["shape"]  ## [z,y,x]
    # Read the binary data from the file into a NumPy array
    try:
        with open(file_path, 'rb') as file:
            raw_data = np.fromfile(file, dtype=dtype)
            # If you have a specific shape, reshape the array accordingly
            if shape:
                raw_data = raw_data.reshape(shape)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        raw_data = None

    return raw_data

def main():
    config = load_config()
    # Define the file path and data type
    file_path = config["file_path"]
    result_path = config["result_path"]
    # Extract filename without the file extension
    filename_without_extension, _ = os.path.splitext(os.path.basename(file_path))

    print("Filename without extension:", filename_without_extension)



    ## Parameters for StuctureTensor analysis
    fiber_diameter = config["fiber_diameter"] # μm
    voxel_size = config["voxel_size"] # μm

    if file_path.endswith('.raw'):
        # Load data from raw file
        raw_data = raw_loader(file_path,config)
    elif file_path.endswith('.hdf5') or file_path.endswith('.h5'):
        # Load data from HDF5 file
        raw_data = hdf5_loader(file_path,config)
    else:
        raise ValueError("Unsupported file format. Supported formats: .raw, .hdf5, .h5")


    if raw_data is not None:
        # Now you have your data in a NumPy array (raw_data)
        print(raw_data.shape,raw_data.dtype,raw_data.nbytes/1024**2)

    ## convert to float64  
    raw_data = raw_data.astype(np.float64)

    ## set parameters for Gaussian Kernel
    r = fiber_diameter / 2 / voxel_size
    sigma = round(np.sqrt(r**2 / 2), 2)
    rho = 4 * sigma

    print('sigma:', sigma)
    print('rho:', rho)

    S = structure_tensor_3d(raw_data, sigma, rho)
    print(f'Structure tensor information is carried in a {S.shape} array.')
    val, vec = eig_special_3d(S, full=False)
    print(f'Orientation information is carried in a {vec.shape} array.')

    ##TODO write export for big data

    # print(vec.nbytes/1024**2)
    # vec = (vec - vec.min()) / (vec.max() - vec.min())
    # vec = (vec*255).astype(np.uint8)
    # print(vec.nbytes/1024**2)
    # convert the data form uint16 to float 32
    
    # print(raw_data.nbytes/1024**2)
    # #normalize the data range 0-1
    # raw_data = (raw_data - raw_data.min()) / (raw_data.max() - raw_data.min())
    # ## convert to uint8 data range 0:255
    # raw_data = (raw_data*255).astype(np.uint8)
    # print(raw_data.nbytes/1024**2)

    
    out = {}
    out['raw'] = raw_data.astype(np.uint8)
    out['vec'] = vec.astype(np.float16)


    with h5.File(os.path.join(result_path,filename_without_extension + ".vec.h5"), 'w') as fout:
        for key in out.keys():
            fout.create_dataset(key, data = out[key],compression="gzip")


if __name__ == '__main__':

    main()