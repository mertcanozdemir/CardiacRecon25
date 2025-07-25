
import h5py
import numpy as np
from matplotlib import pyplot as plt
import hdf5storage
import os


# The CMRxRecon dataset is distributed as a set of HDF5 files and can be read with the h5py package. Here, we first show how to open a file from the multi-coil dataset. Each file corresponds to ntime * nslice of MRI scan and contains the k-space data.

rootdir = "F:/CMRxRecon2025/ChallengeData/MultiCoil/Cine/TrainingSet/"
file_name = os.path.join(rootdir,"FullSample", "Center005","Siemens_30T_Vida","P003","cine_sax.mat")
hf_m = h5py.File(file_name)


print('Keys:', list(hf_m.keys()))


# here to load the kspace data
newvalue = hf_m['kspace']
fullmulti = newvalue["real"] + 1j*newvalue["imag"]


[nframe, nslice, ncoil, ny, nx] = fullmulti.shape
print(nframe, nslice, ncoil, ny, nx)


# In multi-coil MRIs, k-space has the following shape: 
# 
# #number of frame, number of slices, number of coils, ny, nx (ny is the phase encoding direction for undersampling)
# 
# For single-coil MRIs, k-space has the following shape: #number of frame, number of slices, ny, nx
# 

# choose the first frame and the middle slice
slice_kspace = fullmulti[0,5] 


# Let's see what the absolute value of k-space looks like:

def show_coils(data, slice_nums, cmap=None, vmax = 0.0005):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap,vmax=vmax)


show_coils(np.log(np.abs(slice_kspace) + 1e-9), [0, 4, 9], cmap='gray', vmax = 0.0001)  # This shows coils 0, 4 and 9


import fastmri
from fastmri.data import transforms as T


slice_kspace2 = T.to_tensor(slice_kspace)              # Convert from numpy array to pytorch tensor
slice_image = fastmri.ifft2c(slice_kspace2)           # Apply Inverse Fourier Transform to get the complex image
slice_image_abs = fastmri.complex_abs(slice_image)   # Compute absolute value to get a real image


show_coils(slice_image_abs, [0, 1, 2], cmap='gray', vmax = 0.0005)


# As we can see, each coil in a multi-coil MRI scan focusses on a different region of the image. These coils can be combined into the full image using the Root-Sum-of-Squares (RSS) transform.

slice_image_rss = fastmri.rss(slice_image_abs, dim=0)


plt.imshow(np.abs(slice_image_rss.numpy()), cmap='gray', vmax = 0.0015)


# So far, we have been looking at fully-sampled multi-coil data. We can apply the mask to the kspace and get the undersampled images. 

mask_name = os.path.join(rootdir, "Mask_TaskAll", "Center005","Siemens_30T_Vida","P003","cine_sax_mask_ktRadial8.mat")
mask_s = h5py.File(mask_name)


print('Keys:', list(mask_s.keys()))


# here to load the kspace data, the keys are differnt for different acceleration factors
mask = mask_s['mask']
print(mask.shape)


# apply the mask to the kspace to undersample
underslice = slice_kspace * mask[0,:,:]


plt.imshow(np.abs(underslice[0]), cmap='gray', vmax = 0.001)


# Let's see what the subsampled image looks like:

underslice2 = T.to_tensor(underslice)              # Convert from numpy array to pytorch tensor
sampled_image = fastmri.ifft2c(underslice2)           # Apply Inverse Fourier Transform to get the complex image
sampled_image_abs = fastmri.complex_abs(sampled_image)   # Compute absolute value to get a real image
sampled_image_rss = fastmri.rss(sampled_image_abs, dim=0)


plt.imshow(np.abs(sampled_image_rss.numpy()), cmap='gray',vmax = 0.0015)


# Here we try to save the numpy to mat73 file, please use hdf5storage function. 
# For a large numpy array, it may take a longer time (e.g. up to two minutes for a multi-coil file). 

def savenumpy2mat(data, np_var, filepath):
    ''' 
    np_var: str, the name of the variable in the mat file.
    data: numpy, array to save.
    filepath: str, the path to save the mat file.
    '''
    savedict= {}
    savedict[np_var] = data
    hdf5storage.savemat(filepath, savedict)


savenumpy2mat(underslice, 'underkspace', 'underkslice.mat')


