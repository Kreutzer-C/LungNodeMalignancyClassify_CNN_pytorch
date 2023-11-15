import h5py
import numpy as np

f = h5py.File('../dataset/all_patches.hdf5','r')
print(list(f.keys()))

ct_slices = f['ct_slices']
slice_class = f['slice_class']

ct_slices = np.array(ct_slices)
slice_class = np.array(slice_class)

print(ct_slices.shape)
print(slice_class.shape)

np.save("../dataset/ct_images.npy",ct_slices)
