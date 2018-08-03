'''
Add custom augmentations as standalone functions here to be utilized
when specified in the NIfTIImageGenerator arguments

Or pass them in during the generator param function

TODO: figure out best API 

Rotation code taken from: http://nbviewer.jupyter.org/gist/lhk/f05ee20b5a826e4c8b9bb3e528348688
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.ndimage import map_coordinates
from transformations import rotation_matrix, translation_matrix
from itertools import product, combinations
import nibabel as nib
import os

def pad_image(img_data, target_dims):
    left_pad = round(float(target_dims[0] - img_data.shape[0]) / 2)
    right_pad = round(float(target_dims[0] - img_data.shape[0]) - left_pad)
    top_pad = round(float(target_dims[1] - img_data.shape[1]) / 2)
    bottom_pad = round(float(target_dims[1] - img_data.shape[1]) - top_pad)
    front_pad = round(float(target_dims[2] - img_data.shape[2]) / 2)
    back_pad = round(float(target_dims[2] - img_data.shape[2]) - front_pad)

    pads = ((left_pad, right_pad),
            (top_pad, bottom_pad),
            (front_pad, back_pad))

    new_img = np.zeros((target_dims))
    new_img[:,:,:] = np.pad(img_data[:,:,:], pads, 'constant', constant_values=0)

    return new_img

def rotate(img, angle, direction):
    '''
    img is a 3D image tensor
    angle is in degrees
    direction is a 3D vector in [0,1], axes on which to turn
    '''

    dims = img.shape

    coords = np.meshgrid(np.arange(dims[0]), 
                        np.arange(dims[1]), 
                        np.arange(dims[2]),) 

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz = np.vstack([coords[0].reshape(-1)-float(dims[0])/2, # x coordinate, centered
                     coords[1].reshape(-1)-float(dims[1])/2, # y coordinate, centered
                     coords[2].reshape(-1)-float(dims[2])/2, # z coordinate, centered
                     np.ones((dims[0], dims[1], dims[2])).reshape(-1)]) # homogeneous coordinates

    # create transformation matrix
    mat = rotation_matrix(angle, direction)

    transformed_img = np.dot(mat, xyz)

    # extract coordinates, don't use transformed_xyz[3,:] 
    # that's the homogeneous coordinate, always 1
    x = transformed_img[0, :]+float(dims[0])/2
    y = transformed_img[1, :]+float(dims[1])/2
    z = transformed_img[2, :]+float(dims[2])/2

    x = x.reshape((dims[0], dims[1], dims[2]))
    y = y.reshape((dims[0], dims[1], dims[2]))
    z = z.reshape((dims[0], dims[1], dims[2]))

    # the coordinate system seems to be strange, it has to be ordered like this
    new_xyz = [y, x, z]

    # sample
    new_vol = scipy.ndimage.map_coordinates(img, new_xyz, order=0)

    return new_vol


if __name__ == "__main__":

    ####### BRAIN EXAMPLE #######

    img_path = "1021_1a_MR_T1.nii.gz"

    new_img_path = "rotated_" + img_path
    unrotated_img_path = "unrotated_" + img_path

    nii_obj = nib.load(img_path)
    img = nii_obj.get_data()
    # pad out vertically to ensure stability
    target_dims = (256,256,256)
    img = pad_image(img, target_dims)

    # save padded img to disk
    new_nii_obj = nib.Nifti1Image(img,affine=nii_obj.affine) 
    nib.save(new_nii_obj, "padded_" + img_path)

    angle = 15
    direction=np.random.random(3) - 0.5

    # rotate
    new_vol = rotate(img, angle, direction)

    # save to disk
    new_nii_obj = nib.Nifti1Image(new_vol,affine=nii_obj.affine) 
    nib.save(new_nii_obj, new_img_path)


    # rotate
    unrotated_new_vol = rotate(new_vol, angle, -direction)

    # save
    unrotated_nii_obj = nib.Nifti1Image(unrotated_new_vol,affine=nii_obj.affine) 
    nib.save(unrotated_nii_obj, unrotated_img_path)
