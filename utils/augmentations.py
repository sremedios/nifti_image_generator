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
import random
import scipy
from scipy.ndimage import map_coordinates
from .transformations import rotation_matrix, translation_matrix
from itertools import product, combinations
import nibabel as nib
import os
from .pad import pad_image


def get_patch_2D(img, patch_size, num_patches, transpose_chance=0, num_channels=1):
    '''
    Gets num_patches 2D patches from a 3D volume

    img is the 3D image tensor
    patch_size is the size of the 2D patch
    '''
    random.seed()
    mu = 0
    sigma = 10
    center_coords = [x//2 for x in img.shape]

    patches = np.empty((num_patches, *patch_size, num_channels))

    for i in range(num_patches):

        timeout = 50
        patch = np.zeros(patch_size)

        while timeout > 0 and np.sum(patch) < 5:

            horizontal_displacement = int(random.gauss(mu, sigma))
            depth_displacement = int(random.gauss(mu, sigma))
            vertical_displacement = int(random.gauss(mu, sigma))

            c = [center_coords[0] + horizontal_displacement,
                 center_coords[1] + depth_displacement,
                 center_coords[2] + depth_displacement, ]

            if c[0]+patch_size[0]//2 > img.shape[0] or c[0]-patch_size[0]//2 < 0 or\
               c[1]+patch_size[1]//2 > img.shape[1] or c[1]-patch_size[1]//2 < 0 or\
               c[2] > img.shape[2] or c[2] < 0 or\
               img[c[0]-patch_size[0]//2:c[0]+patch_size[0]//2+1,
                   c[1]-patch_size[1]//2:c[1]+patch_size[1]//2+1,
                   c[2]].shape != patch_size or\
               np.sum(img[c[0]-patch_size[0]//2:c[0]+patch_size[0]//2+1,
                          c[1]-patch_size[1]//2:c[1]+patch_size[1]//2+1,
                          c[2]]) < 10:
                timeout -= 1
                if timeout <= 0:
                    print("Failed to find valid patch")
                continue

            patch = img[c[0]-patch_size[0]//2:c[0]+patch_size[0]//2+1,
                        c[1]-patch_size[1]//2:c[1]+patch_size[1]//2+1,
                        c[2]]
            if np.random.random() <= transpose_chance:
                patch = patch.T

        # TODO: handle multiple channels
        patches[i, :, :, 0] = patch

    return patches


def get_patch_3D(img, patch_size):
    '''
    Gets a single 3D patch from a 3D volume

    TODO: make this work for multi patches like above

    img is the 3D image tensor
    patch_size is the size of the 3D patch
    '''
    random.seed()
    mu = 0
    sigma = 40
    center_coords = [x//2 for x in img.shape]

    while True:
        horizontal_displacement = int(random.gauss(mu, sigma))
        depth_displacement = int(random.gauss(mu, sigma))
        vertical_displacement = int(random.gauss(mu, sigma))

        c = [center_coords[0] + horizontal_displacement,
             center_coords[1] + depth_displacement,
             center_coords[2] + depth_displacement, ]

        if not (c[0]+patch_size[0]//2 > img.shape[0] or c[0]-patch_size[0]//2 < 0 or
                c[1]+patch_size[1]//2 > img.shape[1] or c[1]-patch_size[1]//2 < 0 or
                c[2]+patch_size[2]//2 > img.shape[2] or c[2]-patch_size[2]//2 < 0 or
                img[c[0]-patch_size[0]//2:c[0]+patch_size[0]//2+1,
                    c[1]-patch_size[1]//2:c[1]+patch_size[1]//2+1,
                    c[2]-patch_size[2]//2:c[2]+patch_size[2]//2+1, ].shape != patch_size):
            break

    return img[c[0]-patch_size[0]//2:c[0]+patch_size[0]//2+1,
               c[1]-patch_size[1]//2:c[1]+patch_size[1]//2+1,
               c[2]-patch_size[2]//2:c[2]+patch_size[2]//2+1, ]


def rotate_2D(img, max_angle):
    '''
    img is a 2D image tensor
    max_angle is in degrees
    direction is a 2D vector in [0,1], axes on which to turn
    '''

    dims = img.shape
    angle = np.radians(np.random.randint(-max_angle, max_angle))

    coords = np.meshgrid(np.arange(dims[0]),
                         np.arange(dims[1]))

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xy = np.vstack([coords[0].reshape(-1)-float(dims[0])/2,  # x coordinate, centered
                    coords[1].reshape(-1)-float(dims[1])/2, ])  # y coordinate, centered

    # create transformation matrix
    #mat = rotation_matrix(angle, direction)
    mat = np.array([[np.cos(angle), np.sin(angle)],
                    [-np.sin(angle), np.cos(angle)]])

    transformed_img = np.dot(mat, xy)

    # extract coordinates
    x = transformed_img[0, :]+float(dims[0])/2
    y = transformed_img[1, :]+float(dims[1])/2

    x = x.reshape((dims[0], dims[1]))
    y = y.reshape((dims[0], dims[1]))

    # sample
    new_img = scipy.ndimage.map_coordinates(img, [x, y], order=0)

    return new_img


def rotate_3D(img, max_angle, direction_length):
    '''
    img is a 3D image tensor
    angle is in degrees
    direction is a 3D vector in [0,1], axes on which to turn
    '''

    dims = img.shape
    angle = np.radians(np.random.randint(0, max_angle))
    direction = np.random.random(direction_length) - 0.5

    coords = np.meshgrid(np.arange(dims[0]),
                         np.arange(dims[1]),
                         np.arange(dims[2]),)

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz = np.vstack([coords[0].reshape(-1)-float(dims[0])/2,  # x coordinate, centered
                     coords[1].reshape(-1)-float(dims[1]) / \
                     2,  # y coordinate, centered
                     coords[2].reshape(-1)-float(dims[2]) / \
                     2,  # z coordinate, centered
                     np.ones((dims[0], dims[1], dims[2])).reshape(-1)])  # homogeneous coordinates

    # create transformation matrix
    mat = rotation_matrix(np.radians(angle), direction)

    transformed_img = np.dot(mat, xyz)

    # extract coordinates, don't use transformed_xyz[3,:]
    # that's the homogeneous coordinate, always 1
    x = transformed_img[0, :]+float(dims[0])/2
    y = transformed_img[1, :]+float(dims[1])/2
    z = transformed_img[2, :]+float(dims[2])/2

    x = x.reshape((dims[0], dims[1], dims[2]))
    y = y.reshape((dims[0], dims[1], dims[2]))
    z = z.reshape((dims[0], dims[1], dims[2]))

    # sample
    new_vol = scipy.ndimage.map_coordinates(img, [x, y, z], order=0)

    return new_vol


if __name__ == "__main__":

    ####### BRAIN EXAMPLE #######

    img_path = "1021_1a_MR_T1.nii.gz"

    new_img_path = "rotated_" + img_path
    unrotated_img_path = "unrotated_" + img_path

    nii_obj = nib.load(img_path)
    img = nii_obj.get_data()
    # pad out vertically to ensure stability
    target_dims = (256, 256, 256)
    img = pad_image(img, target_dims)

    # save padded img to disk
    new_nii_obj = nib.Nifti1Image(img, affine=nii_obj.affine)
    nib.save(new_nii_obj, "padded_" + img_path)

    angle = 15
    direction = np.random.random(3) - 0.5

    # rotate
    new_vol = rotate(img, angle, direction)

    # save to disk
    new_nii_obj = nib.Nifti1Image(new_vol, affine=nii_obj.affine)
    nib.save(new_nii_obj, new_img_path)

    # rotate
    unrotated_new_vol = rotate(new_vol, angle, -direction)

    # save
    unrotated_nii_obj = nib.Nifti1Image(
        unrotated_new_vol, affine=nii_obj.affine)
    nib.save(unrotated_nii_obj, unrotated_img_path)
