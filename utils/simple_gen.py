'''
Data generator which works with nifti files

TODO: patch collection from images
'''
from keras.utils import to_categorical
from .patch_ops import get_patches
import numpy as np
import nibabel as nib
import os

def generator(list_IDs, batch_size, patch_size, dim, n_channels,
             n_classes, num_patches, class_encodings, shuffle):

    samples_per_epoch = batch_size * num_patches 
    num_batches = samples_per_epoch / (batch_size * num_patches)
    counter = 0

    while True:
        X = np.empty(
            (batch_size * num_patches, *patch_size, n_channels))
        y = np.empty(
            (batch_size * num_patches, n_classes), dtype=int)

        indices = [x for x in range(batch_size)]
        np.random.shuffle(indices)

        for ID in list_IDs:
            # extract a random number of patches in the range:
            cur_num_patches = np.random.randint(1, len(list_IDs)//num_patches)

            img = nib.load(ID).get_data()
            patches = get_patches(img, patch_size, cur_num_patches)

            for patch in patches:
                if len(indices) == 0:
                    break
                cur_idx = indices.pop()
                X[cur_idx] = patch
                y[cur_idx] = to_categorical(class_encodings[ID.split(os.sep)[-2]], 
                                      num_classes=n_classes) 


        yield X, y

        if counter <= num_batches:
            counter = 0
