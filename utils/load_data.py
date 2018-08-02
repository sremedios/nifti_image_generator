'''
Samuel Remedios
NIH CC CNRM
Load images from file
'''

import os
from tqdm import *
import numpy as np
import nibabel as nib
from keras.utils import to_categorical
from sklearn.utils import shuffle

def load_image(filename):
    '''
    Loads a single-channel image and adds a dimension for the implicit "1" dimension
    '''
    img = nib.load(filename).get_data()
    img = np.reshape(img, (1,)+img.shape+(1,))
    #MAX_VAL = 255  # consistent maximum intensity in preprocessing

    # linear scaling so all intensities are in [0,1]
    #return np.divide(img, MAX_VAL)
    return  img

def load_slices(filename):
    '''
    Loads a single-channel image and adds a dimension for the implicit "1" dimension
    Returns a np.array of slices: [slice_idx, height, width, channels==1]
    '''
    img = nib.load(filename).get_data()
    img_slices = np.zeros((img.shape[-1], img.shape[0], img.shape[1], 1), dtype=np.uint8)
    for idx in range(len(img_slices)):
        img_slices[idx, :, :, 0] = img[:,:,idx]

    return img_slices

def load_slice_data(data_dir, classes=None):
    '''
    Loads all 2D image slices from 3D images  and returns them.
    '''

    #################### CLASSIFICATION OF UNKNOWN DATA ####################

    if classes is None:
        filenames = [x for x in os.listdir(data_dir)
                     if not os.path.isdir(os.path.join(data_dir, x))]
        filenames.sort()

        img_shape = nib.load(filenames[0]).get_data().shape
        total_num_slices = len(filenames) * img_shape[-1]
        data = np.zeros(shape=((total_num_slices,) + img_shape[:-1] + (1,)), 
                                dtype=np.uint8)
        all_slice_filenames = [None] * total_num_slices

        for f in tqdm(all_filenames):
            img_slices = load_slices(f)

            for img_slice in img_slices:
                data[indices[cur_idx]] = img_slice
                all_slice_filenames[indices[cur_idx]] = f
                cur_idx += 1

        print(data.shape)
        print(labels.shape)
        return data, all_filenames

    #################### TRAINING OR VALIDATION ####################

    # determine number of classes
    class_directories = [os.path.join(data_dir, x)
                         for x in os.listdir(data_dir)]
    class_directories.sort()

    print(classes)
    num_classes = len(classes)

    # set up all_filenames and class_labels to speed up shuffling
    all_filenames = []
    class_labels = {}
    i = 0
    for class_directory in class_directories:

        if not os.path.basename(class_directory) in classes:
            print("{} not in {}; omitting.".format(
                os.path.basename(class_directory),
                classes))
            continue

        class_labels[os.path.basename(class_directory)] = i
        i += 1
        for filename in os.listdir(class_directory):
            filepath = os.path.join(class_directory, filename)
            all_filenames.append(filepath)

    img_shape = nib.load(all_filenames[0]).get_data().shape

    total_num_slices = len(all_filenames) * img_shape[-1]
    data = np.zeros(shape=((total_num_slices,) + img_shape[:-1] + (1,)), 
                            dtype=np.uint8)
    labels = np.zeros(shape=((total_num_slices,) + (num_classes,)), 
                            dtype=np.uint8)
    all_slice_filenames = [None] * total_num_slices


    indices = np.arange(len(data))
    indices = shuffle(indices, random_state=0)
    cur_idx = 0

    for f in tqdm(all_filenames):
        img_slices = load_slices(f)

        for img_slice in img_slices:
            data[indices[cur_idx]] = img_slice
            cur_label = f.split(os.sep)[-2]
            labels[indices[cur_idx]] = to_categorical(class_labels[cur_label], num_classes=num_classes)
            all_slice_filenames[indices[cur_idx]] = f
            cur_idx += 1

    print(data.shape)
    print(labels.shape)
    return data, labels, all_slice_filenames, num_classes, data[0].shape




def load_data(data_dir, classes=None):
    '''
    Loads in datasets and returns the labeled preprocessed patches for use in the model.

    Determines the number of classes for the problem and assigns labels to each class,
    sorted alphabetically.

    Params:
        - data_dir: string, path to all training class directories
        - task: string, one of modality, T1-contrast, FL-contrast'
        - labels_known: boolean, True if we know the labels, such as for training or
                                 validation.  False if we do not know the labels, such
                                 as loading in data to classify in production
    Returns:
        - data: list of 3D ndarrays, the patches of images to use for training
        - labels: list of 1D ndarrays, one-hot encoding corresponding to classes
        - all_filenames: list of strings, corresponding filenames for use in validation/test
        - num_classes: integer, number of classes
        - img_shape: ndarray, shape of an individual image
    '''

    labels = []

    #################### CLASSIFICATION OF UNKNOWN DATA ####################

    if classes is None:
        all_filenames = []
        data = []
        filenames = [x for x in os.listdir(data_dir)
                     if not os.path.isdir(os.path.join(data_dir, x))]
        filenames.sort()

        for f in tqdm(filenames):
            img = nib.load(os.path.join(data_dir, f)).get_data()
            img = np.reshape(img, img.shape+(1,))
            data.append(img)
            all_filenames.append(f)

        data = np.array(data)

        return data, all_filenames

    #################### TRAINING OR VALIDATION ####################

    # determine number of classes
    class_directories = [os.path.join(data_dir, x)
                         for x in os.listdir(data_dir)]
    class_directories.sort()

    print(classes)
    num_classes = len(classes)

    # set up all_filenames and class_labels to speed up shuffling
    all_filenames = []
    class_labels = {}
    i = 0
    for class_directory in class_directories:

        if not os.path.basename(class_directory) in classes:
            print("{} not in {}; omitting.".format(
                os.path.basename(class_directory),
                classes))
            continue

        class_labels[os.path.basename(class_directory)] = i
        i += 1
        for filename in os.listdir(class_directory):
            filepath = os.path.join(class_directory, filename)
            all_filenames.append(filepath)

    img_shape = nib.load(all_filenames[0]).get_data().shape
    data = np.empty(shape=((len(all_filenames),) +
                           img_shape + (1,)), dtype=np.uint8)

    # shuffle data
    all_filenames = shuffle(all_filenames, random_state=0)

    data_idx = 0  # pointer to index in data

    for f in tqdm(all_filenames):
        img = nib.load(f).get_data()
        img = np.asarray(img, dtype=np.uint8)

        # place this image in its spot in the data array
        data[data_idx] = np.reshape(img, (1,)+img.shape+(1,))
        data_idx += 1

        cur_label = f.split(os.sep)[-2]
        labels.append(to_categorical(
            class_labels[cur_label], num_classes=num_classes))

    labels = np.array(labels, dtype=np.uint8)
    print(data.shape)
    print(labels.shape)
    return data, labels, all_filenames, num_classes, data[0].shape


