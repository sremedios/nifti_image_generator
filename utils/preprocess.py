'''
Samuel Remedios
NIH CC CNRM
Preprocess files
'''

import os
from multiprocessing.pool import ThreadPool
import shutil
from tqdm import tqdm

from .mri_convert import mri_convert
from .robustfov import robust_fov
from .orient import orient
from .warp3d import warp3d

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'


def preprocess(filename, src_dir, dst_dir, tmp_dir, verbose=1, remove_tmp_files=True):
    '''
    Preprocess a single file.
    Can be used in parallel

    Params:
        - filename: string, path to file to preprocess
        - preprocess_dir: string, path to destination directory to save preprocessed image
        - verbose: int, if 0, surpress all output. If 1, display all output
    '''
    MRI_CONVERT_DIR = os.path.join(tmp_dir, "mri_convert")
    ORIENT_DIR = os.path.join(tmp_dir, "orient")
    ROBUST_FOV_DIR = os.path.join(tmp_dir, "robust_fov")
    WARP_3D_DIR = os.path.join(tmp_dir, "warp3d")

    for d in [MRI_CONVERT_DIR, ORIENT_DIR, ROBUST_FOV_DIR, WARP_3D_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    mri_convert(filename, src_dir, MRI_CONVERT_DIR, verbose=verbose)
    orient(filename, MRI_CONVERT_DIR, ORIENT_DIR, verbose=verbose)
    robust_fov(filename, ORIENT_DIR, ROBUST_FOV_DIR, verbose=verbose)
    warp3d(filename, ROBUST_FOV_DIR, WARP_3D_DIR, verbose=verbose)

    final_preprocessing_dir = WARP_3D_DIR 

    # since the intensities are already [0,255] after warp3d,
    # change the file from float to uchar to save space
    call = "fslmaths " +\
            os.path.join(final_preprocessing_dir, filename) + " "+\
            os.path.join(dst_dir, filename) +\
            " -odt char"

    os.system(call)

    # remove the intermediate steps from each of the preprocessing steps
    if remove_tmp_files:
        for d in [MRI_CONVERT_DIR, ORIENT_DIR, ROBUST_FOV_DIR, WARP_3D_DIR]:
            tmp_file = os.path.join(d, filename)
            if os.path.exists(tmp_file):
                os.remove(tmp_file)


def preprocess_dir(train_dir,
                   preprocess_dir,
                   classes,
                   ncores):
    '''
    Preprocesses all files in train_dir into preprocess_dir using prepreocess.sh

    Params:
        - train_dir: string, path to where all the training images are kept
        - preprocess_dir: string, path to where all preprocessed images will be saved
    '''
    TMPDIR = os.path.join(preprocess_dir,
                          "tmp_intermediate_preprocessing_steps")
    if not os.path.exists(TMPDIR):
        os.makedirs(TMPDIR)

    class_directories = [os.path.join(train_dir, x)
                         for x in os.listdir(train_dir)]
    class_directories.sort()

    print(classes)
    num_classes = len(classes)

    # preprocess all images
    print("*** PREPROCESSING ***")
    for class_dir in class_directories:
        # only operate over specified classes
        if not os.path.basename(class_dir) in classes:
            print("{} not in specified {}; omitting.".format(
                os.path.basename(class_dir),
                classes))
            continue

        # create preprocess dir for this class
        preprocess_class_dir = os.path.join(preprocess_dir,
                                            os.path.basename(class_dir))
        if not os.path.exists(preprocess_class_dir):
            os.makedirs(preprocess_class_dir)

        # only process if we've already done it
        if len(os.listdir(class_dir)) <= len(os.listdir(preprocess_class_dir)):
            print("Already preprocessed.")
            continue

        filenames = [os.path.join(class_dir, x)
                     for x in os.listdir(class_dir)]
        filenames = os.listdir(class_dir)

        # preprocess in parallel using all but one cores (n_jobs=-2)
        tp = ThreadPool(30)
        for f in tqdm(filenames):
            tp.apply_async(preprocess(filename=f,
                                      src_dir=class_dir,
                                      dst_dir=preprocess_class_dir,
                                      tmp_dir=TMPDIR,
                                      verbose=0,
                                      remove_tmp_files=True))
        tp.close()
        tp.join()

    # If the preprocessed data already exists, delete tmp_intermediate_preprocessing_steps
    if os.path.exists(TMPDIR):
        shutil.rmtree(TMPDIR)
