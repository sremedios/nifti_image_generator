'''
Samuel Remedios
NIH CC CNRM
Train PhiNet to classify MRI modalities
'''
import os
import json
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.models import model_from_json

from models.phinet import phinet, phinet_2D
from models.multi_gpu import ModelMGPU

from utils.nifti_image import NIfTIImageDataGenerator
from utils.augmentations import *
from utils.utils import parse_args, now

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    ############### DIRECTORIES ###############

    results = parse_args("train")
    NUM_GPUS = 1
    if results.GPUID == None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif results.GPUID == -1:
        NUM_GPUS = 3
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(results.GPUID)

    TRAIN_DIR = os.path.abspath(os.path.expanduser(results.TRAIN_DIR))
    VAL_DIR = os.path.abspath(os.path.expanduser(results.VAL_DIR))

    classes = results.classes.replace(" ", "").split(',')

    WEIGHT_DIR = os.path.abspath(os.path.expanduser(results.OUT_DIR))

    MODEL_NAME = "phinet_model_" + "-".join(classes)
    MODEL_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME+".json")

    SAMPLE_AUG_PATH = os.path.join("data", "augmented_slices")
    AUG_FILE_PREFIX = "augmented_file"

    for d in [WEIGHT_DIR, SAMPLE_AUG_PATH]:
        if not os.path.exists(d):
            os.makedirs(d)

    patch_size = (45, 45)
    num_patches = 100

    ############### MODEL SELECTION ###############

    '''
    if results.model:
        with open(results.model) as json_data:
            model = model_from_json(json.load(json_data))
        model.load_weights(results.weights)
    '''

    LR = 1e-5
    if len(patch_size) == 2:
        model = phinet_2D(model_path=MODEL_PATH,
                          n_classes=len(classes),
                          learning_rate=LR,
                          num_channels=1,
                          num_gpus=NUM_GPUS,
                          verbose=0,)
    elif len(patch_size) == 3:
        model = phinet(model_path=MODEL_PATH,
                       n_classes=len(classes),
                       learning_rate=LR,
                       num_channels=1,
                       num_gpus=NUM_GPUS,
                       verbose=0,)

    if results.weights:
        model.load_weights(results.weights)

    ############### DATA IMPORT ###############

    # augmentations occur in the order they appear
    train_augmentations = {
        rotate_3D: {"max_angle": 30,
                    "direction_length": 3},
        get_patch_2D: {"patch_size": patch_size,
                       "num_patches": num_patches,
                       "transpose_chance": 0.5},
    }
    val_augmentations = {
        get_patch_2D: {"patch_size": patch_size},
    }

    num_files = 2087
    num_val_files = 600
    batch_size = 16

    params = {
        # 'target_size': (256, 256, 256),
        'target_size': patch_size,
        'batch_size': batch_size,
        'class_mode': 'categorical',
        'num_patches': num_patches,
        # 'axial_slice': 2,
        'save_to_dir': SAMPLE_AUG_PATH,
        'save_prefix': AUG_FILE_PREFIX,
    }

    train_params = {'augmentations': train_augmentations}
    val_params = {'augmentations': val_augmentations}

    train_datagen = NIfTIImageDataGenerator()
    test_datagen = NIfTIImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, **params, **train_params)
    validation_generator = test_datagen.flow_from_directory(
        VAL_DIR, **params, **val_params)

    ############### CALLBACKS ###############

    callbacks_list = []

    # Checkpoint
    WEIGHT_NAME = MODEL_NAME.replace("model", "weights") + "_" +\
        now()+"-epoch-{epoch:04d}-val_acc-{val_acc:.4f}.hdf5"
    fpath = os.path.join(WEIGHT_DIR, WEIGHT_NAME)
    checkpoint = ModelCheckpoint(fpath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 save_weights_only=True)
    callbacks_list.append(checkpoint)

    # Early Stopping, used to quantify convergence
    es = EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=20)
    callbacks_list.append(es)

    ############### TRAINING ###############
    model.fit_generator(train_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=num_files//batch_size,  # total number of images
                        epochs=100000,
                        validation_steps=num_val_files//batch_size,  # total number val images
                        callbacks=callbacks_list)

    # TODO: ensure that the classes learned can be predicted upon

    K.clear_session()
