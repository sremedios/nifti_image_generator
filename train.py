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

from models.phinet import phinet
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

    for d in [WEIGHT_DIR, ]:
        if not os.path.exists(d):
            os.makedirs(d)

    ############### MODEL SELECTION ###############

    if results.model:
        model = load_model(results.model)
        model.load_weights(results.weights)
    else:
        model = phinet(model_path=MODEL_PATH,
                       n_classes=len(classes),
                       learning_rate=1e-4,
                       num_channels=1,
                       num_gpus=NUM_GPUS,
                       verbose=0,)

    ############### DATA IMPORT ###############

    # randomly rotate along any axis by 5 degrees
    augmentations = {rotate: {"angle": 5,
                              "direction": np.random.random(3) - 0.5}}


    params = {'target_size': (256, 256, 256),
              'batch_size': 4,
              'class_mode': 'categorical',
              'augmentations': augmentations}

    train_datagen = NIfTIImageDataGenerator()
    test_datagen = NIfTIImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, **params)
    validation_generator = test_datagen.flow_from_directory(VAL_DIR, **params)

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
    es = EarlyStopping(monitor='val_acc', min_delta=1e-8, patience=20)
    callbacks_list.append(es)

    ############### TRAINING ###############
    model.fit_generator(train_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=50,
                        epochs=100000,
                        validation_steps=50,
                        callbacks=callbacks_list)

    # TODO: ensure that the classes learned can be predicted upon

    K.clear_session()
