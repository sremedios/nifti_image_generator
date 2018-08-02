'''
Samuel Remedios
NIH CC CNRM
Predict contrast of an image.
'''

import os
import sys
from tqdm import tqdm
import time
import shutil
import json
from operator import itemgetter
from datetime import datetime
import numpy as np
from sklearn.utils import shuffle

from utils.load_data import load_data, load_image, load_slice_data
from utils.utils import now, parse_args, get_classes, record_results
from utils.preprocess import preprocess_dir
from utils.patch_ops import load_patch_data

from keras.models import load_model, model_from_json
from keras import backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    ############### DIRECTORIES ###############

    results = parse_args("validate")
    NUM_GPUS = 1
    if results.GPUID == None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif results.GPUID == -1:
        NUM_GPUS = 3
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(results.GPUID)

    VAL_DIR = os.path.abspath(os.path.expanduser(results.VAL_DIR))
    CUR_DIR = os.path.abspath(
        os.path.expanduser(
            os.path.dirname(__file__)
        )
    )

    PREPROCESSED_DIR = os.path.join(VAL_DIR, "preprocess")
    if not os.path.exists(PREPROCESSED_DIR):
        os.makedirs(PREPROCESSED_DIR)

    ############### MODEL SELECTION ###############

    with open(results.model) as json_data:
        model = model_from_json(json.load(json_data))
    model.load_weights(results.weights)

    ############### PREPROCESSING ###############

    classes = results.classes.replace(" ", "").split(',')

    preprocess_dir(VAL_DIR,
                   PREPROCESSED_DIR,
                   classes,
                   results.numcores)

    # get class encodings
    class_encodings = get_classes(classes)
    print(class_encodings)

    ############### DATA IMPORT ###############

    patch_size = tuple([int(x) for x in results.patch_size.split('x')])
    X, y, filenames, num_classes, img_shape = load_patch_data(PREPROCESSED_DIR,
                                                              patch_size=patch_size,
                                                              num_patches=results.num_patches,
                                                              classes=classes)

    ############### PREDICT ###############

    PRED_DIR = results.OUT_DIR
    if not os.path.exists(PRED_DIR):
        os.makedirs(PRED_DIR)
    BATCH_SIZE = 2**10

    # make predictions with best weights and save results
    preds = model.predict(X, batch_size=BATCH_SIZE, verbose=1)

    # track overall accuracy
    acc_count = len(set(filenames))
    unsure_count = 0
    total = len(set(filenames))
    total_sure_only = len(set(filenames))

    print("PREDICTION COMPLETE")

    ############### AGGREGATE PATCHES ###############

    print("AGGREGATING RESULTS")
    # initialize aggregate
    final_pred_scores = {}
    final_ground_truth = {}
    pred_shape = preds[0].shape
    for filename in tqdm(set(filenames)):
        final_pred_scores[filename] = np.zeros(pred_shape)

    # posslby faster, must unit test
    from itertools import groupby
    TOTAL_ELEMENTS = len(set(filenames))
    final_pred_scores = {k: v for k, v in
                         (tqdm(map(lambda pair: (pair[0],
                                                 np.mean([p[1] for p in pair[1]], axis=0)),
                                   groupby(zip(filenames, preds), lambda i: i[0])),
                               total=TOTAL_ELEMENTS))}

    print("Num filenames: {}".format(len(filenames)))
    print("Num preds: {}".format(len(preds)))
    print("Shape of y: {}".format(y.shape))
    for i in tqdm(range(len(preds))):
        final_ground_truth[filenames[i]] = y[i]


    print("RECORDING RESULTS")

    ############### RECORD RESULTS ###############
    # mean of all values must be above 80
    surety_threshold = .80

    with open(os.path.join(PRED_DIR, now()+"_results.txt"), 'w') as f:
        with open(os.path.join(PRED_DIR, now()+"_results_errors.txt"), 'w') as e:
            for filename, pred in final_pred_scores.items():

                surety = np.max(pred) - np.min(pred)

                # check for surety
                if surety < surety_threshold:
                    pos = "??"  # unknown
                    f.write("UNSURE for {:<10} with {:<50}".format(
                        pos, filename))
                    unsure_count += 1
                    total_sure_only -= 1
                    acc_count -= 1

                    f.write("{:<10}\t{:<50}".format(pos, filename))
                    confidences = ", ".join(
                        ["{:>5.2f}".format(x*100) for x in pred])
                    f.write("Confidences: {}\n".format(confidences))

                else:
                    # find class of prediction via max
                    max_idx, max_val = max(enumerate(pred), key=itemgetter(1))
                    max_true, val_true = max(
                        enumerate(final_ground_truth[filename]), key=itemgetter(1))
                    pos = class_encodings[max_idx]

                    # record confidences
                    confidences = ", ".join(
                        ["{:>5.2f}".format(x*100) for x in pred])

                    if max_idx == max_true:
                        f.write("CORRECT for {:<10} with {:<50}".format(
                            pos, filename))
                    else:
                        f.write("INCRRCT for {:<10} {:<50}".format(
                            pos, filename))
                        e.write("{:<10}\t{:<50}".format(pos, filename))
                        e.write("Confidences: {}\n".format(confidences))
                        acc_count -= 1

                    f.write("Confidences: {}\n".format(confidences))

            f.write("{} of {} images correctly classified.\nUnsure Number: {}\nAccuracy: {:.2f}\nAccuracy Excluding Unsure: {:.2f}".format(
                str(acc_count),
                str(total),
                str(unsure_count),
                acc_count/total * 100.,
                acc_count/total_sure_only * 100.,))

    print("{} of {} images correctly classified.\nAccuracy: {:.2f}\n".format(
        str(acc_count),
        str(total),
        acc_count/total * 100.))

    # prevent small crash from TensorFlow/Keras session close bug
    K.clear_session()
