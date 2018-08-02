'''
Samuel Remedios
NIH CC CNRM
Utility functions
'''

import os
import csv
import argparse
import sys
from datetime import datetime

def parse_args(session):
    '''
    Parse command line arguments.

    Params:
        - session: string, one of "train", "validate", or "test"
    Returns:
        - parse_args: object, accessible representation of args
    '''
    parser = argparse.ArgumentParser(
        description="Arguments for Training and Testing")

    if session == "train":
        parser.add_argument('--datadir', required=True, action='store', dest='TRAIN_DIR',
                            help='Where the initial unprocessed data is')
        parser.add_argument('--weightdir', required=True, action='store', dest='OUT_DIR',
                            help='Output directory where the trained models are written')
        parser.add_argument('--numcores', required=True, action='store', dest='numcores',
                            default='1', type=int,
                            help='Number of cores to preprocess in parallel with')
        parser.add_argument('--weights', required=False, action='store', dest='weights',
                            help='Learnt weights (.hdf5) file')
        parser.add_argument('--model', required=False, action='store', dest='model',
                            help='Model Architecture (.json) file')
    elif session == "test":
        parser.add_argument('--infile', required=True, action='store', dest='INFILE',
                            help='Image to classify')
        parser.add_argument('--model', required=True, action='store', dest='model',
                            help='Model Architecture (.json) file')
        parser.add_argument('--weights', required=True, action='store', dest='weights',
                            help='Learnt weights (.hdf5) file')
        parser.add_argument('--result_dst', required=True, action='store', dest='OUTFILE',
                            help='Output filename (e.g. result.csv) to where the results are written')
    elif session == "validate":
        parser.add_argument('--datadir', required=True, action='store', dest='VAL_DIR',
                            help='Where the initial unprocessed data is')
        parser.add_argument('--model', required=True, action='store', dest='model',
                            help='Model Architecture (.json) file')
        parser.add_argument('--weights', required=True, action='store', dest='weights',
                            help='Learnt weights (.hdf5) file')
        parser.add_argument('--result_dst', required=True, action='store', dest='OUT_DIR',
                            help='Output directory where the results are written')
        parser.add_argument('--result_file', required=True, action='store', dest='OUTFILE',
                            help='Output directory where the results are written')
        parser.add_argument('--numcores', required=True, action='store', dest='numcores',
                            default='1', type=int,
                            help='Number of cores to preprocess in parallel with')
    else:
        print("Invalid session. Must be one of \"train\", \"validate\", or \"test\"")
        sys.exit()

    parser.add_argument('--patch_size', required=False, action='store', dest='patch_size',
                        default='45x45x5', type=str,
                        help="Patch size; eg: 45x45x5. Each dimension is separated by 'x'")
    parser.add_argument('--num_patches', required=False, action='store', dest='num_patches',
                        default='100', type=int,
                        help="Number of patches to collect from each image volume")
    parser.add_argument('--classes', required=True, action='store', dest='classes',
                        help='Comma separated list of all classes, CASE-SENSITIVE')
    parser.add_argument('--gpuid', required=False, action='store', type=int, dest='GPUID',
                        help='For a multi-GPU system, the trainng can be run on different GPUs.\
                        Use a GPU id (single number), eg: 1 or 2 to run on that particular GPU.\
                        0 indicates first GPU.  Optional argument. Default is the first GPU.\
                        -1 for all GPUs.')
    parser.add_argument('--delete_preprocessed_dir', required=False, action='store', dest='clear',
                        default='n', help='delete all temporary directories. Enter either y or n. Default is n.')

    return parser.parse_args()

def get_classes(classes):
    '''
    Params:
        - classes: list of strings
    Returns:
        - class_encodings: dictionary mapping an integer to a class_string
    '''
    class_list = classes
    class_list.sort()

    class_encodings = {x: class_list[x] for x in range(len(class_list))}

    return class_encodings

def record_results(csv_filename, args):

    filename, ground_truth, prediction, confidences = args

    if ground_truth is not None:
        fieldnames = [
            "filename",
            "ground_truth",
            "prediction",
            "confidences",
        ]

        # write to file the two sums
        if not os.path.exists(csv_filename):
            with open(csv_filename, 'w') as csvfile:
                fieldnames = fieldnames

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        with open(csv_filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                "filename": filename,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "confidences": confidences,
            })
    else:
        fieldnames = [
            "filename",
            "prediction",
            "confidences",
        ]

        # write to file the two sums
        if not os.path.exists(csv_filename):
            with open(csv_filename, 'w') as csvfile:
                fieldnames = fieldnames

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        with open(csv_filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                "filename": filename,
                "prediction": prediction,
                "confidences": confidences,
            })

def now():
    '''
    Returns a string format of current time, for use in checkpoint filenaming
    '''
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")
