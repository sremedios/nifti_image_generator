'''
Samuel Remedios
NIH CC CNRM
MRI convert a file
'''

import os

def mri_convert(filename, src_dir, dst_dir, verbose=0):
    '''
    Converts image to be 256^3 mm^3 coronal with intensity range [0, 255]

    Requires mri_convert 

    Params:
        - filename: string, name of original image
        - src_dir: string, path to data to reorient
        - dst_dir: string, path to where the data will be saved
        - verbose: int, 0 for silent, 1 for verbose
    '''
    infile = os.path.join(src_dir, filename)
    outfile = os.path.join(dst_dir, filename)

    if os.path.exists(outfile):
        if verbose:
            print("Already mri_converted", filename)
        return

    call = "mri_convert -odt uchar --crop 0 0 0 -c" + " " +\
            infile + " " + outfile
    if not verbose:
        call += " >/dev/null 2>&1"
    os.system(call)

    if verbose:
        print("Orientation complete for", filename)
