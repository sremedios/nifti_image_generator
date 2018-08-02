'''
Samuel Remedios
NIH CC CNRM
Robust FOV
'''

import os

def robust_fov(filename, src_dir, dst_dir, verbose=0):
    '''
    Calls FSL's robustfov on all images in the given directory, outputting them 
    into a directory at the same level called "robustfov"

    Params:
        - filename: string, name of file to apply robust_fov to
        - src_dir: string, path to data from which to remove necks
        - dst_dir: string, path to where the data will be saved
        - verbose: int, 0 for silent, 1 for verbose
    '''

    infile = os.path.join(src_dir, filename)
    outfile = os.path.join(dst_dir, filename)

    if os.path.exists(outfile):
        if verbose:
            print("Already applied robustFOV on", filename)
        return

    call = "robustfov -i " + infile + " -r " + outfile  + " -b 160"

    if not verbose:
        call += " >/dev/null 2>&1"
    os.system(call)

    if verbose:
        print("RobustFOV complete for", filename)
