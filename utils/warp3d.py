'''
Samuel Remedios
NIH CC CNRM
3d Warp
'''

import os

def warp3d(filename, src_dir, dst_dir, verbose=0):
    '''
    Calls FSL's 3dwarp 

    This makes images AC-PC alinged.
    Ideally images should be rigid registered to some template for uniformity, 
    but rigid registration is slow.  This is a faster way.

    NOTE: Resample is now deprecated
    The -newgrid 2 resamples the image to 2mm^3 resolution

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
            print("Already applied 3dwarp to", filename)
        return

    #call = "3dWarp -deoblique -NN -newgrid 2 -prefix" + " " + outfile + " " + infile
    call = "3dWarp -deoblique -NN -prefix" + " " + outfile + " " + infile

    if not verbose:
        call += " >/dev/null 2>&1"
    os.system(call)

    if verbose:
        print("3Dwarp complete for", filename)
