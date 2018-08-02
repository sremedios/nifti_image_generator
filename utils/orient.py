'''
Samuel Remedios
NIH CC CNRM
Reorient a file
'''

import os

def orient(filename, src_dir, dst_dir, verbose=0, target_orientation="RAI"):
    '''
    Orients image to RAI using 3dresample

    Requires AFNI 3dressample

    Params:
        - filename: string, name of original image
        - src_dir: string, path to data to reorient
        - dst_dir: string, path to where the data will be saved
        - verbose: int, 0 for silent, 1 for verbose
        - target_orientation: string, three letter orientation code for image
    '''
    infile = os.path.join(src_dir, filename)
    outfile = os.path.join(dst_dir, filename)

    if os.path.exists(outfile):
        if verbose:
            print("Already oriented", filename)
        return

    call = "3dresample -orient" + " " + target_orientation + " " +\
            "-inset" + " " + infile + " " +\
            "-prefix" + " " + outfile
    if not verbose:
        call += " >/dev/null 2>&1"
    os.system(call)

    if verbose:
        print("Orientation complete for", filename)
