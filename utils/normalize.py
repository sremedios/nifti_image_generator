'''
Samuel Remedios
NIH CC CNRM
Normalization
'''

import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate
from scipy.signal import argrelextrema

def normalize_data(img, contrast='T1'):
    '''
    Normalizes 3D images via KDE and clamping
    Params:
        - img: 3D image
    Returns:
        - normalized image
    '''

    if contrast == 'T1':
        CONTRAST = 1
    else:
        CONTRAST = 0

    if (len(np.nonzero(img)[0])) == 0:
        normalized_img = img
    else:
        tmp = np.asarray(np.nonzero(img.flatten()))
        q = np.percentile(tmp, 99.)
        tmp = tmp[tmp <= q]
        tmp = np.asarray(tmp, dtype=float).reshape(-1, 1)

        GRID_SIZE = 80
        bw = float(q) / GRID_SIZE

        kde = KDEUnivariate(tmp)
        kde.fit(kernel='gau', bw=bw, gridsize=GRID_SIZE, fft=True)
        X = 100.*kde.density
        Y = kde.support

        idx = argrelextrema(X, np.greater)
        idx = np.asarray(idx, dtype=int)
        H = X[idx]
        H = H[0]
        p = Y[idx]
        p = p[0]
        x = 0.

        if CONTRAST == 1:
            T1_CLAMP_VALUE = 1.25
            x = p[-1]
            normalized_img = img/x
            normalized_img[normalized_img > T1_CLAMP_VALUE] = T1_CLAMP_VALUE
        else:
            T2_CLAMP_VALUE = 3.5
            x = np.amax(H)
            j = np.where(H == x)
            x = p[j]
            if len(x) > 1:
                x = x[0]
            normalized_img = img/x
            normalized_img[normalized_img > T2_CLAMP_VALUE] = T2_CLAMP_VALUE

    normalized_img /= normalized_img.max()
    return normalized_img
