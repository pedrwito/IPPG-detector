
"""POS
Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). 
Algorithmic principles of remote PPG. 
IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
"""

import math
import numpy as np
from scipy import signal
import Tools.utils as utils
from Tools.signalprocesser import signalprocesser


def POS_WANG(RGB, fs, normalize = False, detrend = True, bandpass = True, derivative = False , plot_steps = False):
    
    color = 'purple'
    WinSec = 1.6
    N = RGB.shape[1]
    H = np.zeros((1, N))
    l = math.ceil(WinSec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            Cn = np.true_divide(RGB[:,m:n], np.mean(RGB[:, m:n], axis=0))
            Cn = np.mat(Cn)
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = h[0, temp] - mean_h
            H[0, m:n] = H[0, m:n] + (h[0])

    BVP = H
    if normalize:
        BVP = signalprocesser.normalize(BVP, fs = fs, plot= plot_steps, color = color )
    
    if detrend:
        BVP = signalprocesser.detrend(np.mat(BVP).H, 100, fs = fs, plot= plot_steps, color = color)
    
    BVP = np.asarray(np.transpose(BVP))[0]
    
    if bandpass:
        b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass') #The bandpass filter frequencies are the ones set by Wang et al. We have to see the influence of changing frequencies and filter order
        BVP = signal.filtfilt(b, a, BVP.astype(np.double))
        
    if derivative:
        BVP = signalprocesser.derivativeFilter(BVP, fs = fs, plot= plot_steps, color = color )
        
    return BVP