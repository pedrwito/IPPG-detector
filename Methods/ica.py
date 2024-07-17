import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from Tools.jadeR import jadeR
import Tools.utils as utils
from Tools.signalprocesser import signalprocesser

def ICA(seriesRGB, fs, plot_steps = False,  method = 'jade', normalize = True, detrending = True, bandpass = True, derivate = True):

    color = 'black'
    seriesRGB = np.array(seriesRGB)
    
    if method == 'jade':   
        B = jadeR(np.array(seriesRGB), 3)
        icaSeries = np.dot(np.transpose(np.array(seriesRGB)),B)
        
        #I select the second component according to bibliography and for the sake of automation
        icaBest = np.array(icaSeries[:,1]).flatten()
        
    elif method == 'fastica':  
        numComponents = 3         
        MaxPx = np.zeros(numComponents)
        ica = FastICA(n_components=len(seriesRGB))

        if numComponents == 3:
        
            stacked_data = np.vstack((seriesRGB[0], seriesRGB[1], seriesRGB[2])).T

        ica.fit(stacked_data)  # estimated independent sources  
        icaSeries = ica.transform(stacked_data)
        
        for component in range(numComponents):
            # Compute FFT
            freqs, fft = utils.FFT(icaSeries[:, component], fs)
            
            # Calculate power spectrum
            N = len(fft)
            Px = np.abs(fft[1:N//2])**2
            
            # Normalize power spectrum
            Px = Px / np.sum(Px)
            
            # Find maximum normalized power
            MaxPx[component] = np.max(Px)
            
        # Find component with maximum normalized power
        MaxComp = np.argmax(MaxPx)
        icaBest = icaSeries[:, MaxComp]
        
    if normalize:
        icaBest = signalprocesser.normalize(icaBest, fs = fs, plot = plot_steps, color = color)
        
    if detrending:
        icaBest = signalprocesser.detrend(icaBest, fs = fs, plot = plot_steps, color = color)

    if bandpass: 
        icaBest = signalprocesser.bandpass(icaBest, fs = fs, plot = plot_steps,color = color)
        
    if derivate:
        icaBest = signalprocesser.derivativeFilter(icaBest, fs, plot = plot_steps, color = color)
        
    return icaBest

            
            
            
            
        
        