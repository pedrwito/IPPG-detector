import numpy as np
from Tools.signalprocesser import signalprocesser


#Based on the Matlab implementation proposed by Mcduff et al in https://github.com/danmcduff/iphys-toolbox 
#Changed the cutoff frequencies of the bandpass filter to 0.5 (30BPM) and 4 (240 BPM)
def CHROM(seriesRGB, fs, plot_steps = False, normalize = True, detrend = True, bandpass = True, derivative = True):

    color = 'blue'
    red = seriesRGB[0]
    green = seriesRGB[1]
    blue = seriesRGB[2]
    
    meanR = np.mean(red)
    meanG = np.mean(green)
    meanB = np.mean(blue)


    Rn = np.array(red)/meanR - 1
    Gn = np.array(green)/meanG -1 
    Bn = np.array(blue)/meanB - 1


    Xs = 3*Rn - 2*Gn
    Ys = 1.5*Rn + Gn - 1.5*Bn

    #optimizar, llamo al bandpass varias veces lo que inicializa el filtro muchas veces.
    Xf = signalprocesser.bandpass(Xs, fs)
    Yf = signalprocesser.bandpass(Ys, fs)

    alpha = np.std(Xf)/np.std(Yf)
    
    S = Xf - alpha*Yf

    if normalize:
        processedSeries = signalprocesser.normalize(S, fs, plot = plot_steps, color = color)
    
    if detrend:
        processedSeries = signalprocesser.detrend(processedSeries, fs, plot = plot_steps)
        
    if bandpass:
        processedSeries = signalprocesser.bandpass(processedSeries, fs, plot = plot_steps, color = color)
        
    if derivative:
        processedSeries = signalprocesser.derivativeFilter(processedSeries, fs, plot = plot_steps, color = color)

    return S