import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft

def plotSignal(series, fs, color, title):
    x = np.linspace(0, len(series)/fs, len(series))
    plt.plot(x, series, color=color)
    plt.title(title)
    plt.grid(True)
    plt.show()
      
def FFT(series, fs):

    fft_series = fft.fft(series)
    freqs = fft.fftfreq(len(fft_series), 1/fs)  # Frequency values

    # Keep only first half
    freqs = freqs[0:int(len(freqs)/2)]
    fft_series = fft_series[0:int(len(fft_series)/2)]

    # keep only the frequencies until 4Hz
    indices_until_4Hz = np.where(freqs <= 4)[0]
    freqs = freqs[indices_until_4Hz]
    fft_series = fft_series[indices_until_4Hz]
    #print(IppgSignalObtainer.__getPeakFrequency__(freqs, fft_series)* 60)

    return freqs, fft_series

def getPeakFrequencyFFT(freqs, fft_series):
    peak_index = np.argmax(np.abs(fft_series))
    return freqs[peak_index]





