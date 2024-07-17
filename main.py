import matplotlib.pyplot as plt
from IppgSignalObtainer import IppgSignalObtainer
import numpy as np
# Specify the path to the MP4 video file
video_path = 'C:/Users/pedro/Documents/IPPG Virtual Sense/ippg-git/P1H1_edited.avi'
fs = 30
freqs, [fft_red_series, fft_green_series, fft_blue_series] = IppgSignalObtainer.fftOfIppgFromVideo(video_path , fs)



plt.plot(freqs, np.abs(fft_green_series), color='green')
plt.title('Frequency Space (FFT)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()

plt.plot(freqs, np.abs(fft_red_series), color='red')
plt.title('Frequency Space (FFT)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()

plt.plot(freqs, np.abs(fft_blue_series), color='blue')
plt.title('Frequency Space (FFT)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()