from IppgSignalObtainer import IppgSignalObtainer
import numpy as np
import matplotlib.pyplot as plt
# Specify the path to the MP4 video file
#video_path = 'googledrive-archive/2. ORIGINAL DATA/P1LC3/P1LC3_edited.avi'
video_path = 'P1LC4_edited.mp4'
fs = 30
#freqs, [fft_red_series, fft_green_series, fft_blue_series] = IppgSignalObtainer.fftOfIppgFromVideo(video_path , fs, play_video = True, plot_steps= True)


IppgSignalObtainer.BenchmarkMethods(video_path = video_path, fs = fs, play_video = False , plot = False, frequency = True)
"""
red_series, green_series, blue_series = IppgSignalObtainer.extractSeriesRoiRGB(video_path , fs, play_video = False, plot = False)



c = IppgSignalObtainer.Chrom(red_series,blue_series,green_series, fs)
asd, asd, ica_series = IppgSignalObtainer.fftOfIppgFromVideo(video_path , fs, play_video = False, plot_steps= False)
x = np.linspace(0, len(c)/fs, len(c))


plt.plot(x, c, label='Chrom', color='red')
plt.grid(True)
plt.show()
plt.plot(x, ica_series[:,0], label='ICA1', color='black')
plt.grid(True)
plt.show()
plt.plot(x, ica_series[:,1], label='ICA2', color='green')
plt.grid(True)
plt.show()
plt.plot(x, ica_series[:,2], label='ICA3', color='yellow')
plt.grid(True)
plt.show()

"""


