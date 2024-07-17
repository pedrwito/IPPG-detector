from IppgSignalObtainer import IppgSignalObtainer
import numpy as np
import matplotlib.pyplot as plt
# Specify the path to the MP4 video file
#video_path = 'googledrive-archive/2. ORIGINAL DATA/P1LC3/P1LC3_edited.avi'


#PATH TO FILES DRIVE
general_path = 'C:/Users/pedro/Documents/IPPG Virtual Sense/ippg-git/googledrive-archive/2. ORIGINAL DATA'
patient =  'P1M1' #'P1H1'
file_video = patient + '_edited.avi'
file_rr = patient + '_Mobi_RR-intervals.rr'

#video_path = 'C:/Users/pedro/Documents/IPPG Virtual Sense/ippg-git/P1LC4_edited.mp4'
video_path = general_path + '/' + patient + '/' + file_video
rr_path =  general_path + '/' + patient + '/' + file_rr

#PATH UMA
video_path = 'C:/Users/pedro/Documents/IPPG Virtual Sense/ippg-git/iggp-uma/TI/' + '0001M_PR_TI_bpm_55,67_spo2_96,67_30.mp4'

#PATH UBFC
folder = '8-gt'
video_path = 'C:/Users/pedro/Documents/IPPG Virtual Sense/ippg-git/'+ folder + '/vid.avi'
GTPath = 'C:/Users/pedro/Documents/IPPG Virtual Sense/ippg-git/'+ folder + '/gtdump.xmp'


#PARAMETERS START AND END
start = 0
end = 30

fs = 30

#freqs, [fft_red_series, fft_green_series, fft_blue_series] = IppgSignalObtainer.fftOfIppgFromVideo(video_path , fs, play_video = True, plot_steps= True)

IppgSignalObtainer.BenchmarkMethods(video_path = video_path, fs = fs,   window_lenght= end - start , start_time = start, play_video = False , plot = False, red = True, blue = True, frequency = True)
peaks = IppgSignalObtainer.HRGroundTruthUBFC(GTPath, 30)
gtTrace, gtTime,gtHR = IppgSignalObtainer.GroundTruthUBFC(GTPath)

t_R = gtTime[peaks]
plt.plot(gtTime, gtTrace)
plt.plot(t_R, gtTrace[peaks], 'or')
plt.show()




#print(IppgSignalObtainer.GetHeartRateFromRRFile(rr_path, start, end))
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


