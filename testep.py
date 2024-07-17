from IppgSignalObtainer import IppgSignalObtainer
import numpy as np
import matplotlib.pyplot as plt

#PATH TO FILES
general_path = 'C:/Users/pedro/Documents/IPPG Virtual Sense/ippg-git/googledrive-archive/2. ORIGINAL DATA'
patient = 'P1H1'
file_video = patient + '_edited.avi'
file_rr = patient + '_Mobi_RR-intervals.rr'

#video_path = 'C:/Users/pedro/Documents/IPPG Virtual Sense/ippg-git/P1LC4_edited.mp4'
video_path = general_path + '/' + patient + '/' + file_video
rr_path =  general_path + '/' + patient + '/' + file_rr


#PARAMETERS START AND END
start = 0
end = 30

fs = 30
print(IppgSignalObtainer.GetHeartRateFromRRFile(rr_path, start, end))