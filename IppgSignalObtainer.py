import cv2
import numpy as np
import scipy
import mediapipe as mp
import matplotlib.pyplot as plt
from Tools.jadeR import jadeR
from scipy import signal
import math

from Methods.pos import POS_WANG
from Methods.green import green
from Methods.ica import ICA
from Methods.chrom import CHROM
import Tools.utils as utils
from Tools.signalprocesser import signalprocesser

class IppgSignalObtainer:

    @staticmethod
    def extractSeriesRoiRGB(video_path, fs, play_video=False, plot = False, window_lenght = 20, start_time = 0):

        if play_video:
            # Flag to check if the video has been played
            video_played = False

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Initialize lists to store time series
        red_series = []
        green_series = []
        blue_series = []

        # Initialize Mediapipe FaceMesh
        
        #Code based on mediapipe doc 
        #https://mediapipe.readthedocs.io/en/latest/solutions/face_mesh.html#:~:text=The%20Face%20Landmark%20Model%20performs,weak%20perspective%20projection%20camera%20model.
        
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

            # Flag to limit the number of frames
            frame_limit = fs*(start_time + window_lenght)
            current_frame = 0
            
            
        
            while True and current_frame < frame_limit:
                ret, frame = cap.read()
                if not ret:
                    # If the video has reached the end, break out of the loop
                    break

                # Convert the frame to RGB for Mediapipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame with Mediapipe FaceMesh
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Extract forehead landmarks
                        forehead_landmarks = [
                            face_landmarks.landmark[10],  # Left forehead
                            face_landmarks.landmark[105],
                            face_landmarks.landmark[334]# Right forehead
                        ]
                        
                        #For denormalization of pixel coordinates, we should multiply x coordinate by width and y coordinate by height.
                        
                        # Convert forehead landmarks to pixel coordinates
                        h, w, _ = frame.shape
                        forehead_points = np.array([(int(l.x * w), int(l.y * h)) for l in forehead_landmarks])

                        margin = 4

                        # Define a custom forehead region
                        forehead_top = forehead_points[1][1] - 2*margin # Top of forehead
                        forehead_left = min(forehead_points[0][0], forehead_points[1][0]) + 2*margin# Leftmost point
                        forehead_right = max(forehead_points[2][0], forehead_points[1][0]) - 2*margin# Rightmost point
                        forehead_bottom = min(forehead_points[0][1], forehead_points[2][1]) + 2*margin # Bottom of forehead

                        roi = frame[forehead_bottom:forehead_top, forehead_left:forehead_right]

                        if play_video:
                            # Draw rectangle around forehead region on the original image
                            cv2.rectangle(frame, (forehead_left, forehead_top), (forehead_right, forehead_bottom), (0, 255, 0), 2)

                    current_frame += 1

                    if current_frame > (start_time)*fs and current_frame < fs*(window_lenght + start_time):
                    # Calculate average intensity for each color channel
                        red_avg = np.mean(roi[:, :, 0])
                        green_avg = np.mean(roi[:, :, 1])
                        blue_avg = np.mean(roi[:, :, 2])

                        # Append the average intensities to the time series lists
                        red_series.append(red_avg)
                        green_series.append(green_avg)
                        blue_series.append(blue_avg)


                    if play_video:
                        # Display the frame with forehead recognition
                        cv2.imshow("Forehead Recognition", frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                        # Check if the window is still open before showing a new frame
                        if cv2.getWindowProperty("Forehead Recognition", cv2.WND_PROP_VISIBLE) < 1:
                            break
                    

            # Release the video capture object
            cap.release()

        if plot:
            x = np.linspace(0, len(red_series)/fs, len(red_series))
            

            plt.plot(x, red_series, label='Red', color='red')
            #plt.show()
            
            plt.plot(x, green_series, label='Green', color='green')
            #plt.show()
            
            plt.plot(x, blue_series, label='Blue', color='blue')
            plt.show()

        return red_series, green_series, blue_series
    

    @staticmethod
    def plotFFT(freqs, fft_series):

        plt.plot(freqs, np.abs(fft_series), color='green')
        plt.title('Frequency Space (FFT)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.show()


    @staticmethod
    def BenchmarkMethods(video_path, fs, play_video = False, start_time = 0, window_lenght = 30, green = True, red = False, blue = False, pos = True, ica = 'jade', chrom = True, frequency = False, plot = False, plotToguether = True):

        seriesRGB = IppgSignalObtainer.extractSeriesRoiRGB(video_path, fs, play_video = play_video, plot = False, start_time= start_time, window_lenght= window_lenght)
        fft_series_RGB = []
        seriesRGBFiltered = []
        colors = ("red","green","blue")
        t = np.linspace(0, len(seriesRGB[0])/fs, len(seriesRGB[0])) #for plotting
        
        ippgSignals = {}
        
        #For green, red and blue method
        for i in range(3):
            
            
            seriesAux = signalprocesser.normalize(seriesRGB[i], fs, color = colors[i])
            seriesAux = signalprocesser.detrend(seriesAux, fs)   
            seriesAux = signalprocesser.bandpass(seriesAux, fs, color = colors[i])
            seriesAux = signalprocesser.derivativeFilter(seriesAux, fs, color = colors[i], plot = plot)
        
            seriesRGBFiltered.append(seriesAux)
        
        #agregar magnificacion euleriana (ver PFC)

        if red:
            ippgSignals["red"] = seriesRGBFiltered[0]

            if plot:
                
                plt.plot(t, ippgSignals["red"], label = "Red", color = "red")
                plt.show()
                
        if green:
            ippgSignals["green"] = seriesRGBFiltered[1]
            
            if plot:

                plt.plot(t, ippgSignals["green"], label = "Green", color = "green")
                plt.show()
                
        if blue:
            ippgSignals["blue"] = seriesRGBFiltered[2]

            if plot:
                plt.plot(t, ippgSignals["blue"], label = "Blue", color = "blue")
                plt.show()
        
        if pos: 
            posSeries = POS_WANG(np.asarray(seriesRGB), fs)
            ippgSignals["pos"] = posSeries
            
            if plot:
                plt.plot(t, ippgSignals["pos"], label = "POS", color = "purple")
                plt.show()
        
        if ica:           
            icaSeries = ICA(seriesRGBFiltered, method = ica, fs = fs,)        
            ippgSignals["ica"] = icaSeries
            
            if plot:
                plt.plot(t, ippgSignals["ica"], label = "ICA", color = "black")
                plt.show()

        if chrom:
            chromSeries = CHROM(seriesRGB = seriesRGB, fs = fs)
            ippgSignals["chrom"] = chromSeries

            if plot:
                plt.plot(t, ippgSignals["chrom"], label = "Chrom", color = "blue")
                plt.show()
            
        if frequency:
            
            if plotToguether:
                fig, axs = plt.subplots(2, 2)
            
            if green:
                freqs, fftSeriesGreen = utils.FFT(ippgSignals["green"], fs)
                print("HR Green: " + str(utils.getPeakFrequencyFFT(freqs, fftSeriesGreen)* 60))
                ippgSignals["fft green"] = fftSeriesGreen
                
                if plotToguether:
                    axs[0, 0].plot(t, ippgSignals["green"],  'tab:green')
                    axs[0, 0].set_title('green')
                
            if ica:
                freqs, fftSeriesICA = utils.FFT(ippgSignals["chrom"], fs)
                print("HR ICA: " + str(utils.getPeakFrequencyFFT(freqs, fftSeriesICA)* 60))
                
                if plotToguether:
                    axs[0, 1].plot(t, ippgSignals["ica"], 'tab:orange')
                    axs[0, 1].set_title('ICA')
            
                
            if chrom:
                freqs, fftSeriesCHROM = utils.FFT(ippgSignals["ica"], fs)
                print("HR CHROM: " + str(utils.getPeakFrequencyFFT(freqs, fftSeriesCHROM)* 60))
                
                if plotToguether:
                    axs[1, 0].plot(t, ippgSignals["chrom"], 'tab:blue')
                    axs[1, 0].set_title('CHROM')
            
            if pos:
                freqs, fftSeriesPOS = utils.FFT(ippgSignals["pos"], fs)
                print("HR POS: " + str(utils.getPeakFrequencyFFT(freqs, fftSeriesPOS)* 60))
            
                if plotToguether:
                    axs[1, 1].plot(t, ippgSignals["pos"], 'tab:purple')
                    axs[1, 1].set_title('POS')
                    
            if plotToguether:
                
                for ax in axs.flat:
                    ax.set(xlabel='x-label', ylabel='y-label')

                # Hide x labels and tick labels for top plots and y ticks for right plots.
                for ax in axs.flat:
                    ax.label_outer()

                plt.show()

        if plotToguether: #Solo con fines visuales de prueba, hay que sacarlo despues o ponerlo en otra funcion
        
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].plot(t, ippgSignals["green"],  'tab:green')
            axs[0, 0].set_title('green')
            
            axs[0, 1].plot(t, ippgSignals["ica"], 'tab:orange')
            axs[0, 1].set_title('ICA')
            
            axs[1, 0].plot(t, ippgSignals["chrom"], 'tab:blue')
            axs[1, 0].set_title('CHROM')
            
            axs[1, 1].plot(t, ippgSignals["pos"], 'tab:purple')
            axs[1, 1].set_title('POS')


            for ax in axs.flat:
                ax.set(xlabel='x-label', ylabel='y-label')

            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in axs.flat:
                ax.label_outer()

            plt.show()
        
        return ippgSignals
        
    def GetHeartRateFromRRFile(file_path : str, start : int = 0, end : int = 30):
        second_column_data = []

        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read each line in the file
            for line in file:
                # Split the line into columns based on whitespace
                columns = line.split()
                # Append the second column (index 1) to the list
                if float(columns[0]) >= start:
                    second_column_data.append(float(columns[1]))
                    
                if float(columns[0]) >= end:
                    break
        
        return 60 / np.mean(second_column_data)
    
    def GroundTruthUBFC(filename):
        gtdata = np.loadtxt(filename, delimiter=',')
    
        # Extract the columns
        gtTrace = gtdata[:, 3]
        gtTime = gtdata[:, 0] / 1000
        gtHR = gtdata[:, 1]
        
        return gtTrace, gtTime,gtHR
    
    def HRGroundTruthUBFC(filename, lenght):
        gtTrace, gtTime, gtHR = IppgSignalObtainer.GroundTruthUBFC(filename)
        h_max = 1*np.mean(gtTrace)
        fs = int(len(gtTime)/gtTime[-1])
        print(fs)
        peaks, _  = signal.find_peaks(gtTrace[:fs*lenght], height = h_max, distance = round(fs*0.30))
        print('HR BVP ground truth: ' + str(len(peaks)/lenght*60))
        
        return peaks
       
       

        
        
                        
        

        
        
        
