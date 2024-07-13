import cv2
import numpy as np
import scipy
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import FastICA

class IppgSignalObtainer:

    @staticmethod
    def extractSeriesRoiRGB(video_path, fs, play_video=False, plot = False, window_lenght = 10, start_time = 0):

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
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()

        # Flag to limit the number of frames
        frame_limit = fs*15
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

                    # Convert forehead landmarks to pixel coordinates
                    h, w, _ = frame.shape
                    forehead_points = np.array([(int(l.x * w), int(l.y * h)) for l in forehead_landmarks])

                    margin = 4

                    # Define a custom forehead region
                    forehead_top = forehead_points[1][1] - 4*margin # Top of forehead
                    forehead_left = min(forehead_points[0][0], forehead_points[1][0]) # Leftmost point
                    forehead_right = max(forehead_points[2][0], forehead_points[1][0]) # Rightmost point
                    forehead_bottom = min(forehead_points[0][1], forehead_points[2][1]) - margin # Bottom of forehead

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
    def fftOfIppgFromVideo(video_path, fs, start_frame=0, play_video = False, plot_steps = True):
        
        series = IppgSignalObtainer.extractSeriesRoiRGB(video_path, fs, play_video = play_video, plot = plot_steps)
        fft_series_RGB = []
        series_filtered = []
        colors = ["red","green","blue"]
        for i in range(3):
            
            #OPTIMIZAR, HAGO 3 VECES LA OBTENCION DE LAS FRECUECIAS DE LA FFT
            series_aux = IppgSignalObtainer.__normalize__(series[i], fs, color = colors[i], plot = plot_steps)
            series_aux = IppgSignalObtainer.__bandpass__(series_aux, fs, color = colors[i], plot = plot_steps)
            series_filtered.append(series_aux)
            series_aux = IppgSignalObtainer.__derivativeFilter__(series_aux, fs, color = colors[i], plot = plot_steps)
            
            freqs, fft_series= IppgSignalObtainer.__fft__(series_aux, fs)
            fft_series_RGB.append(fft_series)
            

        
        icaSeries = IppgSignalObtainer.__ICA__(series_filtered)
        
        numComponents = 3
        MaxPx = np.zeros(numComponents)
        icaFFT = np.zeros(numComponents)
        for component in range(numComponents):
            # Compute FFT
            freqs, fft = IppgSignalObtainer.__fft__(icaSeries[:, component], fs)
            icaFFT[component] = fft
            # Calculate power spectrum
            N = len(fft)
            Px = np.abs(fft[1:N//2])**2
            
            # Normalize power spectrum
            Px = Px / np.sum(Px)
            
            # Find maximum normalized power
            MaxPx[component] = np.max(Px)
            
        # Find component with maximum normalized power
        MaxComp = np.argmax(MaxPx)

        icaFftMaxPx = icaFFT[MaxComp]
        icaMaxPx = icaSeries[:, MaxComp]
        icaMaxPx = IppgSignalObtainer.__bandpass__(icaMaxPx, fs, color = "black")

        for i in range(3):
            icaSeries[:,i] =  IppgSignalObtainer.__bandpass__(icaSeries[:,i], fs)
            IppgSignalObtainer.__plotSignal__(icaSeries[:,i], fs, title= "ica " + str(i), color = "black")
            
        
        #return freqs, fft_series_RGB, ica_series
        return freqs, fft_series_RGB

    @staticmethod
    def plotFFT(freqs, fft_series):

        plt.plot(freqs, np.abs(fft_series), color='green')
        plt.title('Frequency Space (FFT)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.show()

    @staticmethod
    def __normalize__(series, fs,  color = "orange", plot = False):
        mean = np.mean(series)
        std = np.std(series)
        normalized_signal = series - mean
        normalized_signal = normalized_signal / std
        if plot:
            IppgSignalObtainer.__plotSignal__(normalized_signal,fs, color, "Normalized signal")
        return normalized_signal
        

    @staticmethod
    def __bandpass__(series, fs, color = "orange", order=3, lowcut=0.5, highcut=4, plot = False):
        b, a = scipy.signal.butter(order, [lowcut, highcut], btype='band', analog=False, fs=fs)
        filtered_signal = scipy.signal.filtfilt(b, a, series)
        if plot:
            IppgSignalObtainer.__plotSignal__(filtered_signal,fs, color, "Bandpass Filter")

        return filtered_signal

    @staticmethod
    def __derivativeFilter__(series, fs,  color = "orange", L = 1, plot = False): #L represents the order of the derivative
        L = 1
        h = np.zeros(2*L + 1)
        h[0] = 1
        h[-1] = -1
        h = h*fs / (2*L)
        filtered_signal = np.convolve(series, h, 'same')
        
        if plot:
            IppgSignalObtainer.__plotSignal__(filtered_signal,fs, color, "Derivative Filter")
            
        return filtered_signal

    @staticmethod
    def __ICA__(series, numComponents = 3):
        #CHECK JADE
        series = np.array(series)
        ica = FastICA(n_components=len(series))
        if numComponents == 3:
            stacked_data = np.vstack((series[0], series[1], series[2])).T
        elif numComponents == 2:
            stacked_data = np.vstack((series[0], series[1])).T
        ica.fit(stacked_data)  # estimated independent sources
        return ica.transform(stacked_data)
        

    @staticmethod
    def __fft__(series, fs):

        fft_series = scipy.fft.fft(series)
        freqs = scipy.fft.fftfreq(len(fft_series), 1/fs)  # Frequency values

        # Keep only first half
        freqs = freqs[0:int(len(freqs)/2)]
        fft_series = fft_series[0:int(len(fft_series)/2)]

        # keep only the frequencies until 4Hz
        indices_until_4Hz = np.where(freqs <= 4)[0]
        freqs = freqs[indices_until_4Hz]
        fft_series = fft_series[indices_until_4Hz]
        #print(IppgSignalObtainer.__getPeakFrequency__(freqs, fft_series)* 60)

        return freqs, fft_series

    @staticmethod
    def __getPeakFrequency__(freqs, fft_series):
        peak_index = np.argmax(np.abs(fft_series))
        return freqs[peak_index]
    
    @staticmethod
    def __plotSignal__(signal, fs, color, title):
            x = np.linspace(0, len(signal)/fs, len(signal))
            plt.plot(x, signal, color=color)
            plt.title(title)
            plt.grid(True)
            plt.show()
    

    #Based on the Matlab implementation proposed by Mcduff et al in https://github.com/danmcduff/iphys-toolbox 
    #Changed the cutoff frequencies of the bandpass filter to 0.5 (30BPM) and 4 (240 BPM)
    @staticmethod
    def Chrom(red : list, blue : list, green : list, fs):

        meanR = np.mean(red)
        meanB = np.mean(blue)
        meanG = np.mean(green)

        Rn = np.array(red)/meanR - 1
        Bn = np.array(blue)/meanB - 1
        Gn = np.array(green)/meanG -1 

        Xs = 3*Rn - 2*Gn
        Ys = 1.5*Rn + Gn - 1.5*Bn

        #optimizar, llamo al bandpass varias veces lo que inicializa el filtro muchas veces.
        Xf = IppgSignalObtainer.__bandpass__(Xs, fs)
        Yf = IppgSignalObtainer.__bandpass__(Ys, fs)

        alpha = np.std(Xf)/np.std(Yf)
        
        S = Xf - alpha*Yf

        return S

    @staticmethod
    def BenchmarkMethods(video_path, fs, play_video = False, green = True, red = False, blue = False, ica = True, chrom = True, frequency = False, scale = True, plot = False):

        series = IppgSignalObtainer.extractSeriesRoiRGB(video_path, fs, play_video = play_video, plot = False)
        fft_series_RGB = []
        seriesRGBFiltered = []
        colors = ("red","green","blue")
        ippgSignals = {}
        #For green and ICA method
        
        for i in range(3):
            
            #OPTIMIZAR, HAGO 3 VECES LA OBTENCION DE LAS FRECUECIAS DE LA FFT
            seriesAux = IppgSignalObtainer.__normalize__(series[i], fs, color = colors[i])
            seriesAux = IppgSignalObtainer.__bandpass__(seriesAux, fs, color = colors[i])
            seriesRGBFiltered.append(seriesAux)
            
            #series_aux = IppgSignalObtainer.__derivativeFilter__(series_aux, fs, color = colors[i], plot = plot_steps)

        #if plot:
        t = np.linspace(0, len(series[0])/fs, len(series[0]))

        def Scale(series):
            minVal = min(series)
            maxVal = max(series)

            # Step 2: Use map and lambda to scale each value
            scaledSeries = list(map(lambda x: (x - minVal) / (maxVal - minVal), series))
            return scaledSeries

        if red:
            ippgSignals["red"] = seriesRGBFiltered[0]

            if plot:
                
                plt.plot(t, seriesRGBFiltered[0], label = "Red", color = "red")
                

        if green:
            ippgSignals["green"] = seriesRGBFiltered[1]
            
            if plot:
                greenScaled = Scale(seriesRGBFiltered[0])
                plt.plot(t, greenScaled, label = "Green", color = "green")
                

        if blue:
            ippgSignals["blue"] = seriesRGBFiltered[2]

            if plot:
                plt.plot(t, seriesRGBFiltered[0], label = "Blue", color = "blue")

        if ica:
            icaSeries = IppgSignalObtainer.__ICA__(seriesRGBFiltered)
            numComponents = 3
            MaxPx = np.zeros(numComponents)

            for component in range(numComponents):
                # Compute FFT
                freqs, fft = IppgSignalObtainer.__fft__(icaSeries[:, component], fs)
                
                # Calculate power spectrum
                N = len(fft)
                Px = np.abs(fft[1:N//2])**2
               
                # Normalize power spectrum
                Px = Px / np.sum(Px)
                
                # Find maximum normalized power
                MaxPx[component] = np.max(Px)
                
            # Find component with maximum normalized power
            MaxComp = np.argmax(MaxPx)
            icaMaxPx = icaSeries[:, MaxComp]
            icaMaxPx = IppgSignalObtainer.__bandpass__(icaMaxPx, fs, color = "black")
            ippgSignals["ica"] = icaMaxPx

            if plot:
                
                icaScaled = Scale(icaMaxPx)
                plt.plot(t, icaScaled, label = "ICA", color = "black")


        if chrom:
            chromSeries = IppgSignalObtainer.Chrom(seriesRGBFiltered[0], seriesRGBFiltered[1], seriesRGBFiltered[2], fs)
            ippgSignals["chrom"] = chromSeries

            if plot:
                chromScaled = Scale(chromSeries)
                plt.plot(t, chromScaled, label = "Chrom", color = "purple")
            
        if frequency:
            freqs, fftSeriesGreen = IppgSignalObtainer.__fft__(ippgSignals["green"], fs)
            print("HR Green: " + str(IppgSignalObtainer.__getPeakFrequency__(freqs, fftSeriesGreen)* 60))
            freqs, fftSeriesICA = IppgSignalObtainer.__fft__(ippgSignals["chrom"], fs)
            print("HR ICA: " + str(IppgSignalObtainer.__getPeakFrequency__(freqs, fftSeriesICA)* 60))
            freqs, fftSeriesCHROM = IppgSignalObtainer.__fft__(ippgSignals["ica"], fs)
            print("HR Chrom: " + str(IppgSignalObtainer.__getPeakFrequency__(freqs, fftSeriesCHROM)* 60))
            ippgSignals["fft green"] = fftSeriesGreen
        
        #for i in range(3):
        #    ica_series[:,i] =  IppgSignalObtainer.__bandpass__(ica_series[:,i], fs)
        #    IppgSignalObtainer.__plotSignal__(ica_series[:,i], fs, title= "ica " + str(i), color = "black")
        if plot:
            
            plt.grid(True)
            plt.show()


        if plot:

            newSeriesIca = [ippgSignals["ica"], ippgSignals["chrom"]]
            numComponents = 2
            icaSeries = IppgSignalObtainer.__ICA__(newSeriesIca, numComponents= numComponents)
            
            MaxPx = np.zeros(numComponents)

            for component in range(numComponents):
                # Compute FFT
                freqs, fft = IppgSignalObtainer.__fft__(icaSeries[:, component], fs)
                
                # Calculate power spectrum
                N = len(fft)
                Px = np.abs(fft[1:N//2])**2
                # Normalize power spectrum
                Px = Px / np.sum(Px)
                
                # Find maximum normalized power
                MaxPx[component] = np.max(Px)
                
            # Find component with maximum normalized power
            MaxComp = np.argmax(MaxPx)
            icaMaxPx = icaSeries[:, MaxComp]
            icaMaxPx = IppgSignalObtainer.__bandpass__(icaMaxPx, fs, color = "black")
            icaScaled = Scale(icaMaxPx)
            plt.plot(t, icaScaled, label = "ICA", color = "black")
            plt.show()


        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(t, seriesRGBFiltered[0])
        axs[0, 0].set_title('green')
        axs[0, 1].plot(t, icaMaxPx, 'tab:orange')
        axs[0, 1].set_title('ICA')
        axs[1, 0].plot(t, chromSeries, 'tab:green')
        axs[1, 0].set_title('CHROM')
        axs[1, 1].plot(freqs, fftSeriesGreen, 'tab:green')
        axs[1, 1].set_title('fft green')


        for ax in axs.flat:
            ax.set(xlabel='x-label', ylabel='y-label')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        plt.show()
        
        return ippgSignals
    
    @staticmethod        
    def __detrendin__(signal):
        pass
