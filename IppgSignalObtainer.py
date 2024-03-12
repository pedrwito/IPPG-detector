import cv2
import numpy as np
import scipy
import mediapipe as mp
import matplotlib.pyplot as plt


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
        frame_limit = fs*20
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

            if current_frame > start_time*fs and current_frame < fs*(window_lenght + start_time):
            # Calculate average intensity for each color channel
                red_avg = np.mean(roi[:, :, 2])
                green_avg = np.mean(roi[:, :, 1])
                blue_avg = np.mean(roi[:, :, 0])

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
            x = np.linspace(0, len(red_series)/30, len(red_series))
            
            plt.plot(x, green_series, label='Green', color='green')
            plt.show()
            plt.plot(x, red_series, label='Red', color='red')
            plt.show()
            plt.plot(x, blue_series, label='Blue', color='blue')
            plt.show()

        return red_series, green_series, blue_series
    
    @staticmethod
    def fftOfIppgFromVideo(video_path, fs, start_frame=0, plot_steps = True):

        series = IppgSignalObtainer.extractSeriesRoiRGB(video_path, fs, plot = plot_steps)
        fft_series_RGB = []
        for i in range(3):
            
            #OPTIMIZAR, HAGO 3 VECES LA OBTENCION DE LAS FRECUECIAS DE LA FFT
            series_aux = IppgSignalObtainer.__bandpass_(series[i], fs, plot = plot_steps)
            series_aux = IppgSignalObtainer.__derivativeFilter__(series_aux, fs, plot = plot_steps)
            freqs, fft_series= IppgSignalObtainer.__fft__(series_aux, fs)
            fft_series_RGB.append(fft_series)

        return freqs, fft_series_RGB

    @staticmethod
    def plotFFT(freqs, fft_series):

        plt.plot(freqs, np.abs(fft_series), color='green')
        plt.title('Frequency Space (FFT)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.show()

    @staticmethod
    def __bandpass_(series, fs, order=3, lowcut=0.5, highcut=3.5, plot = False):
        b, a = scipy.signal.butter(order, [lowcut, highcut], btype='band', analog=False, fs=fs)
        filtered_signal = scipy.signal.filtfilt(b, a, series)
        if plot:
            x = np.linspace(0, len(filtered_signal)/fs, len(filtered_signal))
            plt.plot(x, filtered_signal, label='Bandpass filter', color='orange')
            plt.show()

        return filtered_signal

    @staticmethod
    def __derivativeFilter__(series, fs, L = 1, plot = False): #L represents the order of the derivative
        L = 1
        h = np.zeros(2*L + 1)
        h[0] = 1
        h[-1] = -1
        h = h*fs / (2*L)
        filtered_signal = np.convolve(series, h, 'same')
        
        if plot:
            x = np.linspace(0, len(filtered_signal)/fs, len(filtered_signal))
            plt.plot(x, filtered_signal, label='Derivative filter', color='orange')
            plt.show()
            
        return filtered_signal

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

        return freqs, fft_series

    @staticmethod
    def __getPeakFrequency__(freqs, fft_series):
        peak_index = np.argmax(np.abs(fft_series))
        return freqs[peak_index]
