import cv2
import numpy as np
import scipy


class IppgSignalObtainer:

    def extractSeriesRoiRGB(self, video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Initialize lists to store time series
        red_series = []
        green_series = []
        blue_series = []

        # Read frames until the end of the video
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            #cv2.imshow("IMAGE",frame)

            # waits for user to press any key
            # (this is necessary to avoid Python kernel form crashing)
            cv2.waitKey(0)

            # closing all open windows
            cv2.destroyAllWindows()

            # Calculate average intensity for each color channel
            red_avg = np.mean(frame[:,:,2])
            green_avg = np.mean(frame[:,:,1])
            blue_avg = np.mean(frame[:,:,0])

            # Append the average intensities to the time series lists
            red_series.append(red_avg)
            green_series.append(green_avg)
            blue_series.append(blue_avg)

        # Release the video capture object
        cap.release()

        return red_series, green_series, blue_series

    def fftIppgRGB(self, video_path, fs):
        
        series = self.extractSeriesRoiRGB(video_path)
        series = self.__bandpass_(series, fs)
        series = self.__derivativeFilter__(series, fs)
        freqs, fft_series = self.__fft__(series, fs)
        
        return freqs, fft_series

    def __bandpass_(self, series, fs, order=3, lowcut=0.5, highcut=3.5):
        b, a = scipy.signal.butter(order, [lowcut, highcut], btype='band', analog=False, fs=fs)
        return scipy.signal.filtfilt(b, a, series)

    def __derivativeFilter__(self, series, fs, L = 1): #L represents the order of the derivative
        L = 1
        h = np.zeros(2*L + 1)
        h[0] = 1
        h[-1] = -1
        h = h*fs / (2*L)
        return np.convolve(series, h, 'same')

    def __fft__(self, series, fs):
        
        fft_series = scipy.fft.fft(series)
        freqs = scipy.fft.fftfreq(len(fft_series), 1/fs)  # Frequency values

        # Keep only first half
        freqs = freqs[0:int(len(freqs)/2)]
        fft_series = fft_series[0:int(len(fft_series)/2)]

        #keep only the frequencies until 4Hz
        indices_until_4Hz = np.where(freqs <= 4)[0]
        freqs = freqs[indices_until_4Hz]
        fft_series = fft_series[indices_until_4Hz]

        return freqs, fft_series

    def __getPeakFrequency__(self, freqs, fft_series):
        peak_index = np.argmax(np.abs(fft_series))
        return freqs[peak_index]
