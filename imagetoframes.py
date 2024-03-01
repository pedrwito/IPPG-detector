import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy


def extract_time_series(video_path):
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
        
        cv2.imshow("IMAGE",frame) 
  
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


# Specify the path to the MP4 video file
video_path = 'videoP1M3.mp4'

# Extract the time series
red_series, green_series, blue_series = extract_time_series(video_path)

fs = 30
x = np.linspace(0, len(red_series)/fs, len(red_series))

#bandpasss filter
order1 = 3
lowcut = 0.7
highcut = 2.5
b, a = scipy.signal.butter(order1, [lowcut, highcut], btype='band', analog=False, fs=fs)
green_series_bp = scipy.signal.filtfilt(b, a, green_series)
red_series_bp = scipy.signal.filtfilt(b, a, red_series)
blue_series_bp = scipy.signal.filtfilt(b, a, blue_series)

# derivative filter

L = 1
h = np.zeros(2*L + 1)
h[0] = 1
h[-1] = -1
h = h*fs / (2*L)
green_series_derivative = np.convolve(green_series_bp, h, 'same')
red_series_derivative = np.convolve(red_series_bp, h, 'same')
blue_series_derivative = np.convolve(blue_series_bp, h, 'same')

#plt.plot(x, red_series_bp, label='Red', color='red')
plt.plot(x, green_series, label='Green', color='green')
plt.show()
plt.plot(x, red_series, label='Red', color='red')
plt.show()
plt.plot(x, blue_series, label='Blue', color='blue')
plt.show()

# Compute FFT
fft_green_derivative = scipy.fft.fft(green_series_derivative)
fft_red_derivative = scipy.fft.fft(red_series_derivative)
fft_blue_derivative = scipy.fft.fft(blue_series_derivative)
freqs = scipy.fft.fftfreq(len(fft_green_derivative), 1/fs)  # Frequency values
freqs = freqs[0:int(len(freqs)/2)]  # Keep only first half
fft_green_derivative = fft_green_derivative[0:int(len(fft_green_derivative)/2)]
fft_red_derivative = fft_red_derivative[0:int(len(fft_red_derivative)/2)]
fft_blue_derivative = fft_blue_derivative[0:int(len(fft_blue_derivative)/2)]

indices_until_4Hz = np.where(freqs <= 4)[0]
freqs = freqs[indices_until_4Hz]
fft_green_derivative = fft_green_derivative[indices_until_4Hz]
fft_red_derivative = fft_red_derivative[indices_until_4Hz]
fft_blue_derivative = fft_blue_derivative[indices_until_4Hz]


plt.plot(freqs, np.abs(fft_green_derivative), color='green')
plt.title('Frequency Space (FFT)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()

plt.plot(freqs, np.abs(fft_red_derivative), color='red')
plt.title('Frequency Space (FFT)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()

plt.plot(freqs, np.abs(fft_blue_derivative), color='blue')
plt.title('Frequency Space (FFT)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()
