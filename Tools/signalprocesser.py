import Tools.utils as utils 
import numpy as np
import scipy.signal as signal
from scipy import sparse

class signalprocesser():
    
    def bandpass(series, fs, color = "orange", order=3, lowcut=0.5, highcut=4, plot = False):
        b, a = signal.butter(order, [lowcut, highcut], btype='band', analog=False, fs=fs)
        filtered_series = signal.filtfilt(b, a, series)
        if plot:
            utils.plotSignal(filtered_series,fs, color, "Bandpass Filter")

        return filtered_series
    
    def derivativeFilter(series, fs,  color = "orange", L = 1, plot = False): #L represents the order of the derivative
        L = 1
        h = np.zeros(2*L + 1)
        h[0] = 1
        h[-1] = -1
        h = h*fs / (2*L)
        filtered_series = np.convolve(series, h, 'same')
        
        if plot:
            utils.plotSignal(filtered_series,fs, color, "Derivative Filter")
            
        return filtered_series
    

    def normalize(series, fs,  color = "orange", plot = False):
        mean = np.mean(series)
        std = np.std(series)
        normalized_series = series - mean
        normalized_series = normalized_series / std
        
        if plot:
            utils.plotSignal(normalized_series,fs, color, "Normalized signal")
        
        return normalized_series


    def detrend(series, lambda_value = 100, fs = 30, method = 'mcduff', color = 'orange', plot= False):
        
        if method == 'mcduff':
            series_length = series.shape[0]
            # observation matrix
            H = np.identity(series_length)
            ones = np.ones(series_length)
            minus_twos = -2 * np.ones(series_length)
            diags_data = np.array([ones, minus_twos, ones])
            diags_index = np.array([0, 1, 2])
            D = sparse.spdiags(diags_data, diags_index,
                        (series_length - 2), series_length).toarray()
            detrended_series = np.dot(
                (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), series)
            
        if method == 'haan':
            #Use detrendign method proposed by de Haan, G. and Van Leest, A. (2014). 
            # Improved motion robustness of remote-PPG by using the blood volume pulse signature, Physiological measurement 35(9): 1913.
            #L is assumed as 1 second
            L = L * fs       
            
            detrended_series = np.zeros(len(signal))
            
            for i in range(len(signal)):
                if i < L:
                    L_aux = i
                else: 
                    L_aux = L
                m = np.sum(signal[i-L_aux:i])
                detrended_series[i]= signal[i] - m / m
            
        if plot:
            utils.plotSignal(detrended_series, fs, color, "Detrended signal")
            
        return detrended_series
    
    def ScaleMinMax(series):
        minVal = min(series)
        maxVal = max(series)

        # Step 2: Use map and lambda to scale each value
        scaledSeries = list(map(lambda x: (x - minVal) / (maxVal - minVal), series))
        return scaledSeries