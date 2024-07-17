from Tools.signalprocesser import signalprocesser

def green(series, fs, plot_steps = False, normalize = True, detrend = True, bandpass = True, derivative = True):
    
    color = 'green'
    
    if normalize:
        processedSeries = signalprocesser.normalize(series, fs, plot = plot_steps, color = color)
    
    if detrend:
        processedSeries = signalprocesser.detrend(processedSeries, fs, plot = plot_steps)
        
    if bandpass:
        processedSeries = signalprocesser.bandpass(processedSeries, fs, plot = plot_steps, color = color)
        
    if derivative:
        processedSeries = signalprocesser.derivativeFilter(processedSeries, fs, plot = plot_steps, color = color)
    
    
    return processedSeries