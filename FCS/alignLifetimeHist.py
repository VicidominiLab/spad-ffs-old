import numpy as np
from movingAverage import movingAverage


def alignLifetimeHist(data, n=15):
    """
    calculate moving average from list of values
    ===========================================================================
    Input       Meaning
    ----------  ---------------------------------------------------------------
    data        data object with microtime histograms
    n           width of the window for the calculation of the moving average
    ===========================================================================
    Output      Meaning
    ----------  ---------------------------------------------------------------
    data        same data object as input, but with histograms aligned in time
    ===========================================================================
    """
    
    histList = [i for i in list(data.__dict__.keys()) if i.startswith('hist')]
    
    for hist in histList:
        # get histogram
        histD = getattr(data, hist)
        # calculate moving average
        IAv = movingAverage(histD[:,1], n=15)
        # calculate derivative
        derivative = IAv[1:] - IAv[0:-1]
        # find maximum of derivative
        maxInd = np.argmax(derivative)
        # shift histogram
        histD[:,1] = np.roll(histD[:,1], -maxInd)
        # store shifted histogram in data object
        setattr(data, hist, histD)
    
    return data
