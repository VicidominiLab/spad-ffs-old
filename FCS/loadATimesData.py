import numpy as np
import pandas as pd
from FCS2ArrivalTimes import aTimesData
import h5py


def loadATimesData(fname, channels=21):
    """
    Load multichannel arrival times data from .hdf5 file and store in data object
    for further processing
    ===========================================================================
    Input       Meaning
    ----------  ---------------------------------------------------------------
    fname       Filename [.hdf5]
                containing the tables ['det0', 'det1', 'det2', ..., 'det20']
                with each table containing the macro and microtimes
    channels    Either number describing the number of channels (typically 21)
                or list of channels that have to be loaded, e.g. [15, 17, 18]
    ===========================================================================
    Output      Meaning
    ----------  ---------------------------------------------------------------
    data        Data object with a field for each channel
                Each field contains a [Np x 2] np.array
                with Np the number of photons,
                column 1 the absolute macrotimes,
                column 2 the aboslute microtimes
    ===========================================================================
    """
    
    if isinstance(channels, int):
        # total number of channels is given, e.g. 21
        channels = [str(x) for x in range(channels)]
    else:
        # individual channel numbers are given, e.g. [15, 17, 18]
        channels = [str(x) for x in channels]
    
    data = aTimesData()
    
    with h5py.File(fname, 'r') as f:
        for ch in channels:
            print('Loading channel ' + ch)
            setattr(data, 'det' + ch, f['det' + ch][()])
    f.close()
    
    return data


def writeATimesData(data, channels, fname):
    """
    Write multichannel arrival times data to .hdf5
    ===========================================================================
    Input       Meaning
    ----------  ---------------------------------------------------------------
    data        Arrival times data, i.e. output from loadATimesDataPandas()
    channels    Either number describing the number of channels (typically 21)
                or list of channels that have to be loaded, e.g. [15, 17, 18]
    fname       Filename [.hdf5]
    ===========================================================================
    Output      Meaning
    ----------  ---------------------------------------------------------------
    data        Data object with a field for each channel
                Each field contains a [Np x 2] np.array
                with Np the number of photons,
                column 1 the absolute macrotimes,
                column 2 the aboslute microtimes
    ===========================================================================
    """
    if isinstance(channels, int):
        # total number of channels is given, e.g. 21
        channels = [str(x) for x in range(channels)]
    else:
        # individual channel numbers are given, e.g. [15, 17, 18]
        channels = [str(x) for x in channels]
    
    with h5py.File(fname, 'w') as f:
        for ch in range(21):
            f.create_dataset('det' + str(ch), data=getattr(data, 'det' + str(ch)))


def loadATimesDataPandas(fname, chunksize=1000000, macroFreq=240e6):
    """
    Load multichannel arrival times data from h5 file and store in data object
    for further processing
    ===========================================================================
    Input       Meaning
    ----------  ---------------------------------------------------------------
    fname       Filename
    chunksize   Number of rows to read in a single chunk
    macroFreq   Conversion factor to go from relative macrotimes to absolute
                macrotimes
    ===========================================================================
    Output      Meaning
    ----------  ---------------------------------------------------------------
    data        Data object with a field for each channel
                Each field contains a [Np x 2] np.array
                with Np the number of photons,
                column 1 the absolute macrotimes,
                column 2 the aboslute microtimes
    ===========================================================================
    """

    # convert macrotime frequency to macrotime step size
    macroStep = 1 / macroFreq
    
    # read hdf file
    dataR = pd.read_hdf(fname, iterator=True, chunksize=chunksize)
    
    # number of data chunks to read
    Nchunks = int(np.ceil(len(dataR.coordinates) / chunksize))
    
    myIter = iter(dataR)
    
    chunk = 1
    for dataChunk in myIter:
        print('Loading data chunk ' + str(chunk) + '/' + str(Nchunks))
        if chunk == 1:
            # initialize data object
            data = aTimesData()
            listOfChannels = [name[12:] for name in dataChunk.columns if name.startswith('microtime_ch')]
            for chNr in listOfChannels:
                setattr(data, "det" + chNr, np.array([]))
        # go through each channel
        cumstep = dataChunk['cumulative_step']
        for chNr in listOfChannels:
            dataSingleCh = dataChunk['microtime_ch' + chNr]
            microtime = dataSingleCh[dataSingleCh.notna()]
            macrotime = macroStep * cumstep[dataSingleCh.notna()]
            dataSingleCh = np.transpose([macrotime, microtime])
            dataSingleChTot = getattr(data, "det" + chNr)
            setattr(data, "det" + chNr, np.vstack([dataSingleChTot, dataSingleCh]) if dataSingleChTot.size else dataSingleCh)
        chunk += 1
    
    return data


def aTimesH5toHDF5(fname, chunksize=1000000, macroFreq=240e6, channels=21):
    """
    Load multichannel arrival times data from .h5 file, remove NaN and
    store as .hdf5 file
    ===========================================================================
    Input       Meaning
    ----------  ---------------------------------------------------------------
    fname       .h5 filename
    chunksize   Number of rows to read in a single chunk
    macroFreq   Conversion factor to go from relative macrotimes to absolute
                macrotimes
    ===========================================================================
    Output      Meaning
    ----------  ---------------------------------------------------------------
    .hdf5 file
    ===========================================================================
    """
    data = loadATimesDataPandas(fname, chunksize, macroFreq)
    writeATimesData(data, channels, fname[:-3] + '.hdf5')

