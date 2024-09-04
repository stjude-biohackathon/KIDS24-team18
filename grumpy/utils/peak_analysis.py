import os
import logging
import inspect
from pybedtools import BedTool

def determinePkCalling(reportDir, sampleName, nocPrefix="noC_", subdirName="Peaks"):
    """
    Determines the type of peak calling (narrow, broad, or both) for a given sample.

    Parameters:
    -----------
    reportDir : str
        The directory containing the report data.
    sampleName : str
        The name of the sample to analyze.
    nocPrefix : str, optional
        Prefix for files without control. Default is "noC_".
    subdirName : str, optional
        Name of the subdirectory where peak files are stored. Default is "Peaks".

    Returns:
    --------
    str
        The type of peak calling detected ("narrow", "broad", "both narrow and broad - check manually", or "unknown").
    
    Raises:
    -------
    SystemExit
        If the `nocPrefix` is set incorrectly, the program will abort.
    """
    lgr = logging.getLogger(inspect.currentframe().f_code.co_name)
    if nocPrefix == "" or nocPrefix == "noC_":
        peakFileNamePattern = os.path.join(reportDir, subdirName, sampleName)
        narrowPeakFile = f"{peakFileNamePattern}.macs2.filter.narrowPeak"
        broadPeakFile = f"{peakFileNamePattern}.sicer.filter.bed"

        peakFileNamePattern = os.path.join(reportDir, subdirName, f"{nocPrefix}{sampleName}")
        narrowPeakFileNoC = f"{peakFileNamePattern}.macs2.filter.narrowPeak"
        broadPeakFileNoC = f"{peakFileNamePattern}.sicer.filter.bed"
       
        if (os.path.exists(narrowPeakFile) or os.path.exists(narrowPeakFileNoC)) and (os.path.exists(broadPeakFile) or os.path.exists(broadPeakFileNoC)):
            return "both narrow and broad - check manually"
        elif os.path.exists(narrowPeakFile) or os.path.exists(narrowPeakFileNoC):
            return "narrow"
        elif os.path.exists(broadPeakFile) or os.path.exists(broadPeakFileNoC):
            return "broad"
        else:
            return "unknown"
    else:
        lgr.critical("The prefix of the peak files should be either empty string or the 'noC_', while here it was set to '{}'. Program was aborted.".format(nocPrefix))
        exit()

def getPeakNumber(reportDir, sampleName, nocPrefix="", subdirName="Peaks"):
    """
    Retrieves the number of peaks for a given sample, either with or without control.

    Parameters:
    -----------
    reportDir : str
        The directory containing the report data.
    sampleName : str
        The name of the sample to analyze.
    nocPrefix : str, optional
        Prefix for files without control. Default is an empty string.
    subdirName : str, optional
        Name of the subdirectory where peak files are stored. Default is "Peaks".

    Returns:
    --------
    int or str
        The number of peaks for the sample, or "n/a" if no peak files are found.
    
    Raises:
    -------
    SystemExit
        If the `nocPrefix` is set incorrectly, the program will abort.
    """
    lgr = logging.getLogger(inspect.currentframe().f_code.co_name)
    if nocPrefix == "" or nocPrefix == "noC_":
        peakFileNamePattern = os.path.join(reportDir, subdirName, f"{nocPrefix}{sampleName}")
        narrowPeakFile = f"{peakFileNamePattern}.macs2.filter.narrowPeak"
        broadPeakFile = f"{peakFileNamePattern}.sicer.filter.bed"

        if os.path.exists(narrowPeakFile):
            peakFileName = narrowPeakFile
        elif os.path.exists(broadPeakFile):
            peakFileName = broadPeakFile
        else:
            return "n/a"
        
        peaks = BedTool(peakFileName)
        return len(peaks)
    else:
        lgr.critical("The prefix of the peak files should be either empty string or the 'noC_', while here it was set to '{}'. Program was aborted.".format(nocPrefix))
        exit()
