import os
import glob
import logging
import inspect
import pandas as pd

def parseStandardRepDir(reportDir, protocol, outfilesPrefix, force, outputDirectory, hidden=False):
    """
    Parses the standard report directory to extract relevant statistics and peak information.

    Parameters:
    -----------
    reportDir : str
        The directory containing the report data.
    protocol : str
        The protocol used (e.g., "cutrun", "chip") to determine the specific columns to extract.
    outfilesPrefix : str
        Prefix for the output files.
    force : bool
        If True, the program will overwrite existing files. If False, existing files will be reused.
    outputDirectory : str
        The directory where the output file will be saved.
    hidden : bool, optional
        If True, the output file will be saved with a dot prefix, making it hidden on UNIX-like systems. Default is False.

    Returns:
    --------
    str
        The path to the generated TSV file containing the parsed data.
    
    Raises:
    -------
    SystemExit
        If the protocol is not recognized, the program will abort.
    """
    lgr = logging.getLogger(inspect.currentframe().f_code.co_name)
    lgr.info("Parsing the standard report directory '{}'.".format(reportDir))
    if hidden:
        outfileName = os.path.join(outputDirectory, f".{outfilesPrefix}.meta.tsv")
    else:
        outfileName = os.path.join(outputDirectory, f"{outfilesPrefix}.meta.tsv")

    if os.path.exists(outfileName) and not force:
        lgr.info("The output file '{}' already exists and the force parameter was set to False, so the program will re-use the preexisting file.".format(outfileName))
    else:
        ### Merge the content of all the *.StatsAll.dat files in Stats subdirectory:
        statFiles = []
        for file in glob.glob(os.path.join(reportDir, "Stats", "*.StatsAll.dat")): 
            statFiles.append(file)

        # Read each TSV file into a DataFrame
        dataframes = [pd.read_csv(file, sep='\t') for file in statFiles]

        # Concatenate all DataFrames into one
        combined_df = pd.concat(dataframes)

        # Sort the DataFrame by the 'Sample' column
        sorted_df = combined_df.sort_values(by='Sample')

        # Remove columns that are completely empty
        statsDf = sorted_df.dropna(axis=1, how='all')

        ### Cleaning of the content of the table to get the most relevant protocol-related information included for Grumpy:
        if protocol == "cutrun":
            statsDf = statsDf[["Sample", "Reads", "NonDupMapped", "Mpd%", "Dup%", "Final_Fragments", "<2kb"]].copy()
            statsDf.rename(columns={"Sample": "Sample", "Reads": "Total(M)", "NonDupMapped": "Unique Reads(M)", "Mpd%": "Mapping Rate(%)", "Dup%": "Duplication Rate(%)", "Final_Fragments": "Final Fragments(M)", "<2kb": "<2kb(M)"}, inplace=True)
        elif protocol == "chip":
            statsDf = statsDf[["Sample", "Reads", "NonDupMapped", "Mpd%", "Dup%", "FinalRead", "FragmentSize", "Qtag", "RSC"]].copy()
            statsDf.rename(columns={"Sample": "Sample", "Reads": "Total(M)", "NonDupMapped": "Unique Reads(M)", "Mpd%": "Mapping Rate(%)", "Dup%": "Duplication Rate(%)", "FinalRead": "Final Reads(M)", "FragmentSize": "Fragment Size(bp)", "Qtag": "Qtag", "RSC": "RSC"}, inplace=True)
        else:
            lgr.critical("The protocol '{}' is not recognized. Program was aborted.".format(protocol))
            exit()
        
        ### Scan the "Peaks" subdirectory to identify the number of filered peaks (not FDR50) for each sample, while determining also the type of peaks (broad peaks come from sicer, and narrow peaks come from macs2). Also, the noC_peaks are called without control, while the ones without prefix are called with control, but here we dont care which one.
        statsDf["pkCalling"] = statsDf["Sample"].apply(lambda x: determinePkCalling(reportDir, x))
        statsDf["PeaksControl"] = statsDf["Sample"].apply(lambda x: getPeakNumber(reportDir, x))
        statsDf["PeaksNoControl"] = statsDf["Sample"].apply(lambda x: getPeakNumber(reportDir, x, nocPrefix="noC_"))

        ### Save the DataFrame to a TSV file
        statsDf.to_csv(outfileName, sep='\t', index=False)

        lgr.info("Successfully parsed the report directory with information for {} samples.".format(len(statsDf)))

    return outfileName
