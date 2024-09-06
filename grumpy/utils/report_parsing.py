import os
import glob
import logging
import inspect
import pandas as pd

def parseMultiQCReportDir(reportDir, protocol, protocolFullName, keyFile, apiType, gptModel, outfilesPrefix, outputDirectory, globPattern="multiqc_*.txt", hidden=False):
    """
    Function to identify and parse all the summary stat files from MultiQC report directory, the example directory for ATAC-seq could look like this:
    |-- multiqc_data
    |   |-- mqc_fastqc_per_base_n_content_plot_1.txt
    |   |-- mqc_fastqc_per_base_sequence_quality_plot_1.txt
    |   |-- mqc_fastqc_per_sequence_gc_content_plot_Counts.txt
    |   |-- mqc_fastqc_per_sequence_gc_content_plot_Percentages.txt
    |   |-- mqc_fastqc_per_sequence_quality_scores_plot_1.txt
    |   |-- mqc_fastqc_sequence_counts_plot_1.txt
    |   |-- mqc_fastqc_sequence_duplication_levels_plot_1.txt
    |   |-- mqc_fastqc_sequence_length_distribution_plot_1.txt
    |   |-- multiqc.log
    |   |-- multiqc_citations.txt
    |   |-- multiqc_data.json
    |   |-- multiqc_fastqc.txt
    |   |-- multiqc_general_stats.txt
    |   |-- multiqc_macs.txt
    |   |-- multiqc_samtools_flagstat.txt
    |   |-- multiqc_software_versions.txt
    |   `-- multiqc_sources.txt
    `-- multiqc_report.html
    The function parse all the files starting with pattern "multiqc_*.txt" and if they contain the tabular data (TSV), with the first column named "Sample", it will be parsed into a single DataFrame. Then the dataframe is added to the list and the list is returned.

    Parameters:
    -----------
    reportDir : str
        The directory containing the report data.
    globPattern : str, optional
        The glob pattern to match the files to parse. Defaults to "multiqc_*.txt".
    hidden : bool, optional
        If True, the output file will be saved with a dot prefix, making it hidden on UNIX-like systems. Default is False.

    Returns:
    --------
    list
        A list of DataFrames containing the parsed data.

    Raises:
    -------
    SystemExit
        If no files are found matching the glob pattern, the program will abort.
    """
    lgr = logging.getLogger(inspect.currentframe().f_code.co_name)
    lgr.info("Parsing the MultiQC report directory '{}'.".format(reportDir))

    # Search the reportDir recursively for all matching files, even in subdirectories
    files = glob.glob(os.path.join(reportDir, "**", globPattern), recursive=True)

    # Check if any files were found
    if len(files) == 0:
        lgr.critical("No files matching the glob pattern '{}' were found in the directory '{}'. Program was aborted.".format(globPattern, reportDir))
        exit()

    # Initialize an empty list to store the metadata files from multiqc
    metaFiles = []

    if protocol == "other":
        prot = protocolFullName
    else:
        prot = protocol

    # set the output file name prefix
    if hidden:
        outfileNamePRefix = os.path.join(outputDirectory, f".{outfilesPrefix}.")
    else:
        outfileNamePRefix = os.path.join(outputDirectory, f"{outfilesPrefix}.")

    # Loop through each file
    for file in files:
        # Read the TSV file into a DataFrame
        df = pd.read_csv(file, sep='\t')

        # Check if the DataFrame contains a 'Sample' column
        if 'Sample' in list(df):
            # remove the columns with all NaN values
            df = df.dropna(axis=1, how='all')

            # Run basic evaluation for each sample
            # dfEval = assess_sample_quality(df, prot, keyFile, apiType, gptModel, hidden=True, sampleColName='Sample')
            dfEval = df.copy()

            ### Save the DataFrame to a TSV file
            outfileName = f"{outfileNamePRefix}{os.path.basename(file).replace('.txt', '.eval.tsv')}"
            dfEval.to_csv(outfileName, sep='\t')#, index=False)

            lgr.info("Successfully parsed the report directory with information for {} samples.".format(len(dfEval)))

            # Append the DataFrame to the list
            metaFiles.append(outfileName)

    # Return the list of DataFrames
    return metaFiles


# Function to generate a prompt for each sample
def generate_sample_prompt(row, protocol):
    # Dynamically create the list of columns and values
    sample_data = "\n".join([f"- {col}: {row[col]}" for col in row.index])

    # Generate the prompt
    prompt = f"""
    You are given the following sample data:
    {sample_data}

    The protocol used is {protocol}. Based on this data, evaluate the quality of the sample simply describing it as 'High Quality' or 'Low Quality', no other description whatsoever - its critical for downstream processing. Remember to be a critique about it.
    """
    return prompt

# Function to process each row and return the quality assessment
def assess_sample_quality(df, protocol, keyFile, apiType, gptModel, outfileName=".tmpSampleEval.txt", hidden=False, sampleColName='SAMPLE'):
    assessments = []
    from grumpy import grumpyConnect

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():

        # Generate a prompt for the current row
        prompt = generate_sample_prompt(row, protocol)

        grumpyRole = "You are a QC expert for NGS data."

        response = grumpyConnect(keyFile, apiType, gptModel, grumpyRole, prompt, outfileName,
                                 max_tokens=2000, hidden=hidden, temperature=0.1, top_p=0.6, saveResponse=False)

        # Append the result for this row
        assessments.append({
            sampleColName: row[sampleColName],
            'Assessment': response
        })

    # Return a DataFrame with the assessments
    return pd.DataFrame(assessments)

def parseStandardRepDir(reportDir, protocol, protocolFullName, outfilesPrefix, force, outputDirectory, keyFile, apiType, gptModel, hidden=False):
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

    ### Check if the file with "*.report" (e.g. CABDEV-GSE247821-ATACSEQ.2024-08-28_22-13-12.report) exists, if yes parse it because the summary of mapping is already present, if not, parse the StatsAll.dat files becasue we need to assume its standard QC report directory. if more than one report file is found, merge them together:

    if os.path.exists(outfileName) and not force:
        lgr.info("The output file '{}' already exists and the force parameter was set to False, so the program will re-use the preexisting file.".format(outfileName))
    else:
        if len(glob.glob(os.path.join(reportDir, "*.report"))) == 0:
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

        else:

            statsDf = []
            for file in glob.glob(os.path.join(reportDir, "*.report")):
                df = pd.read_csv(file, sep='|', skiprows=2)
                df.columns = df.columns.str.strip()
                df = df.dropna(axis=1, how='all')
                df.iloc[:, 0] = df.iloc[:, 0].astype(str)
                df = df[~df.iloc[:, 0].str.contains('-')]
                statsDf.append(df.copy())
            if len(glob.glob(os.path.join(reportDir, "*.report"))) > 1:
                statsDf = pd.concat(statsDf)
                lgr.info("More than one report file was found in the report directory. The program will merge them together.")
            else:
                statsDf = statsDf[0]

        ### Save the DataFrame to a TSV file
        statsDf.to_csv(outfileName, sep='\t')#, index=False)

        lgr.info("Successfully parsed the report directory with information for {} samples.".format(len(statsDf)))


    return outfileName
