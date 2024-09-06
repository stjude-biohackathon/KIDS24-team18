import string
import os
import logging
import inspect
from random import choice
from math import ceil

import pandas as pd

from pathlib import Path

import datetime

from grumpy.utils.logger import CustomFormatter
from grumpy.utils.html_processing import extract_section, decodeHTML, format_html, write_html_file, load_html_template
from grumpy.utils.utils import id_generator, load_template, str2bool, caesar_cipher
from grumpy.utils.tokenization import getMaxTokenPerModel
from grumpy.utils.compression import compress_text, decompress_text
from grumpy.utils.peak_analysis import determinePkCalling, getPeakNumber
from grumpy.utils.report_parsing import parseStandardRepDir

def callGrumpyMEA(motifsFile, outfilesPrefix, force, keyFile, apiType, gptModel, hidden=False):
    """
    description

    Parameters:
    -----------
    metaFile : str
        Path to the metafile containing data for QC evaluation.
    protocol : str
        Type of protocol used (e.g., 'cutrun', 'chip'). Determines the basic role for Grumpy.
    protocolFullName : str
        Full name of the protocol used. Required if 'protocol' is set to 'other'.
    outfilesPrefix : str
        Prefix for the output files.
    force : bool
        If True, force the generation of reports even if previous reports exist.
    keyFile : str
        Path to the file containing the API key for Grumpy.
    apiType : str
        Type of API used for Grumpy.
    gptModel : str
        Name of the GPT model used for generating reports.
    outfileName : str
        Name of the file where the full assessment report will be saved.
    outfileNameShort : str
        Name of the file where the concise assessment report will be saved.
    hidden : bool, optional
        If True, certain actions are hidden or not logged. Default is False.

    Returns:
    --------
    None
    """
    from grumpy.connect import grumpyConnect

    # # Initialize logger for this function
    # lgr = logging.getLogger(inspect.currentframe().f_code.co_name)
    # lgr.info("Calling the Grumpy for the standard report metafile '{}'.".format(metaFile))
    
    # ### Renaming old assessments if they already exist
    # if os.path.exists(outfileName):
    #     movedOutfileName = f"{outfileName}.{datetime.datetime.fromtimestamp(os.path.getctime(outfileName)).strftime('%Y%m%d')}.{id_generator()}.txt"
    #     os.rename(outfileName, movedOutfileName)
    #     lgr.info("The output file '%s' already existed, so it was renamed to '%s'.", outfileName, movedOutfileName)
    
    # if os.path.exists(outfileNameShort):
    #     movedOutfileName = f"{outfileNameShort}.{datetime.datetime.fromtimestamp(os.path.getctime(outfileNameShort)).strftime('%Y%m%d')}.{id_generator()}.txt"
    #     os.rename(outfileNameShort, movedOutfileName)
    #     lgr.info("The output file '%s' already existed, so it was renamed to '%s'.", outfileNameShort, movedOutfileName)
    
    ### Define descriptions for the basic role for Grumpy based on the protocol
    # basicRole = load_template(protocol)
    # if protocol == "other":
    basicRole = f"First and foremost, you are here going to analyze the data from the fgfghgfhghf protocol. The following are the general guidelines for the analysis that you have to run, without any specific protocol in mind, so please adjust accordingly to the procol specified here as ''."

    grumpyRole = f"""
   Y ou are an AI assistant that acts as the Computational Biology expert in the area of Epigenetics. Your goal is to help people with the QC evaluation for their data and in providing recommendations. Please be as concise as possible in providing your assessment (not extending 300 words). 
    Moreover, please be as critique, as skeptical and as realistic as possible, I want you to be able to provide focus on the low-quality aspects of the data for the human recipient of your message. If you don't find any issues with the data, don't make them up, instead just please write that it all rather looks good etc.

    Finally, when you mention the actual sample names, always put two vertical bars (i.e. "||") before and after the name, e.g. ||123451_H3K27Ac_rep1||. This is critical for the proper identification of mentioned names by the subsequent script and proper formatting of the report.

    # {basicRole}
    """

    print(basicRole)

    # 1. grumpyRole
    # 2. curatedListOfMotifs
    # 3. outfileName e.g. grumpyMEA.md

    # grumpyConnect(keyFile, apiType, gptModel, grumpyRole, curatedListOfMotifs, outfileName)













    # ### Read the metafile as a simple text file
    # with open(metaFile, 'r') as f:
    #     QC_table = f.read()

    # ### Process the metafile as a pandas DataFrame and evaluate duplication rates
    # df = pd.read_csv(metaFile, sep="\t")
    # dupColName = "ignore"
    # for col in ["DUPLICATION (%)", "Duplication Rate(%)"]:
    #     if col in df.columns:
    #         dupColName = col
    # if dupColName != "ignore":
    #     try:
    #         df[dupColName] = df[dupColName].str.replace("%", "").astype(float)
    #     except AttributeError:
    #         pass
    #     highDuplicationSamples = df[df[dupColName] > 30].shape[0]
    #     if highDuplicationSamples > 0:
    #         highDupNote = f"Additional Note: There are {highDuplicationSamples} samples with duplication rates higher than 30%."
    #         QC_table += f"\n\n{highDupNote}\n"

    # ### Evaluate mapping rates and append notes if applicable
    # mapColName = "ignore"
    # for col in ["Mapping Rate(%)", "MAPPED (%)"]:
    #     if col in df.columns:
    #         mapColName = col
    # if mapColName != "ignore":
    #     try:
    #         df[mapColName] = df[mapColName].str.replace("%", "").astype(float)
    #     except AttributeError:
    #         pass
    #     lowMappingSamples = df[df[mapColName] < 80].shape[0]
    #     if lowMappingSamples > 0:
    #         lowMapNote = f"Additional Note: There are {lowMappingSamples} samples with mapping rates lower than 80%."
    #         QC_table += f"\n\n{lowMapNote}\n"

    ### Connect to Grumpy AI and generate the reports
    # grumpyConnect(keyFile, apiType, gptModel, grumpyRole, QC_table, outfileName)
