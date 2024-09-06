import string
import os
import logging
import inspect
from random import choice
from math import ceil
import glob

import pandas as pd

from pathlib import Path

import datetime

from utils.logger import CustomFormatter
from utils.html_processing import extract_section, decodeHTML, format_html, write_html_file, load_html_template
from utils.utils import id_generator, load_template, str2bool, caesar_cipher
from utils.tokenization import getMaxTokenPerModel
from utils.compression import compress_text, decompress_text
from utils.peak_analysis import determinePkCalling, getPeakNumber
from utils.report_parsing import parseStandardRepDir

from openai import AzureOpenAI, AuthenticationError, OpenAI



def callGrumpyDPKQC(inputDirectory, outputDirectory, outfilesPrefix, force, keyFile, apiType, gptModel, hidden=False):

    from connect import grumpyConnect


    grumpyRole=f"""
In this task, you are focusing on assessing if the biological replicates have high reproducibility for each group. Your goal is to report the biological replicates that have low replicates so that biological scientists can have further investigation. 
Moreover, please be relaxed, for example, if replicate 1 has less than 60% peaks of replicate 2, then these two replicates have low reproducibility. But be as realistic as possible in this task, dont make things up. 

Also, you don't need to give any reason why this happen when the replicates have low reproducibility. 
When you find replicates that have low reproducibilty, try to separate report them in a format below: 
1. **WT_rep1 & WT_rep2 have high reproducibility:**
   - WT_rep1 has 184060 peaks.
   - WT_rep2 has 190000 peaks.
   - Warning: These two replicates need further investigation. 
2. **KO_rep1 & KO_rep2 have low reproducibility:**
   - WT_rep1 has 74060 peaks.
   - WT_rep2 has 20000 peaks.
   - Warning: These two replicates need further investigation.
    """
    
    allReproStats = pd.read_table(glob.glob(inputDirectory + "/reproduciblePeakCounts/*.allReproStats.tsv")[0])
    allReproStats_table = allReproStats.to_csv(index = False, sep = "\t")

    outfileName = outputDirectory + "/" + outfilesPrefix+".ReplicatesQC.md"
    #outfileNameShort = outfilesPrefix+".ReplicatesQC.short.md"

    grumpyConnect(keyFile, apiType, gptModel, grumpyRole, allReproStats_table, outfileName)
    #grumpyConnect(keyFile, apiType, gptModel, grumpyRoleShorter, allReproStats_table, outfileNameShort)



def callGrumpyDPKExtract(inputDirectory, outputDirectory, outfilesPrefix, force, keyFile, apiType, gptModel, context, hidden=False):
    
    from connect import grumpyConnect
    ### load the basicRole for the Grumpy for differentail peak analysis
    basicRole = load_template("dpk")

    ## add additional biological information to extract genes, here the context needs to be a path to a file
    #if os.path.exsit(context):
        #with open(context, 'r') as file:
            #biocontext = file.read()
    #else:
        #biocontext = "ignore"


    contextDescription = f"""
    Additionally, please analyze differential peak analysis results using the following biological context in mind:
    ```
    {context}
    ```"""

    grumpyRole = f"""
You are an AI assistant that acts as the Computational Biology expert in the area of Epigenetics. 
Your goal in this task is to help people check if the differential peak analysis work as expected, as well as help people find out peaks that are relavant to specific biological process.
More importantly, I want you to be as critique, realistic as possible. 
If the contextDescription is not 'ignore', I also need you to describe what biological process that I am interested in at the begining of the report. 
Finally, when you mention regions, always put two vertical bars (i.e. "||") before and after the region, e.g. ||chr1:12345-12867||. This is critical for the proper identification of mentioned names by the subsequent script and proper formatting of the report.

--------
{basicRole}
--------
{contextDescription}
    """

    comparisonFiles = glob.glob(inputDirectory + "/supplementaryFiles/*.vout.anno.Ranks.tsv")
    print(comparisonFiles)
    ## for each comparison, generate a query
    for f in comparisonFiles:
        DF = pd.read_table(f, sep = "\t") ## since the table is already sorted, 
        DF_kept = DF[['Region','Regulation', 'Closest_Gene', 'Distance','FeatureAssignment']]
        Top_DPK_Regions = DF_kept.head(100)
        Bottom_DPK_Regions = DF_kept.tail(100)
        DPK_Regions = pd.concat([Top_DPK_Regions, Bottom_DPK_Regions])
        AnnoRank_table = DPK_Regions.to_csv(index = False, sep = "\t")

        outfileName_precise = outputDirectory + "/" + os.path.basename(f).replace('vout.anno.Ranks.tsv', 'precise.md')
        outfileName_balanced = outputDirectory + "/" + os.path.basename(f).replace('vout.anno.Ranks.tsv', 'balanced.md')
        outfileName_creative= outputDirectory + "/" + os.path.basename(f).replace('vout.anno.Ranks.tsv', 'creative.md')

        ### Running the Grumpy in the Precise mode:
        grumpyConnect(keyFile, apiType, gptModel, grumpyRole, AnnoRank_table, outfileName_precise,
                     hidden=hidden, temperature=0.1, top_p=0.6)

        ### Running the Grumpy in the Balanced mode:
        grumpyConnect(keyFile, apiType, gptModel, grumpyRole, AnnoRank_table, outfileName_balanced,
                    hidden=hidden, temperature=0.5, top_p=0.8)

        ### Running the Grumpy in the Creative mode:
        grumpyConnect(keyFile, apiType, gptModel, grumpyRole, AnnoRank_table, outfileName_creative,
                     hidden=hidden, temperature=0.85, top_p=0.9)

