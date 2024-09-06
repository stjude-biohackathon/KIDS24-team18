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

from grumpy.utils.logger import CustomFormatter
from grumpy.utils.html_processing import extract_section, decodeHTML, format_html, write_html_file, load_html_template
from grumpy.utils.utils import id_generator, load_template, str2bool, caesar_cipher
from grumpy.utils.tokenization import getMaxTokenPerModel
from grumpy.utils.compression import compress_text, decompress_text
from grumpy.utils.peak_analysis import determinePkCalling, getPeakNumber
from grumpy.utils.report_parsing import parseStandardRepDir

from openai import AzureOpenAI, AuthenticationError, OpenAI



def callGrumpyDPKQC(inputDirectory, outfilesPrefix, force, keyFile, apiType, gptModel, hidden=False):

    from grumpy.connect import grumpyConnect


    grumpyRole=f"""
In this task, you are focusing on assessing if the biological replicates have high reproducibility for each group. Your goal is to report the biological replicates that have low replicates so that biological scientists can have further investigation. 
Moreover, please be relaxed, for example, if replicate 1 has less than 60% peaks of replicate 2, then these two replicates have low reproducibility. But be as realistic as possible in this task, dont make things up. 

Also, just report the replicates having low reproducibility, you don't need to give any reason why this happen. If the biological replicates have high reproducibility, you don't need report them for the sake of consice information.
When you find replicates that have low reproducibilty, try to separate report them in a format below: 
1. **WT_rep1 & WT_rep2 have low reproducibility:**
   - WT_rep1 has 184060 peaks.
   - WT_rep2 has 90000 peaks.
   - Warning: These two replicates need further investigation. 
2. **KO_rep1 & KO_rep2 have low reproducibility:**
   - WT_rep1 has 74060 peaks.
   - WT_rep2 has 20000 peaks.
   - Warning: These two replicates need further investigation.
    """
    
    allReproStats = pd.read_table(glob.glob(inputDirectory + "/reproduciblePeakCounts/*.allReproStats.tsv")[0])
    allReproStats_table = allReproStats.to_csv(index = False, sep = "\t")

    outfileName = outfilesPrefix+".ReplicatesQC.md"
    #outfileNameShort = outfilesPrefix+".ReplicatesQC.short.md"

    grumpyConnect(keyFile, apiType, gptModel, grumpyRole, allReproStats_table, outfileName)
    #grumpyConnect(keyFile, apiType, gptModel, grumpyRoleShorter, allReproStats_table, outfileNameShort)




def callGrumpyDPKExtract(inputDirectory, outfilesPrefix, force, keyFile, apiType, gptModel, context, hidden=False):
    
    from grumpy import grumpyConnect
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
--------
{basicRole}
--------
{contextDescription}
    """

    comparisonFiles = glob.glob(inputDirectory + "/supplementaryFiles/*.vout.anno.Ranks.tsv")
    print(comparisonFiles)
    ## for each comparison, generate a query
    for f in comparisonFiles:
        DF = pd.read_table(f, sep = "\t")
        DF_kept = DF[['Region','log2FC', 'log2AveExpr', 't.value', 'p.value', 'q.value', 'Regulation', 'Gene_2kb', 'Gene_2-50kb', 'Closest_Gene', 'Distance','FeatureAssignment']]
        AnnoRank_table = DF_kept.to_csv(index = False, sep = "\t")

        outfileName_precise = os.path.basename(f).replace('vout.anno.Ranks.tsv', '.precise.md')
        outfileName_balanced = os.path.basename(f).replace('vout.anno.Ranks.tsv', '.balanced.md')
        outfileName_creative= os.path.basename(f).replace('vout.anno.Ranks.tsv', '.creative.md')

        #outfileName_precise = outfilesPrefix + '.precise.md'
        #outfileName_balanced = outfilesPrefix '.balanced.md'
        #outfileName_creative= outfilesPrefix '.creative.md'

        ### Running the Grumpy in the Precise mode:
        grumpyConnect(keyFile, apiType, gptModel, grumpyRole, AnnoRank_table, outfileName_precise,
                     hidden=hidden, temperature=0.1, top_p=0.6)

        ### Running the Grumpy in the Balanced mode:
        grumpyConnect(keyFile, apiType, gptModel, grumpyRole, AnnoRank_table, outfileName_balanced,
                    hidden=hidden, temperature=0.5, top_p=0.8)

        ### Running the Grumpy in the Creative mode:
        grumpyConnect(keyFile, apiType, gptModel, grumpyRole, AnnoRank_table, outfileName_creative,
                     hidden=hidden, temperature=0.85, top_p=0.9)
