"""The purpose of this script is to be a helper script for the GPT-based evaluation of standard or standardized reports."""

from sys import argv, stdout
import os, glob
import re
import subprocess
import argparse

import logging
import inspect

import string
from random import choice

import datetime

import zlib
import base64
from pathlib import Path

from math import ceil
from grumpy.utils.logger import CustomFormatter
from grumpy.utils.html_processing import decodeHTML
from grumpy.utils.report_parsing import parseStandardRepDir
from grumpy.utils.utils import str2bool
from grumpy.modules.qc import callGrumpySTD
from grumpy.connect import grumpyConnect
from grumpy.modules.gsea import GrumpyGSEA
from grumpy.modules.dpk import callGrumpyDPKQC, callGrumpyDPKExtract
from grumpy.modules.mea import callGrumpyMEA
from grumpy.version import __version__

from pathlib import Path

def parseArgs():
    lgr = logging.getLogger(inspect.currentframe().f_code.co_name)
    lgr.info("Current working directory: {}".format(os.getcwd()))
    lgr.info("Command used to run the program: python {}".format(' '.join(str(x) for x in argv)))

    parser = argparse.ArgumentParser()

    # Common parser for the shared arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("-o", "--outputDirectory", help="The output directory where the results of the GrumPy's evaluation will be stored. By default='./'.", default='./', action="store", type=str, required=False, dest="outputDirectory")
    common_parser.add_argument("-i", "--inputDirectory", help="The input directory where the results of the protocol are stored or the path to the text file. By default='./'.", default='./', action="store", type=str, required=False, dest="inputDirectory")
    common_parser.add_argument("-f", "--force", help="If set to True, the program wil overwrite the existing GPT-based evaluation files. By default = True.", action="store", type=str2bool, required=False, dest="force", default=True)
    common_parser.add_argument("--outfilesPrefix", help="The prefix for the output files. By default = 'grumpy'.", default="grumpy", action="store", type=str, required=False, dest="outfilesPrefix")
    common_parser.add_argument("--hidden", help="If set to True, the output files will be hidden (i.e. dot will prefix the output files). By default = True.", action="store", type=str2bool, required=False, dest="hidden", default=False)
    common_parser.add_argument("-k", "--apikey", help="Full path to the super-secret API-KEY file. By default = '/research_jude/rgs01_jude/groups/cab/projects/Control/common/cab_epi/APIKEY/key'.", default="/research_jude/rgs01_jude/groups/cab/projects/Control/common/cab_epi/APIKEY/key", action="store", type=str, required=False, dest="apikey")
    common_parser.add_argument("--apiType", help="Type of API, currently either 'openai' for direct linking with OpenAI, or 'azure' for the test st. Jude dedicated instance, those influence how the connection with API is established. By default='RECOMMENDED', which sets it to 'openai'.", default="RECOMMENDED", action="store", type=str, required=False, dest="apiType", choices=['RECOMMENDED', 'azure', 'openai', 'ollama'])
    common_parser.add_argument("--gptModel", help="Type of the model, currently either 'GPT-4-32k-API' for the test st. Jude dedicated instance, or 'gpt-3.5-turbo' and 'gpt-4o' for the direct OpenAI connections. By default='RECOMMENDED', which will set it to 'gpt-3.5-turbo' - the most applicable here.", default="RECOMMENDED", action="store", type=str, required=False, dest="gptModel", choices=['RECOMMENDED', 'GPT-4-32k-API', 'gpt-3.5-turbo', 'gpt-4o', 'gpt4o-api', 'llama3', 'meditron', 'medllama2', 'gpt-4o-mini'])


    subparsers = parser.add_subparsers(help='Availible modes/tools/modules', dest='mode')

    ### QC Parser for Automapper and standard reports.
    qcParser = subparsers.add_parser('QC', help='Run evaluation of the Quality Control (QC) for standard report or Automapper output/summary QC table.', parents=[common_parser])
    
    requiredParams_qc = qcParser.add_argument_group('REQUIRED QC parameters')
    requiredParams_qc.add_argument("-p", "--protocol", help="What protocol is to be considered? if you have protocol not listed, use 'other' and add a name of the protocol in the -n flag.", action="store", type=str, required=True, dest="protocol", choices = ['cutandrun', 'chipseq', 'atacseq', 'other'])
    
    optionalParams_qc = qcParser.add_argument_group("Optional QC parameters")
    optionalParams_qc.add_argument("-n", "--protocolFullName", help="If you specified the 'protocol' (-p) as 'other', you should add here what the type of protocol we deal with here, like for example Hi-C, which will help the LLM to establish the best set of reference guidelines to analyze it. By default = 'unspecified'.", action="store", type=str, required=False, dest="protocolFullName", default='unspecified')
    optionalParams_qc.add_argument("--inputType", help="Type of the input file, so either this should point directly to the text file / table with some statistics that is relevant, or the directory where the mapping statistics from automapper are located, or to the directory that has the multiqc report inside. By default = 'automapper'.", action="store", type=str, required=False, dest="inputType", default='automapper', choices=['automapper', 'classicStdReportDir', 'multiqc'])
    

    ### PE Parser for Automapper and standard reports.
    peParser = subparsers.add_parser('PE', help='Run evaluation of the Pathway Enrichment (PE) analyses for either GSEA results, or typical Pathway Enrichment.', parents=[common_parser])
    
    requiredParams_pe = peParser.add_argument_group('REQUIRED PE parameters')
    requiredParams_pe.add_argument("--inputType", help="What protocol is to be considered? if you have protocol not listed, use 'other' and add a name of the protocol in the -n flag.", action="store", type=str, required=True, dest="inputType", choices = ['genes', 'pathways', 'deseq2'])
    requiredParams_pe.add_argument("-p", "--protocol", help="What protocol is to be considered? if you have protocol not listed, use 'other' and add a name of the protocol in the -n flag.", action="store", type=str, required=True, dest="protocol", choices = ['gsea', 'gsealist', 'other'])
   
    optionalParams_pe = peParser.add_argument_group("Optional PE parameters")
    optionalParams_pe.add_argument("-r", "--reportType", help="the type of the report to be parsed, currently only 'std' is availible, which stands for standard report. By default='auto', which will set the actual value to 'std' if 'chip' or 'cutrun' is set in '-p' flag, or if the 'gsea' is set in '-p' flag, then it the value will be set to 'gsealist' if the list of pathways was supplied to '-i' flag, or it will be set to 'gseareport' if the '-i' flag pointed to the directory with the GSEApy report. Set to 'decode' and point '-i' to the previous output report from Grumpy to convert HTML to TXT for debugging purposes.", default='auto', action="store", type=str, required=False, dest="reportType", choices = ['auto', 'std', 'gsealist', 'gseareport', 'decode'])
    optionalParams_pe.add_argument("--context", help="Optional context to pass to Grumpy evaluation.", default=None, action="store", type=str, required=False, dest="context")
    optionalParams_pe.add_argument("--species", help="[GSEA specific parameter - not used for typical PE] Define the species which information will be used to try to correctly identify the reference external links to all recognized MSigDB pathways. By default='human_mouse', which is used to provide the broadest spectrum, including links to all known MSigDB signatures from both human and mice, with the priority toward the human descriptions, thus if you did use the mice data, please make sure to change the setting to ppint to 'mouse' specifically. Specify as 'other' for other species, custom gene sets or if you simply wish to skip the attempt to link the pathways to external reference all together.", default="human_mouse", action="store", type=str, required=False, dest="species", choices=["human", "mouse", "human_mouse", "other"])
    optionalParams_pe.add_argument("--inputFile", help="[GSEA specific parameter - not used for typical PE] Define the species which information will be used to try to correctly identify the reference external links to all recognized MSigDB pathways. By default='human_mouse', which is used to provide the broadest spectrum, including links to all known MSigDB signatures from both human and mice, with the priority toward the human descriptions, thus if you did use the mice data, please make sure to change the setting to ppint to 'mouse' specifically. Specify as 'other' for other species, custom gene sets or if you simply wish to skip the attempt to link the pathways to external reference all together.", default="", action="store", type=str, required=False, dest="inputFile", choices=[""])


    ### MEA Parser for conversational evaluation.
    meaParser = subparsers.add_parser('MEA', help='Run evaluation of the Motif Enrichment Analysis (MEA) for the standard reports from Homer tool.', parents=[common_parser])

    optionalParams_mea = meaParser.add_argument_group("Optional MEA parameters")
    

    ### DEG Parser for conversational evaluation.
    degParser = subparsers.add_parser('DEG', help='Run evaluation of the Differentially Expressed Genes (DEG) for the typical DEG tables from tools such as limma, DEseq2 etc.', parents=[common_parser])

    optionalParams_deg = degParser.add_argument_group("Optional DEG parameters")
    

    ### DPK Parser for conversational evaluation.
    dpkParser = subparsers.add_parser('DPK', help='Run evaluation of the Differentially Peaks (DPK), so either differentially binding regions from protocols like ChIP-seq or differentially accessible ones from protocols like ATAC-seq.', parents=[common_parser])
     
    requiredParams_dpk = dpkParser.add_argument_group('REQUIRED PE parameters')
    requiredParams_dpk.add_argument("-p", "--protocol", help="What protocol is to be considered? if you have protocol not listed, use 'other' and add a name of the protocol in the -n flag.", action="store", type=str, required=True, dest="protocol", choices = ['cutandrun', 'chipseq', 'atacseq', 'other'])
    
    optionalParams_dpk = dpkParser.add_argument_group("Optional DPK parameters")
    optionalParams_dpk.add_argument("-c", "--context", help="specific biological context of interest, should be a full path to the file.", action="store", type=str, required=False, default="ignore")
    
    
    ### GrumpyChat Parser for conversational evaluation.
    chatParser = subparsers.add_parser('chat', help='Nice chat with the Grumpy AI', parents=[common_parser])

    optionalParams_chat = chatParser.add_argument_group("Optional Chat parameters")
    
    
    ### GrumpyChat Parser for conversational evaluation.
    decodeParser = subparsers.add_parser('decode', help='Small tool to decode / extract the information from the HTML file.', parents=[common_parser])

    

    params = vars(parser.parse_args())

    lgr.info("Parsed arguments: {}".format(params))

    errors = False
    if not os.path.exists(params["inputDirectory"]):
        lgr.error("The input directory '{}' does not exist. Program was aborted.".format(params["inputDirectory"]))
        errors = True
    lgr.info("Input directory (--inputDirectory flag): {}".format(params["inputDirectory"]))

    if not os.path.exists(params["outputDirectory"]):
        lgr.error("The output directory '{}' does not exist. Program was aborted.".format(params["outputDirectory"]))
        errors = True
    lgr.info("Output directory (--outputDirectory flag): {}".format(params["outputDirectory"]))

    if params["mode"] == "QC": # if params["reportType"] == "auto":
        params["reportType"] = "std"
        ### Get the list of *txt files within the "templates" of the scripts directory and add them as protocols:
        protocolReferencesFound = ["other"]
        for file in glob.glob(os.path.join(os.path.dirname(Path( __file__ ).absolute()), "..", "templates", "qc.*.txt")):
            if len(os.path.basename(file).split(".")) == 3:
                protocolReferencesFound.append(os.path.basename(file).split(".")[1])
        if params["protocol"] in protocolReferencesFound:
            lgr.info(f"Found the reference template for the '{params['protocol']}' protocol.")
        else:
            lgr.error(f"The reference template for the '{params['protocol']}' protocol was not found. Program was aborted.")
            errors = True
        if params["protocol"] == "other":
            if params["protocolFullName"] == "unspecified":
                lgr.error("The 'protocol' flag was set to 'other', but the 'protocolFullName' was not provided. Program was aborted.")
                errors = True
            else:
                lgr.info("The 'protocol' flag was set to 'other', and seemingly valid 'protocolFullName' was provided as '{}'. Program will proceed with the evaluation.".format(params["protocolFullName"]))

        if params["reportType"] == "std":
            for subdir in ["Stats", "Peaks"]:
                if not os.path.exists(os.path.join(params["inputDirectory"], subdir)):
                    lgr.info("The classic STDreport '{}' should have '{}' subdirectory, to be analyzed as 'classicSTDrepDir', assumin input file is the directory with *report file from Automapper.".format(params["inputDirectory"], subdir))


    if params["mode"] == "PE":
        if params["protocol"] == "gsea":
            if params["inputType"] == "pathways":
                params["reportType"] = "gseareport"
            else:
                params["reportType"] = "gsealist"

        if params["reportType"] == "gseareport":
            if not os.path.exists(os.path.join(params["inputDirectory"], "gseapy.gene_set.prerank.report.filtered.csv")):
                lgr.error("The input directory '{}' should have 'gseapy.gene_set.prerank.report.filtered.csv' file, which does not exist. Program was aborted.".format(params["inputDirectory"]))
                errors = True

        if params["reportType"] == "gsealist":
            if not os.path.exists(params["inputDirectory"]):
                lgr.error("The input file with the list of pathways to be checked'{}' does not exist. Program was aborted.".format(params["inputDirectory"]))
                errors = True

    # lgr.info("Report type (--reportType flag): {}".format(params["reportType"]))
    
    if params["mode"] == "decode":
        if params["inputDirectory"].endswith(".html"):
            pass
        else:
            lgr.error(f"The input file '{params['inputDirectory']}' should be an HTML file when working in 'decode' mode. Program was aborted.")
            errors = True

    if params["mode"] == "DPK":
        if os.path.exists(params["inputDirectory"]):
            params["reportType"] = "dpk"
            pass
        else:
            lgr.error(f"The input directory doesn't exist, Program was aborted.")
            errors = True
    
    params["keyFilePresent"] = True
    if not os.path.exists(params["apikey"]):
        lgr.error("The API-KEY file '{}' does not exist. Program was aborted.".format(params["apikey"]))
        params["keyFilePresent"] = False
    # lgr.info("API-KEY file (--apikey flag): {}".format(params["apikey"]))

    # check if context is a file or a string:
    if "context" in params:
        if os.path.exists(params["context"]):
            with open(params["context"], 'r') as file:
                params["context"] = file.read()
        else:
            params["context"] = params["context"]

    if "gptModel" in params:
        if params["gptModel"] == "RECOMMENDED" and params['mode'] == "QC":
            params["gptModel"] = 'gpt-4o-mini'
        elif params["gptModel"] == "RECOMMENDED" and params['mode'] == "chat":
            params["gptModel"] = "llama3"
    
    if "apiType" in params:
        if params["apiType"] == "RECOMMENDED" and params['mode'] == "QC":
            params["apiType"] = "openai"
        elif params["apiType"] == "RECOMMENDED" and params['mode'] == "chat":
            params["apiType"] = "ollama"

    if errors:
        lgr.critical("Errors found while parsing parameters -- see more details above. Program was aborted.")
        raise Exception("Errors found while parsing parameters")

    return params

def main():
    CustomFormatter.configure_logging()
    lgr = logging.getLogger(inspect.currentframe().f_code.co_name)
    params = parseArgs()

    # if params["hidden"]:
    #     outfileName = os.path.join(params['outputDirectory'], f".{params['outfilesPrefix']}.elaborate.md")
    #     outfileNameShort = os.path.join(params['outputDirectory'], f".{params['outfilesPrefix']}.concise.md")
    # else:
    outfileName = os.path.join(params['outputDirectory'], f"{params['outfilesPrefix']}.elaborate.md")
    outfileNameShort = os.path.join(params['outputDirectory'], f"{params['outfilesPrefix']}.concise.md")

    if params["keyFilePresent"]:
        if params["mode"] == 'QC':
            if params["inputType"] == "automapper" or params["inputType"] == "classicStdReportDir":
                metaFile = parseStandardRepDir(params["inputDirectory"], params["protocol"], params['protocolFullName'], params["outfilesPrefix"], params["force"], params['outputDirectory'], params['apikey'], params["apiType"], params["gptModel"], hidden=params["hidden"])
                callGrumpySTD([metaFile], params["inputType"], params["protocol"], params['protocolFullName'], params["outfilesPrefix"], params["force"], params["apikey"], params["apiType"], params["gptModel"], outfileName, outfileNameShort, hidden=params["hidden"])
            else:
                metaFiles = parseMultiQCReportDir(params["inputDirectory"], params["protocol"], params['protocolFullName'], params["apikey"], params["apiType"], params["gptModel"], params["outfilesPrefix"], params['outputDirectory'], hidden=params["hidden"])
                callGrumpySTD(metaFiles, params["inputType"], params["protocol"], params['protocolFullName'], params["outfilesPrefix"], params["force"], params["apikey"], params["apiType"], params["gptModel"], outfileName, outfileNameShort, hidden=params["hidden"])
        elif params["mode"] == 'PE':
            if params["reportType"] in ['gsealist', 'gseareport']:
                GrumpyGSEA().run_gsea(params["inputType"], params["reportType"], params["inputFile"], params["protocol"], params["inputDirectory"], params["force"], params["apikey"],
                                        params["apiType"], params["gptModel"], params["context"], params['outfilesPrefix'], params["hidden"], params["species"])
        elif params["mode"] == 'decode':
            decodeHTML(params["protocol"], params["inputDirectory"])
        if params["mode"] == "MEA":
            callGrumpyMEA(params["inputDirectory"], params['outputDirectory'], params["force"], params["apikey"], params["apiType"], params["gptModel"], hidden=params["hidden"])
        elif params["reportType"] == 'dpk':
            callGrumpyDPKQC(params["inputDirectory"], params['outfilesPrefix'], params["force"], params["apikey"], params["apiType"], params["gptModel"], params["hidden"])
            callGrumpyDPKExtract(params["inputDirectory"], params['outfilesPrefix'], params["force"], params["apikey"], params["apiType"], params["gptModel"], params["context"], params["hidden"])
    else:
        outfile = open(outfileName, "w")
        outfile.write("API key file was not provided or not accessible, GrumPy's evaluation cannot be performed.")
        outfile.close()

        outfile = open(outfileNameShort, "w")
        outfile.write("API key file was not provided or not accessible, GrumPy's evaluation cannot be performed.")
        outfile.close()

    lgr.info("All done, thank you!")


if __name__ == "__main__":
    main()
