"""The purpose of this script is to be a helper script for the GPT-based evaluation of standard or standardized reports."""

showLoadingTimes = True # for debugging purposes only

### Load basic libraries:
if showLoadingTimes:
    from datetime import datetime
    start = datetime.now()
    from sys import argv, stdout
    print(f"Loading: `from sys import argv, stdout` took {datetime.now()-start}")

    start = datetime.now()
    import os, glob
    print(f"Loading: `import os, glob` took {datetime.now()-start}")

    start = datetime.now()
    import re
    print(f"Loading: `import re` took {datetime.now()-start}")

    start = datetime.now()
    import subprocess
    print(f"Loading: `import subprocess` took {datetime.now()-start}")

    start = datetime.now()
    import argparse
    print(f"Loading: `import argparse` took {datetime.now()-start}")

    start = datetime.now()
    import logging
    print(f"Loading: `import logging` took {datetime.now()-start}")

    start = datetime.now()
    import inspect
    print(f"Loading: `import inspect` took {datetime.now()-start}")

    start = datetime.now()
    import string
    print(f"Loading: `import string` took {datetime.now()-start}")

    start = datetime.now()
    from random import choice
    print(f"Loading: `from random import choice` took {datetime.now()-start}")

    start = datetime.now()
    import zlib
    print(f"Loading: `import zlib` took {datetime.now()-start}")

    start = datetime.now()
    import base64
    print(f"Loading: `import base64` took {datetime.now()-start}")

    start = datetime.now()
    from math import ceil
    print(f"Loading: `from math import ceil` took {datetime.now()-start}")

    start = datetime.now()
    from pathlib import Path
    print(f"Loading: `from pathlib import Path` took {datetime.now()-start}")

    start = datetime.now()
    from utils.logger import CustomFormatter
    from utils.html_processing import extract_section, decodeHTML, format_html, write_html_file, load_html_template
    from utils.utils import id_generator, load_template, str2bool, caesar_cipher
    from utils.tokenization import getMaxTokenPerModel
    from utils.compression import compress_text, decompress_text
    from utils.peak_analysis import determinePkCalling, getPeakNumber
    from utils.report_parsing import parseStandardRepDir
    print(f"Loading: `import CustomFormatter` took {datetime.now()-start}")

    #### This one had to be loaded in the end, because it was causing the error with the datetime.datetime.now() function
    start = datetime.now()
    import datetime
    print(f"Loading: `import datetime` took {datetime.datetime.now()-start}")


else:

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

    from math import ceil
    from utils.logger import CustomFormatter
    from utils.html_processing import extract_section, decodeHTML, format_html, write_html_file, load_html_template
    from utils.utils import id_generator, load_template, str2bool
    from utils.tokenization import getMaxTokenPerModel
    from utils.compression import compress_text, decompress_text
    from utils.peak_analysis import determinePkCalling, getPeakNumber
    from utils.report_parsing import parseStandardRepDir

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
    
    ### PE Parser for Automapper and standard reports.
    peParser = subparsers.add_parser('PE', help='Run evaluation of the Pathway Enrichment (PE) analyses for either GSEA results, or typical Pathway Enrichment.', parents=[common_parser])
    
    requiredParams_pe = peParser.add_argument_group('REQUIRED PE parameters')
    requiredParams_pe.add_argument("-p", "--protocol", help="What protocol is to be considered? if you have protocol not listed, use 'other' and add a name of the protocol in the -n flag.", action="store", type=str, required=True, dest="protocol", choices = ['cutandrun', 'chipseq', 'atacseq', 'other'])
    
    optionalParams_pe = peParser.add_argument_group("Optional PE parameters")
    requiredParams_qc.add_argument("-r", "--reportType", help="the type of the report to be parsed, currently only 'std' is availible, which stands for standard report. By default='auto', which will set the actual value to 'std' if 'chip' or 'cutrun' is set in '-p' flag, or if the 'gsea' is set in '-p' flag, then it the value will be set to 'gsealist' if the list of pathways was supplied to '-i' flag, or it will be set to 'gseareport' if the '-i' flag pointed to the directory with the GSEApy report. Set to 'decode' and point '-i' to the previous output report from Grumpy to convert HTML to TXT for debugging purposes.", default='auto', action="store", type=str, required=False, dest="reportType", choices = ['auto', 'std', 'gsealist', 'gseareport', 'decode'])
    optionalParams_pe.add_argument("--species", help="[GSEA specific parameter - not used for typical PE] Define the species which information will be used to try to correctly identify the reference external links to all recognized MSigDB pathways. By default='human_mouse', which is used to provide the broadest spectrum, including links to all known MSigDB signatures from both human and mice, with the priority toward the human descriptions, thus if you did use the mice data, please make sure to change the setting to ppint to 'mouse' specifically. Specify as 'other' for other species, custom gene sets or if you simply wish to skip the attempt to link the pathways to external reference all together.", default="human_mouse", action="store", type=str, required=False, dest="species", choices=["human", "mouse", "human_mouse", "other"])
    

    ### MEA Parser for conversational evaluation.
    meaParser = subparsers.add_parser('MEA', help='Run evaluation of the Motif Enrichment Analysis (MEA) for the standard reports from Homer tool.', parents=[common_parser])

    optionalParams_mea = meaParser.add_argument_group("Optional MEA parameters")
    

    ### DEG Parser for conversational evaluation.
    degParser = subparsers.add_parser('DEG', help='Run evaluation of the Differentially Expressed Genes (DEG) for the typical DEG tables from tools such as limma, DEseq2 etc.', parents=[common_parser])

    optionalParams_deg = degParser.add_argument_group("Optional DEG parameters")
    

    ### DEG Parser for conversational evaluation.
    dpkParser = subparsers.add_parser('DPK', help='Run evaluation of the Differentially Peaks (DPK), so either differentially binding regions from protocols like ChIP-seq or differentially accessible ones from protocols like ATAC-seq.', parents=[common_parser])

    optionalParams_dpk = dpkParser.add_argument_group("Optional DEG parameters")
    

    ### GrumpyChat Parser for conversational evaluation.
    chatParser = subparsers.add_parser('chat', help='Nice chat with the Grumpy AI', parents=[common_parser])

    optionalParams_chat = chatParser.add_argument_group("Optional Chat parameters")
    
    
    ### GrumpyChat Parser for conversational evaluation.
    decodeParser = subparsers.add_parser('decode', help='Small tool to decode / extract the information from the HTML file.', parents=[common_parser])

    
    
    params = vars(parser.parse_args())

    lgr.info("Parsed arguments: {}".format(params))
    lgr.info("Protocol (--protocol flag): {}".format(params["protocol"]))

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
            if os.path.isdir(params["inputDirectory"]):
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
    else:
        global tiktoken
        global AzureOpenAI
        global OpenAI
        global pd
        global BedTool
        global AuthenticationError
        global np

        lgr.info("Loading additional libreries needed for the Grumpy evaluation, please wait...")
        if showLoadingTimes:
            start = datetime.datetime.now()

            import tiktoken
            lgr.info("Loaded: `import tiktoken` took %s", datetime.datetime.now()-start)

            start = datetime.datetime.now()
            from openai import AzureOpenAI, AuthenticationError, OpenAI
            lgr.info("Loaded: `from openai import AzureOpenAI, AuthenticationError, OpenAI` took %s", datetime.datetime.now()-start)

            start = datetime.datetime.now()
            import pandas as pd
            lgr.info(f"Loaded: `import pandas as pd` took {datetime.datetime.now()-start}")

            start = datetime.datetime.now()
            from pybedtools import BedTool
            lgr.info(f"Loaded: `from pybedtools import BedTool` took {datetime.datetime.now()-start}")

            start = datetime.datetime.now()
            import numpy as np
            lgr.info(f"Loaded: `import numpy as np` took {datetime.datetime.now()-start}")

        else:

            # import openai
            import tiktoken
            from openai import AzureOpenAI, AuthenticationError

            import pandas as pd

            from pybedtools import BedTool

            import numpy as np

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
            params["gptModel"] = "gpt-3.5-turbo"
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


def callGrumpySTD(metaFile, protocol, protocolFullName, outfilesPrefix, force, keyFile, apiType, gptModel, outfileName, 
                  outfileNameShort, hidden=False):
    """
    Function to call the Grumpy AI for generating a standard report based on a metafile. 
    The function handles renaming of old assessments, setting up the role for Grumpy, 
    processing QC tables, and finally generating both detailed and concise assessment reports.

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
    # Initialize logger for this function
    lgr = logging.getLogger(inspect.currentframe().f_code.co_name)
    lgr.info("Calling the Grumpy for the standard report metafile '{}'.".format(metaFile))
    
    ### Renaming old assessments if they already exist
    if os.path.exists(outfileName):
        movedOutfileName = f"{outfileName}.{datetime.datetime.fromtimestamp(os.path.getctime(outfileName)).strftime('%Y%m%d')}.{id_generator()}.txt"
        os.rename(outfileName, movedOutfileName)
        lgr.info("The output file '%s' already existed, so it was renamed to '%s'.", outfileName, movedOutfileName)
    
    if os.path.exists(outfileNameShort):
        movedOutfileName = f"{outfileNameShort}.{datetime.datetime.fromtimestamp(os.path.getctime(outfileNameShort)).strftime('%Y%m%d')}.{id_generator()}.txt"
        os.rename(outfileNameShort, movedOutfileName)
        lgr.info("The output file '%s' already existed, so it was renamed to '%s'.", outfileNameShort, movedOutfileName)
    
    ### Define descriptions for the basic role for Grumpy based on the protocol
    basicRole = load_template(protocol)
    if protocol == "other":
        basicRole = f"First and foremost, you are here going to analyze the data from the '{protocolFullName}' protocol. The following are the general guidelines for the analysis that you have to run, without any specific protocol in mind, so please adjust accordingly to the procol specified here as '{protocolFullName}'."

    grumpyRole = f"""
    You are an AI assistant that acts as the Computational Biology expert in the area of Epigenetics. Your goal is to help people with the QC evaluation for their data and in providing recommendations. Please be as concise as possible in providing your assessment (not extending 300 words). 
    Moreover, please be as critique, as skeptical and as realistic as possible, I want you to be able to provide focus on the low-quality aspects of the data for the human recipient of your message. If you don't find any issues with the data, don't make them up, instead just please write that it all rather looks good etc.

    Finally, when you mention the actual sample names, always put two vertical bars (i.e. "||") before and after the name, e.g. ||123451_H3K27Ac_rep1||. This is critical for the proper identification of mentioned names by the subsequent script and proper formatting of the report.

    {basicRole}
    """

    ### Read the metafile as a simple text file
    with open(metaFile, 'r') as f:
        QC_table = f.read()

    ### Process the metafile as a pandas DataFrame and evaluate duplication rates
    df = pd.read_csv(metaFile, sep="\t")
    dupColName = "ignore"
    for col in ["DUPLICATION (%)", "Duplication Rate(%)"]:
        if col in df.columns:
            dupColName = col
    if dupColName != "ignore":
        try:
            df[dupColName] = df[dupColName].str.replace("%", "").astype(float)
        except AttributeError:
            pass
        highDuplicationSamples = df[df[dupColName] > 30].shape[0]
        if highDuplicationSamples > 0:
            highDupNote = f"Additional Note: There are {highDuplicationSamples} samples with duplication rates higher than 30%."
            QC_table += f"\n\n{highDupNote}\n"

    ### Evaluate mapping rates and append notes if applicable
    mapColName = "ignore"
    for col in ["Mapping Rate(%)", "MAPPED (%)"]:
        if col in df.columns:
            mapColName = col
    if mapColName != "ignore":
        try:
            df[mapColName] = df[mapColName].str.replace("%", "").astype(float)
        except AttributeError:
            pass
        lowMappingSamples = df[df[mapColName] < 80].shape[0]
        if lowMappingSamples > 0:
            lowMapNote = f"Additional Note: There are {lowMappingSamples} samples with mapping rates lower than 80%."
            QC_table += f"\n\n{lowMapNote}\n"

    ### Connect to Grumpy AI and generate the reports
    grumpyConnect(keyFile, apiType, gptModel, grumpyRole, QC_table, outfileName)
    # grumpyConnect(keyFile, apiType, gptModel, grumpyRoleShorter, QC_table, outfileNameShort)

def grumpyConnect(keyFile, apiType, gptModel, grumpyRole, query, outfileName, max_tokens=28000, top_p=0.95, frequency_penalty=0, presence_penalty=1, temperature=0.1, hidden=True):
    lgr = logging.getLogger(inspect.currentframe().f_code.co_name)

    if hidden == False:
        ### saving the prompt:
        promptFile = f"{outfileName}.prompt.txt"
        outfile = open(promptFile, "w")
        outfile.write(grumpyRole)
        outfile.write(query)
        outfile.close()
        lgr.info("The prompt was saved to the file '{}'.".format(promptFile))

    APIKEY = open(keyFile).readlines()[0].strip()
    if apiType == "azure":
        ### Initiate connection to the Azure OpenAI
        os.environ["OPENAI_API_KEY"] = ""
        os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://oa-northcentral-dev.openai.azure.com/"

        client = AzureOpenAI(
            api_version = "2023-07-01-preview",
            api_key = APIKEY,
            azure_endpoint = "https://oa-northcentral-dev.openai.azure.com/"  # Your Azure OpenAI resource's endpoint value.
        )
    elif apiType == "ollama":
        ### Direct connection with the OpenAI API using private key - use with caution!
        client = OpenAI(api_key='ollama', base_url = 'http://localhost:11434/v1')
        maxTok = max_tokens
    else:
        ### Direct connection with the OpenAI API using private key - use with caution!
        client = OpenAI(api_key=APIKEY, base_url = "https://api.openai.com/v1")
        maxTok = min([max_tokens, getMaxTokenPerModel(gptModel)]) ### In case of "Object of type int64 is not JSON serializable" refer to here: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable

    ### Call Grumpy for his assesment of the data QC
    # Grumpy - full assesment
    message_text = [{"role":"system","content":grumpyRole},
                    {"role":"user","content":query}]

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    total_tokens = sum(len(tokenizer.encode(message['content'])) for message in message_text)
    lgr.info(f"Total tokens in the prompt: {total_tokens}; maxTok for model is {maxTok}")
    requestedCompletionTokens = min([maxTok - total_tokens, 4096]) # some references to 4k: https://community.openai.com/t/how-the-max-tokens-are-considered/313514/4 and https://community.openai.com/t/max-tokens-chat-completion-gpt4o/758066

    if requestedCompletionTokens < 0:
        lgr.error(f"The prompt is too long to fit into the '{gptModel}' model. Please shorten the prompt or increase the max number of tokes assigned (if possible).")
        outfile = open(outfileName, "w")
        outfile.write(f"The prompt is too long to fit into the '{gptModel}' model. Please shorten the prompt or increase the max number of tokes assigned (if possible). DEBUG info: Total tokens in the prompt: {total_tokens}; max_tokens set in grumpyConnect function: {max_tokens}; tokens assigned per model: {getMaxTokenPerModel(gptModel)}")
        outfile.close()
    else:
        try:
            # print("#!#!#gptModel: ", gptModel)  # DEBUG
            # print("#!#!#message_text: ", message_text)  # DEBUG
            # print("#!#!#temperature: ", temperature)  # DEBUG
            # print("#!#!#requestedCompletionTokens: ", requestedCompletionTokens)  # DEBUG
            # print("#!#!#top_p: ", top_p)  # DEBUG
            # print("#!#!#frequency_penalty: ", frequency_penalty)  # DEBUG
            # print("#!#!#presence_penalty: ", presence_penalty)  # DEBUG
            # print("#!#!#stop: ", None)  # DEBUG
            completion = client.chat.completions.create(
                                                          model=gptModel,
                                                          messages = message_text,
                                                          temperature=temperature,
                                                          max_tokens=requestedCompletionTokens,
                                                          top_p=top_p,
                                                          frequency_penalty=frequency_penalty,
                                                          presence_penalty=presence_penalty,
                                                          stop=None
                                                        )
            # print(completion.choices[0].message.content)
            outfile = open(outfileName, "w")
            outfile.write(completion.choices[0].message.content)
            outfile.close()
            lgr.info("The full assesment was saved to the file '{}'.".format(outfileName))

        except AuthenticationError as e:
            lgr.error(f"Failed to authenticate with OpenAI API. Please check your API key and permissions. Error details: {e}")
            outfile = open(outfileName, "w")
            outfile.write("Failed to authenticate with OpenAI API. Please check your API key and permissions. Also, most likely the API key is expired.")
            outfile.close()
        except Exception as e:
            lgr.error(f"An unexpected error occurred: {e}")
            outfile = open(outfileName, "w")
            outfile.write("An unexpected error occurred while calling the OpenAI API.")
            outfile.close()

def callGrumpyGSEA_sanityCheck(referencePathwaysList, grumpyEvaluationFile, pattern=r'\|\|([^|]+)\|\|'):
    lgr = logging.getLogger(inspect.currentframe().f_code.co_name)

    ### Read in the Grumpy evaluation file
    with open(grumpyEvaluationFile, 'r') as file:
        grumpyEvaluation = file.read()
    matches = re.findall(pattern, grumpyEvaluation)

    mismatchedPathways = []
    for match in matches:
        if match not in referencePathwaysList:
            mismatchedPathways.append(match)

    if len(mismatchedPathways) > 0:
        lgr.warning(f"The following pathways were mentioned in the Grumpy evaluation, but were not present in the reference list: {mismatchedPathways}")

    # Remove all "||" from the text
    cleaned_text = grumpyEvaluation.replace('||', '')

    ### Append summary of the sanity check to the Grumpy evaluation file:
    with open(grumpyEvaluationFile, 'w') as file:
        file.write(cleaned_text)
        file.write(f"\n\n### Sanity Check Summary:\n")
        file.write(f"The Grumpy evaluation contained {len(matches)} pathways.\n")
        if len(mismatchedPathways) > 0:
            file.write(f"The following pathways were mentioned in the Grumpy evaluation, but were not present in the reference list: {mismatchedPathways}\n")
        else:
            file.write("All pathways mentioned in the Grumpy evaluation were present in the reference list.\n")

def callGrumpyGSEA_reporter(referencePathwaysList, species, grumpyEvaluationFile_precise, grumpyEvaluationFile_balanced, grumpyEvaluationFile_creative, outfileName, grumpyRole, pathwaysList, contextDescription, outfileNamePrefix, pattern=r'\|\|([^|]+)\|\|'):
    lgr = logging.getLogger(inspect.currentframe().f_code.co_name)
    scriptsDir = os.path.dirname(os.path.realpath(__file__))

    checkIcon = """<i class='bi bi-check-lg' style='color: green;' title='Verified: gene signature name present in input list'></i>"""
    warningIcon = """<i class='bi bi-exclamation-triangle' style='color: orange;' title='Warning: sanity check was unable to verify the presence of this gene signature in your original input - please proceed with caution'></i>"""


    ### get the collection of the external signature links from MSigDB:
    externalSignatureLinks = {}
    refSignFiles = []
    if "human" in species:
        refSignFiles.append(os.path.join(scriptsDir, "data", "MSigDB", "msigdb.v2023.2.Hs.links.tsv"))
    if "mouse" in species:
        refSignFiles.append(os.path.join(scriptsDir, "data", "MSigDB", "msigdb.v2023.2.Mm.links.tsv"))

    for refSignFile in refSignFiles:
        with open(refSignFile, 'r') as file:
            for line in file:
                pathway, link = line.strip().split("\t")
                if pathway not in externalSignatureLinks:
                    externalSignatureLinks[pathway] = link
    lgr.info(f"External '{species}' signature links were loaded for {len(externalSignatureLinks)} pathways.")

    ### Read in the Grumpy evaluation file(s)
    processedEvals = {}

    for evalType, grumpyEvaluationFile in zip(["precise", "balanced", "creative"], [grumpyEvaluationFile_precise, grumpyEvaluationFile_balanced, grumpyEvaluationFile_creative]):
        with open(grumpyEvaluationFile, 'r') as file:
            grumpyEvaluation = file.read()
        matches = re.findall(pattern, grumpyEvaluation)

        mismatchedPathways = []
        confirmedSignatures = []
        for match in matches:
            if match not in referencePathwaysList:
                mismatchedPathways.append(match)
            else:
                confirmedSignatures.append(match)
        processedEvals[f"{evalType}_confirmed"] = ','.join(str(x) for x in confirmedSignatures)

        if len(set(mismatchedPathways)) > 0:
            lgr.warning(f"The following pathways were mentioned in the Grumpy evaluation, but were not present in the reference list: {mismatchedPathways}")

        for foundSignature in matches:

            if foundSignature in referencePathwaysList:
                icon = checkIcon
            else:
                icon = warningIcon

            if foundSignature in externalSignatureLinks:
                linkFront = f"<a href='{externalSignatureLinks[foundSignature]}' target='_blank'>"
                linkBack = "</a>"
            else:
                linkFront = ""
                linkBack = ""

            grumpyEvaluation = grumpyEvaluation.replace(f"||{foundSignature}||", f"{linkFront}{foundSignature} {icon}{linkBack}")

        grumpyEvaluation += f"""
            <hr>
            <h6 class='text-grey'>Sanity Check Summary</h6>
            <p class='text-grey' class='small-font'>The Grumpy evaluation contained {len(set(matches))} pathways."""
        if len(mismatchedPathways) > 0:
            grumpyEvaluation += f"<br>The following pathways were mentioned in the Grumpy evaluation, but were not present in the reference list: {mismatchedPathways}"
        else:
            grumpyEvaluation += "<br>All pathways mentioned in the Grumpy evaluation were present in the reference list."
        grumpyEvaluation += f"""
            <br>In addition:
            <ul class='text-grey' class='small-font'>
            <li>If any of the pathways were not present in the reference list, the following warning was displayed: {warningIcon}</li>
            <li>If any of the pathways were present in the reference list, the following checkmark was displayed: {checkIcon}</li>
            <li>If neither of the above icons is displayed next to pathway name, it means that the sanity-check code was unable to find it - its highly recommended to check those manually.</li>
            <li>If any of the pathways were present in the external signature links, the pathway name was hyperlinked to the external source.</li>
            <li>If any of the pathways were not present in the external signature links, the pathway name was not hyperlinked, but it doesnt mean its wrong, but rather that we were unable to find a reference link to the most up to date version of the MSigDB for that signature.</li>
            </ul>
            </p>
            """

        processedEvals[evalType] = grumpyEvaluation


        # Remove all "||" from the text
        # cleaned_text = grumpyEvaluation.replace('||', '')

        ### Append summary of the sanity check to the Grumpy evaluation file:
        # with open(grumpyEvaluationFile, 'w') as file:
        #     file.write(cleaned_text)
        #     file.write(f"\n\n### Sanity Check Summary:\n")
        #     file.write(f"The Grumpy evaluation contained {len(matches)} pathways.\n")
        #     if len(mismatchedPathways) > 0:
        #         file.write(f"The following pathways were mentioned in the Grumpy evaluation, but were not present in the reference list: {mismatchedPathways}\n")
        #     else:
        #         file.write("All pathways mentioned in the Grumpy evaluation were present in the reference list.\n")

        ### generate the final report:

    # Path to the HTML file template
    html_template_path = "templates/grumpy_template.html"

    # Load the base HTML template
    base_html = load_html_template(html_template_path)

    # Other data to replace in the template
    replacements = {
    "context_description": contextDescription,
    "outfile_name_prefix": outfileNamePrefix,
    "version": __version__,
    "processedEvals": processedEvals  # Pass processedEvals as part of the dictionary
    }

    # Format the HTML with the processedEvals values
    formatted_html = format_html(html_content=base_html,
                                 replacements=replacements)

    # Write the formatted HTML to the output file
    write_html_file(outfileName, formatted_html)

    lgr.info(f"The Grumpy report was saved to the file '{outfileName}'.")


def callGrumpyGSEA(reportType, protocol, inputDirectory, force, keyFile, apiType, gptModel, context, outfileNamePrefix, hidden, species, topPaths=250, FDRcut=0.05, pValCut=0.05, max_tokens=32000):
    lgr = logging.getLogger(inspect.currentframe().f_code.co_name)
    lgr.info("Calling the Grumpy for the GSEA mode directory or pathways list: '{}'.".format(inputDirectory))

    ### Renaming the old assesments (if applicable)
    outfileName_precise = f"{outfileNamePrefix}.precise.md"
    outfileName_balanced = f"{outfileNamePrefix}.balanced.md"
    outfileName_creative = f"{outfileNamePrefix}.creative.md"
    outfileName_report = f"{outfileNamePrefix}.evaluation.html"
    # for outFile in [outfileName_precise, outfileName_balanced, outfileName_creative, outfileName_report]:
    #     if os.path.exists(outFile):
    #         if outFile.endswith(".html"):
    #             movedOutfileName = f"{outFile}.{datetime.datetime.fromtimestamp(os.path.getctime(outFile)).strftime('%Y%m%d')}.{id_generator()}.html"
    #         else:
    #             movedOutfileName = f"{outFile}.{datetime.datetime.fromtimestamp(os.path.getctime(outFile)).strftime('%Y%m%d')}.{id_generator()}.md"
    #         os.rename(outFile, movedOutfileName)
    #         lgr.info("The output file '{}' already existed, so it was renamed to '{}'.".format(outFile, movedOutfileName))



    # if os.path.exists(outfileName):
    #     ### Extract the creation date of the existing file, and embed into new file name with format: YYYYMMDD
    #     # movedOutfileName = f"{outfileName}.{id_generator()}.txt"
    #     movedOutfileName = f"{outfileName}.{datetime.datetime.fromtimestamp(os.path.getctime(outfileName)).strftime('%Y%m%d')}.{id_generator()}.txt"
    #     os.rename(outfileName, movedOutfileName)
    #     lgr.info("The output file '{}' already existed, so it was renamed to '{}'.".format(outfileName, movedOutfileName))
    # if os.path.exists(outfileNameShort):
    #     #movedOutfileName = f"{outfileNameShort}.{id_generator()}.txt"
    #     movedOutfileName = f"{outfileNameShort}.{datetime.datetime.fromtimestamp(os.path.getctime(outfileNameShort)).strftime('%Y%m%d')}.{id_generator()}.txt"
    #     os.rename(outfileNameShort, movedOutfileName)
    #     lgr.info("The output file '{}' already existed, so it was renamed to '{}'.".format(outfileNameShort, movedOutfileName))

    ### Define the basic role for the Grumpy:
    basicRole = """

        """

    if reportType == 'gsealist':
        ### Here we are just using the list provided by the user as-is:
        with open(inputDirectory, 'r') as file:
            pathwaysList = file.read()
            referencePathwaysList = pathwaysList.split("\n")
    elif reportType == 'gseareport':
        ### Check if `gseapy.gene_set.prerank.report.filtered.csv` file is present, and open it to get the list of pathways
        ### This CVS file has the following columns: Name,Term,ES,NES,NOM p-val,FDR q-val,FWER p-val,Tag %,Gene %,Lead_genes,inGMT

        if os.path.isfile(os.path.join(inputDirectory, "gseapy.gene_set.prerank.report.filtered.csv")):
            pathwaysDf = pd.read_csv(os.path.join(inputDirectory, "gseapy.gene_set.prerank.report.filtered.csv"))
            referencePathwaysList = list(pathwaysDf["Term"])

            signUpPaths = pathwaysDf[(pathwaysDf["FDR q-val"] < FDRcut) & (pathwaysDf["NOM p-val"] < pValCut) & (pathwaysDf["NES"] > 0)]
            signUpPaths.sort_values(by="NES", inplace=True, ascending=False)
            upPathways = list(signUpPaths.head(topPaths)["Term"])
            if len(upPathways) == 0:
                upPathways = ["No significant upregulated pathways were found."]

            signDownPaths = pathwaysDf[(pathwaysDf["FDR q-val"] < FDRcut) & (pathwaysDf["NOM p-val"] < pValCut) & (pathwaysDf["NES"] < 0)]
            signDownPaths.sort_values(by="NES", inplace=True, ascending=True)
            downPathways = list(signDownPaths.head(topPaths)["Term"])
            if len(downPathways) == 0:
                downPathways = ["No significant downregulated pathways were found."]

            pathwaysList = f"""
Found {len(signUpPaths)} of significantly upregulated pathways. Top {np.min([topPaths, len(signUpPaths)])} are:
{upPathways}

Found {len(signDownPaths)} of significantly downregulated pathways. Top {np.min([topPaths, len(signDownPaths)])} are:
{downPathways}
            """
    else:
        pathwaysList = "unrecognized report type - report an error"

    if context == "ignore":
        contextDescription = ""
    else:
        contextDescription = f"""
        Additionally, please analyze the GSEA results with the following biological context in mind:
        ```
        {context}
        ```"""


    basicRole = f"""
In this task, you are focusing on the list of gene signatures / gene sets / pathways, coming from the Gene Set Enrichment Analysis (GSEA). Your goal is to analyze all those pathways presented to you, and to highlight the most relevant ones, emphasizing why do you think they are relevant.
Moreover, please be as critique, as skeptical and as realistic as possible, dont make things up. If you find some potentially interesting patterns, mention them. If you find something that is worht further exploring, mention that as well. If something doesnt make sense, e.g. you identify contradictory results of some sort, please feel free to mention that as well. But if you dont find anything interesting, just say that you dont find anything interesting and that is much better than making things up.

Assuming that you indeed identify the pathways worth highlighting, try to separate them into specific categories, like pathways related with cell proliferation, pathways relevant for immune system or for signalling etc. But be flexible while categorizing and take the biological context into account.

Finally, when you mention the actual pathway's name, always put two vertical bars (i.e. "||")  before and after the name, e.g. ||KEGG_CELL_CYCLE||. This is critical for the proper identification of mentioned names by the subsequent script and proper formatting of the report.
    """

#     grumpyRole = f"""
# You are an AI assistant that acts as the Computational Biology expert in the area of Epigenetics. Your goal is to help people with the evaluation for their data, in better understanding them and in finding patterns relevant for further studies. Please be as detailed as needed in your evaluation.
# --------
# {basicRole}
# --------
# {contextDescription}
#     """

    grumpyRole = f"""
You are an AI assistant that acts as the Computational Biology expert in the area of Epigenetics. Your goal is to help people with the evaluation for their data, in better understanding them and in finding patterns relevant for further studies.
--------
{basicRole}
--------
{contextDescription}
    """
    ## Please be concise in your evaluation.

    ### Running the Grumpy in the Precise mode:
    grumpyConnect(keyFile, apiType, gptModel, grumpyRole, pathwaysList, outfileName_precise,
                  max_tokens=max_tokens, hidden=hidden, temperature=0.1, top_p=0.6)

    ### Running the Grumpy in the Balanced mode:
    grumpyConnect(keyFile, apiType, gptModel, grumpyRole, pathwaysList, outfileName_balanced,
                  max_tokens=max_tokens, hidden=hidden, temperature=0.5, top_p=0.8)

    ### Running the Grumpy in the Creative mode:
    grumpyConnect(keyFile, apiType, gptModel, grumpyRole, pathwaysList, outfileName_creative,
                   max_tokens=max_tokens, hidden=hidden, temperature=0.85, top_p=0.9)

    ### Compiling the final report:
    callGrumpyGSEA_reporter(referencePathwaysList, species, outfileName_precise, outfileName_balanced,
                            outfileName_creative, outfileName_report, grumpyRole, pathwaysList, context, outfileNamePrefix)

__version__ = "0.3.0-alpha"

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
        if params["reportType"] == 'std':
            metaFile = parseStandardRepDir(params["inputDirectory"], params["protocol"], params["outfilesPrefix"], params["force"], params['outputDirectory'], hidden=params["hidden"])
            callGrumpySTD(metaFile, params["protocol"], params['protocolFullName'], params["outfilesPrefix"], params["force"], params["apikey"], params["apiType"], params["gptModel"], outfileName, outfileNameShort, hidden=params["hidden"])
        elif params["reportType"] in ['gsealist', 'gseareport']:
            callGrumpyGSEA(params["reportType"], params["protocol"], params["inputDirectory"], params["force"], params["apikey"], params["apiType"], params["gptModel"], params["context"], params['outfilesPrefix'], params["hidden"], params["species"])
        elif params["reportType"] == 'decode':
            decodeHTML(params["protocol"], params["inputDirectory"])
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
