import logging
import os
import re
import pkg_resources
import inspect

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from grumpy.utils.html_processing import (format_html, load_html_template,
                                   write_html_file)

from grumpy.connect import grumpyConnect
from grumpy.version import __version__


class GrumpyGSEA:
    def __init__(self, organism="human", gene_sets=None):
        """
        Initializes the GrumpyGSEA class with organism and gene sets.

        Parameters:
        -----------
        organism : str, optional
            The organism for which to perform the analysis. Default is 'human'.
        gene_sets : list, optional
            List of gene sets to use for enrichment analysis. Default includes 'Reactome_2022' and 'KEGG_2021_Human'.
        """
        self.organism = organism
        self.gene_sets = gene_sets or ['Reactome_2022', 'KEGG_2021_Human']
        self.results = None

    def _prep_input_file(self, input_file, col_name):
        """
        Prepares the input file by reading either a CSV or TSV format and extracting the 'Genes' column.

        Parameters:
        -----------
        input_file : str
            Path to the input file (CSV or TSV) containing the 'Genes' column.
        col_name : str
            The column name that contains the gene symbols.

        Returns:
        --------
        list
            A list of genes extracted from the specified column.
        """
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith('.tsv'):
            df = pd.read_csv(input_file, sep='\t')
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or TSV file.")
        
        if col_name not in df.columns:
            raise ValueError(f"The column '{col_name}' was not found in the input file.")
     
        return df

    def enrichr_ora(self, input_file=None, gene_list=None, col_name="geneSymbol", output_file=None):
        """
        Performs Over-Representation Analysis (ORA) using the Enrichr web service.

        Parameters:
        -----------
        input_file : str, optional
            Path to the input file containing gene data. If provided, `gene_list` will be ignored.
        gene_list : list, optional
            List of gene symbols. If provided, it will be used instead of `input_file`.
        col_name : str, optional
            The column name containing the gene symbols. Default is 'geneSymbol'.
        output_file : str, optional
            If provided, the Enrichr results will be saved to this file.

        Returns:
        --------
        pandas.DataFrame
            The Enrichr results as a DataFrame.
        """
        if gene_list is None:
            if input_file is None:
                raise ValueError("Either input_file or gene_list must be provided.")
            # Process the input file to extract the gene list
            df = self._prep_input_file(input_file=input_file, col_name=col_name)

        if not gene_list:
            raise ValueError("Gene list is empty. Please provide a valid input.")
        
        # Define thresholds
        fdr_threshold = 0.05
        log2fc_threshold = 0.05

        # Filter for upregulated genes (log2FC > log2fc_threshold and FDR < fdr_threshold)
        upregulated_genes = df[(df['log2FC'] > log2fc_threshold) & (df['FDR'] < fdr_threshold)][col_name].tolist()

        # Filter for downregulated genes (log2FC < -log2fc_threshold and FDR < fdr_threshold)
        downregulated_genes = df[(df['log2FC'] < -log2fc_threshold) & (df['FDR'] < fdr_threshold)][col_name].tolist()


        down_enr = gp.enrichr(gene_list=downregulated_genes,
                        gene_sets=self.gene_sets,
                        organism=self.organism,
                        outdir=None)  # Don't write to disk
        
        up_enr = gp.enrichr(gene_list=upregulated_genes,
                        gene_sets=self.gene_sets,
                        organism=self.organism,
                        outdir=None)  # Don't write to disk

        downreg_results = down_enr.results
        upreg_results = up_enr.results

        if output_file:
            downreg_results.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")

        return downreg_results, upreg_results


    def call_sanity_check(self, input_file, inputType, referencePathwaysList,
                          grumpyEvaluationFile, pattern=r'\|\|([^|]+)\|\|'):
        """
        Performs a sanity check by comparing the results to a reference pathway list or gene list.

        Parameters:
        -----------
        input_file : str
            The path to the file containing either gene list, deseq2 output, or pathways list.
        inputType : str
            The type of input provided in the input_file. Should be 'genes', 'deseq2', or 'pathways'.
        referencePathwaysList : list
            The list of reference pathways for comparison.
        grumpyEvaluationFile : str
            Path to the Grumpy evaluation file to be checked.
        pattern : str, optional
            Regex pattern to extract pathway names from the evaluation file. Default is r'\\|\\|([^|]+)\\|\\|'.
        """
        lgr = logging.getLogger("sanity_check")

        # Load input file based on the input type
        if inputType == 'genes':
            with open(input_file, 'r') as file:
                genes_list = [line.strip() for line in file.readlines()]
                lgr.info(f"Loaded gene list with {len(genes_list)} genes.")
            comparison_list = genes_list

        elif inputType == 'deseq2':
            pass

        elif inputType == 'pathways':
            with open(input_file, 'r') as file:
                pathways_list = [line.strip() for line in file.readlines()]
                lgr.info(f"Loaded pathways list with {len(pathways_list)} pathways.")
            referencePathwaysList = pathways_list

        else:
            raise ValueError("Invalid inputType. Must be one of 'genesList', 'deseq2output', or 'pathwaysList'.")

        # Sanity check logic
        with open(grumpyEvaluationFile, 'r') as file:
            grumpyEvaluation = file.read()
        matches = re.findall(pattern, grumpyEvaluation)

        mismatchedPathways = [match for match in matches if match not in referencePathwaysList]

        if mismatchedPathways:
            lgr.warning(f"The following pathways were mentioned in the Grumpy evaluation but were not present in the reference list: {mismatchedPathways}")

        # Clean the Grumpy evaluation file
        cleaned_text = grumpyEvaluation.replace('||', '')

        with open(grumpyEvaluationFile, 'w') as file:
            file.write(cleaned_text)
            file.write(f"\n\n### Sanity Check Summary:\n")
            file.write(f"The Grumpy evaluation contained {len(matches)} pathways.\n")
            if mismatchedPathways:
                file.write(f"The following pathways were mentioned in the Grumpy evaluation, but were not present in the reference list: {mismatchedPathways}\n")
            else:
                file.write("All pathways mentioned in the Grumpy evaluation were present in the reference list.\n")

    def get_msigdb(self, species):
        # Get the collection of the external signature links from MSigDB:
        human_msigdb = pkg_resources.resource_filename('grumpy.data', 'MSigDB/msigdb.v2023.2.Hs.links.tsv')
        mouse_msigdb = pkg_resources.resource_filename('grumpy.data', 'MSigDB/msigdb.v2023.2.Mm.links.tsv')

        externalSignatureLinks = {}
        refSignFiles = []
        if "human" in species:
            refSignFiles.append(human_msigdb)
        if "mouse" in species:
            refSignFiles.append(mouse_msigdb)

        for refSignFile in refSignFiles:
            with open(refSignFile, 'r') as file:
                for line in file:
                    pathway, link = line.strip().split("\t")
                    if pathway not in externalSignatureLinks:
                        externalSignatureLinks[pathway] = link
        
        return externalSignatureLinks

    def call_reporter(self, inputType, referencePathwaysList, species, grumpyEvaluationFile_precise, 
                      grumpyEvaluationFile_balanced, grumpyEvaluationFile_creative, 
                      outfileName, grumpyRole, pathwaysList, contextDescription, 
                      outfileNamePrefix, pattern=r'\|\|([^|]+)\|\|'):
        """
        Generates a report based on the Grumpy evaluations and enrichr results.

        Parameters:
        -----------
        referencePathwaysList : list
            The list of reference pathways for comparison.
        species : str
            Species name (e.g., 'human' or 'mouse').
        grumpyEvaluationFile_precise, grumpyEvaluationFile_balanced, grumpyEvaluationFile_creative : str
            Paths to the Grumpy evaluation files for precise, balanced, and creative modes.
        outfileNamePrefix : str
            Prefix for the output report file name.
        contextDescription : str, optional
            Additional context for the report.
        pattern : str, optional
            Regex pattern to extract pathway names from the evaluation files.
        """
        lgr = logging.getLogger(inspect.currentframe().f_code.co_name)
        scriptsDir = os.path.dirname(os.path.realpath(__file__))

        checkIcon = """<i class='bi bi-check-lg' style='color: green;' title='Verified: gene signature name present in input list'></i>"""
        warningIcon = """<i class='bi bi-exclamation-triangle' style='color: orange;' title='Warning: sanity check was unable to verify the presence of this gene signature in your original input - please proceed with caution'></i>"""

        # Read in the Grumpy evaluation file(s)
        processedEvals = {}

        # Load input file based on the input type
        if inputType == 'genes':
            pass

        elif inputType == 'deseq2':
            pass

        elif inputType == 'pathways':
            externalSignatureLinks = self.get_msigdb(species=species)
            lgr.info(f"External '{species}' signature links were loaded for {len(externalSignatureLinks)} pathways.")

            for evalType, grumpyEvaluationFile in zip(["precise", "balanced", "creative"],
                                                    [grumpyEvaluationFile_precise, grumpyEvaluationFile_balanced, grumpyEvaluationFile_creative]):
                with open(grumpyEvaluationFile, 'r') as file:
                    grumpyEvaluation = file.read()
                matches = re.findall(pattern, grumpyEvaluation)

                mismatchedPathways = [match for match in matches if match not in referencePathwaysList]
                confirmedSignatures = [match for match in matches if match in referencePathwaysList]
                processedEvals[f"{evalType}_confirmed"] = ','.join(str(x) for x in confirmedSignatures)

                if mismatchedPathways:
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
                
        # Generating final report
        outfileName = f"{outfileNamePrefix}.evaluation.html"
        html_template_path = "templates/grumpy_template.html"
        base_html = load_html_template(html_template_path)

        replacements = {
            "version": __version__,
            "context_description": contextDescription,
            "outfile_name_prefix": outfileNamePrefix,
            "processedEvals": processedEvals  
        }

        formatted_html = format_html(html_content=base_html, replacements=replacements)
        write_html_file(outfileName, formatted_html)

        lgr.info(f"The Grumpy report was saved to the file '{outfileName}'.")

    def run_grumpy_mode(self, mode, keyFile, apiType, gptModel, grumpyRole, 
                        pathwaysList, outfileName, max_tokens=28000, top_p=0.95,
                        frequency_penalty=0, presence_penalty=1, temperature=0.1, hidden=True):
        """
        Run the Grumpy analysis in a specific mode with the provided parameters.

        Parameters:
        -----------
        mode : str
            The mode to run Grumpy in (e.g., 'Precise', 'Balanced', 'Creative').
        keyFile : str
            Path to the API key file.
        apiType : str
            Type of API to use.
        gptModel : str
            The GPT model to use.
        grumpyRole : str
            Role of Grumpy.
        pathwaysList : list
            List of pathways for analysis.
        outfileName : str
            Path to the output file.
        max_tokens : int
            Maximum number of tokens.
        hidden : bool
            Whether the hidden state should be used.
        temperature : float
            Sampling temperature for creativity.
        top_p : float
            Probability mass for nucleus sampling.

        Returns:
        --------
        None
        """
        logging.info(f"Running Grumpy in {mode} mode")
        grumpyConnect(keyFile, apiType, gptModel, grumpyRole, pathwaysList, outfileName,
                        max_tokens=max_tokens, temperature=temperature, top_p=top_p, hidden=hidden)


    def run_gsea(self, inputType, reportType, inputFile, protocol, inputDirectory, force, keyFile, apiType, gptModel, 
                context, outfileNamePrefix, hidden, species, topPaths=250, FDRcut=0.05, pValCut=0.05, 
                max_tokens=32000):      
        """
        Runs the GSEA analysis using Grumpy, followed by sanity check and reporting.

        Parameters:
        -----------
        inputType : str
            Type of input provided (e.g., 'deseq2results' or other).
        reportType : str
            Type of report ('gsealist' or 'gseareport').
        inputDirectory : str
            Directory containing input data.
        keyFile, apiType, gptModel : str
            Various input parameters for running Grumpy.
        outfileNamePrefix : str
            Prefix for the output file names.
        species : str
            The species used for analysis ('human' or 'mouse').
        context : str, optional
            Biological context for the analysis.
        topPaths : int, optional
            Number of top pathways to analyze.
        FDRcut, pValCut : float
            Cutoffs for filtering pathways.
        max_tokens : int, optional
            Maximum number of tokens for the GPT model in Grumpy.
        hidden : bool, optional
            Whether to use hidden output.

        Returns:
        --------
        None
        """
        lgr = logging.getLogger("run_gsea")
        lgr.info(f"Running GSEA analysis with input: {inputDirectory}")

        # Load genes or pathways list based on input type and report type
        if reportType == 'gsealist':
            if inputType == 'deseq2':
                if os.path.isfile(os.path.join(inputFile)):
                    down, up = self.enrichr_ora(inputFile)
                    down_sig_results = down[down['Adjusted P-value'] < 0.05]
                    up_sig_results = up[up['Adjusted P-value'] < 0.05]
                    significant_terms_with_genes = down_sig_results[['Term', 'Gene_set', 'Genes']]

            elif inputType == 'genes':
                with open(inputDirectory, 'r') as file:
                    genesList = file.read().split("\n")
            else:
                raise ValueError("Unsupported input type for 'gsealist'")
        elif reportType == 'gseareport':
            ### Check if `gseapy.gene_set.prerank.report.filtered.csv` file is present, and open it to get the list of pathways
            ### This CVS file has the following columns: Name,Term,ES,NES,NOM p-val,FDR q-val,FWER p-val,Tag %,Gene %,Lead_genes,inGMT
            if inputType == "pathways":
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
                    raise ValueError("Unsupported input type for 'gseareport'")
        else:
            pathwaysList = "unrecognized report type - report an error"

        if context:
            contextDescription = f"Additionally, please analyze the GSEA results with the following biological context in mind: {context}"
        else:
            contextDescription = ""
        
        basicRole = """
        In this task, you are focusing on the list of gene signatures / gene sets / pathways, coming from the Gene Set Enrichment Analysis (GSEA). Your goal is to analyze all those pathways presented to you, and to highlight the most relevant ones, emphasizing why do you think they are relevant.
        Moreover, please be as critique, as skeptical and as realistic as possible, dont make things up. If you find some potentially interesting patterns, mention them. If you find something that is worht further exploring, mention that as well. If something doesnt make sense, e.g. you identify contradictory results of some sort, please feel free to mention that as well. But if you dont find anything interesting, just say that you dont find anything interesting and that is much better than making things up.

        Assuming that you indeed identify the pathways worth highlighting, try to separate them into specific categories, like pathways related with cell proliferation, pathways relevant for immune system or for signalling etc. But be flexible while categorizing and take the biological context into account.

        Finally, when you mention the actual pathway's name, always put two vertical bars (i.e. "||")  before and after the name, e.g. ||KEGG_CELL_CYCLE||. This is critical for the proper identification of mentioned names by the subsequent script and proper formatting of the report.
        """

        grumpyRole = f"""
        You are an AI assistant that acts as the Computational Biology expert in the area of Epigenetics. Your goal is to help people with the evaluation for their data, in better understanding them and in finding patterns relevant for further studies.
        --------
        {basicRole}
        --------
        {contextDescription}
        """

        # Define output file names for each evaluation mode
        outfileName_precise = f"{outfileNamePrefix}.precise.md"
        outfileName_balanced = f"{outfileNamePrefix}.balanced.md"
        outfileName_creative = f"{outfileNamePrefix}.creative.md"
        outfileName_report = f"{outfileNamePrefix}.evaluation.html"


        # Define parameters for each mode
        modes_params = {
            'Precise': {'temperature': 0.1, 'top_p': 0.6, 'outfileName': outfileName_precise},
            'Balanced': {'temperature': 0.5, 'top_p': 0.8, 'outfileName': outfileName_balanced},
            'Creative': {'temperature': 0.85, 'top_p': 0.9, 'outfileName': outfileName_creative},
        }

        # Common parameters for GrumpyConnect
        common_params = {
            'keyFile': keyFile,
            'apiType': apiType,
            'gptModel': gptModel,
            'grumpyRole': grumpyRole,
            'pathwaysList': pathwaysList,
            'max_tokens': max_tokens,
            'hidden': hidden,
        }

        # Running Grumpy in different modes
        for mode, params in modes_params.items():
            self.run_grumpy_mode(mode, **common_params, **params)
        
        # Generate the final HTML report based on the evaluations
        self.call_reporter(inputType, referencePathwaysList, species, outfileName_precise, outfileName_balanced,
                            outfileName_creative, outfileName_report, grumpyRole, pathwaysList, context, outfileNamePrefix)
