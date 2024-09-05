import os
import logging
import re

import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np


from grumpy.utils.html_processing import format_html, write_html_file, load_html_template
from grumpy import grumpyConnect, __version__

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
        self.gene_list = []
        self.organism = organism
        self.gene_sets = gene_sets or ['Reactome_2022', 'KEGG_2021_Human']
        self.results = None

    def _prep_input(self, input_file, col_name):
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
     
        self.gene_list = df[col_name].dropna().tolist()  # Remove NaN values if any
        return self.gene_list

    def enrichr_ora(self, input_file, col_name="geneSymbol", output_file=None):
        """
        Performs Over-Representation Analysis (ORA) using the Enrichr web service.

        Parameters:
        -----------
        input_file : str
            Path to the input file containing gene data.
        col_name : str, optional
            The column name containing the gene symbols. Default is 'geneSymbol'.
        output_file : str, optional
            If provided, the Enrichr results will be saved to this file.

        Returns:
        --------
        pandas.DataFrame
            The Enrichr results as a DataFrame.
        """
        gene_list = self._prep_input(input_file=input_file, col_name=col_name)

        if not gene_list:
            raise ValueError("Gene list is empty. Please prepare the input file first.")
        
        enr = gp.enrichr(gene_list=self.gene_list,
                         gene_sets=self.gene_sets,
                         organism=self.organism,
                         outdir=None)  # Don't write to disk

        self.results = enr.results

        if output_file:
            self.results.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")

        return self.results

    def call_sanity_check(self, referencePathwaysList, grumpyEvaluationFile, pattern=r'\|\|([^|]+)\|\|'):
        """
        Performs a sanity check by comparing the results to a reference pathway list.

        Parameters:
        -----------
        referencePathwaysList : list
            The list of reference pathways for comparison.
        grumpyEvaluationFile : str
            Path to the Grumpy evaluation file to be checked.
        pattern : str, optional
            Regex pattern to extract pathway names from the evaluation file.
        """
        lgr = logging.getLogger("sanity_check")

        with open(grumpyEvaluationFile, 'r') as file:
            grumpyEvaluation = file.read()
        matches = re.findall(pattern, grumpyEvaluation)

        mismatchedPathways = [match for match in matches if match not in referencePathwaysList]

        if mismatchedPathways:
            lgr.warning(f"The following pathways were mentioned in the Grumpy evaluation, but were not present in the reference list: {mismatchedPathways}")

        cleaned_text = grumpyEvaluation.replace('||', '')

        with open(grumpyEvaluationFile, 'w') as file:
            file.write(cleaned_text)
            file.write(f"\n\n### Sanity Check Summary:\n")
            file.write(f"The Grumpy evaluation contained {len(matches)} pathways.\n")
            if mismatchedPathways:
                file.write(f"The following pathways were mentioned in the Grumpy evaluation, but were not present in the reference list: {mismatchedPathways}\n")
            else:
                file.write("All pathways mentioned in the Grumpy evaluation were present in the reference list.\n")

    def call_reporter(self, referencePathwaysList, species, grumpyEvaluationFile_precise, grumpyEvaluationFile_balanced, grumpyEvaluationFile_creative, outfileNamePrefix, contextDescription="", pattern=r'\|\|([^|]+)\|\|'):
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
        lgr = logging.getLogger("reporter")
        processedEvals = {}

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

    def run_gsea(self, reportType, protocol, inputDirectory, keyFile, apiType, gptModel, 
                 outfileNamePrefix, species, context="", topPaths=250, FDRcut=0.05, pValCut=0.05, 
                 max_tokens=32000, hidden=False):
        """
        Runs the GSEA analysis using Grumpy, followed by sanity check and reporting.

        Parameters:
        -----------
        reportType : str
            Type of report ('gsealist' or 'gseareport').
        protocol, inputDirectory, keyFile, apiType, gptModel : str
            Various input parameters for running Grumpy.
        outfileNamePrefix : str
            Prefix for the output file names.
        species : str
            The species used for analysis ('human' or 'mouse').
        context : str, optional
            Biological context for the analysis.
        topPaths, FDRcut, pValCut : float
            Parameters for filtering pathways.
        max_tokens : int, optional
            Maximum number of tokens for the GPT model in Grumpy.
        hidden : bool, optional
            Whether to use hidden output.
        """
        lgr = logging.getLogger("run_gsea")
        lgr.info(f"Running GSEA analysis with input: {inputDirectory}")

        # Determine which type of input to use (list or report)
        if reportType == 'gsealist':
            with open(inputDirectory, 'r') as file:
                pathwaysList = file.read().split("\n")
        elif reportType == 'gseareport':
            if os.path.isfile(os.path.join(inputDirectory, "gseapy.gene_set.prerank.report.filtered.csv")):
                pathwaysDf = pd.read_csv(os.path.join(inputDirectory, "gseapy.gene_set.prerank.report.filtered.csv"))
                referencePathwaysList = pathwaysDf["Term"].tolist()

                upPathways = pathwaysDf[(pathwaysDf["FDR q-val"] < FDRcut) & (pathwaysDf["NES"] > 0)].head(topPaths)["Term"].tolist()
                downPathways = pathwaysDf[(pathwaysDf["FDR q-val"] < FDRcut) & (pathwaysDf["NES"] < 0)].head(topPaths)["Term"].tolist()
            else:
                raise ValueError("GSEA report file not found.")
        else:
            raise ValueError("Unrecognized report type. Please use 'gsealist' or 'gseareport'.")

        # Description for context (optional)
        contextDescription = f"Additionally, please analyze the GSEA results with the following biological context in mind: {context}"

        # Define output file names for each evaluation mode
        outfileName_precise = f"{outfileNamePrefix}.precise.md"
        outfileName_balanced = f"{outfileNamePrefix}.balanced.md"
        outfileName_creative = f"{outfileNamePrefix}.creative.md"
        outfileName_report = f"{outfileNamePrefix}.evaluation.html"

        # Define Grumpy role with context and basic biological analysis task
        grumpyRole = f"""
        You are an AI assistant that acts as a Computational Biology expert in the area of Epigenetics. Your goal is to help people with the evaluation of their data, better understanding them, and finding patterns relevant for further studies.
        --------
        In this task, you are focusing on the list of gene signatures / gene sets / pathways, coming from the Gene Set Enrichment Analysis (GSEA). Your goal is to analyze all those pathways presented to you, and to highlight the most relevant ones, emphasizing why do you think they are relevant.
        {contextDescription}
        """

        # Running the Grumpy in Precise mode
        grumpyConnect(keyFile, apiType, gptModel, grumpyRole, pathwaysList, outfileName_precise,
                    max_tokens=max_tokens, hidden=hidden, temperature=0.1, top_p=0.6)

        # Running the Grumpy in Balanced mode
        grumpyConnect(keyFile, apiType, gptModel, grumpyRole, pathwaysList, outfileName_balanced,
                    max_tokens=max_tokens, hidden=hidden, temperature=0.5, top_p=0.8)

        # Running the Grumpy in Creative mode
        grumpyConnect(keyFile, apiType, gptModel, grumpyRole, pathwaysList, outfileName_creative,
                    max_tokens=max_tokens, hidden=hidden, temperature=0.85, top_p=0.9)

        # Generate the final HTML report based on the evaluations
        self.call_reporter(referencePathwaysList, species, outfileName_precise, outfileName_balanced,
                           outfileName_creative, outfileName_report, grumpyRole, pathwaysList, 
                           context, outfileNamePrefix)
