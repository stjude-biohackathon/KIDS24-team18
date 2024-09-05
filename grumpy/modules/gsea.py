import logging
import os
import re

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.html_processing import (format_html, load_html_template,
                                   write_html_file)
from version import __version__

from grumpy import grumpyConnect


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
     
        self.gene_list = df[col_name].dropna().tolist()  # Remove NaN values if any
        return self.gene_list

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
            gene_list = self._prep_input_file(input_file=input_file, col_name=col_name)

        if not gene_list:
            raise ValueError("Gene list is empty. Please provide a valid input.")

        enr = gp.enrichr(gene_list=gene_list,
                        gene_sets=self.gene_sets,
                        organism=self.organism,
                        outdir=None)  # Don't write to disk

        self.results = enr.results

        if output_file:
            self.results.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")

        return self.results


    def call_sanity_check(self, input_file, input_type, referencePathwaysList,
                          grumpyEvaluationFile, pattern=r'\|\|([^|]+)\|\|'):
        """
        Performs a sanity check by comparing the results to a reference pathway list or gene list.

        Parameters:
        -----------
        input_file : str
            The path to the file containing either gene list, deseq2 output, or pathways list.
        input_type : str
            The type of input provided in the input_file. Should be 'genes', 'deseq2output', or 'pathways'.
        referencePathwaysList : list
            The list of reference pathways for comparison.
        grumpyEvaluationFile : str
            Path to the Grumpy evaluation file to be checked.
        pattern : str, optional
            Regex pattern to extract pathway names from the evaluation file. Default is r'\\|\\|([^|]+)\\|\\|'.
        """
        lgr = logging.getLogger("sanity_check")

        # Load input file based on the input type
        if input_type == 'genes':
            with open(input_file, 'r') as file:
                genes_list = [line.strip() for line in file.readlines()]
                lgr.info(f"Loaded gene list with {len(genes_list)} genes.")
            comparison_list = genes_list

        elif input_type == 'deseq2results':
            # Assuming deseq2 output is a CSV or TSV file
            deseq2_data = pd.read_csv(input_file, sep='\t' if input_file.endswith('.tsv') else ',')
            genes_list = deseq2_data['gene'].tolist()
            lgr.info(f"Loaded DESeq2 output with {len(genes_list)} genes.")
            comparison_list = genes_list

        elif input_type == 'pathways':
            with open(input_file, 'r') as file:
                pathways_list = [line.strip() for line in file.readlines()]
                lgr.info(f"Loaded pathways list with {len(pathways_list)} pathways.")
            referencePathwaysList = pathways_list

        else:
            raise ValueError("Invalid input_type. Must be one of 'genesList', 'deseq2output', or 'pathwaysList'.")

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

    def call_reporter(self, input_type, referencePathwaysList, species, grumpyEvaluationFile_precise, 
                      grumpyEvaluationFile_balanced, grumpyEvaluationFile_creative, outfileNamePrefix, 
                      contextDescription="", pattern=r'\|\|([^|]+)\|\|'):
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

        # Load input file based on the input type
        if input_type == 'genes':
            pass

        elif input_type == 'deseq2results':
            pass

        elif input_type == 'pathways':
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

    def run_grumpy_mode(mode, keyFile, apiType, gptModel, grumpyRole, pathwaysList, outfileName, max_tokens, hidden, temperature, top_p):
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
                        max_tokens=max_tokens, hidden=hidden, temperature=temperature, top_p=top_p)


    def run_gsea(self, input_type, reportType, inputFile=None, inputDirectory=None, keyFile, apiType, gptModel, 
                outfileNamePrefix, species, context="", topPaths=250, FDRcut=0.05, pValCut=0.05, 
                max_tokens=32000, hidden=False):
        """
        Runs the GSEA analysis using Grumpy, followed by sanity check and reporting.

        Parameters:
        -----------
        input_type : str
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
            if input_type == 'deseq2results':
                if os.path.isfile(os.path.join(inputFile)):
                    deseq2Df = pd.read_csv(os.path.join(inputFile))
                    referenceList = deseq2Df["geneSymbol"].tolist()
            elif input_type == 'genes':
                with open(inputDirectory, 'r') as file:
                    genesList = file.read().split("\n")
            else:
                raise ValueError("Unsupported input type for 'gsealist'")
        elif reportType == 'gseareport':
            if os.path.isfile(os.path.join(inputDirectory, "gseapy.gene_set.prerank.report.filtered.csv")):
                pathwaysDf = pd.read_csv(os.path.join(inputDirectory, "gseapy.gene_set.prerank.report.filtered.csv"))
                referenceList = pathwaysDf["Term"].tolist()

                upPathways = pathwaysDf[(pathwaysDf["FDR q-val"] < FDRcut) & (pathwaysDf["NES"] > 0)].head(topPaths)["Term"].tolist()
                downPathways = pathwaysDf[(pathwaysDf["FDR q-val"] < FDRcut) & (pathwaysDf["NES"] < 0)].head(topPaths)["Term"].tolist()
            else:
                raise ValueError("GSEA report file not found.")
        else:
            raise ValueError("Unrecognized report type. Please use 'gsealist' or 'gseareport'.")

        # Create context description if provided
        contextDescription = f"Additionally, please analyze the GSEA results with the following biological context in mind: {context}"

        # Define output file names for each evaluation mode
        outfileName_precise = f"{outfileNamePrefix}.precise.md"
        outfileName_balanced = f"{outfileNamePrefix}.balanced.md"
        outfileName_creative = f"{outfileNamePrefix}.creative.md"
        outfileName_report = f"{outfileNamePrefix}.evaluation.html"

        basicRole = """
        In this task, you are focusing on the list of gene signatures / gene sets / pathways, coming from the Gene Set Enrichment Analysis (GSEA). Your goal is to analyze all those pathways presented to you, and to highlight the most relevant ones, emphasizing why do you think they are relevant.
        Moreover, please be as critique, as skeptical and as realistic as possible, dont make things up. If you find some potentially interesting patterns, mention them. If you find something that is worth further exploring, mention that as well. If something doesn't make sense, e.g., you identify contradictory results, please feel free to mention that as well. But if you don't find anything interesting, just say that you don't find anything interesting and that is much better than making things up.

        Assuming that you indeed identify the pathways worth highlighting, try to separate them into specific categories, like pathways related with cell proliferation, pathways relevant for immune system or for signaling, etc. Be flexible while categorizing and take the biological context into account.

        Finally, when you mention the actual pathway's name, always put two vertical bars (i.e. "||")  before and after the name, e.g. ||KEGG_CELL_CYCLE||. This is critical for the proper identification of mentioned names by the subsequent script and proper formatting of the report.
        """

        # Define Grumpy role with context and basic biological analysis task
        grumpyRole = f"""
        You are an AI assistant that acts as the Computational Biology expert in the area of Epigenetics. Your goal is to help people with the evaluation for their data, in better understanding them, and in finding patterns relevant for further studies.
        --------
        {basicRole}
        --------
        {contextDescription}
        """

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
            run_grumpy_mode(mode, **common_params, **params)

        # Generate the final HTML report based on the evaluations
        self.call_reporter(input_type, referenceList, species, outfileName_precise, outfileName_balanced,
                            outfileName_creative, outfileName_report, grumpyRole, pathwaysList,
                            context, outfileNamePrefix)
