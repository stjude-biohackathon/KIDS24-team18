import gseapy as gp
import pandas as pd
import matplotlib.pyplot as plt

class Enrichr:
    def __init__(self, organism="human", gene_sets=None):
        """
        Initializes the Enrichr class with organism and gene sets.

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

        Raises:
        -------
        ValueError
            If the file format is not supported or the column is not found.
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

        Raises:
        -------
        ValueError
            If the gene list is empty or input has not been prepared.
        """
        gene_list = self._prep_input(input_file=input_file, col_name=col_name)

        if not gene_list:
            raise ValueError("Gene list is empty. Please prepare the input file first.")
        
        enr = gp.enrichr(gene_list=self.gene_list,
                         gene_sets=self.gene_sets,
                         organism=self.organism,
                         outdir=None)  # Don't write to disk

        results = enr.results

        if output_file:
            results.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")

        return results

    def plot_results(self, results, top_n=10):
        """
        Plots the top enriched pathways from the Enrichr results.

        Parameters:
        -----------
        results : pandas.DataFrame
            DataFrame containing the Enrichr results.
        top_n : int, optional
            Number of top enriched pathways to plot. Default is 10.
        """
        if results.empty:
            print("No results to plot.")
            return
