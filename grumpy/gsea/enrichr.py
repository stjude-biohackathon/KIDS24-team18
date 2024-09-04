import gseapy as gp
import matplotlib.pyplot as plt

gene_list = []

def prep_input(input_file):
    """Prepares the input file and only takes specific formats."""
    return None

def enrichr_ora(gene_list):
    """Over-representation analysis via Enrichr web services."""
    enr = gp.enrichr(gene_list=gene_list,
                    gene_sets=['Reactome_2022','KEGG_2021_Human'],
                    organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                    outdir=None, # don't write to disk
                    )
    
    return enr.results