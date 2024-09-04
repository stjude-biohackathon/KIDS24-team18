import logging
import re
import os
from utils.compression import compress_text, decompress_text
from jinja2 import Template

def extract_section(file_path, flankPattern):
    """
    Extracts a section of text from a file that is flanked by a specific pattern.

    Parameters:
    -----------
    file_path : str
        The path to the file from which to extract the section.
    flankPattern : str
        The pattern that flanks the section of text to be extracted.

    Returns:
    --------
    str or None
        The extracted section of text if found, otherwise None.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Define the regex pattern to search for the text between the flanking pattern
    pattern = re.escape(flankPattern) + r'(.*?)' + re.escape(flankPattern)

    # Search for the pattern in the content
    match = re.search(pattern, content, re.DOTALL)

    # Extract and return the section if found, otherwise return None
    if match:
        return match.group(1).strip()
    else:
        return None

def decodeHTML(protocol, inputHtml):
    """
    Decodes specific sections of an HTML file based on the protocol and saves the decoded content to a text file.

    Parameters:
    -----------
    protocol : str
        The protocol type that determines which sections to decode (e.g., "gsea").
    inputHtml : str
        The path to the HTML file to be decoded.

    Returns:
    --------
    None
    """
    lgr = logging.getLogger(inspect.currentframe().f_code.co_name)
    lgr.info("Decoding the HTML file '{}'.".format(inputHtml))
    
    # Depending on the protocol, extract and decode specific sections of the HTML file
    if protocol == "gsea":
        grumpyRole = decompress_text(extract_section(inputHtml, "||::GR::||"))
        inputSignaturesList = decompress_text(extract_section(inputHtml, "||::IS::||"))
        preciseConfirmed = decompress_text(extract_section(inputHtml, "||::CP::||"))
        balancedConfirmed = decompress_text(extract_section(inputHtml, "||::CB::||"))
        creativeConfirmed = decompress_text(extract_section(inputHtml, "||::CC::||"))
    
    # Create an output file to save the decoded content
    with open(inputHtml.replace(".html", ".decoded.txt"), "w") as outfile:
        outfile.write(f"Grumpy's role:\n{grumpyRole}\n\n#############################################################################\n\n")
        outfile.write(f"Input signatures list:\n{inputSignaturesList}\n\n#############################################################################\n\n")
        outfile.write(f"Precise mode confirmed:\n{preciseConfirmed}\n\n#############################################################################\n\n")
        outfile.write(f"Balanced mode confirmed:\n{balancedConfirmed}\n\n#############################################################################\n\n")
        outfile.write(f"Creative mode confirmed:\n{creativeConfirmed}\n\n#############################################################################\n\n")
    
    lgr.info("The HTML file was decoded and saved as '{}.decoded.txt'.".format(inputHtml.replace(".html", "")))

def format_html(html_content, replacements):
    """
    Replaces placeholders in the HTML content with corresponding values from the processed_evals dictionary.

    Parameters:
    -----------
    html_content : str
        The HTML content with placeholders.
    replacements : dict
        A dictionary where the keys correspond to placeholder names in the HTML content,
        and the values are the strings that will replace the placeholders.

    Returns:
    --------
    str
        The modified HTML content with placeholders replaced by the corresponding values.
    """
    # Create a Jinja2 Template object
    template = Template(html_content)
    # Render the template with the provided data
    return template.render(replacements)

def write_html_file(file_path, content):
    """
    Writes the modified HTML content to a file.

    Parameters:
    -----------
    file_path : str
        The path to the file where the modified HTML content will be saved.
    content : str
        The modified HTML content to be written to the file.

    Returns:
    --------
    None
    """
    with open(file_path, 'w') as file:
        file.write(content)

def load_html_template(file_path):
    """
    Reads the content of an HTML template file and returns it as a string.

    Parameters:
    -----------
    file_path : str
        The path to the HTML template file.

    Returns:
    --------
    str
        The content of the HTML file as a string.
    """
    with open(file_path, 'r') as file:
        html_content = file.read()
    return html_content