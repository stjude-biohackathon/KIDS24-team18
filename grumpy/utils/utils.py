import string
import os
import logging
import inspect
from random import choice
from math import ceil

from pathlib import Path

def id_generator(size=7, chars=string.ascii_uppercase + string.digits):
    """
    Generates a random string of specified size using the given characters.

    Parameters:
    -----------
    size : int, optional
        The length of the generated string. Default is 7.
    chars : str, optional
        The characters to be used for generating the string. Default is uppercase letters and digits.

    Returns:
    --------
    str
        The randomly generated string.
    """
    return ''.join(choice(chars) for _ in range(size))

def caesar_cipher(text, shift):
    """
    Applies a Caesar cipher to a text string, shifting each character by a specified value.

    Parameters:
    -----------
    text : str
        The text string to be obfuscated.
    shift : int
        The number of positions to shift each character.

    Returns:
    --------
    str
        The text string after applying the Caesar cipher.
    """
    def shift_char(c, direction):
        if 'a' <= c <= 'z':
            return chr((ord(c) - ord('a') + shift) % 26 + ord('a'))
        elif 'A' <= c <= 'Z':
            return chr((ord(c) - ord('A') - direction * (ceil(abs(shift)/2))) % 26 + ord('A'))
        elif '0' <= c <= '9':
            return chr((ord(c) - ord('0') + direction * (ceil(abs(shift)/2))) % 10 + ord('0'))
        else:
            return c
    direction = 1 if shift > 0 else -1
    return ''.join(shift_char(c, direction) for c in text)

def load_template(protocol, folder="templates"):
    """
    Load a template file based on the given protocol.

    Parameters:
    -----------
    protocol : str
        The protocol to determine the file to load.
    folder : str, optional
        The folder where the template files are located. Defaults to "templates".

    Returns:
    --------
    str
        The content of the loaded template file.
    """
    file_path = os.path.join(os.path.dirname(Path( __file__ ).absolute()), "..", "..", folder, f"qc.{protocol}.txt")
    
    with open(file_path, 'r') as file:
        content = file.read()
    
    return content

def str2bool(v):
    """
    Convert a string representation of a boolean value to its corresponding boolean value.

    Parameters:
    -----------
    v : str
        The string representation of the boolean value.

    Returns:
    --------
    bool
        The corresponding boolean value.

    Raises:
    -------
    ValueError
        If the input string is not a recognized boolean value.
    """
    lgr = logging.getLogger(inspect.currentframe().f_code.co_name)
    if str(v).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str(v).lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        lgr.critical("Unrecognized parameter was set for '%s'. Program was aborted.", v)
        exit()
