import zlib
import base64

def compress_text(text, s=17):
    """
    Compresses and encodes a text string using zlib compression and base64 encoding,
    then applies a Caesar cipher for additional obfuscation.

    Parameters:
    -----------
    text : str
        The text string to be compressed and encoded.
    s : int, optional
        The shift value for the Caesar cipher. Default is 17.

    Returns:
    --------
    str
        The compressed, encoded, and obfuscated text string.
    """
    compressed_data = zlib.compress(text.encode('utf-8'))
    encoded_data = base64.b64encode(compressed_data).decode('utf-8')
    cesar_data = caesar_cipher(encoded_data, s)
    return cesar_data

def decompress_text(cesar_data, s=-17):
    """
    Decodes and decompresses a text string that was previously encoded using compress_text.

    Parameters:
    -----------
    cesar_data : str
        The compressed and encoded text string to be decompressed and decoded.
    s : int, optional
        The shift value for reversing the Caesar cipher. Default is -17.

    Returns:
    --------
    str
        The original, decompressed text string.
    """
    text = caesar_cipher(cesar_data, s)
    compressed_data = base64.b64decode(text)
    decoded_data = zlib.decompress(compressed_data).decode('utf-8')
    return decoded_data

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
