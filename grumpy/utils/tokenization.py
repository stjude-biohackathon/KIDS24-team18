import logging
import inspect

def getMaxTokenPerModel(gptModel):
    """
    Returns the maximum token limit for the specified GPT model.

    Parameters:
    -----------
    gptModel : str
        The name of the GPT model (e.g., 'GPT-4-32k-API', 'gpt-3.5-turbo').

    Returns:
    --------
    int
        The maximum number of tokens allowed for the specified model.
    
    Notes:
    ------
    - The token values are slightly rounded down to account for possible discrepancies 
      when calculating tokens using the tiktoken library.
    - Reference values were accessed on 2024-06-13.
    """
    lgr = logging.getLogger(inspect.currentframe().f_code.co_name)
    if gptModel == "GPT-4-32k-API":
        return 32000 # actual value is 32,768
    elif gptModel == "gpt-3.5-turbo":
        return 16000 # actual value is 16,385
    elif gptModel == "gpt-4o" or gptModel == "gpt4o-api":
        return 127000 # actual value is 128,000
    else:
        return 30000
