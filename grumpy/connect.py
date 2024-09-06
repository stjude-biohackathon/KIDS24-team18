import os
import logging
import tiktoken
import inspect

from openai import AzureOpenAI, OpenAI, AuthenticationError
from grumpy.utils.tokenization import getMaxTokenPerModel  # Ensure this is in your utils

def grumpyConnect(keyFile, apiType, gptModel, grumpyRole, query, outfileName, max_tokens=28000, 
                  top_p=0.95, frequency_penalty=0, presence_penalty=1, temperature=0.1, hidden=True):
    lgr = logging.getLogger(inspect.currentframe().f_code.co_name)

    if not hidden:
        # Saving the prompt
        promptFile = f"{outfileName}.prompt.txt"
        with open(promptFile, "w") as outfile:
            outfile.write(grumpyRole)
            outfile.write(query)
        lgr.info(f"The prompt was saved to the file '{promptFile}'.")

    APIKEY = open(keyFile).readlines()[0].strip()

    if apiType == "azure":
        # Initiate connection to Azure OpenAI
        os.environ["OPENAI_API_KEY"] = ""
        os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://oa-northcentral-dev.openai.azure.com/"

        client = AzureOpenAI(
            api_version="2023-07-01-preview",
            api_key=APIKEY,
            azure_endpoint="https://oa-northcentral-dev.openai.azure.com/"
        )
    elif apiType == "ollama":
        client = OpenAI(api_key='ollama', base_url='http://localhost:11434/v1')
        maxTok = max_tokens
    else:
        client = OpenAI(api_key=APIKEY, base_url="https://api.openai.com/v1")
        maxTok = min([max_tokens, getMaxTokenPerModel(gptModel)])

    # Construct message text
    message_text = [{"role": "system", "content": grumpyRole},
                    {"role": "user", "content": query}]

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    total_tokens = sum(len(tokenizer.encode(message['content'])) for message in message_text)
    lgr.info(f"Total tokens in the prompt: {total_tokens}; maxTok for model is {maxTok}")
    requestedCompletionTokens = min([maxTok - total_tokens, 4096])

    if requestedCompletionTokens < 0:
        lgr.error(f"The prompt is too long to fit into the '{gptModel}' model. Please shorten the prompt.")
        with open(outfileName, "w", encoding="utf-8") as outfile:
            outfile.write(f"Prompt too long to fit into '{gptModel}' model. Total tokens: {total_tokens}.")
    else:
        try:
            completion = client.chat.completions.create(
                model=gptModel,
                messages=message_text,
                temperature=temperature,
                max_tokens=requestedCompletionTokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=None
            )
            with open(outfileName, "w") as outfile:
                outfile.write(completion.choices[0].message.content)
            lgr.info(f"The full assessment was saved to the file '{outfileName}'.")

        except AuthenticationError as e:
            lgr.error(f"Failed to authenticate with OpenAI API. Error details: {e}")
            with open(outfileName, "w") as outfile:
                outfile.write("Failed to authenticate with OpenAI API. Check your API key and permissions.")

        except Exception as e:
            lgr.error("An unexpected error occurred: %s", e)
            with open(outfileName, "w") as outfile:
                outfile.write("An unexpected error occurred while calling the OpenAI API.")
