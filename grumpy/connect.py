import os
import logging
import tiktoken
import inspect

from openai import AzureOpenAI, OpenAI, AuthenticationError
from grumpy.utils.tokenization import getMaxTokenPerModel  # Ensure this is in your utils

def grumpyConnect(keyFile, apiType, gptModel, grumpyRole, query, outfileName, max_tokens=128000, top_p=0.95, frequency_penalty=0, presence_penalty=1, temperature=0.1, hidden=True, saveResponse=True):
    lgr = logging.getLogger(inspect.currentframe().f_code.co_name)

    if hidden == False:
        ### saving the prompt:
        promptFile = f"{outfileName}.prompt.txt"
        outfile = open(promptFile, "w")
        outfile.write(grumpyRole)
        outfile.write(query)
        outfile.close()
        lgr.info("The prompt was saved to the file '{}'.".format(promptFile))

    APIKEY = open(keyFile).readlines()[0].strip()
    if apiType == "azure":
        ### Initiate connection to the Azure OpenAI
        os.environ["OPENAI_API_KEY"] = ""
        os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://oa-northcentral-dev.openai.azure.com/"

        client = AzureOpenAI(
            api_version = "2023-07-01-preview",
            api_key = APIKEY,
            azure_endpoint = "https://oa-northcentral-dev.openai.azure.com/"  # Your Azure OpenAI resource's endpoint value.
        )
    elif apiType == "ollama":
        ### Direct connection with the OpenAI API using private key - use with caution!
        client = OpenAI(api_key='ollama', base_url = 'http://localhost:11434/v1')
        maxTok = max_tokens
    else:
        ### Direct connection with the OpenAI API using private key - use with caution!
        client = OpenAI(api_key=APIKEY, base_url = "https://api.openai.com/v1")
        maxTok = min([max_tokens, getMaxTokenPerModel(gptModel)]) ### In case of "Object of type int64 is not JSON serializable" refer to here: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable

    ### Call Grumpy for his assesment of the data QC
    # Grumpy - full assesment
    message_text = [{"role":"system","content":grumpyRole},
                    {"role":"user","content":query}]

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    total_tokens = sum(len(tokenizer.encode(message['content'])) for message in message_text)
    lgr.info(f"Total tokens in the prompt: {total_tokens}; maxTok for model is {maxTok}")
    requestedCompletionTokens = min([maxTok - total_tokens, 4096]) # some references to 4k: https://community.openai.com/t/how-the-max-tokens-are-considered/313514/4 and https://community.openai.com/t/max-tokens-chat-completion-gpt4o/758066

    if requestedCompletionTokens < 0:
        lgr.error(f"The prompt is too long to fit into the '{gptModel}' model. Please shorten the prompt or increase the max number of tokes assigned (if possible).")
        response = f"The prompt is too long to fit into the '{gptModel}' model. Please shorten the prompt or increase the max number of tokes assigned (if possible). DEBUG info: Total tokens in the prompt: {total_tokens}; max_tokens set in grumpyConnect function: {max_tokens}; tokens assigned per model: {getMaxTokenPerModel(gptModel)}"
        if saveResponse:
            outfile = open(outfileName, "w")
            outfile.write(response)
            outfile.close()
        else:
            return response
    else:
        try:
            # print("#!#!#gptModel: ", gptModel)  # DEBUG
            # print("#!#!#message_text: ", message_text)  # DEBUG
            # print("#!#!#temperature: ", temperature)  # DEBUG
            # print("#!#!#requestedCompletionTokens: ", requestedCompletionTokens)  # DEBUG
            # print("#!#!#top_p: ", top_p)  # DEBUG
            # print("#!#!#frequency_penalty: ", frequency_penalty)  # DEBUG
            # print("#!#!#presence_penalty: ", presence_penalty)  # DEBUG
            # print("#!#!#stop: ", None)  # DEBUG
            completion = client.chat.completions.create(
                                                          model=gptModel,
                                                          messages = message_text,
                                                          temperature=temperature,
                                                          max_tokens=requestedCompletionTokens,
                                                          top_p=top_p,
                                                          frequency_penalty=frequency_penalty,
                                                          presence_penalty=presence_penalty,
                                                          stop=None
                                                        )
            # print(completion.choices[0].message.content)
            if saveResponse:
                outfile = open(outfileName, "w")
                outfile.write(completion.choices[0].message.content)
                outfile.close()
                lgr.info("The full assesment was saved to the file '{}'.".format(outfileName))
            else:
                return completion.choices[0].message.content

        except AuthenticationError as e:
            lgr.error(f"Failed to authenticate with OpenAI API. Please check your API key and permissions. Error details: {e}")
            if saveResponse:
                outfile = open(outfileName, "w")
                outfile.write("Failed to authenticate with OpenAI API. Please check your API key and permissions. Also, most likely the API key is expired.")
                outfile.close()
            else:
                return "Failed to authenticate with OpenAI API. Please check your API key and permissions. Also, most likely the API key is expired."
        except Exception as e:
            lgr.error(f"An unexpected error occurred: {e}")
            if saveResponse:
                outfile = open(outfileName, "w")
                outfile.write("An unexpected error occurred while calling the OpenAI API.")
                outfile.close()
            else:
                return "An unexpected error occurred while calling the OpenAI API."