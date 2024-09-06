#!/bin/bash

# Define variables for paths and parameters
GRUMPY_SCRIPT=grumpy
# Update the API Key File Path
API_KEY_FILE=.env
API_TYPE=openai
MODEL_NAME=gpt-4o-mini #choose from gpt-4o, gpt-4o-mini, llama3, meditron, medllama2

INPUT_FILE_2="./examples/DPK_examples"
OUTFILES_PREFIX_2="GSE247821"
CONTEXT_FILE_2="./examples/DPK_examples/biological_context_example"
PROTOCOL_2="atacseq"
MODE="DPK"

# Run Grumpy on the updated dataset
$GRUMPY_SCRIPT \
$MODE \
-i $INPUT_FILE_2 \
--outfilesPrefix $OUTFILES_PREFIX_2 \
--context $CONTEXT_FILE_2 \
-p $PROTOCOL_2 \
-k $API_KEY_FILE \
--apiType $API_TYPE \
--gptModel $MODEL_NAME
