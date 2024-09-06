#!/bin/bash

# Define variables for paths and parameters
GRUMPY_SCRIPT=grumpy
# Update the API Key File Path
API_KEY_FILE=.env
API_TYPE=openai
MODEL_NAME=gpt-4o #choose from gpt-4o, gpt-4o-mini, llama3, meditron, medllama2

INPUT_FILE_2="./examples/PE_examples/GMT_h.all.v2023.2.Hs.symbols"
OUTFILES_PREFIX_2="./examples/PE_examples/GSE164073_sclera"
CONTEXT_FILE_2="./examples/PE_examples/context.txt"
PROTOCOL_2="gsea"
REPORT_2="gseareport"
MODE="PE"
INPUT_TYPE="pathways"

# Run Grumpy on a test dataset
$GRUMPY_SCRIPT \
$MODE \
-i $INPUT_FILE_2 \
--outfilesPrefix $OUTFILES_PREFIX_2 \
--context $CONTEXT_FILE_2 \
--inputType $INPUT_TYPE \
-p $PROTOCOL_2 \
-r $REPORT_2 \
-k $API_KEY_FILE \
--apiType $API_TYPE \
--gptModel $MODEL_NAME