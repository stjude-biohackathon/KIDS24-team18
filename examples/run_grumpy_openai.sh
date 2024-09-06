#!/bin/bash

# Define variables for paths and parameters
GRUMPY_SCRIPT=./grumpy/grumpy.py
# Update the API Key File Path
API_KEY_FILE=.env
API_TYPE=openai
MODEL_NAME=gpt-4o #choose from gpt-4o, gpt-4o-mini, llama3, meditron, medllama2

INPUT_FILE_2="./examples/RNK_GSE164073_diff.sclera_CoV2_vs_sclera_mock.regulation.rank_/GMT_h.all.v2023.2.Hs.symbols"
OUTFILES_PREFIX_2="./examples/GSE164073_sclera"
CONTEXT_FILE_2="./examples/context.txt"
PROTOCOL_2="gsea"
REPORT_2="gseareport"
MODE="PE"

# Run Grumpy on a test dataset
python $GRUMPY_SCRIPT \
$MODE \
-i $INPUT_FILE_2 \
--outfilesPrefix $OUTFILES_PREFIX_2 \
--context $CONTEXT_FILE_2 \
-p $PROTOCOL_2 \
-r $REPORT_2 \
-k $API_KEY_FILE \
--apiType $API_TYPE \
--gptModel $MODEL_NAME