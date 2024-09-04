#!/bin/bash

# Define variables for paths and parameters
GRUMPY_SCRIPT=./grumpy/grumpy.py
# Update the API Key File Path
API_KEY_FILE=.env
API_TYPE=openai

INPUT_FILE_2="./examples/RNK_GSE164073_diff.sclera_CoV2_vs_sclera_mock.regulation.rank_/GMT_h.all.v2023.2.Hs.symbols"
OUTFILES_PREFIX_2="GSE164073_sclera"
CONTEXT_FILE_2="context.txt"
PROTOCOL_2="gsea"
REPORT_2="gseareport"

# Run Grumpy on a test dataset
python $GRUMPY_SCRIPT \
-i $INPUT_FILE_2 \
--outfilesPrefix $OUTFILES_PREFIX_2 \
--context $CONTEXT_FILE_2 \
-p $PROTOCOL_2 \
-r $REPORT_2 \
-k $API_KEY_FILE \
--apiType $API_TYPE \
--gptModel gpt-4o
