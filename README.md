# KIDS24-Team 18: Grumpy (Generative Research Utility Model in Python)

Grumpy (Generative Research Utility Model in Python) is a tool designed to conduct Biological Context Analysis (BCA).
It utilizes Large Language Models (LLMs) such as OpenAI's GPT-4 (St. Jude Dedicated Instance) or other models like Llama
from Meta.

Grumpy is composed of a set of LLM Agents, each customized for specific tasks within the realm of biological data
analysis. These agents are AI-driven entities designed to perform specialized tasks or interact with users by
leveraging the capabilities of large language models to understand and generate human-like text.

## What is Biological Context Analysis (BCA)?

Biological Context Analysis, or BCA for short, refers to the process of analyzing and interpreting biological data with
an emphasis on its relevance to biological functions and systems. Grumpy facilitates BCA by utilizing advanced LLMs
to provide context-aware insights and interpretations of complex biological datasets.

## Installation

To set up Grumpy, you need to create a Python environment and install the required dependencies. Follow the steps below:

### Quick Install

To use grumpy without getting grumpy, you can install directly from github.

```bash
pip install git+https://github.com/stjude-biohackathon/KIDS24-team18.git
```

Now, you can use grumpy on your command line.

```console
grumpier@MacOSx ~/Documents/Git-Repos/st-jude-hackathon/KIDS24-team18/ grumpy -h
###     [2024-09-06 00:24:22,571] cli.py:37: parseArgs INFO: Current working directory: /Users/shutchens/Documents/Git-Repos/st-jude-hackathon/KIDS24-team18
###     [2024-09-06 00:24:22,572] cli.py:38: parseArgs INFO: Command used to run the program: python /Users/shutchens/Documents/Git-Repos/st-jude-hackathon/KIDS24-team18/grumpy_env/bin/grumpy -h
usage: grumpy [-h] {QC,PE,MEA,DEG,DPK,chat,decode} ...

positional arguments:
  {QC,PE,MEA,DEG,DPK,chat,decode}
                        Availible modes/tools/modules
    QC                  Run evaluation of the Quality Control (QC) for standard report or Automapper output/summary QC table.
    PE                  Run evaluation of the Pathway Enrichment (PE) analyses for either GSEA results, or typical Pathway Enrichment.
    MEA                 Run evaluation of the Motif Enrichment Analysis (MEA) for the standard reports from Homer tool.
    DEG                 Run evaluation of the Differentially Expressed Genes (DEG) for the typical DEG tables from tools such as limma, DEseq2 etc.
    DPK                 Run evaluation of the Differentially Peaks (DPK), so either differentially binding regions from protocols like ChIP-seq or differentially
                        accessible ones from protocols like ATAC-seq.
    chat                Nice chat with the Grumpy AI
    decode              Small tool to decode / extract the information from the HTML file.

optional arguments:
  -h, --help            show this help message and exit
```

### Use the script installing to your user `bin`

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/stjude-biohackathon/KIDS24-team18.git
   cd KIDS24-team18
   ```

2. **Create a Virtual Environment**:

   ```bash
   python -m venv grumpy_env
   ```

3. **Activate the Virtual Environment**:
   - On Windows:

     ```bash
     .\grumpy_env\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source grumpy_env/bin/activate
     ```

4. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Run Grumpy**:
   Once the dependencies are installed, you can start using grumpy!

## Usage

Grumpy is designed to be flexible and extensible, allowing you to define and run LLM Agents tailored to your specific
research needs. Detailed usage instructions will be provided in future updates.

### Running an example for PE (Pathway Enrichment)

To run grumpy on an example dataset `PE_examples` with an `OPENAI API Key`, use the below in
the top level repository folder.

:warning: Make sure you have a file that includes your API Key and update it in `examples/run_grumpy.sh`.

```bash
chmod +x examples/run_grumpy_pe_openai.sh
sh examples/run_grumpy_pe_openai.sh
```

## Running Grumpy chatbot

Please follow instructions in `grumpybot/readme.md` !


## License

This project is licensed under [MIT](./LICENSE).

## Contributing

Contributions to Grumpy are welcome! If you have suggestions, bug reports, or would like to contribute code, please
open an issue or submit a pull request.

## Contact

For any inquiries, please contact:

## Team Members
