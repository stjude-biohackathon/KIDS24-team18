# KIDS24-Team 18: Grumpy (Generative Research Utility Model in Python)

Grumpy (Generative Research Utility Model in Python) is a tool designed to conduct Biological Context Analysis (BCA).
It utilizes Large Language Models (LLMs) such as OpenAI's GPT-4 (St. Jude Dedicated Instance) or other models like Llama
from Meta.

Grumpy is composed of a set of LLM Agents, each customized for specific tasks within the realm of biological data analysis. These agents are AI-driven entities designed to perform specialized tasks or interact with users by leveraging the capabilities of large language models to understand and generate human-like text.

## What is Biological Context Analysis (BCA)?

Biological Context Analysis, or BCA for short, refers to the process of analyzing and interpreting biological data with
an emphasis on its relevance to biological functions and systems. Grumpy facilitates BCA by utilizing advanced LLMs
to provide context-aware insights and interpretations of complex biological datasets.

## Installation

To set up Grumpy, you need to create a Python environment and install the required dependencies. Follow the steps below:

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd grumpy
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
   Once the dependencies are installed, you can start using Grumpy to perform Biological Context Analysis.

## Usage

Grumpy is designed to be flexible and extensible, allowing you to define and run LLM Agents tailored to your specific
research needs. Detailed usage instructions will be provided in future updates.

### Running an example

To run grumpy on an example dataset `RNK_GSE164073_diff.sclera_CoV2_vs_sclera_mock.regulation.rank_`, use the below in
the top level repository folder.

:warning: Make sure you have a file that includes your API Key and update it in `examples/run_grumpy.sh`.

```bash
chmod +x examples/run_grumpy.sh
sh examples/run_grumpy.sh
```

## License

This project is licensed under [MIT](./LICENSE).

## Contributing

Contributions to Grumpy are welcome! If you have suggestions, bug reports, or would like to contribute code, please
open an issue or submit a pull request.

## Contact

For any inquiries, please contact:
