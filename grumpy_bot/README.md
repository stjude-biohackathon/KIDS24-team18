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

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/stjude-biohackathon/KIDS24-team18.git
   cd KIDS24-team18
   ```

2. **Create a Virtual Environment**:

   ```bash
   python -m venv grumpy_chatbot
   ```

3. **Activate the Virtual Environment**:
   - On Windows:

     ```bash
     .\grumpy_chatbot\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source grumpy_chatbot/bin/activate
     ```

4. **Install Dependencies**:

   Install tesseract for the code to work - https://stackoverflow.com/questions/50951955/pytesseract-tesseractnotfound-error-tesseract-is-not-installed-or-its-not-i

   ```bash
   pip install -r grumpy_bot/requirements.txt
   ```

5. **Run Grumpy**:
   Once the dependencies are installed, you can start using grumpy!

## Usage

Grumpy is designed to be flexible and extensible, allowing you to define and run LLM Agents tailored to your specific
research needs. Detailed usage instructions will be provided in future updates.


To run grumpy, start the chatbot using the below command -

   ```bash
   streamlit run grumpy_bot/chat.py
   ```

**Note:** Upload an example dataset with an `OPENAI API Key`, use the below file in the top level repository folder.

`examples/GSE247821.StaticReport.20240829.pdf`

## License

This project is licensed under [MIT](./LICENSE).

## Contributing

Contributions to Grumpy are welcome! If you have suggestions, bug reports, or would like to contribute code, please
open an issue or submit a pull request.

## Contact

For any inquiries, please contact:

## Team Members
