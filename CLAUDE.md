# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GRUMPY (Generative Research Utility Model in Python)** is a bioinformatics analysis platform that uses Large Language Models (LLMs) to perform Biological Context Analysis (BCA) on genomic research data. It's a hybrid project combining a CLI tool, Python library, and web application (Streamlit).

**Key Technologies:** Python 3.9+, OpenAI/Azure/Ollama APIs, Streamlit, pandas, gseapy, pybedtools
**Version:** 0.3.0-alpha
**License:** MIT

## Architecture

The project consists of two main components:

1. **Main CLI Tool (`grumpy/`)** - Core analysis modules for bioinformatics workflows
2. **Web Application (`grumpy_bot/`)** - Streamlit-based chatbot with RAG capabilities

### Core CLI Modules

- **QC**: Quality Control analysis for ChIP-seq, ATAC-seq, CUT&RUN protocols
- **PE**: Pathway Enrichment analysis (GSEA integration)
- **DPK**: Differentially Peaks analysis for chromatin dynamics
- **MEA**: Motif Enrichment Analysis (Homer tool integration)
- **DEG**: Differentially Expressed Genes (placeholder)
- **chat**: Interactive LLM conversation mode
- **decode**: HTML to text conversion utility

### Key Components

- `grumpy/cli.py` - Main CLI entry point with 7 subcommands
- `grumpy/connect.py` - Multi-provider LLM connection handler (OpenAI/Azure/Ollama)
- `grumpy/modules/` - Analysis modules (qc.py, gsea.py, dpk.py, mea.py)
- `grumpy/utils/` - Utilities (logging, parsing, tokenization, HTML processing)
- `templates/` - Protocol-specific LLM prompt templates
- `grumpy_bot/` - Streamlit web application with vector database

## Development Commands

### Installation & Setup

```bash
# Quick install from GitHub
pip install git+https://github.com/stjude-biohackathon/KIDS24-team18.git

# From source (recommended for development)
git clone https://github.com/stjude-biohackathon/KIDS24-team18.git
cd KIDS24-team18
python -m venv grumpy_env
source grumpy_env/bin/activate  # Linux/macOS
# OR: grumpy_env\Scripts\activate  # Windows
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### CLI Usage

```bash
# Basic help
grumpy -h

# Quality Control analysis
grumpy QC -i /path/to/qc/data -o /path/to/output -k .env --apiType openai --gptModel gpt-4o -p chipseq --inputType multiqc

# Pathway Enrichment analysis
grumpy PE -i ./data/gene_list.csv --outfilesPrefix ./output/results --context ./context.txt --inputType genes -p gsea -r pathways -k .env --apiType openai --gptModel gpt-4o

# Differentially Peaks analysis
grumpy DPK -i /path/to/peaks -o /path/to/output --context ./biological_context.txt -k .env --apiType openai

# Interactive chat mode
grumpy chat -k .env --apiType openai --gptModel gpt-4o
```

### Web Application

```bash
# Install web app dependencies
pip install -r grumpy_bot/requirements.txt

# Additional system dependency for OCR
# macOS: brew install tesseract
# Ubuntu: sudo apt-get install tesseract-ocr
# Windows: Download from GitHub

# Run the chatbot
streamlit run grumpy_bot/chat.py

# Run specific analysis pages
streamlit run grumpy_bot/qc.py
streamlit run grumpy_bot/pathways.py
```

### Testing

**Note:** No automated test framework is currently implemented. Testing is done manually using example datasets.

```bash
# Run example workflow
chmod +x examples/run_grumpy_pe_openai.sh
sh examples/run_grumpy_pe_openai.sh

# Test with different example datasets
ls examples/  # PE_examples/, DPK_examples/, MultiQC_examples/
```

## Code Structure & Patterns

### LLM Integration Pattern

All analysis modules follow this pattern:
1. Parse input data using `grumpy/utils/report_parsing.py`
2. Load protocol-specific templates from `templates/`
3. Call `grumpy.connect.grumpyConnect()` with constructed prompts
4. Generate HTML/text reports using `grumpy/utils/html_processing.py`

### API Configuration

The tool supports multiple LLM providers:
- **OpenAI**: Direct API (requires OPENAI_API_KEY)
- **Azure**: St. Jude dedicated instance (requires Azure config)
- **Ollama**: Local models (llama3, meditron, medllama2)

API keys can be provided via:
- `.env` file (recommended)
- Environment variables
- File path (default: `/research_jude/rgs01_jude/groups/cab/projects/Control/common/cab_epi/APIKEY/key`)

### Adding New Analysis Modules

1. Create new module in `grumpy/modules/new_module.py`
2. Add CLI parser in `grumpy/cli.py` `parseArgs()` function
3. Create prompt template in `templates/new_module.txt`
4. Follow existing patterns from `grumpy/modules/qc.py` (simplest example)

### Template System

Protocol-specific prompts are stored in `templates/`:
- `qc.chipseq.txt` - ChIP-seq QC evaluation instructions
- `qc.cutandrun.txt` - CUT&RUN specific guidelines
- `qc.dpk.txt` - Differentially peaks analysis
- `grumpy_template.html` - HTML report template

Templates contain domain expertise and evaluation criteria that guide LLM analysis.

## Important Considerations

### Token Management
- Uses `tiktoken` for accurate token counting
- Automatically prevents token overflow errors
- Supports text compression for large reports
- Different models have different token limits (handled in `grumpy/utils/tokenization.py`)

### Bioinformatics Context
- Designed for St. Jude research workflows
- Handles standard bioinformatics formats (MultiQC, Automapper, GSEA)
- Protocol-aware analysis (ChIP-seq vs ATAC-seq vs CUT&RUN)
- Biological context files provide experimental background

### Output Formats
- HTML reports for web viewing (using Jinja2 templates)
- Plain text summaries for programmatic use
- Optional intermediate files for debugging
- Compressed text for very large outputs

### Development Status
- Alpha version (0.3.0-alpha)
- Active development, recent focus on bug fixes
- Research-grade tool, not production-hardened
- No automated testing infrastructure yet

## File Organization

```
KIDS24-team18/
├── grumpy/                    # Main Python package
│   ├── cli.py                 # CLI entry point (265 lines)
│   ├── connect.py             # LLM connection handler (207 lines)
│   ├── modules/               # Analysis modules
│   │   ├── qc.py             # Quality Control (165 lines)
│   │   ├── gsea.py           # GSEA analysis (489 lines, most complex)
│   │   ├── dpk.py            # Differentially Peaks (115 lines)
│   │   └── mea.py            # Motif Enrichment (142 lines)
│   ├── utils/                 # Utility functions
│   └── data/                  # Reference data (MSigDB)
├── grumpy_bot/               # Streamlit web application
│   ├── chat.py               # Main chat interface
│   ├── qc.py                 # QC analysis UI
│   ├── pathways.py           # Pathway analysis UI
│   └── vectorstore.py        # RAG vector database
├── templates/                # LLM prompt templates
├── examples/                 # Sample datasets and workflows
├── setup.py                 # Package configuration
└── requirements.txt         # Core dependencies
```

## Common Development Tasks

### Debugging LLM Interactions
- Use `--hidden false` to see intermediate prompt files
- Check `grumpy/utils/tokenization.py` for token limits
- Review templates in `templates/` for prompt engineering
- Enable detailed logging (automatic in CLI mode)

### Working with Different Protocols
- Protocol templates are in `templates/qc.{protocol}.txt`
- Protocol-specific logic is in individual modules
- Use `--inputType` to specify data format (multiqc, automapper, etc.)

### Extending Web Application
- Streamlit pages are in `grumpy_bot/pages/`
- Vector database code is in `grumpy_bot/vectorstore.py`
- RAG functionality uses langchain + Chroma
- Document processing supports PDF, text, and other formats

This codebase combines domain expertise in bioinformatics with modern LLM capabilities to provide intelligent analysis of genomic data. The modular design allows for easy extension with new analysis types and protocols.