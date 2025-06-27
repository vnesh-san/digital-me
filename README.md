# Digital Me Dataset Tools

This repository contains utilities to extract text from PDF and EPUB books and
prepare a dataset for OpenAI fine-tuning.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Create a YAML configuration file specifying the author, output dataset file, and
books with page ranges to include. An example configuration is provided in
`example_config.yaml`.

Run the dataset generation script:

```bash
python generate_dataset.py path/to/config.yaml
```

Each entry in the output `.jsonl` will contain a `prompt` instructing the model
to write in the style of the given author and a `completion` holding the text
extracted from the book pages.
