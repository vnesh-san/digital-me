# Digital Me Dataset Tools

This repository contains utilities to extract text from PDF and EPUB books and
prepare a dataset for OpenAI fine-tuning. Extraction and splitting are powered
by [LangChain](https://python.langchain.com/) loaders and text splitters so
multiple books or entire directories can be processed efficiently.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Create a YAML configuration file specifying the author, output dataset file and
books with page ranges to include. You can also control how the text is split
into chunks using the optional `chunk_size` and `overlap` settings. `chunk_size`
is measured in characters (default `1024`) and words are never split across
chunks. The `overlap` value determines the fraction of each chunk's characters
that appear at the start of the next chunk (default `0.25`). A report of
the planned splits is written before the dataset is generated. An example
configuration is provided in `example_config.yaml`.

Run the dataset generation script:

```bash
python generate_dataset.py path/to/config.yaml
```

Paths in the configuration may reference individual files or entire directories.
When a directory is provided, all PDF and EPUB books inside are loaded using
LangChain's `DirectoryLoader`.

The script shows progress for each book and the pages processed using `tqdm` and
leverages all available CPU cores to convert multiple books in parallel. Each
book is split into overlapping chunks, and a report describing the split is
saved alongside the dataset.

Each entry in the output `.jsonl` will contain a `prompt` instructing the model
to write in the style of the given author and a `completion` holding the text
extracted from the book pages.
