import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
from PyPDF2 import PdfReader
from ebooklib import epub
from bs4 import BeautifulSoup
from tqdm import tqdm


def extract_pdf_text(path, start, end, position=0):
    reader = PdfReader(path)
    num_pages = len(reader.pages)
    extracted = []
    start = max(start, 1)
    end = min(end, num_pages)
    page_range = range(start - 1, end)
    for i in tqdm(page_range, desc=os.path.basename(path), position=position, leave=False):
        page = reader.pages[i]
        text = page.extract_text() or ""
        extracted.append(text)
    return extracted


def extract_epub_text(path, start, end, position=0, words_per_page=700):
    book = epub.read_epub(path)
    texts = []
    for item in book.get_items():
        if item.get_type() == epub.EpubHtml:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            texts.append(soup.get_text())
    words = " ".join(texts).split()
    pages = [" ".join(words[i:i + words_per_page]) for i in range(0, len(words), words_per_page)]
    start = max(start, 1)
    end = min(end, len(pages))
    extracted = []
    page_range = range(start - 1, end)
    for i in tqdm(page_range, desc=os.path.basename(path), position=position, leave=False):
        extracted.append(pages[i])
    return extracted


def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    author = config.get("author", "")
    output = config.get("output_file", "dataset.jsonl")
    books = config.get("books", [])

    prompt = f"Write a passage in the style of {author}."

    def process_book(book, position):
        path = book["path"]
        start = int(book.get("start_page", 1))
        end = int(book.get("end_page", 1))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Book not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            pages = extract_pdf_text(path, start, end, position=position)
        elif ext == ".epub":
            pages = extract_epub_text(path, start, end, position=position)
        else:
            raise ValueError(f"Unsupported file type: {path}")

        records = []
        for text in pages:
            record = {"prompt": prompt, "completion": text.strip()}
            records.append(record)
        return records

    all_records = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
        futures = [executor.submit(process_book, book, idx + 1) for idx, book in enumerate(books)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Books"):
            all_records.extend(f.result())

    with open(output, "w", encoding="utf-8") as out_f:
        for record in all_records:
            json_line = json.dumps(record, ensure_ascii=False)
            out_f.write(json_line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create fine-tuning dataset from books.")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
