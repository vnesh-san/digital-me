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
    chunk_size = int(config.get("chunk_size", 1024))
    overlap = float(config.get("overlap", 0.25))
    report_file = config.get("report_file", os.path.splitext(output)[0] + "_report.json")

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

        text = "\n".join(pages)
        words = text.split()

        records = []
        splits = []
        i = 0
        chunk_index = 1
        while i < len(words):
            char_count = 0
            j = i
            chunk_words = []
            while j < len(words):
                word = words[j]
                add_len = len(word) if char_count == 0 else len(word) + 1
                if char_count + add_len > chunk_size:
                    break
                char_count += add_len
                chunk_words.append(word)
                j += 1

            if not chunk_words:
                chunk_words.append(words[i])
                j = i + 1

            record = {"prompt": prompt, "completion": " ".join(chunk_words).strip()}
            records.append(record)
            splits.append({"chunk_index": chunk_index, "start_word": i + 1, "end_word": j})
            chunk_index += 1

            if j >= len(words):
                break

            overlap_words = int(len(chunk_words) * overlap)
            if overlap_words >= len(chunk_words):
                overlap_words = len(chunk_words) - 1
            i = j - overlap_words

        report_entry = {
            "book": path,
            "start_page": start,
            "end_page": end,
            "total_words": len(words),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "num_chunks": len(splits),
            "chunks": splits,
        }

        return records, report_entry

    all_records = []
    report = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
        futures = [executor.submit(process_book, book, idx + 1) for idx, book in enumerate(books)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Books"):
            records, rep = f.result()
            all_records.extend(records)
            report.append(rep)

    with open(report_file, "w", encoding="utf-8") as rep_f:
        json.dump(report, rep_f, ensure_ascii=False, indent=2)

    with open(output, "w", encoding="utf-8") as out_f:
        for record in all_records:
            json_line = json.dumps(record, ensure_ascii=False)
            out_f.write(json_line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create fine-tuning dataset from books.")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
