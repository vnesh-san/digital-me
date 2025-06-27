import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
from tqdm import tqdm
from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredEPubLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_pdf_text(path, start, end, position=0):
    loader = PyPDFLoader(path)
    pages = loader.load()
    start = max(start, 1)
    end = min(end, len(pages))
    extracted = []
    for doc in tqdm(pages[start - 1:end], desc=os.path.basename(path), position=position, leave=False):
        extracted.append(doc.page_content)
    return extracted


def extract_epub_text(path, start, end, position=0, words_per_page=700):
    loader = UnstructuredEPubLoader(path)
    docs = loader.load()
    text = " ".join(doc.page_content for doc in docs)
    words = text.split()
    pages = [" ".join(words[i:i + words_per_page]) for i in range(0, len(words), words_per_page)]
    start = max(start, 1)
    end = min(end, len(pages))
    extracted = []
    for i in tqdm(range(start - 1, end), desc=os.path.basename(path), position=position, leave=False):
        extracted.append(pages[i])
    return extracted


def extract_directory_text(path, position=0):
    pdf_loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    epub_loader = DirectoryLoader(path, glob="**/*.epub", loader_cls=UnstructuredEPubLoader)
    docs = []
    for loader in [pdf_loader, epub_loader]:
        for doc in tqdm(loader.load(), desc=os.path.basename(path), position=position, leave=False):
            docs.append(doc.page_content)
    return "\n".join(docs)


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

        if os.path.isdir(path):
            text = extract_directory_text(path, position=position)
            pages = [text]
            start = end = 1
        else:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                pages = extract_pdf_text(path, start, end, position=position)
            elif ext == ".epub":
                pages = extract_epub_text(path, start, end, position=position)
            else:
                raise ValueError(f"Unsupported file type: {path}")

        text = "\n".join(pages)
        words = text.split()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * overlap),
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_text(text)

        records = []
        splits = []
        start_char = 0
        overlap_chars = int(chunk_size * overlap)
        for idx, chunk in enumerate(chunks, start=1):
            end_char = start_char + len(chunk)
            records.append({"prompt": prompt, "completion": chunk.strip()})
            splits.append({"chunk_index": idx, "start_char": start_char, "end_char": end_char})
            start_char = end_char - overlap_chars

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
